#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque  # optional for fixed-size history
import threading
import time
import mediapipe as mp
import torch
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Bool
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from tf.transformations import quaternion_from_matrix

print("YOLO running on:", "GPU" if torch.cuda.is_available() else "CPU")
print("OpenCV CUDA available:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

class Realsense:
    def __init__(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config   = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(self.config)

        # intrinsics once
        color_stream   = profile.get_stream(rs.stream.color)
        video_profile  = color_stream.as_video_stream_profile()
        self.intrinsics = video_profile.get_intrinsics()

        # shared state
        self.lock        = threading.Lock()
        self.running     = True
        self.color_frame = None
        self.depth_frame = None

        # start grab thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            frames   = self.pipeline.wait_for_frames()      # blocks at ~33 ms
            aligned  = self.align.process(frames)
            color    = aligned.get_color_frame()
            depth    = aligned.get_depth_frame()
            if not color or not depth:
                continue

            arr = np.asanyarray(color.get_data())
            with self.lock:
                # always overwrite with the very latest
                self.color_frame = arr
                self.depth_frame = depth

    def get_frame(self):
        # wait until the grab thread has produced at least one pair
        while True:
            with self.lock:
                cf = self.color_frame
                df = self.depth_frame
            if cf is not None and df is not None:
                # return a copy so downstream can’t clobber our buffer
                #return cf.copy(), df
                return cf, df

            #time.sleep(0.001)

    def get_intrinsics(self):
        return self.intrinsics

    def close(self):
        # stop grab thread and join, then stop the pipeline
        self.running = False
        self.thread.join()
        self.pipeline.stop()


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.lock = threading.Lock()
        self.running = True
        # self.thread = threading.Thread(target=self.run, daemon=True)
        # self.thread.start()

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        selected = [0, 4, 9]
        keypoints = np.zeros((3, 4))
        palm_confidence = 0.000000000001 # Default confidence (no hand detected)
        if results.multi_handedness:
            palm_confidence = results.multi_handedness[0].classification[0].score

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            for hand_landmark in results.multi_hand_landmarks:
                for i, idx in enumerate(selected):
                    x = hand_landmark.landmark[idx].x
                    y = hand_landmark.landmark[idx].y
                    z = hand_landmark.landmark[idx].z
                    keypoints[i] = [x, y, z, palm_confidence]
                break  # Only process one hand
        else:
            keypoints[:,3] = 0.000000000001
            landmarks = None

        return landmarks, keypoints #hand_landmarks, palm_confidence

    # def run(self):
    #     while self.running:
    #         time.sleep(0.001)  # Avoid excessive CPU usage
    #
    # def stop(self):
    #     self.running = False
    #     self.thread.join()

class TomatoDetector:
    def __init__(self):
        self.model = YOLO('/home/carpos/catkin_ws/src/perception_node/models/keypoints_new.pt')
        self.lock = threading.Lock()
        self.running = True
        # self.thread = threading.Thread(target=self.run, daemon=True)
        # self.thread.start()

    def process_frame(self, image):
        results = self.model.predict(source=image, conf=0.7, verbose=False, save=False, save_txt=False, show=False)[0]
        return results

    # def run(self):
    #     while self.running:
    #         time.sleep(0.001)  # Avoid excessive CPU usage
    #
    # def stop(self):
    #     self.running = False
    #     self.thread.join()

class GraspDetector:
    def __init__(self, show_frame=True):
        self.show_frame = show_frame
        self.last_frame = None
        self.keypoint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.process_times = []
        self.last_keypoint_set = None           # (each keypoint is a dict with 3D coordinates and confidence)
        self.hand_3d_points = None
        self.hand_pose = np.eye(4)  # Start with identity
        self.scale_factor = 0.1                 # Adjust as needed for real-world scaling of hand keypoints
        self.tomato_reference_frame = np.eye(4)
        self.hand_reference_frame = np.eye(4)
        # Optional: store a history of keypoint sets with a maximum length to avoid unbounded memory usage.
        #self.keypoints_history = deque(maxlen=history_size)
        self.camera = Realsense()
        self.intrinsics = self.camera.get_intrinsics()
        self.tomato_detector = TomatoDetector()
        self.hand_detector = HandDetector()  # Initialize hand detector thread
        # self.thread = threading.Thread(target=self.run, daemon=True)
        # self.thread.start()
        # Initialize ROS node and ROS publisher for tomato pose
        self.enabled = False
        rospy.init_node('grasp_detector', anonymous=True)
        self.hand_pub = rospy.Publisher('/hand_pose', PoseStamped, queue_size=1)
        self.keypoints_pub = rospy.Publisher('/keypoints', Float32MultiArray, queue_size=1)
        #self.image_pub = rospy.Publisher('/detections', PoseStamped, queue_size=1)
        self.tomato_pub = rospy.Publisher('/tomato_pose', PoseStamped, queue_size=1)
        self.enable_sub = rospy.Subscriber('/grasp_enable', Bool, self._enable_callback, queue_size=1)
        self.image_pub = rospy.Publisher('/rs_image', CompressedImage, queue_size=1)

    def deproject_to_3d(self, x, y, depth_frame, intrinsics):
        frame_width = depth_frame.width
        frame_height = depth_frame.height
        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
            return None
        
        region_size = 20
        depths = []
        for dx in range(-region_size // 2, region_size // 2 + 1):
            for dy in range(-region_size // 2, region_size // 2 + 1):
                depth = depth_frame.get_distance(x + dx, y + dy)
                if depth > 0:
                    depths.append(depth)

        if depths is None:
            depth = min(depths)  #depth_frame.get_distance(x, y)

        if depth > 0:
            return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        return None

    def get_average_depth(self, x1, x2, y1, y2, depth_frame):
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        region_size = 5
        depths = []
        for dx in range(-region_size // 2, region_size // 2 + 1):
            for dy in range(-region_size // 2, region_size // 2 + 1):
                depth = depth_frame.get_distance(center_x + dx, center_y + dy)
                if depth > 0:
                    depths.append(depth)
        if depths:
            return sum(depths) / len(depths)
        return 0

    def calculate_object_size(self, x1, x2, y1, y2, depth_frame, intrinsics):
        depth = self.get_average_depth(x1, x2, y1, y2, depth_frame)
        if depth <= 0:
            return None
        fx = intrinsics.fx
        pixel_length = abs(x2 - x1)
        object_size = (pixel_length * depth) / fx
        return object_size

    def get_hand_pose(self, hand_landmarks, image, width, height):
        # get the cordinates of the specific landmarks 0, 1, 5 and 17
        j = 0
        pw = np.array([hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])
        j = 1
        pbth = np.array([hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])
        j = 5
        pbi = np.array([hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])
        j = 17
        pbs = np.array([hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])

        # compute the vectors to the finger bases
        p1 = (pbth - pw) / np.linalg.norm(pbth - pw)
        p2 = (pbi - pw) / np.linalg.norm(pbi - pw)
        p3 = (pbs - pw) / np.linalg.norm(pbs - pw)


        # z will be towards the mid vector from those vectors (giving more weight to the thumb)
        z = (0.5 * p1 + 0.25 * p2 + 0.25 * p3)
        z = z / np.linalg.norm(z)
        z.shape = (3, 1)

        # x will be towards the thumb
        x = (np.identity(3) - z @ z.T) @ p1
        x = x / np.linalg.norm(x)
        x.shape = (3, 1)

        # y will be according to the right hand rule
        y = np.cross(z.T, x.T)
        y = y / np.linalg.norm(y)
        y.shape = (3, 1)

        # create the rotation matrix
        R = np.hstack((x, y, z))

        # print the lines of the frame to the image
        w_pixel = np.array([int(pw[0] * width), int(pw[1] * height)])
        x_pixel = np.array([int(x[0] * 100), int(x[1] * 100)])
        y_pixel = np.array([int(y[0] * 100), int(y[1] * 100)])
        z_pixel = np.array([int(z[0] * 100), int(z[1] * 100)])
        cv2.line(image, w_pixel, w_pixel + x_pixel, color=(0, 0, 200), thickness=6)
        cv2.line(image, w_pixel, w_pixel + y_pixel, color=(0, 200, 0), thickness=6)
        cv2.line(image, w_pixel, w_pixel + z_pixel, color=(200, 0, 0), thickness=6)

        return w_pixel, R

    def calculate_reference_frame(self, p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        z_axis = p1 - p2
        z_axis /= np.linalg.norm(z_axis)
        zz_t = np.outer(z_axis, z_axis)
        I = np.eye(3)
        nul = I - zz_t
        x_axis = nul @ (p1 - p3)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        T = np.eye(4)
        T[0:3, 0] = x_axis
        T[0:3, 1] = y_axis
        T[0:3, 2] = z_axis
        T[0:3, 3] = p1  # origin
        return T
    
    def calculate_reference_frame_2points(self, p1, p2):
        p1, p2 = np.array(p1), np.array(p2)
        z_axis = p1 - p2
        z_axis /= np.linalg.norm(z_axis)
        zz_t = np.outer(z_axis, z_axis)
        I = np.eye(3)
        nul = I - zz_t
        x_axis = nul @ np.array([1,0,0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        T = np.eye(4)
        T[0:3, 0] = x_axis
        T[0:3, 1] = y_axis
        T[0:3, 2] = z_axis
        T[0:3, 3] = p1  # origin
        return T

    def draw_reference_frame(self, image, origin, x_axis, y_axis, z_axis, intrinsics):
        def project_to_2d(point):
            return rs.rs2_project_point_to_pixel(intrinsics, point)
        origin_2d = project_to_2d(origin)
        x_2d = project_to_2d((np.array(origin) + np.array(x_axis) * 0.1).tolist())
        y_2d = project_to_2d((np.array(origin) + np.array(y_axis) * 0.1).tolist())
        z_2d = project_to_2d((np.array(origin) + np.array(z_axis) * 0.1).tolist())
        cv2.arrowedLine(image, tuple(map(int, origin_2d)), tuple(map(int, x_2d)), (0, 0, 255), 2)
        cv2.arrowedLine(image, tuple(map(int, origin_2d)), tuple(map(int, y_2d)), (0, 255, 0), 2)
        cv2.arrowedLine(image, tuple(map(int, origin_2d)), tuple(map(int, z_2d)), (255, 0, 0), 2)

    def process_frame(self):
        # initialize keypoints array, so if keypoint is not detected it defaults to 0,0,0,0.01
        current_keypoints = np.zeros((6, 4))  # Tomato: 0–2, Hand: 3–5
        current_keypoints[:, 3] = 0.000000000001
        color_frame, depth_frame = self.camera.get_frame()

        color_frame_print = color_frame.copy()

        # Hand Predictions
        landmarks, hand_keypoints = self.hand_detector.process_frame(color_frame)
        if np.any(hand_keypoints[:, 3] > 0.1):
            # Get wrist absolute depth from RealSense
            wrist_x, wrist_y, wrist_relative_z = hand_keypoints[0, :3]
            px_wrist = int(wrist_x * color_frame.shape[1])
            py_wrist = int(wrist_y * color_frame.shape[0])
            if px_wrist > 0 and py_wrist > 0 and px_wrist < color_frame.shape[1] and py_wrist < color_frame.shape[0]:
                wrist_depth = depth_frame.get_distance(px_wrist, py_wrist)
                if wrist_depth > 0:  # Ensure wrist has a valid depth
                    for i in range(hand_keypoints.shape[0]):
                        x, y, relative_z, confidence = hand_keypoints[i]
                        px = int(x * color_frame.shape[1])
                        py = int(y * color_frame.shape[0])
                        estimated_depth = wrist_depth + (relative_z * self.scale_factor)
                        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [px, py], estimated_depth)
                        current_keypoints[i + 3, :3] = point_3d
                        current_keypoints[i + 3, 3] = confidence
                        cv2.circle(color_frame_print, (px, py), 5, (0, 255, 255), -1)
                        cv2.putText(color_frame_print, str(i+3), (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # hand frame
                    pw, R = self.get_hand_pose(landmarks, color_frame_print, 640, 480)
                    if pw is not None and R is not None:
                        self.hand_pose[0:3, 0:3] = R  # Set the rotation
                        wrist_point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pw[0], pw[1]], wrist_depth)
                        self.hand_pose[0:3, 3] = wrist_point  # Set the translation
                    else:
                        self.hand_pose = np.eye(4)

        # YOLO Predictions
        detections = self.tomato_detector.process_frame(color_frame)

        if detections.keypoints is not None:
            # filter detections for the lowest hanging tomato
            best_i = -1  # invalid index at start
            best_cy = -1  # smaller than any possible center_y

            for i, box in enumerate(detections.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(color_frame_print, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cy = (y1 + y2) // 2
                if cy > best_cy:
                    best_cy, best_i = cy, i

            if best_i >= 0 and detections.keypoints[best_i] is not None:
                det = detections.keypoints.data[best_i]
                box = detections.boxes.xyxy[best_i]

            # if detections.keypoints is not None:
                #for det, box in zip(detections.keypoints.data, detections.boxes.xyxy):
                keypoints = det.cpu().numpy()
                points_3d = []
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(color_frame_print, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # cv2.putText(color_frame_print, "tomato", (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                object_size = self.calculate_object_size(x1, x2, y1, y2, depth_frame, self.intrinsics)
                if object_size is not None:
                    object_radius = object_size / 2
                    cv2.putText(color_frame_print, f"R:{object_radius:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 2)
                else:
                    object_radius = 0.04 
                    #continue

                for idx, kp in enumerate(keypoints):
                    kp_x, kp_y, kp_conf = int(kp[0]), int(kp[1]), kp[2]
                    color = self.keypoint_colors[idx % len(self.keypoint_colors)]
                    if kp_conf > 0.2:
                        point_3d = self.deproject_to_3d(kp_x, kp_y, depth_frame, self.intrinsics)
                        if point_3d:
                            points_3d.append(point_3d)
                            current_keypoints[idx, :3] = point_3d
                            current_keypoints[idx, 3] = kp_conf
                            cv2.circle(color_frame_print, (kp_x, kp_y), 5, color, -1)
                            cv2.putText(color_frame_print, f"{idx}", (kp_x, kp_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color, 2)

                # calculate tomato reference frame
                # if len(points_3d) == 3:
                #     surface_normal = np.array(points_3d[0]) / np.linalg.norm(points_3d[0])
                #     adjusted_point_0 = np.array(points_3d[0]) + surface_normal * object_radius
                #     self.tomato_reference_frame = self.calculate_reference_frame(adjusted_point_0, points_3d[1],
                #                                                                  points_3d[2])
                #     self.draw_reference_frame(
                #         color_frame_print,
                #         self.tomato_reference_frame[0:3, 3],  # origin
                #         np.zeros(3), np.zeros(3),
                #         # self.tomato_reference_frame[0:3, 0],  # x_axis
                #         # self.tomato_reference_frame[0:3, 1],  # y_axis
                #         self.tomato_reference_frame[0:3, 2],  # z_axis
                #         self.intrinsics
                #     )

                # calculate tomato reference frame
                if len(points_3d) == 2 or len(points_3d) == 3:
                    surface_normal = np.array(points_3d[0]) / np.linalg.norm(points_3d[0])
                    adjusted_point_0 = np.array(points_3d[0]) + surface_normal * object_radius
                    temp_point = points_3d[1]
                    # temp_point[2] = adjusted_point_0[2]
                    # print('adjusted_point_0=',adjusted_point_0)
                    # print('temp_point=',temp_point)
                    self.tomato_reference_frame = self.calculate_reference_frame_2points(adjusted_point_0, temp_point)
                    self.draw_reference_frame(
                        color_frame_print,
                        self.tomato_reference_frame[0:3, 3],  # origin
                        np.zeros(3), np.zeros(3), 
                        # self.tomato_reference_frame[0:3, 0],  # x_axis
                        # self.tomato_reference_frame[0:3, 1],  # y_axis
                        self.tomato_reference_frame[0:3, 2],  # z_axis
                        self.intrinsics
                    )

        # Update the last keypoint set
        self.last_keypoint_set = current_keypoints
        #self.keypoints_history.append(current_keypoints)
        self.last_frame = color_frame_print
        self.publish_frame_data(self.tomato_reference_frame, self.hand_pose, self.last_keypoint_set, self.last_frame)
        return color_frame_print

    def get_last_keypoint_set(self):
        return self.last_keypoint_set

    def get_last_frame(self):
        return self.last_frame

    def get_process_time(self):
        return self.process_times

    def _enable_callback(self, msg: Bool):
        # Option A: set directly from the incoming message
        self.enabled = msg.data
        # Option B: flip the flag every time a message arrives:
        # self.enabled = not self.enabled
        rospy.loginfo(f"Grasp enabled? {self.enabled}")

    def publish_frame_data(self, tomato_frame, hand_frame, keypoints, image):
        """Publish tomato reference frame as a PoseStamped message."""
        tomato_pose_msg = PoseStamped()
        tomato_pose_msg.header = Header()
        tomato_pose_msg.header.stamp = rospy.Time.now()
        tomato_pose_msg.header.frame_id = "camera_link"  # Update with your camera frame
        # Set position (origin)
        tomato_pose_msg.pose.position.x = tomato_frame[0,3]
        tomato_pose_msg.pose.position.y = tomato_frame[1,3]
        tomato_pose_msg.pose.position.z = tomato_frame[2,3]
        # Compute orientation (rotation matrix to quaternion)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, 0] = tomato_frame[0:3,0]
        rotation_matrix[:3, 1] = tomato_frame[0:3,1]
        rotation_matrix[:3, 2] = tomato_frame[0:3,2]
        quaternion = quaternion_from_matrix(rotation_matrix)
        tomato_pose_msg.pose.orientation = Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3]
        )
        self.tomato_pub.publish(tomato_pose_msg)

        """Publish hand reference frame as a PoseStamped message."""
        hand_pose_msg = PoseStamped()
        hand_pose_msg.header = Header()
        hand_pose_msg.header.stamp = rospy.Time.now()
        hand_pose_msg.header.frame_id = "camera_link"  # Update with your camera frame
        # Set position (origin)
        hand_pose_msg.pose.position.x = hand_frame[0,3]
        hand_pose_msg.pose.position.y = hand_frame[1,3]
        hand_pose_msg.pose.position.z = hand_frame[2,3]
        # Compute orientation (rotation matrix to quaternion)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, 0] = hand_frame[0:3,0]
        rotation_matrix[:3, 1] = hand_frame[0:3,1]
        rotation_matrix[:3, 2] = hand_frame[0:3,2]
        quaternion = quaternion_from_matrix(rotation_matrix)
        hand_pose_msg.pose.orientation = Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3]
        )
        self.hand_pub.publish(hand_pose_msg)

        """Publish keypoints."""
        keypoint_msg = Float32MultiArray()
        # describe the shape: 6 rows × 4 cols
        keypoint_msg.layout.dim = [
            MultiArrayDimension(label='rows',    size=6, stride=6*4),
            MultiArrayDimension(label='columns', size=4, stride=4),
        ]
        keypoint_msg.layout.data_offset = 0
        # flatten in row-major order
        keypoint_msg.data = keypoints.flatten().tolist()
        self.keypoints_pub.publish(keypoint_msg)

        """Publish Image"""
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            rospy.logwarn("Failed to encode frame")
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = encoded_image.tobytes()
        self.image_pub.publish(msg)


    def run(self):
            rate = rospy.Rate(500)  # Publish at 10 Hz
            try:
                while not rospy.is_shutdown():
                    if self.enabled:
                        start_time = time.time()
                        image = self.process_frame()
                        end_time = time.time() - start_time
                        #print(f"time: {end_time}")
                        self.process_times.append(end_time)
                        self.last_frame = image
                        if image is None:
                            continue
                        if self.show_frame:
                            # cv2.namedWindow('RealSense Tomato Detection', cv2.WINDOW_NORMAL)
                            # resized_image = cv2.resize(image, (1280, 960))
                            cv2.imshow('RealSense Tomato Detection', image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    rate.sleep()
            finally:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = GraspDetector(show_frame=False)
    # rospy.spin()  # Keep ROS node alive
    detector.run()
    # Keep the main thread alive
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Exiting...")

