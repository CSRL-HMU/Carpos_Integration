#!/usr/bin/env python

# import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque  # optional for fixed-size history
import threading
import time
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix
import pyzed.sl as sl  # ZED SDK

class TomatoDetector:
    def __init__(self, model_path='/home/carpos/catkin_ws/src/tomato_detector/models/keypoints_new.pt', show_frame=True, history_size=50):
        # Initialize ROS node and ROS publisher for tomato pose
        rospy.init_node('tomato_detector', anonymous=True)
        self.pose_pub = rospy.Publisher('/tomato_pose', PoseStamped, queue_size=10)

        # Load parameters from launch file
        #self.model_path = rospy.get_param("~model_path", "models/keypoints_new.pt")
        #self.show_frame = rospy.get_param("~show_frame", True)
        self.frame_id = rospy.get_param("~frame_id", "zed_camera")
        model_path='/home/carpos/catkin_ws/src/tomato_detector/models/keypoints_new.pt'
        show_frame = True
        # Initialize ZED camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # High-quality depth
        init_params.coordinate_units = sl.UNIT.METER  # Depth in meters
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("ZED Camera failed to initialize!")
            # rospy.logerr("ZED Camera failed to initialize!")
            exit(1)

        self.runtime_params = sl.RuntimeParameters()
        self.depth_map = sl.Mat()
        self.image_zed = sl.Mat()

        # Get camera intrinsics
        cam_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
        self.fx = cam_params.left_cam.fx  # Focal length (x)
        self.fy = cam_params.left_cam.fy  # Focal length (y)
        self.cx = cam_params.left_cam.cx  # Optical center (x)
        self.cy = cam_params.left_cam.cy  # Optical center (y)

        # Load YOLOv8 Pose model
        self.model = YOLO(model_path)
        self.show_frame = show_frame

        # Define colors for keypoints
        self.keypoint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        # Variable to store the last keypoint set (each keypoint is a dict with 3D coordinates and confidence)
        self.last_keypoint_set = None
        # Optional: store a history of keypoint sets with a maximum length to avoid unbounded memory usage.
        self.keypoints_history = deque(maxlen=history_size)

        # Last detected reference frame
        self.last_pose = None
        self.pose_history = deque(maxlen=history_size)

        # Start the detection loop in a new thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def __del__(self):
        self.zed.close()
        cv2.destroyAllWindows()

    def pixel_to_3d(self, px, py):
        """Convert pixel (x, y) to 3D world coordinates using ZED depth."""
        # err, depth_value = self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH).get_value(x, y)
        # if err != sl.ERROR_CODE.SUCCESS or depth_value <= 0:
        #     return None  # No valid depth

        depth_value = self.depth_map.get_value(px, py)[1]  # Correct way to get depth
        if np.isnan(depth_value) or np.isinf(depth_value) or depth_value <= 0:
            print(f"Invalid depth at ({px}, {py}): {depth_value}")
            return None  # Skip if depth is invalid

        # Convert pixel to real-world coordinates
        point_3d = [(px - self.cx) * depth_value / self.fx,
                    (py - self.cy) * depth_value / self.fy,
                    depth_value]
        return point_3d

    def get_average_depth(self, x1, x2, y1, y2):
        """Calculate the average depth within a 5x5 region around the center of the bounding box."""
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        region_size = 5  # 5x5 sampling region

        depths = []
        for dx in range(-region_size // 2, region_size // 2 + 1):
            for dy in range(-region_size // 2, region_size // 2 + 1):
                px = min(max(center_x + dx, 0), self.depth_map.get_width() - 1)
                py = min(max(center_y + dy, 0), self.depth_map.get_height() - 1)

                depth_value = self.depth_map.get_value(px, py)[1]  # Correct way to get depth
                if depth_value > 0:  # Ignore invalid depth values
                    depths.append(depth_value)

        return np.mean(depths) if depths else 0

    def calculate_object_size(self, x1, x2, y1, y2):
        """Estimate real-world object size based on bounding box and depth."""
        depth = self.get_average_depth(x1, x2, y1, y2)
        if depth <= 0:
            return None
        pixel_length = abs(x2 - x1)
        object_size = (pixel_length * depth) / self.fx  # Convert pixel size to meters
        return object_size

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
        return {
            "origin": p1.tolist(),
            "x_axis": x_axis.tolist(),
            "y_axis": y_axis.tolist(),
            "z_axis": z_axis.tolist()
        }

    def draw_reference_frame(self, image, origin, x_axis, y_axis, z_axis):
        """Draws the reference frame (X, Y, Z axes) projected onto the 2D image."""

        def project_to_2d(point_3d):
            """Projects 3D world coordinates to 2D image coordinates using the ZED intrinsic matrix."""
            x, y, z = point_3d
            if np.isnan(z) or np.isinf(z) or z <= 0:  # Avoid invalid depth values
                return None
            u = int((x * self.fx / z) + self.cx)
            v = int((y * self.fy / z) + self.cy)
            return (u, v)

        # Compute projected 2D points
        origin_2d = project_to_2d(origin)
        x_2d = project_to_2d((np.array(origin) + np.array(x_axis) * 0.1).tolist())
        y_2d = project_to_2d((np.array(origin) + np.array(y_axis) * 0.1).tolist())
        z_2d = project_to_2d((np.array(origin) + np.array(z_axis) * 0.1).tolist())

        # Draw the coordinate frame if projection is valid
        if None not in (origin_2d, x_2d, y_2d, z_2d):
            cv2.arrowedLine(image, origin_2d, x_2d, (0, 0, 255), 2)  # X-axis (Red)
            cv2.arrowedLine(image, origin_2d, y_2d, (0, 255, 0), 2)  # Y-axis (Green)
            cv2.arrowedLine(image, origin_2d, z_2d, (255, 0, 0), 2)  # Z-axis (Blue)

    def process_frame(self):
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None

        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)

        color_image = self.image_zed.get_data()[:, :, :3]
        color_image = np.array(color_image, dtype=np.uint8)  # Ensure NumPy format
        results = self.model.predict(source=color_image, conf=0.5, save=False, save_txt=False, show=False,
                                     verbose=False)
        detections = results[0]
        # Dictionary to store keypoints with their original YOLO index
        current_keypoints = {idx: {"idx": idx, "point": [0, 0, 0], "confidence": 0.01} for idx in
                          range(3)}  # Default missing keypoints
        # Temporary list to hold keypoints for the current frame
        #current_keypoints = []

        if detections.keypoints is not None:
            for det, box in zip(detections.keypoints.data, detections.boxes.xyxy):
                keypoints = det.cpu().numpy()
                points_3d = []
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(color_image, "tomato", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                object_size = self.calculate_object_size(x1, x2, y1, y2)
                if object_size is not None:
                    object_radius = object_size / 2
                    print(f"Bounding box length: {object_radius:.2f} m")
                    cv2.putText(color_image, f"L:{object_radius:.2f}", (x1, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    continue

                for idx, kp in enumerate(keypoints):
                    kp_x, kp_y, kp_conf = int(kp[0]), int(kp[1]), kp[2]
                    color = self.keypoint_colors[idx % len(self.keypoint_colors)]
                    if kp_conf > 0.8:
                        point_3d = self.pixel_to_3d(kp_x, kp_y)
                        if point_3d:
                            points_3d.append(point_3d)
                            # Save each keypoint as a dict with its 3D coordinates and confidence
                            current_keypoints[idx] = {
                                "idx": idx,
                                "point": point_3d,
                                "confidence": kp_conf
                            }
                            cv2.circle(color_image, (kp_x, kp_y), 5, color, -1)
                            cv2.putText(color_image, f"conf: {kp_conf:.2f}", (kp_x, kp_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(f"Keypoint Dict: {current_keypoints}")

                if len(points_3d) == 3:
                    surface_normal = np.array(points_3d[0]) / np.linalg.norm(points_3d[0])
                    adjusted_point_0 = np.array(points_3d[0]) + surface_normal * object_radius
                    reference_frame = self.calculate_reference_frame(adjusted_point_0, points_3d[1], points_3d[2])
                    self.draw_reference_frame(
                        color_image,
                        reference_frame["origin"],
                        reference_frame["x_axis"],
                        reference_frame["y_axis"],
                        reference_frame["z_axis"]
                    )
                    #print()
                    self.publish_pose(reference_frame)

        # Update the last keypoint set and optionally store it in history
        self.last_keypoint_set = current_keypoints
        self.keypoints_history.append(current_keypoints)
        return color_image

    def publish_pose(self, reference_frame):
        """Publish the reference frame as a PoseStamped message."""
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "camera_link"  # Update with your camera frame

        # Set position (origin)
        pose_msg.pose.position.x = reference_frame["origin"][0]
        pose_msg.pose.position.y = reference_frame["origin"][1]
        pose_msg.pose.position.z = reference_frame["origin"][2]

        # Compute orientation (rotation matrix to quaternion)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, 0] = reference_frame["x_axis"]
        rotation_matrix[:3, 1] = reference_frame["y_axis"]
        rotation_matrix[:3, 2] = reference_frame["z_axis"]
        quaternion = quaternion_from_matrix(rotation_matrix)

        pose_msg.pose.orientation = Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3]
        )

        self.pose_pub.publish(pose_msg)
        self.last_pose = pose_msg

    def get_last_keypoint_set(self):
        """Returns the most recent set of 3D keypoints with their confidence."""
        return self.last_keypoint_set

    def run(self):

        rate = rospy.Rate(500)  # Publish at 10 Hz
        try:
            while not rospy.is_shutdown():
                image = self.process_frame()
                if image is None:
                    continue
                if self.show_frame:
                    cv2.namedWindow('RealSense Tomato Detection', cv2.WINDOW_NORMAL)
                    resized_image = cv2.resize(image, (1280, 960))
                    cv2.imshow('RealSense Tomato Detection', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                rate.sleep()
        finally:
            self.zed.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        detector = TomatoDetector(model_path='models/keypoints_new.pt', show_frame=False)
        rospy.spin()  # Keep ROS node alive
    except rospy.ROSInterruptException:
        print("Shutting down TomatoDetector...")


    # Keep the main thread alive
    # try:
    #     detector = TomatoDetector(model_path='models/keypoints_new.pt', show_frame=True)
    #     while True:
    #         time.sleep(0.02)
    # except KeyboardInterrupt:
    #     print("Exiting...")


