#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO
import torch
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2


class KeypointDetectorNode:
    def __init__(self):
        # Initialize the YOLO model
        self.model = YOLO('models/best.pt')

        # Initialize ROS node
        rospy.init_node('keypoint_detector', anonymous=True)

        # Set up subscriber and publisher
        self.image_sub = rospy.Subscriber('/input_image_topic', Image, self.image_callback)
        self.keypoints_pub = rospy.Publisher('/keypoints_coordinates', Float32MultiArray, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

    def image_callback(self, img_msg):
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("Failed to convert image: %s", e)
            return

        # Run the YOLO model on the image
        results = self.model(cv_image)

        # Extract keypoints if available
        if results[0].keypoints is not None:
            keypoints_tensor = results[0].keypoints.data  # Tensor with shape [num_objects, num_keypoints, 3]
            keypoints_np = keypoints_tensor.cpu().numpy()  # Convert to numpy array if needed

            # Flatten the keypoints array for publishing
            keypoints_flat = keypoints_np.flatten().tolist()

            # Prepare and publish the keypoints coordinates
            keypoints_msg = Float32MultiArray(data=keypoints_flat)
            self.keypoints_pub.publish(keypoints_msg)
            rospy.loginfo("Published keypoints: %s", keypoints_flat)
        else:
            rospy.logwarn("No keypoints found in the image.")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        detector = KeypointDetectorNode()
        detector.run()
    except rospy.ROSInterruptException:
        pass
