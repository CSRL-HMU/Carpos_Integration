#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose

class ArmController:
    def __init__(self):
        rospy.init_node('arm_controller', anonymous=True)
        self.pub = rospy.Publisher('/arm_start_movement', Pose, queue_size=10)
        rospy.sleep(1)  # Χρονική καθυστέρηση για την αρχικοποίηση του publisher

    def send_target_position(self):
        target_pose = Pose()
        target_pose.position.x = 0.6
        target_pose.position.y = 0.2
        target_pose.position.z = 0.3
        target_pose.orientation.w = 1.0  # Χρησιμοποιούμε ένα έγκυρο quaternion

        rospy.loginfo("Sending target position to arm...")
        self.pub.publish(target_pose)

if __name__ == '__main__':
    controller = ArmController()
    controller.send_target_position()
    rospy.sleep(1)  # Μικρή καθυστέρηση για να στείλει το μήνυμα πριν τερματιστεί

