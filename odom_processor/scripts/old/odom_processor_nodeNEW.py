#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose
import random

class ArmMovementNode:
    def __init__(self):
        rospy.init_node('arm_movement_node', anonymous=True)
        self.status_pub = rospy.Publisher('/arm_movement_status', Pose, queue_size=10)
        self.start_sub = rospy.Subscriber('/arm_start_movement', Pose, self.command_callback)

    def move_arm(self, target_position):
        rospy.loginfo(f"Moving arm to position: x={target_position.position.x}, y={target_position.position.y}, z={target_position.position.z}")
        movement_time = random.uniform(1.0, 5.0)  # Προσομοίωση χρόνου κίνησης
        rospy.sleep(movement_time)
        rospy.loginfo("Movement complete.")
        self.status_pub.publish(target_position)
        rospy.signal_shutdown("Arm movement node completed execution.")

    def command_callback(self, msg):
        rospy.loginfo("Received movement command from controller.")
        self.move_arm(msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    arm_movement = ArmMovementNode()
    arm_movement.run()

