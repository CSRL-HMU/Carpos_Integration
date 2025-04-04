#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
import random

class ArmStateMachine:
    def __init__(self):
        rospy.init_node('arm_movement_controller', anonymous=True)
        self.state = "IDLE"
        
        rospy.Subscriber('/arm_target_position', Pose, self.command_callback)
        self.status_pub = rospy.Publisher('/arm_movement_status', Pose, queue_size=10)
        
    def change_state(self, new_state):
        rospy.loginfo(f"State changed: {self.state} -> {new_state}")
        self.state = new_state
    
    def execute_movement(self, target_position):
        self.change_state("MOVING")
        rospy.loginfo(f"Moving arm to position: x={target_position.position.x}, y={target_position.position.y}, z={target_position.position.z}")
        movement_time = random.uniform(1.0, 5.0)  # Simulating dynamic movement duration
        rospy.sleep(movement_time)
        success = random.choice([True, False])  # Random success/failure simulation
        return success, movement_time

    def command_callback(self, msg):
        if self.state != "IDLE":
            rospy.logwarn("Arm is already in motion. Ignoring new command.")
            return

        target_position = msg
        success, movement_time = self.execute_movement(target_position)
        
        if success:
            self.change_state("SUCCESS")
            rospy.loginfo(f"Movement to x={target_position.position.x}, y={target_position.position.y}, z={target_position.position.z} completed in {movement_time:.2f} seconds.")
            self.status_pub.publish(target_position)
        else:
            self.change_state("FAILURE")
            rospy.logwarn(f"Failed to move arm to the specified position.")
        
        rospy.sleep(1)  # Short delay before returning to IDLE
        self.change_state("IDLE")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    arm_controller = ArmStateMachine()
    arm_controller.run()