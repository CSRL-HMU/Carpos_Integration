#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from custom_msgs.msg import HarvestCommand

class GripperController:
    def __init__(self):
        rospy.init_node('gripper_controller', anonymous=True)
        
        self.gripper_pub = rospy.Publisher('/gripper_command', String, queue_size=10)
        self.status_pub = rospy.Publisher('/gripper_status', String, queue_size=10)
        self.command_sub = rospy.Subscriber('/harvest_command', HarvestCommand, self.process_harvest_command)

    def process_harvest_command(self, msg):
        rospy.loginfo(f"Moving gripper to pre-harvest position: x={msg.pre_harvest_pose.position.x}, y={msg.pre_harvest_pose.position.y}, z={msg.pre_harvest_pose.position.z}")
        rospy.sleep(2)  # Simulating movement
        
        rospy.loginfo("Aligning gripper for grasping...")
        rospy.sleep(1)
        
        rospy.loginfo("Gripper closing...")
        self.gripper_pub.publish(msg.gripper_command)
        rospy.sleep(1)  # Simulating gripper closing time
        
        self.status_pub.publish("Gripper closed")
        rospy.loginfo("Gripper closed. Notifying cutting system.")
        cutting_pub = rospy.Publisher('/start_cutting', String, queue_size=10, latch=True)
        rospy.sleep(1)
        cutting_pub.publish("start")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    gripper = GripperController()
    gripper.run()
