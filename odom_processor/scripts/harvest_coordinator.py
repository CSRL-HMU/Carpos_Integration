#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from custom_msgs.msg import HarvestCommand

class HarvestCoordinator:
    def __init__(self):
        rospy.init_node('harvest_coordinator', anonymous=True)
        
        self.command_pub = rospy.Publisher('/harvest_command', HarvestCommand, queue_size=10)
        self.perception_sub = rospy.Subscriber('/tomato_frame', Pose, self.process_tomato)
        
    def process_tomato(self, msg):
        rospy.loginfo(f"Received tomato position: x={msg.position.x}, y={msg.position.y}, z={msg.position.z}")
        
        harvest_msg = HarvestCommand()
        harvest_msg.tomato_pose = msg
        harvest_msg.pre_harvest_pose.position.x = msg.position.x - 0.1
        harvest_msg.pre_harvest_pose.position.y = msg.position.y
        harvest_msg.pre_harvest_pose.position.z = msg.position.z + 0.2
        
        harvest_msg.grasp_transform.position.x = msg.position.x
        harvest_msg.grasp_transform.position.y = msg.position.y
        harvest_msg.grasp_transform.position.z = msg.position.z
        
        harvest_msg.gripper_command = "close"
        
        rospy.loginfo("Publishing harvest command...")
        self.command_pub.publish(harvest_msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    coordinator = HarvestCoordinator()
    coordinator.run()
