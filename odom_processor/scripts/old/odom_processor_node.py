#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry

class OdomProcessorNode:
    def __init__(self):
        # Initialize the ROS Node
        rospy.init_node('odom_processor', anonymous=True)
        
        # Subscriber for odometry
        self.odom_sub = rospy.Subscriber('/husky_velocity_controller/odom', Odometry, self.odom_callback)
        
        # Publisher for processed odometry
        self.odom_pub = rospy.Publisher('/processed_odom', Odometry, queue_size=10)
    
    def odom_callback(self, msg):
        # Log the received odometry data
        rospy.loginfo("Received odometry: Position x: %f, y: %f, z: %f", 
                      msg.pose.pose.position.x, 
                      msg.pose.pose.position.y, 
                      msg.pose.pose.position.z)
        
        # Here you can add some processing of the data
        processed_msg = msg  # In this case, we are just republishing the message
        
        # Publish the processed message
        self.odom_pub.publish(processed_msg)
        rospy.loginfo("Published processed odometry")
    
    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    try:
        odom_processor = OdomProcessorNode()
        odom_processor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("OdomProcessorNode interrupted.")
