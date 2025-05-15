#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose

def publish_dummy_tomato_pose():
    rospy.init_node('dummy_tomato_publisher', anonymous=True)
    pub = rospy.Publisher('/tomato_frame', Pose, queue_size=10)
    rate = rospy.Rate(1)  # 1Hz

    while not rospy.is_shutdown():
        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0.0
        pose.position.z = 0.2

        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        rospy.loginfo("Publishing dummy tomato pose...")
        pub.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_tomato_pose()
    except rospy.ROSInterruptException:
        pass