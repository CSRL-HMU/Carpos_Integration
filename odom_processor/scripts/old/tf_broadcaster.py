#!/usr/bin/env python3

import rospy
import tf

def broadcast_tf():
    rospy.init_node('tf_broadcaster', anonymous=True)
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)  # 10 Hz

    rospy.loginfo("Broadcasting transform: world -> tomato_frame")
    rospy.loginfo("Broadcasting transform: world -> gripper_frame")

    while not rospy.is_shutdown():
        # Μετασχηματισμός για το tomato_frame
        br.sendTransform(
            (-0.7, 0.0, 1.0),  # x, y, z
            (1.0, 0.0, 0.0, 0.0),  # quaternion
            rospy.Time.now(),
            "tomato_frame",
            "world"
        )

        # Μετασχηματισμός για το gripper_frame
        br.sendTransform(
            (0.1, 0.0, 0.0),  # x, y, z
            (-0.7071068, 0, 0.7071068, 0),  # quaternion
            rospy.Time.now(),
            "gripper_frame",
            "world"
        )

        rate.sleep()

if __name__ == '__main__':
    try:
        broadcast_tf()
    except rospy.ROSInterruptException:
        pass
