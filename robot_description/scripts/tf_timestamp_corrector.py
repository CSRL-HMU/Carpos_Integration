import rospy
import tf2_ros
from tf2_msgs.msg import TFMessage

def tf_callback(msg):
    # Ενημέρωση όλων των timestamps στο παρόν ROS χρόνο
    for transform in msg.transforms:
        transform.header.stamp = rospy.Time.now()
    pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node('tf_timestamp_corrector')

    # Publisher για τα διορθωμένα TF frames
    pub = rospy.Publisher('/tf_corrected', TFMessage, queue_size=10)

    # Subscriber για τα αρχικά TF frames
    rospy.Subscriber('/tf', TFMessage, tf_callback)

    rospy.spin()

