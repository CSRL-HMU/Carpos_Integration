import rospy
from nav_msgs.msg import Odometry

class OdomProcessorNode:
    def __init__(self):
        # Αρχικοποίηση του ROS Node
        rospy.init_node('odom_processor', anonymous=True)
        
        # Subscriber για την οδόμετρία
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Publisher για την επεξεργασμένη οδόμετρία
        self.odom_pub = rospy.Publisher('/processed_odom', Odometry, queue_size=10)
    
    def odom_callback(self, msg):
        rospy.loginfo("Λήψη οδόμετρου: Θέση x: %f, y: %f, z: %f", 
                      msg.pose.pose.position.x, 
                      msg.pose.pose.position.y, 
                      msg.pose.pose.position.z)
        
        # Εδώ μπορεί να γίνει κάποια επεξεργασία των δεδομένων
        processed_msg = msg  # Σε αυτή την περίπτωση, απλά αναμεταδίδουμε το μήνυμα
        
        # Δημοσίευση στο νέο topic
        self.odom_pub.publish(processed_msg)
        rospy.loginfo("Δημοσιεύτηκε η επεξεργασμένη οδόμετρια")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        odom_processor = OdomProcessorNode()
        odom_processor.run()
    except rospy.ROSInterruptException:
        pass
