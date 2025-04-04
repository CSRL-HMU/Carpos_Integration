#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion

class OdomProcessorNode:
    def __init__(self):
        # Αρχικοποίηση του ROS Node
        rospy.init_node('odom_processor', anonymous=True)
        
        # Subscriber για την οδόμετρία
        self.odom_sub = rospy.Subscriber('/husky_velocity_controller/odom', Odometry, self.odom_callback)
        
        # Publisher για την επεξεργασμένη οδόμετρία
        self.odom_pub = rospy.Publisher('/processed_odom', Odometry, queue_size=10)
        
        # Publisher για Marker (για οπτικοποίηση στο RViz)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        
        # Αρχικοποίηση του Marker
        self.marker = Marker()
        self.marker.header.frame_id = "odom"  # Το σύστημα συντεταγμένων του Marker
        self.marker.type = Marker.SPHERE      # Τύπος Marker (σφαίρα)
        self.marker.action = Marker.ADD       # Προσθήκη Marker
        self.marker.scale.x = 0.2             # Μέγεθος στον άξονα x
        self.marker.scale.y = 0.2             # Μέγεθος στον άξονα y
        self.marker.scale.z = 0.2             # Μέγεθος στον άξονα z
        self.marker.color.a = 1.0             # Αδιαφάνεια (1 = αδιαφανές)
        self.marker.color.r = 1.0             # Χρώμα (κόκκινο)
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        
        # Αρχικοποίηση του quaternion (ταυτότητα)
        self.marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    
    def odom_callback(self, msg):
        # Λήψη δεδομένων οδομετρίας
        position = msg.pose.pose.position
        
        # Εμφάνιση των δεδομένων
        rospy.loginfo("Λήψη οδόμετρου: Θέση x: %f, y: %f, z: %f", 
                      position.x, position.y, position.z)
        
        # Επεξεργασία των δεδομένων (εδώ απλά αναμεταδίδουμε το μήνυμα)
        processed_msg = msg
        self.odom_pub.publish(processed_msg)
        
        # Δημιουργία Marker για την τρέχουσα θέση
        self.marker.header.stamp = rospy.Time.now()
        self.marker.pose.position.x = position.x
        self.marker.pose.position.y = position.y
        self.marker.pose.position.z = position.z
        
        # Δημοσίευση του Marker
        self.marker_pub.publish(self.marker)
        rospy.loginfo("Δημοσιεύτηκε Marker στο RViz")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        odom_processor = OdomProcessorNode()
        odom_processor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("OdomProcessorNode interrupted.")
