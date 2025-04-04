import rospy
from std_msgs.msg import String

class CuttingNode:
    def __init__(self):
        rospy.init_node('cutting_node', anonymous=True)
        self.sub = rospy.Subscriber('/start_cutting', String, self.cut_tomato)
        self.status_pub = rospy.Publisher('/cutting_status', String, queue_size=10)

    def cut_tomato(self, msg):
        if msg.data == "start":
            rospy.loginfo("Starting to cut the tomato...")
            rospy.sleep(3)  # Simulating cutting time
            rospy.loginfo("Cutting complete.")
            self.status_pub.publish("Cutting complete")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    cutting = CuttingNode()
    cutting.run()
