import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose

class GripperController:
    def __init__(self):
        rospy.init_node('gripper_controller', anonymous=True)
        
        self.gripper_pub = rospy.Publisher('/gripper_command', String, queue_size=10)
        self.status_pub = rospy.Publisher('/gripper_status', String, queue_size=10)
        self.perception_sub = rospy.Subscriber('/tomato_frame', Pose, self.move_to_tomato)

    def move_to_tomato(self, msg):
        rospy.loginfo(f"Moving gripper to tomato at x={msg.position.x}, y={msg.position.y}, z={msg.position.z}")
        rospy.sleep(2)  # Simulating movement
        
        rospy.loginfo("Gripper closing...")
        self.gripper_pub.publish("close")
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

