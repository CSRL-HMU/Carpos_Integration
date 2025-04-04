import rospy
from geometry_msgs.msg import Pose
import random

class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node', anonymous=True)
        self.pub = rospy.Publisher('/tomato_frame', Pose, queue_size=10)
        rospy.sleep(1)  # Allow time for connections

    def detect_tomato(self):
        tomato_pose = Pose()
        tomato_pose.position.x = random.uniform(0.2, 1.0)
        tomato_pose.position.y = random.uniform(-0.5, 0.5)
        tomato_pose.position.z = random.uniform(0.1, 0.5)
        rospy.loginfo(f"Detected tomato at x={tomato_pose.position.x}, y={tomato_pose.position.y}, z={tomato_pose.position.z}")
        self.pub.publish(tomato_pose)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    perception = PerceptionNode()
    rospy.sleep(2)  # Simulating detection delay
    perception.detect_tomato()
    perception.run()