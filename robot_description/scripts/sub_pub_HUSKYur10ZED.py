import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from visualization_msgs.msg import Marker
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib.simple_action_client import SimpleActionClient
from actionlib_msgs.msg import GoalID

bridge = CvBridge()

# Publishers
marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
pose_image_pub = rospy.Publisher("/pose_image", Image, queue_size=10)
stop_action_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)

# Global variables
marker = Marker()
stop_action = GoalID()
goal = MoveBaseGoal()
ac = SimpleActionClient('move_base', MoveBaseAction)

received_messages = {
    'image': None,
    'depth': None,
    'pose': None,
    'camera_info_color': None,
    'camera_info_depth': None
}

# Callback for RGB image
def image_callback(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    received_messages['image'] = cv_image

# Callback for depth image
def depth_image_callback(msg):
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    received_messages['depth'] = np.array(depth_image, dtype=np.float32)

# Callback for pose
def pose_callback(msg):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    received_messages['pose'] = {
        'position': [position.x, position.y, position.z],
        'orientation': [orientation.x, orientation.y, orientation.z, orientation.w]
    }

# Camera parameters (color)
def camera_params_color(cameraInfo):
    received_messages['camera_info_color'] = cameraInfo

# Camera parameters (depth)
def camera_params_depth(cameraInfo):
    received_messages['camera_info_depth'] = cameraInfo

# Subscribe to topics
def subscribe_to_topics():
    rospy.init_node("husky_ur10_zed_listener", anonymous=True)

    rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color", Image, image_callback)
    rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, depth_image_callback)
    rospy.Subscriber("/zed2/zed_node/rgb/camera_info", CameraInfo, camera_params_color)
    rospy.Subscriber("/zed2/zed_node/depth/camera_info", CameraInfo, camera_params_depth)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, pose_callback)

    rospy.spin()

# Visualize position in RViz
def to_rviz(xyz, color='g'):
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.type = 2  # Sphere
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5

    if color == 'g':
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif color == 'r':
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

    marker.color.a = 1.0
    marker.pose.position.x = xyz[0]
    marker.pose.position.y = xyz[1]
    marker.pose.position.z = xyz[2]
    marker.pose.orientation.w = 1.0

    marker_pub.publish(marker)

# Main function
if __name__ == "__main__":
    try:
        subscribe_to_topics()
    except rospy.ROSInterruptException:
        pass
