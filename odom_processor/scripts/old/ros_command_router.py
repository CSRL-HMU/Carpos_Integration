#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Int32

# Mapping of commands to integers
COMMAND_MAPPING = {
    "Command 1": 1,
    "Command 2": 2,
    "Command 3": 3,
    "Command 4": 4,
    "Command 5": 5,
    "Command 6": 6,
    "Command 7": 7,
    "Command 8": 8,
    "Command 9": 9,
}

def command_callback(msg):
    """Callback function that listens to /command_topic and maps it to an integer."""
    command_str = msg.data.strip()
    if command_str in COMMAND_MAPPING:
        command_int = COMMAND_MAPPING[command_str]
        rospy.loginfo(f"Received: {command_str} -> Sending: {command_int}")
        pub.publish(command_int)
    else:
        rospy.logwarn(f"Unknown command received: {command_str}")

if __name__ == '__main__':
    rospy.init_node('command_router', anonymous=True)
    
    # Subscriber to listen to command_topic
    rospy.Subscriber('/command_topic', String, command_callback)
    
    # Publisher to send integer commands
    pub = rospy.Publisher('/processed_command_topic', Int32, queue_size=10)
    
    rospy.spin()
