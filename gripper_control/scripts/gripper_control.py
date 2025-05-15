#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float32MultiArray
import numpy as np
import serial
import time

class GripperControl:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('gripper_control', anonymous=True)

        # Set up serial communication with Arduino
        self.serial_port = '/dev/ttyUSB0'  # Adjust as needed
        self.baud_rate = 115200
        self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        time.sleep(4)  # Allow time for the serial connection to initialize

        # Set initial position for the thumb placement
        #self.x_start, self.y_start, self.phi_start = 5.3, 2.5, -np.pi / 3
        #Q = self.inverse_kinematics(self.x_start, self.y_start, self.phi_start)
        #if Q is not None:
        #    self.send_commands(10 + np.degrees(Q[0]), 140 + np.degrees(Q[1]), 145 + np.degrees(Q[2]))
        self.send_angles(45,-45,-45)
        self.send_actuator_command(0)  # Initial close fingers command
        rospy.sleep(1)
        
        # Set up subscribers for ROS topics
        rospy.Subscriber('finger_control', Bool, self.finger_callback)
        rospy.Subscriber('thumb_position', Float32MultiArray, self.thumb_callback)

    ####################### Functions ########################

    def inverse_kinematics(self, x, y, phi):
        l1, l2, l3 = 3.9, 3.4, 2.5
        Xw = x - l3 * np.cos(phi)
        Yw = y - l3 * np.sin(phi)
        C2 = (Xw**2 + Yw**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if C2 < -1 or C2 > 1:
            rospy.logwarn("Target not reachable. C2 = %f", C2)
            return None

        theta2 = -np.arctan2(np.sqrt(1 - C2**2), C2)
        A = Xw * (l1 + l2 * np.cos(theta2)) + Yw * l2 * np.sin(theta2)
        B = Yw * (l1 + l2 * np.cos(theta2)) - Xw * l2 * np.sin(theta2)
        theta1 = np.arctan2(B, A)
        theta3 = phi - theta1 - theta2

        return np.array([theta1, theta2, theta3])

    def move_along_trajectory(self, x_start, y_start, phi_start, x_end, y_end, phi_end, steps=10):
        x_points = np.linspace(x_start, x_end, steps)
        y_points = np.linspace(y_start, y_end, steps)
        phi_points = np.linspace(phi_start, phi_end, steps)
        for x, y, phi in zip(x_points, y_points, phi_points):
            Qs = self.inverse_kinematics(x, y, phi)
            if Qs is None:
                rospy.logwarn("Point unreachable, skipping")
                continue
            angle1, angle2, angle3 = map(np.degrees, Qs)
            self.send_commands(angle1, angle2, angle3)
            rospy.sleep(0.1)

    def send_commands(self, angle1, angle2, angle3):
        command = f"{angle1},{angle2},{angle3}\n"
        self.ser.write(command.encode())
        self.ser.flush()
        rospy.loginfo("Sent Command: %s", command.strip())

    def send_angles(self, angle1, angle2, angle3):
        command = f"{angle1},{angle2},{angle3}\n"
        self.ser.write(command.encode())
        self.ser.flush()  # Ensure the command is sent
        print(command)

    def send_actuator_command(self, state):
        command = f"{state}\n"
        self.ser.write(command.encode())
        self.ser.flush()
        rospy.loginfo("Sent Actuator Command: %s", command.strip())
        time.sleep(0.5)

    ################ ROS Callbacks ################

    def finger_callback(self, msg):
        state = 1 if msg.data else 0  # Assuming True means close and False means open
        self.send_actuator_command(state)

    def thumb_callback(self, msg):
        x, y, phi = msg.data
        #x, y, phi = np.rad2deg(msg.data)
        Q = np.rad2deg(self.inverse_kinematics(x, y, phi))
        #self.move_along_trajectory(self.x_start, self.y_start, self.phi_start, x, y, phi, steps=10)
        self.send_angles(Q[0], Q[1], Q[2])

    ################ Main Loop ################

    def run(self):
        rospy.spin()  # Keep the node running

if __name__ == "__main__":
    try:
        robot_arm_control = GripperControl()
        robot_arm_control.run()
    except rospy.ROSInterruptException:
        pass
