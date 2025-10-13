#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import String, Int32, Float64
from custom_msgs.msg import HarvestCommand
import numpy as np
from spatialmath import *
from CSRL_orientation import *
import scipy.io
from spatialmath.base import r2q
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Bool, Float32MultiArray
import numpy as np
import serial
import time
import subprocess

class HarvestMotionNode:
    def __init__(self):
        rospy.init_node('harvest_motion_node', anonymous=True)
        rospy.loginfo("Harvest Motion Node has started!")

        # Mapping των commands
        self.COMMAND_MAPPING = {
            "Command 1": 1,
            "Command 2": 2,
            "Command 3": 3,
            "Command 4": 4,
            "Command 5": 5,
            "Command 6": 6,
            "Command 7": 7,
            "Command 8": 8,  # Execute Harvest
            "Command 9": 9,
        }

        # Publishers και Subscribers
        self.motion_pub = rospy.Publisher('/amg_command', HarvestCommand, queue_size=10)
        self.status_sub = rospy.Subscriber('/motion_status', String, self.motion_status_callback)
        self.tomato_sub = rospy.Subscriber('/tomato_pose', PoseStamped, self.tomato_pose_callback)
        self.gripper_sub = rospy.Subscriber('/gripper_pose', PoseStamped, self.gripper_callback)
        self.camera_sub = rospy.Subscriber('/camera_pose', PoseStamped, self.camera_callback)
        self.camera_ur_sub = rospy.Subscriber('/camera_pose_ur', PoseStamped, self.camera_ur_callback)
        self.enable_pub = rospy.Publisher('/grasp_enable', Bool, queue_size=1)           
        self.enable_robot_pub = rospy.Publisher('/amg_enable_robot', Bool, queue_size=1)           
        self.visual_obs_pub = rospy.Publisher('/start_active_perception', Float32MultiArray, queue_size=10, latch=True)
        self.stop_logger = rospy.Publisher('/stop_logger', Bool, queue_size=1) 

        self.visual_obs_status_sub = rospy.Subscriber('/start_active_perception_status', String, self.visual_obs_status_callback)
    
        self.command_sub = rospy.Subscriber('/command_topic', String, self.command_callback)
        
        self.kinesthetic_status_sub = rospy.Subscriber('/kinesthetic_status', String, self.kinesthetic_status_callback)  # Για Kinesthetic
        
        self.kinesthetic_pub = rospy.Publisher('/start_kinesthetic_correction', String, queue_size=10, latch=True)
        
        self.finger_pub = rospy.Publisher('/finger_control', Bool, queue_size=10)

        #self.thumb_pub = rospy.Publisher('/thumb_position', Float32MultiArray, queue_size=10)
        self.thumb_pub_2 = rospy.Publisher('/thumb_position', Int32, queue_size=1)

        self.finger_status_sub = rospy.Subscriber('/finger_status', String, self.finger_status_callback)

        self.thumb_status_sub = rospy.Subscriber('/thumb_status', String, self.thumb_status_callback)

        self.ok_redetect_sub = rospy.Subscriber('/ok_redetect', String, self.ok_redetect_callback)

        self.tomato_radius_sub = rospy.Subscriber('/tomato_radius', Float64, self.tomato_radius_callback)
        #command_int = None

        # Αρχικές καταστάσεις αναμονής
        self.awaiting_finger_confirmation = False
        self.awaiting_thumb_confirmation = False

        self.gPregrasp = None
        self.gGrasp = None

        # Καταστάσεις και μεταβλητέςα
        self.current_phase = "idle"
        self.g0T = None
        self.g_0camera = None
        self.g_0gripper = None
        self.g_URcamera = None
        self.awaiting_confirmation = False  # Για κινήσεις
        self.awaiting_kinesthetic_confirmation = False  # Για Kinesthetic
        self.awaiting_visual_obs_confirmation = False


        self.ok_redect_pressed = False
        self.detection_decision = None  # None: no input yet, 0: re-detect, 1: ok

        self.observation_counter = 0

        #### ADDITION FOR SCALING 
        self.tomato_r = 0.04


    
    #### ADDITION FOR SCALING 
    def tomato_radius_callback(self, msg):
        self.tomato_r = msg.data

        # print("Received self.tomato_r=", self.tomato_r)


    def send_cameraEN(self,state): 
        msg = Bool()
        msg.data = bool(state)  # Μετατροπή σε True/False
        if state not in [0, 1]:
            rospy.logwarn("Invalid camera command! Must be 1 (open) or 0 (close).")
            return
        rospy.loginfo(f"Sending Camera Enable: {'OPEN (1)' if state == 1 else 'CLOSE (0)'}")
        self.enable_pub.publish(msg)
        #self.awaiting_finger_confirmation = True  # Μπαίνουμε σε αναμονή
        #self.wait_for_confirmation('/finger_status')

    def send_finger_command(self, state):
        """ 
        Στέλνει εντολή στο /finger_control 
        - 0 = Άνοιγμα 
        - 1 = Κλείσιμο 
        """

        if state not in [0, 1]:
            rospy.logwarn("Invalid finger command! Must be 1 (open) or 0 (close).")
            return

        msg = Bool()
        msg.data = bool(state)  # Μετατροπή σε True/False

        rospy.loginfo(f"Sending Finger Command: {'OPEN (1)' if state == 1 else 'CLOSE (0)'}")
        self.finger_pub.publish(msg)
        #self.awaiting_finger_confirmation = True  # Μπαίνουμε σε αναμονή
        #self.wait_for_confirmation('/finger_status')

    def send_thumb_position(self, x, y, angle):
        """ 
        Στέλνει εντολή στο /thumb_position ([x, y, angle]) και περιμένει επιβεβαίωση 
        """
        # x = 6
        # y = -2
        # angle = -np.pi/3
        msg = Float32MultiArray()
        msg.data = [x, y, angle]

        rospy.loginfo(f"Sending Thumb Position: x={x}, y={y}, angle={angle}")
        self.thumb_pub.publish(msg)
        #self.awaiting_thumb_confirmation = True  # Μπαίνουμε σε αναμονή
        #self.wait_for_confirmation('/thumb_status')

    def send_thumb_position_2(self, cmd):
        """ 
        Στέλνει εντολή στο /thumb_position ([x, y, angle]) και περιμένει επιβεβαίωση 
        """
        msg = Int32()
        msg.data = cmd

        rospy.loginfo(f"Sending Thumb Position: {cmd}")
        self.thumb_pub_2.publish(msg)
        #self.awaiting_thumb_confirmation = True  # Μπαίνουμε σε αναμονή
        #self.wait_for_confirmation('/thumb_status')


    def finger_status_callback(self, msg):
        """ 
        Callback για το /finger_status. Περιμένουμε 'success' ή 'error' για να συνεχίσουμε. 
        """
        if msg.data in ["success", "error"]:
            self.awaiting_finger_confirmation = False  # Αποδέσμευση αναμονής
            rospy.loginfo(f"Finger action confirmation received: {msg.data}")

    def thumb_status_callback(self, msg):
        """ 
        Callback για το /thumb_status. Περιμένουμε 'success' ή 'error' για να συνεχίσουμε. 
        """
        if msg.data in ["success", "error"]:
            self.awaiting_thumb_confirmation = False  # Αποδέσμευση αναμονής
            rospy.loginfo(f"Thumb action confirmation received: {msg.data}")

    def visual_obs_status_callback(self, msg):
        if msg.data in ["success", "error"]:
            self.awaiting_visual_obs_confirmation = False
            rospy.loginfo(f"Visual Observation confirmation received: {msg.data}")

    def load_urdf_model(self):
        """ Εκκινεί το URDF μοντέλο του ρομπότ μέσω roslaunch. """
        try:
            rospy.loginfo("Launching URDF model...")
            subprocess.Popen(["roslaunch", "robot_description", "ROBOT.launch"])
            rospy.loginfo("URDF model launched successfully!")
        except Exception as e:
            rospy.logerr(f"Failed to launch URDF model: {e}")
    def kinesthetic_status_callback(self, msg):
        """ Callback για το /kinesthetic_status """
        if msg.data in ["success", "error"]:
            self.awaiting_kinesthetic_confirmation = False  # Αποδεσμεύουμε την αναμονή για Kinesthetic
            rospy.loginfo(f"Kinesthetic confirmation received: {msg.data}")

   

    def motion_status_callback(self, msg):
        """ Διαχείριση μηνυμάτων κατάστασης από το /motion_status """
        rospy.loginfo(f"Received Motion Status: {msg.data}")
        if msg.data in ["success", "error"]:
            self.awaiting_confirmation = False  # Αποδεσμεύουμε την αναμονή

    def wait_for_confirmation(self, topic, timeout=10.0):
        """ Περιμένει για επιβεβαίωση από ένα συγκεκριμένο topic """
        rospy.loginfo(f"Waiting for confirmation from {topic}...")
        try:
            status_msg = rospy.wait_for_message(topic, String, timeout=timeout)
            if status_msg.data in ["success", "error"]:
                rospy.loginfo(f"Received confirmation from {topic}: {status_msg.data}")
                return status_msg.data
            else:
                rospy.logwarn(f"Received unknown confirmation message from {topic}.")
                return None
        except rospy.ROSException:
            rospy.logwarn(f"Timeout while waiting for confirmation from {topic}.")
            return None

    def send_motion_command(self, space, motion_type, target_pose, target_config, duration, tool):
        """ Στέλνει εντολή κίνησης στο /amg_command """
        motion_msg = HarvestCommand()
        motion_msg.space = space
        motion_msg.motion_type = motion_type
        motion_msg.target_pose.position.x = target_pose.t[0]
        motion_msg.target_pose.position.y = target_pose.t[1]
        motion_msg.target_pose.position.z = target_pose.t[2]
        quaternion = r2q(target_pose.R)  # Μετατροπή από Rotation Matrix σε Quaternion
        motion_msg.target_pose.orientation.x = quaternion[1]
        motion_msg.target_pose.orientation.y = quaternion[2]
        motion_msg.target_pose.orientation.z = quaternion[3]
        motion_msg.target_pose.orientation.w = quaternion[0]
        motion_msg.target_config = target_config
        motion_msg.duration = duration
        motion_msg.end_effector = tool

        #### ADDITION FOR SCALING
        print("self.tomato_r=", self.tomato_r)
        motion_msg.tomato_r = self.tomato_r

        rospy.loginfo(f"Sending motion command: {motion_msg}")
        self.motion_pub.publish(motion_msg)
        self.awaiting_confirmation = True  # Ενεργοποίηση αναμονής επιβεβαίωσης

    
    def send_kinesthetic_command(self, command):
        """ Στέλνει εντολή στο Kinesthetic και περιμένει επιβεβαίωση """
        rospy.loginfo(f"Sending kinesthetic command: {command}")
        self.kinesthetic_pub.publish(command)
        self.awaiting_kinesthetic_confirmation = True  # Ενεργοποίηση αναμονής για Kinesthetic

   
    def ok_redetect_callback(self, msg):
        self.ok_redect_pressed = True

        self.detection_decision = 0
        if msg.data == 'ok':
            self.detection_decision = 1

        return
     

    def command_callback(self, msg):
        """ Διαχείριση εισερχόμενων εντολών """
        command_str = msg.data.strip()
        if command_str in self.COMMAND_MAPPING:
            command_int = self.COMMAND_MAPPING[command_str]
            rospy.loginfo(f"Received: {command_str} -> Executing {command_int}")
            if command_int == 1:  # Open URDF + Move Home
                
                print("========================================================================== COMMAND 4 =========================================================================")
                self.enable_robot_pub.publish(True)

                rospy.loginfo("OPEN Finger...")
                self.send_finger_command(0)  # 1
                #thumb_open_position = [8, 4.5, 0]
                #self.send_thumb_position(*thumb_open_position)
                thumb_open_position = 3 
                self.send_thumb_position_2(thumb_open_position)
                rospy.loginfo(f"Moving Thumb to Open Position: {thumb_open_position}")
                
                
                rospy.loginfo("Go to HOME POSE...")

                home_pose = SE3(np.eye(4))
                home_config = np.array([-183, -90, 120, 140, -90, 0])
                home_config = home_config * math.pi / 180
                self.send_motion_command("joint", "home", home_pose, home_config, 3.0, 'camera')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
            
                rospy.loginfo("STATUS:Home Position.")
                self.enable_robot_pub.publish(False)

                time.sleep(4)
            #elif command_int == 2:  # Re-detect
            #    self.detection_decision = 0
            #    rospy.loginfo("Detection decision set to: RE-DETECT")

            #elif command_int == 3:  # Detection OK
            #    self.detection_decision = 1
            #    rospy.loginfo("Detection decision set to: OK")
            ########################################################################################
            #elif command_int in [2, 3]:
            #    self.detection_decision = command_int
            elif command_int == 4:  # Observation Pose

                print("========================================================================== COMMAND 4 =========================================================================")
                self.enable_robot_pub.publish(True)

                self.observation_counter =  self.observation_counter + 1



                time.sleep(4)
                
                rospy.loginfo("Executing Observation Pose...")
                home_pose = SE3(np.eye(4))
                home_config = np.array([-183, -90, 120, 140, -90, 0])
                home_config = home_config * math.pi / 180
                self.send_motion_command("joint", "home", home_pose, home_config, 2.0, 'camera')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)

                
                #self.detection_decision = None  # reset flag
                # while True:
                #     self.send_cameraEN(True)
                #     rospy.sleep(2.0)
                #     self.send_cameraEN(False)

                #     rospy.loginfo("Waiting for detection decision (Command 2 = RE-DETECT, Command 3 = OK)...")
                #     rate = rospy.Rate(10)
                #     timeout = rospy.Time.now() + rospy.Duration(30)
                #     while self.detection_decision is None and rospy.Time.now() < timeout:
                #         rate.sleep()

                #     if self.detection_decision == 3:
                #         rospy.loginfo("Detection confirmed. Proceeding.")
                #         break
                #     elif self.detection_decision == 2:
                #         rospy.loginfo("Re-detecting as requested...")
                #         self.detection_decision = None
                #     else:
                #         rospy.logwarn("No decision received. Repeating vision.")
                #         self.detection_decision = None
                continue_flag = False
                while not continue_flag:
                  
                    self.send_cameraEN(True)
                    visionrate = 30
                    rate = rospy.Rate(visionrate)
                    for i in range(60):
                        rate.sleep()
                    self.send_cameraEN(False)
                 

                

                    #if  input('Press: 0 -> re-detect,  1 -> Detection is ok .. ') != '0':
                    #     break
                    while True:
                        if self.ok_redect_pressed:

                            self.ok_redect_pressed = False

                            if self.detection_decision != 0:
                                continue_flag = True

                            break


                if confirmation != "success":
                    rospy.logerr("Failed to return to home position. Aborting sequence.")
                    return
                
                if self.g0T is None:
                    rospy.logwarn("No tomato pose available! Aborting observation sequence.")
                    return

               


                # Βήμα 1: Υπολογισμός observation pose
                self.gObservation = self.get_observation_pose()
                rospy.loginfo(f"Computed Observation Pose: {self.gObservation}")

                # Βήμα 2: Μετακίνηση στην observation pose
                rospy.loginfo("Moving to Observation Pose...")

                

                self.send_motion_command("task", "poly", self.gObservation, [0, 0, 0, 0, 0, 0], 2.0, 'camera')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10)
                if confirmation != "success":
                    rospy.logerr("Failed to move to observation pose. Aborting sequence.")
                    return
                self.current_phase = "observation"
                rospy.loginfo("Triggering visual observation node...")
                rospy.sleep(1.0)  # Μικρή αναμονή για να είναι έτοιμος ο publisher

                msg = Float32MultiArray()
                #msg.data = self.g0T.data[0:3,3]
                msg.data=self.gURT.A[0:3, 3]
                #msg.data.A[0:3, 3] = msg.data[0:3, 3]


                self.enable_robot_pub.publish(False)

                time.sleep(4)



                self.visual_obs_pub.publish(msg)
                rospy.loginfo("Visual observation node triggered successfully.")
                self.awaiting_visual_obs_confirmation = True
                confirmation = self.wait_for_confirmation('/active_perception_status', timeout=1000.0)
                self.ok_redect_pressed = False
                if confirmation != "success":
                    rospy.logerr("Visual observation failed.")
                return
                
            elif command_int == 5:  # Go Home


                print("========================================================================== COMMAND 5 =========================================================================")
                self.enable_robot_pub.publish(True)

                time.sleep(4)


                ########ANOIGMA gripper######
                rospy.loginfo("OPEN Finger...")
                self.send_finger_command(0)  # 1
                #thumb_open_position = [8, 4.5, 0]
                #self.send_thumb_position(*thumb_open_position)
                thumb_open_position = 3 
                self.send_thumb_position_2(thumb_open_position)
                rospy.loginfo(f"Moving Thumb to Open Position: {thumb_open_position}")


           
                
                
                home_pose = SE3(np.eye(4))
                home_config = np.array([-183, -90, 120, 140, -90, 0])
                home_config = home_config * math.pi / 180
                self.send_motion_command("joint", "home", home_pose, home_config, 3.0, 'camera')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                
                
                
                
                #self.detection_decision = None  # reset flag
                #while True:
                     
                #     self.send_cameraEN(True)
                #     rospy.sleep(2.0)
                #     self.send_cameraEN(False)

                #     rospy.loginfo("Waiting for detection decision (Command 2 = RE-DETECT, Command 3 = OK)...")
                     
                #     timeout = rospy.Time.now() + rospy.Duration(30)
                #     rate = rospy.Rate(10)
                #     while self.detection_decision is None and rospy.Time.now() < timeout:
                #         rate.sleep()

                #     if self.detection_decision == 3:
                #         rospy.loginfo("Detection confirmed. Proceeding.")
                #         break
                #     elif self.detection_decision == 2:
                #         rospy.loginfo("Re-detecting as requested...")
                #         self.detection_decision = None
                #     else:
                #         rospy.logwarn("No decision received. Repeating vision.")
                #         self.detection_decision = None
                    
                #print("self.gPregrasp = ", self.gPregrasp)
                #print("self.gGrasp = ", self.gGrasp)
                continue_flag = False
                while not continue_flag:
                  
                    self.send_cameraEN(True)
                    visionrate = 30
                    rate = rospy.Rate(visionrate)
                    for i in range(60):
                        rate.sleep()
                    self.send_cameraEN(False)


                    # print("self.gPregrasp = ", self.gPregrasp)
                    # print("self.gGrasp = ", self.gGrasp)
                 

                    # rospy.loginfo("Executing Harvest Sequence...")
                    # if self.g0T is None:
                    #     rospy.logwarn("No tomato pose available! Aborting harvest sequence.")
                    #     return
                 
                    # if  input('Press: 0 -> re-detect,  1 -> Detection is ok .. ') != '0':
                    #     break
                    while True:
                        if self.ok_redect_pressed:

                            self.ok_redect_pressed = False

                            if self.detection_decision != 0:
                                continue_flag = True

                            break



                # Βήμα 1: Μετακίνηση στην θέση pregrasp
                rospy.loginfo("Moving to Pregrasp Pose...")
                self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 3.0, 'gripper')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to pregrasp pose. Aborting sequence.")
                    return
                self.current_phase = "pre_grasp"

                # Βήμα 2: Μετακίνηση στην θέση grasp
                rospy.loginfo("Moving to Grasp Pose...")
                self.send_motion_command("task", "poly", self.gGrasp, [0, 0, 0, 0, 0, 0], 3.0, 'gripper')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to grasp pose. Aborting sequence.")
                    return
                self.current_phase = "grasp"

               
                #rospy.loginfo("Triggering Kinesthetic Correction...")
                #self.send_kinesthetic_command("start")
                #confirmation = self.wait_for_confirmation('/kinesthetic_status', timeout=10.0)
                #if confirmation != "success":
                #    rospy.logerr("Failed to start kinesthetic correction. Aborting sequence.")
                #    return

                

                
                rospy.loginfo("CLOSE Finger...")
                

                self.send_finger_command(1)  # 1 = Άνοιγμα Finger (αντί για "OPEN")

                # confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                # if confirmation != "success":
                #     rospy.logerr("Failed to open finger. Aborting sequence.")
                #     return

                # self.current_phase = "finger_open"


                # rospy.loginfo("Opening Finger...")
                # self.send_finger_command(1)  # 1 = Άνοιγμα Finger

                # confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                # if confirmation != "success":
                #     rospy.logerr("Failed to open finger. Aborting sequence.")
                #     return

                # Θέση του Thumb για ανοιχτό Finger
                # thumb_open_position = [6, -1, -np.pi/4]
                # self.send_thumb_position(*thumb_open_position)
                thumb_open_position = 2 
                self.send_thumb_position_2(thumb_open_position)
                rospy.loginfo(f"Moving Thumb to CLOSE Position: {thumb_open_position}")

                #rospy.loginfo("Go Home process completed.")
                # Βήμα 4: Μετακίνηση στην θέση καλαθιού

                self.enable_robot_pub.publish(False)
                
                time.sleep(1)
                # Βήμα 6: Εκκίνηση Kinesthetic Correction
                rospy.loginfo("Triggering Kinesthetic Correction...")
                self.send_kinesthetic_command("start")
                confirmation = self.wait_for_confirmation('/kinesthetic_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to start kinesthetic correction. Aborting sequence.")
                    return

                rospy.loginfo("Go Home process completed.")
                self.ok_redect_pressed = False

            elif command_int == 8:  # Execute Harvest

                print("========================================================================== COMMAND 8 =========================================================================")


                self.enable_robot_pub.publish(True)

                time.sleep(4)


                ########ANOIGMA gripper######
                rospy.loginfo("OPEN Finger...")
                self.send_finger_command(0)  # 1
                #thumb_open_position = [8, 4.5, 0]
                #self.send_thumb_position(*thumb_open_position)
                thumb_open_position = 3 
                self.send_thumb_position_2(thumb_open_position)
                rospy.loginfo(f"Moving Thumb to Open Position: {thumb_open_position}")

           
                
                
                home_pose = SE3(np.eye(4))
                home_config = np.array([-183, -90, 120, 140, -90, 0])
                home_config = home_config * math.pi / 180
                self.send_motion_command("joint", "home", home_pose, home_config, 3.0, 'camera')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                
                
                
                
                #self.detection_decision = None  # reset flag
                #while True:
                     
                #     self.send_cameraEN(True)
                #     rospy.sleep(2.0)
                #     self.send_cameraEN(False)

                #     rospy.loginfo("Waiting for detection decision (Command 2 = RE-DETECT, Command 3 = OK)...")
                     
                #     timeout = rospy.Time.now() + rospy.Duration(30)
                #     rate = rospy.Rate(10)
                #     while self.detection_decision is None and rospy.Time.now() < timeout:
                #         rate.sleep()

                #     if self.detection_decision == 3:
                #         rospy.loginfo("Detection confirmed. Proceeding.")
                #         break
                #     elif self.detection_decision == 2:
                #         rospy.loginfo("Re-detecting as requested...")
                #         self.detection_decision = None
                #     else:
                #         rospy.logwarn("No decision received. Repeating vision.")
                #         self.detection_decision = None
                    
                #print("self.gPregrasp = ", self.gPregrasp)
                #print("self.gGrasp = ", self.gGrasp)
                continue_flag = False

                                
                while not continue_flag:

                    #print('2. continue_flag  = ', continue_flag)
                    #print('2. self.detection_decision = ', self.detection_decision)
                    #print('1. self.ok_redect_pressed = ', self.ok_redect_pressed)
                  
                    self.send_cameraEN(True)
                    visionrate = 30
                    rate = rospy.Rate(visionrate)
                    for i in range(60):
                        rate.sleep()
                    self.send_cameraEN(False)


                    # print("self.gPregrasp = ", self.gPregrasp)
                    # print("self.gGrasp = ", self.gGrasp)
                 

                    # rospy.loginfo("Executing Harvest Sequence...")
                    # if self.g0T is None:
                    #     rospy.logwarn("No tomato pose available! Aborting harvest sequence.")
                    #     return
                 
                    # if  input('Press: 0 -> re-detect,  1 -> Detection is ok .. ') != '0':
                    #     break
                    while True:
                        
                        if self.ok_redect_pressed:

                            
                            self.ok_redect_pressed = False

                            if self.detection_decision != 0:
                                continue_flag = True

                            break

                    



                # Βήμα 1: Μετακίνηση στην θέση pregrasp
                rospy.loginfo("Moving to Pregrasp Pose...")
                self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 3.0, 'gripper')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to pregrasp pose. Aborting sequence.")
                    return
                self.current_phase = "pre_grasp"

                # Βήμα 2: Μετακίνηση στην θέση grasp
                rospy.loginfo("Moving to Grasp Pose...")
                self.send_motion_command("task", "poly", self.gGrasp, [0, 0, 0, 0, 0, 0], 3.0, 'gripper')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                # if confirmation != "success":
                #     rospy.logerr("Failed to move to grasp pose. Aborting sequence.")
                #     return
                self.current_phase = "grasp"

               
                #rospy.loginfo("Triggering Kinesthetic Correction...")
                #self.send_kinesthetic_command("start")
                #confirmation = self.wait_for_confirmation('/kinesthetic_status', timeout=10.0)
                #if confirmation != "success":
                #    rospy.logerr("Failed to start kinesthetic correction. Aborting sequence.")
                #    return

                

                
                rospy.loginfo("CLOSE Finger...")
                self.send_finger_command(1)  # 1 = Άνοιγμα Finger (αντί για "OPEN")

                # confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                # if confirmation != "success":
                #     rospy.logerr("Failed to open finger. Aborting sequence.")
                #     return

                # self.current_phase = "finger_open"


                # rospy.loginfo("Opening Finger...")
                # self.send_finger_command(1)  # 1 = Άνοιγμα Finger

                # confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                # if confirmation != "success":
                #     rospy.logerr("Failed to open finger. Aborting sequence.")
                #     return

                # Θέση του Thumb για ανοιχτό Finger
                #thumb_open_position = [7, -1, -np.pi/4]
                #self.send_thumb_position(*thumb_open_position)
                thumb_open_position = 2 
                self.send_thumb_position_2(thumb_open_position)
                rospy.loginfo(f"Moving Thumb to CLOSE Position: {thumb_open_position}")

                #rospy.loginfo("Go Home process completed.")
                # Βήμα 4: Μετακίνηση στην θέση καλαθιού
                
                time.sleep(1)

                #rospy.loginfo("Moving to Basket Position...")
                #basket_pose = self.get_basket_pose()
                #self.send_motion_command("task", "poly", basket_pose, [0, 0, 0, 0, 0, 0], 3.0, 'gripper')
                #confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                #if confirmation != "success":
                #    rospy.logerr("Failed to move to basket position. Aborting sequence.")
                #    return
                #self.current_phase = "basket"

               
                rospy.loginfo("Executing DMP...")
                self.send_motion_command("task", "dmp", SE3(np.eye(4)), [0, 0, 0, 0, 0, 0], 3.0, 'gripper')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10000.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to DMP pose. Aborting sequence.")
                    return
                self.current_phase = "DMP"


                #################################
                # Βήμα 6: Επιστροφή στην αρχική θέση (home)
                rospy.loginfo("Returning to Home Position...")

                home_pose = SE3(np.eye(4))
                home_config = qT = np.array([-270, -150, 150, 213, -95, 0])
                home_config = home_config * math.pi / 180
                self.send_motion_command("joint", "poly", home_pose, home_config, 3.0, 'camera')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to go to home. Aborting sequence.")
                    return
                self.current_phase = "home_again"


                rospy.loginfo("Leave to basket...")

                basket_pose = SE3(np.eye(4))
                basket_config = np.array([-192, -56, 111, 88, 82, 0])
                basket_config = basket_config * math.pi / 180
                self.send_motion_command("joint", "poly", basket_pose, basket_config, 5.0, 'gripper')
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to go to the basket. Aborting sequence.")
                    return
                self.current_phase = "basket"




                 ########ANOIGMA gripper######
                rospy.loginfo("OPEN Finger...")
                self.send_finger_command(0)  # 1
                #thumb_open_position = [8, 4.5, 0]
                # self.send_thumb_position(*thumb_open_position)
                thumb_open_position = 3 
                self.send_thumb_position_2(thumb_open_position)
                rospy.loginfo(f"Moving Thumb to Open Position: {thumb_open_position}")
                
                rospy.loginfo("Go Basket process completed.")
                

                # Stop logger script
                rospy.loginfo("Harvest Sequence Completed.")
                msg_log = Bool()
                msg_log.data = True  # Μετατροπή σε True/False
                self.stop_logger.publish(msg_log)
        else:
            rospy.logwarn(f"Unknown command received: {command_str}")

    def gripper_callback(self, msg):

        
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        p.shape = (3, 1)
        Q = UnitQuaternion(msg.pose.orientation.w, [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        R = Q.R
        
        gtemp = np.hstack((R, p))
        gtemp = np.vstack((gtemp, np.array([0, 0, 0, 1])))
        

        self.g_0gripper = SE3(gtemp)
  
    def camera_callback(self, msg):

        
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        p.shape = (3, 1)
        Q = UnitQuaternion(msg.pose.orientation.w, [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        R = Q.R
        
        gtemp = np.hstack((R, p))
        gtemp = np.vstack((gtemp, np.array([0, 0, 0, 1])))
       
        

        self.g_0camera = SE3(gtemp)

    def camera_ur_callback(self, msg):

        
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        p.shape = (3, 1)
        Q = UnitQuaternion(msg.pose.orientation.w, [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        R = Q.R
        
        gtemp = np.hstack((R, p))
        gtemp = np.vstack((gtemp, np.array([0, 0, 0, 1])))
       
        

        self.g_URcamera = SE3(gtemp)
       


    def tomato_pose_callback(self, msg):
        """ Υπολογίζει τους ομογενείς μετασχηματισμούς όταν εντοπιστεί η ντομάτα """
        #rospy.loginfo("Processing Tomato Pose...")
        p0T = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        p0T.shape = (3, 1)
        Q0T = UnitQuaternion(msg.pose.orientation.w, [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        R0T = Q0T.R

        ################3 SYMMETRY CHANGE !!!!!! ###################

        z_vis_tom = R0T[0:3,2].T
        z_vis_tom.shape = (3,1)

        N = np.eye(3)- z_vis_tom @ z_vis_tom.T
        # print('N= ', N)

        # print('N @ (np.array([0,0,0]) - p0T) =', N @ (np.array([0,0,0]) - np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])) )
        # This has to change when odom is up
        R0T[0:3,0] = N @ (np.array([0,0,0]) - np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])) 
        R0T[0:3,0] = R0T[0:3,0] / np.linalg.norm(R0T[0:3,0])

        
        R0T[0:3,1] = np.cross(R0T[0:3,2],R0T[0:3,0])

        #############################################################

        # print('R0T=',R0T )
        
        gtemp = np.hstack((R0T, p0T))
        gtemp = np.vstack((gtemp, np.array([0, 0, 0, 1])))
        #print(gtemp)
        try:
            # print('============================____________________=====')
            self.g0T =  self.g_0camera * SE3(gtemp)
            # print("self.g0T", self.g0T)
            self.gURT =  self.g_URcamera * SE3(gtemp)
            # print(" self.gURT ",  self.gURT )
            self.gPregrasp = self.get_pregrasp_pose()
            # print("self.gPregrasp ",  self.gPregrasp )
            self.gGrasp = self.get_grasp_pose()
            self.gObservation = self.get_observation_pose()
        except Exception as ex:
            print(f"Something went wrong: {ex}")
            return


        #print('gtemp : ' , gtemp )
        #print('self.g0T : ' , self.g0T )

        # rospy.loginfo(f"Tomato pose: \n{self.g0T}")
        # rospy.loginfo(f"Pre-grasp pose: \n{self.gPregrasp}")
        # rospy.loginfo(f"Grasp pose: \n{self.gGrasp}")
        # rospy.loginfo(f"Observation pose: \n{self.gObservation}")

    def get_grasp_pose(self):
        """ Υπολογίζει τη θέση grasp """
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/task_knowledge/graspPose.mat')
        ggrasp = f['ggrasp']
        ggrasp[0:3,3] = np.zeros(3)
        
        gTE = SE3(ggrasp)
        g0E = self.g0T * gTE
        return g0E

    def get_pregrasp_pose(self):
        """ Υπολογίζει τη θέση pregrasp """
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/task_knowledge/graspPose.mat')
        ggrasp = np.array(f['ggrasp'],dtype = float)
        
        ggrasp[0:3,3] = np.zeros(3)

        # ---------------------------------NEW
        ggrasp[0:3,3] = -0.05*ggrasp[0:3,0]
        ggrasp[0:3,3] = ggrasp[0:3,3] - 0.1*ggrasp[0:3,2]


        # print('ggrasp', ggrasp)
        # ---------------------------------NEW

        ###### ggrasp[0, 3] = ggrasp[0, 3] + 0.1
        # ---------------------------------OLD
        # ggrasp[0, 3] = ggrasp[0, 3] + 0.1
        # ggrasp[2, 3] = ggrasp[2, 3] + 0.05
        # ---------------------------------OLD
        # print('ggrasp=', ggrasp)
        # print('ggrasp shape=', ggrasp.shape)
        # print('g0T', self.g0T)
        
        gTE = SE3(ggrasp)
        # print('gTE', gTE)
        g0E = self.g0T * gTE

        # print('g0E', g0E)

        return g0E

    def get_observation_pose(self):
        """ Υπολογίζει τη θέση observation """
        g0E = SE3()
        g0T_mat = self.g0T.A  #

        
        g0E.A[0:3, 3] = g0T_mat[0:3, 3]

        
        xy = g0E.A[0:2, 3]
        z_dir = np.append(xy, 0)  # (x, y, 0)
        z_dir = z_dir / np.linalg.norm(z_dir)  # 

        
        y_dir = np.array([0, 0, -1])

        x_dir = np.cross(y_dir, z_dir)

        g0E.A[0:3, 0] = x_dir
        g0E.A[0:3, 1] = y_dir
        g0E.A[0:3, 2] = z_dir

        g0E.A[0:3, 3] = g0T_mat[0:3, 3] - 0.4 * z_dir

        if self.observation_counter == 2:
            g0E.A[0:3, 3] = g0E.A[0:3, 3] - 0.1 * y_dir


        return SE3(g0E.A)

    
    def get_basket_pose(self):
        """ Ορίζει τη θέση του καλαθιού """
        return self.g0T * SE3.Tx(0.5)  # Προσαρμογή της θέσης καλαθιού

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    harvest_motion = HarvestMotionNode()
    harvest_motion.run()