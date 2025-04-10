#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import String
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
        self.tomato_sub = rospy.Subscriber('/tomato_frame', Pose, self.tomato_pose_callback)
                   
        self.visual_obs_pub = rospy.Publisher('/start_active_perception', String, queue_size=10, latch=True)

        self.visual_obs_status_sub = rospy.Subscriber('/start_active_perception_status', String, self.visual_obs_status_callback)
    
        self.command_sub = rospy.Subscriber('/command_topic', String, self.command_callback)
        
        self.kinesthetic_status_sub = rospy.Subscriber('/kinesthetic_status', String, self.kinesthetic_status_callback)  # Για Kinesthetic
        
        self.kinesthetic_pub = rospy.Publisher('/kinesthetic_correction', String, queue_size=10, latch=True)
        
        self.finger_pub = rospy.Publisher('/finger_control', Bool, queue_size=10)

        self.thumb_pub = rospy.Publisher('/thumb_position', Float32MultiArray, queue_size=10)

        self.finger_status_sub = rospy.Subscriber('/finger_status', String, self.finger_status_callback)

        self.thumb_status_sub = rospy.Subscriber('/thumb_status', String, self.thumb_status_callback)

        # Αρχικές καταστάσεις αναμονής
        self.awaiting_finger_confirmation = False
        self.awaiting_thumb_confirmation = False

        # Καταστάσεις και μεταβλητέςα
        self.current_phase = "idle"
        self.g0T = None
        self.awaiting_confirmation = False  # Για κινήσεις
        self.awaiting_kinesthetic_confirmation = False  # Για Kinesthetic
        self.awaiting_visual_obs_confirmation = False

    def send_finger_command(self, state):
        """ 
        Στέλνει εντολή στο /finger_control 
        - 1 = Άνοιγμα 
        - 0 = Κλείσιμο 
        """

        if state not in [0, 1]:
            rospy.logwarn("Invalid finger command! Must be 1 (open) or 0 (close).")
            return

        msg = Bool()
        msg.data = bool(state)  # Μετατροπή σε True/False

        rospy.loginfo(f"Sending Finger Command: {'OPEN (1)' if state == 1 else 'CLOSE (0)'}")
        self.finger_pub.publish(msg)
        self.awaiting_finger_confirmation = True  # Μπαίνουμε σε αναμονή
        self.wait_for_confirmation('/finger_status')

    def send_thumb_position(self, x, y, angle):
        """ 
        Στέλνει εντολή στο /thumb_position ([x, y, angle]) και περιμένει επιβεβαίωση 
        """

        msg = Float32MultiArray()
        msg.data = [x, y, angle]

        rospy.loginfo(f"Sending Thumb Position: x={x}, y={y}, angle={angle}")
        self.thumb_pub.publish(msg)
        self.awaiting_thumb_confirmation = True  # Μπαίνουμε σε αναμονή
        self.wait_for_confirmation('/thumb_status')

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

    def send_motion_command(self, space, motion_type, target_pose, target_config, duration):
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

        rospy.loginfo(f"Sending motion command: {motion_msg}")
        self.motion_pub.publish(motion_msg)
        self.awaiting_confirmation = True  # Ενεργοποίηση αναμονής επιβεβαίωσης

    
    def send_kinesthetic_command(self, command):
        """ Στέλνει εντολή στο Kinesthetic και περιμένει επιβεβαίωση """
        rospy.loginfo(f"Sending kinesthetic command: {command}")
        self.kinesthetic_pub.publish(command)
        self.awaiting_kinesthetic_confirmation = True  # Ενεργοποίηση αναμονής για Kinesthetic

   
    def command_callback(self, msg):
        """ Διαχείριση εισερχόμενων εντολών """
        command_str = msg.data.strip()
        if command_str in self.COMMAND_MAPPING:
            command_int = self.COMMAND_MAPPING[command_str]
            rospy.loginfo(f"Received: {command_str} -> Executing {command_int}")
            if command_int == 1:  # Open URDF + Move Home
                rospy.loginfo("Opening URDF model and moving to Home Position...")

                # Εκκίνηση του URDF μοντέλου
                self.load_urdf_model()

                # Μετακίνηση στη θέση home
                home_pose = SE3(np.eye(4))  # Ομογενής μετασχηματισμός ταυτότητας
                self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "home"
                self.wait_for_confirmation()  # Περιμένουμε success ή error
            
                rospy.loginfo("URDF Model loaded and Robot moved to Home Position.")
        
            elif command_int == 4:  # Observation Pose
                rospy.loginfo("Executing Observation Pose...")
                home_pose = SE3(np.eye(4))
                self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 3.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
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
                self.send_motion_command("task", "poly", self.gObservation, [0, 0, 0, 0, 0, 0], 10.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to observation pose. Aborting sequence.")
                    return
                self.current_phase = "observation"
                rospy.loginfo("Triggering visual observation node...")
                rospy.sleep(1.0)  # Μικρή αναμονή για να είναι έτοιμος ο publisher
                self.visual_obs_pub.publish("start")
                rospy.loginfo("Visual observation node triggered successfully.")
                self.awaiting_visual_obs_confirmation = True
                confirmation = self.wait_for_confirmation('/start_active_perception_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Visual observation failed.")
                return
                
            elif command_int == 5:  # Go Home
                rospy.loginfo("Going Home...")
                if self.g0T is None:
                    rospy.logwarn("No tomato pose available! Aborting command execution.")
                    return

                # Βήμα 1: Μετακίνηση στην αρχική θέση (home)
                rospy.loginfo("Moving to Home Position...")
                home_pose = SE3(np.eye(4))  # Ταυτότητα 4x4
                self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 10.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to home position. Aborting sequence.")
                    return
                self.current_phase = "home"

                # Βήμα 2: Υπολογισμός pregrasp και grasp poses
                rospy.loginfo("Computing pregrasp and grasp poses...")
                self.gPregrasp = self.get_pregrasp_pose()
                self.gGrasp = self.get_grasp_pose()

                # Βήμα 3: Μετακίνηση στην θέση pregrasp
                rospy.loginfo("Moving to Pregrasp Pose...")
                self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 10.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to pregrasp pose. Aborting sequence.")
                    return
                self.current_phase = "pre_grasp"

                # Βήμα 4: Μετακίνηση στην θέση grasp
                rospy.loginfo("Moving to Grasp Pose...")
                self.send_motion_command("task", "poly", self.gGrasp, [0, 0, 0, 0, 0, 0], 10.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to grasp pose. Aborting sequence.")
                    return
                self.current_phase = "grasp"

                # Βήμα 5: Κλείσιμο του Gripper
                rospy.loginfo("Closing Finger...")
                self.send_finger_command(0)  # 0 = Κλείσιμο Finger (αντί για "CLOSE")

                confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to close finger. Aborting sequence.")
                    return

                self.current_phase = "finger_close"
                #Κλεισιμο finger
                rospy.loginfo("Closing Finger...")
                self.send_finger_command(0)  # 0 = Κλείσιμο Finger

                confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to close finger. Aborting sequence.")
                    return

                # Θέση του Thumb για κλειστό Finger
                thumb_closed_position = [0.2, 0.1, -30.0] # na orisoume se ti pose tha kleinei
                rospy.loginfo(f"Moving Thumb to Closed Position: {thumb_closed_position}")
                self.send_thumb_position(*thumb_closed_position)
                # Βήμα 6: Εκκίνηση Kinesthetic Correction
                rospy.loginfo("Triggering Kinesthetic Correction...")
                self.send_kinesthetic_command("start")
                confirmation = self.wait_for_confirmation('/kinesthetic_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to start kinesthetic correction. Aborting sequence.")
                    return

                rospy.loginfo("Go Home process completed.")

            elif command_int == 8:  # Execute Harvest
                rospy.loginfo("Executing Harvest Sequence...")
                if self.g0T is None:
                    rospy.logwarn("No tomato pose available! Aborting harvest sequence.")
                    return

                # Βήμα 1: Μετακίνηση στην θέση pregrasp
                rospy.loginfo("Moving to Pregrasp Pose...")
                self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 3.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to pregrasp pose. Aborting sequence.")
                    return
                self.current_phase = "pre_grasp"

                # Βήμα 2: Μετακίνηση στην θέση grasp
                rospy.loginfo("Moving to Grasp Pose...")
                self.send_motion_command("task", "poly", self.gGrasp, [0, 0, 0, 0, 0, 0], 3.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to grasp pose. Aborting sequence.")
                    return
                self.current_phase = "grasp"

               
                rospy.loginfo("Triggering Kinesthetic Correction...")
                self.send_kinesthetic_command("start")
                confirmation = self.wait_for_confirmation('/kinesthetic_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to start kinesthetic correction. Aborting sequence.")
                    return

                rospy.loginfo("Go Home process completed.")
                # Βήμα 4: Μετακίνηση στην θέση καλαθιού
                rospy.loginfo("Moving to Basket Position...")
                basket_pose = self.get_basket_pose()
                self.send_motion_command("task", "poly", basket_pose, [0, 0, 0, 0, 0, 0], 3.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to move to basket position. Aborting sequence.")
                    return
                self.current_phase = "basket"

                
                rospy.loginfo("Opening Finger...")
                self.send_finger_command(1)  # 1 = Άνοιγμα Finger (αντί για "OPEN")

                confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to open finger. Aborting sequence.")
                    return

                self.current_phase = "finger_open"


                rospy.loginfo("Opening Finger...")
                self.send_finger_command(1)  # 1 = Άνοιγμα Finger

                confirmation = self.wait_for_confirmation('/finger_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to open finger. Aborting sequence.")
                    return

                # Θέση του Thumb για ανοιχτό Finger
                thumb_open_position = [0.5, 0.3, 0.0]
                rospy.loginfo(f"Moving Thumb to Open Position: {thumb_open_position}")
                self.send_thumb_position(*thumb_open_position)

                # Βήμα 6: Επιστροφή στην αρχική θέση (home)
                rospy.loginfo("Returning to Home Position...")
                home_pose = SE3(np.eye(4))  # Ταυτότητα 4x4
                self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 3.0)
                confirmation = self.wait_for_confirmation('/motion_status', timeout=10.0)
                if confirmation != "success":
                    rospy.logerr("Failed to return to home position. Aborting sequence.")
                    return
                self.current_phase = "home"

                rospy.loginfo("Harvest Sequence Completed.")
        else:
            rospy.logwarn(f"Unknown command received: {command_str}")

    def tomato_pose_callback(self, msg):
        """ Υπολογίζει τους ομογενείς μετασχηματισμούς όταν εντοπιστεί η ντομάτα """
        rospy.loginfo("Processing Tomato Pose...")
        p0T = np.array([msg.position.x, msg.position.y, msg.position.z])
        p0T.shape = (3, 1)
        Q0T = UnitQuaternion(msg.orientation.w, [msg.orientation.x, msg.orientation.y, msg.orientation.z])
        R0T = Q0T.R
        
        gtemp = np.hstack((R0T, p0T))
        gtemp = np.vstack((gtemp, np.array([0, 0, 0, 1])))
        print(gtemp)
        

        self.g0T = SE3(gtemp)
        self.gPregrasp = self.get_pregrasp_pose()
        self.gGrasp = self.get_grasp_pose()
        self.gObservation = self.get_observation_pose()

        rospy.loginfo(f"Tomato pose: \n{self.g0T}")
        rospy.loginfo(f"Pre-grasp pose: \n{self.gPregrasp}")
        rospy.loginfo(f"Grasp pose: \n{self.gGrasp}")
        rospy.loginfo(f"Observation pose: \n{self.gObservation}")

    def get_grasp_pose(self):
        """ Υπολογίζει τη θέση grasp """
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/odom_processor/scripts/graspPose.mat')
        ggrasp = f['ggrasp']
        gTE = SE3(ggrasp)
        g0E = self.g0T * gTE
        return g0E

    def get_pregrasp_pose(self):
        """ Υπολογίζει τη θέση pregrasp """
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/odom_processor/scripts/graspPose.mat')
        ggrasp = f['ggrasp']
        ggrasp[0, 3] = ggrasp[0, 3] + 0.1
        gTE = SE3(ggrasp)
        g0E = self.g0T * gTE
        return g0E

    def get_observation_pose(self):
        """ Υπολογίζει τη θέση observation """
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/odom_processor/scripts/graspPose.mat')
        ggrasp = f['ggrasp']
        ggrasp[0, 3] = ggrasp[0, 3] - 0.4
        gTE = SE3(ggrasp)
        g0E = self.g0T * gTE
        return g0E

    def get_basket_pose(self):
        """ Ορίζει τη θέση του καλαθιού """
        return self.g0T * SE3.Tx(0.5)  # Προσαρμογή της θέσης καλαθιού

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    harvest_motion = HarvestMotionNode()
    harvest_motion.run()