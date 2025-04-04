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
        
        self.motion_pub = rospy.Publisher('/amg_command', HarvestCommand, queue_size=10)
        self.status_sub = rospy.Subscriber('/motion_status', String, self.motion_status_callback)
        self.tomato_sub = rospy.Subscriber('/tomato_frame', Pose, self.tomato_pose_callback)
        self.command_sub = rospy.Subscriber('/command_topic', String, self.command_callback)  # Listen to GUI
        self.gripper_pub = rospy.Publisher('/gripper_command', String, queue_size=10, latch=True)
        self.detect_grasp_pub = rospy.Publisher('/decGRASP', String, queue_size=10, latch=True)
        self.kinesthetic_pub = rospy.Publisher('/kinesthetic_correction', String, queue_size=10, latch=True)
        self.dmp_pub = rospy.Publisher('/active_node', String, queue_size=10, latch=True)

        self.current_phase = "idle"
        self.g0T=None
        self.awaiting_confirmation = False  # Flag για αναμονή επιβεβαίωσης

        # self.gripper_initial_pose = None
        # self.listener = tf.TransformListener()

    #def process_gripper_pose(self, msg):
        #rospy.loginfo(f"Received gripper initial pose: x={msg.position.x}, y={msg.position.y}, z={msg.position.z}")
       # self.gripper_initial_pose = msg

    #def process_tomato(self, msg):
        #if self.gripper_initial_pose is None:
           # rospy.logwarn("Waiting for gripper initial pose...")
           # return

        #rospy.loginfo(f"Received tomato pose: x={msg.position.x}, y={msg.position.y}, z={msg.position.z}")
        
        #try:
           # self.listener.waitForTransform("tomato_frame", "gripper_frame", rospy.Time(0), rospy.Duration(1.0))
           # (trans, rot) = self.listener.lookupTransform("tomato_frame", "gripper_frame", rospy.Time(0))
           # rospy.loginfo(f"TF Transform Success: {trans}")
        #except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
           # rospy.logwarn("Failed to lookup transform between gripper and tomato frame. Retrying...")
           # return
        
        #pre_grasp_pose = Pose()
        #pre_grasp_pose.position.x = trans[0] + 0.1
        #pre_grasp_pose.position.y = trans[1]
        #pre_grasp_pose.position.z = trans[2]
        #pre_grasp_pose.orientation.x = rot[0]
        #pre_grasp_pose.orientation.y = rot[1]
        #pre_grasp_pose.orientation.z = rot[2]
        #pre_grasp_pose.orientation.w = rot[3]


        
        #self.send_motion_command("task", "poly", pre_grasp_pose, [0, 0, 0, 0, 0, 0], 2.0)
        #self.current_phase = "grasp"
    def wait_for_confirmation(self):
        rospy.loginfo("Waiting for success or error confirmation...")
        status_msg = rospy.wait_for_message('/motion_status', String)
        if status_msg.data in ["success", "error"]:
            self.awaiting_confirmation = False
            rospy.loginfo(f"Received confirmation: {status_msg.data}")
    
    def command_callback(self, msg):
        """ Διαχείριση εισερχόμενων εντολών """
        command_str = msg.data.strip()
        if command_str in self.COMMAND_MAPPING:
            command_int = self.COMMAND_MAPPING[command_str]
            rospy.loginfo(f"Received: {command_str} -> Executing {command_int}")

            if command_int == 4:  # Observation Pose
                rospy.loginfo("Executing Observation Pose...")

                # Έλεγχος αν έχουμε το tomato frame (self.g0T)
                if self.g0T is None:
                    rospy.logwarn("No tomato pose available! Aborting observation sequence.")
                    return
            
                # Υπολογισμός observation pose
                self.gObservation = self.get_observation_pose()
                rospy.loginfo(f"Computed Observation Pose: {self.gObservation}")

                # Αποστολή εντολής για μετάβαση στο observation pose
                rospy.loginfo("Moving to Observation Pose...")
                self.send_motion_command("task", "poly", self.gObservation, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "observation"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                # Trigger decGRASP & active_node
                rospy.loginfo("Triggering decGRASP node...")
                self.send_dec_grasp_command("start")

                rospy.loginfo("Triggering active_node...")
                self.send_dmp_command("execute")

                self.current_phase = "dec_grasp_active"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                rospy.loginfo("Observation process completed.")

            elif command_int == 5:
                rospy.loginfo("Going Home...")
                if self.g0T is None:
                    rospy.logwarn("No tomato pose available! Aborting command execution.")
                    return
                self.process_go_home()

            elif command_int == 8:
                rospy.loginfo("Starting Full Harvesting Sequence...")
                if self.g0T is None:
                    rospy.logwarn("No tomato pose available! Aborting command execution.")
                    return
                # Βήμα 1: Πήγαινε στο home
                rospy.loginfo("Moving to Home Position...")
                home_pose = SE3(np.eye(4))  # Ομογενής μετασχηματισμός ταυτότητας
                self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "home"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                # Βήμα 2: Υπολογισμός pregrasp & grasp
                rospy.loginfo("Computing pregrasp and grasp poses...")
                self.gPregrasp = self.get_pregrasp_pose()
                self.gGrasp = self.get_grasp_pose()

                # Βήμα 3: Πήγαινε στο pregrasp
                rospy.loginfo("Moving to Pregrasp Pose...")
                self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "pre_grasp"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                # Βήμα 4: Πήγαινε στο grasp
                rospy.loginfo("Moving to Grasp Pose...")
                self.send_motion_command("task", "poly", self.gGrasp, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "grasp"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                # Βήμα 5: Κλείσε τον gripper
                rospy.loginfo("Closing Gripper...")
                self.send_gripper_command("close")
                self.current_phase = "gripper_close"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                # Βήμα 6: Πήγαινε στο basket
                rospy.loginfo("Moving to Basket Position...")
                basket_pose = self.get_basket_pose()
                self.send_motion_command("task", "poly", basket_pose, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "basket"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                # Βήμα 7: Άνοιξε τον gripper
                rospy.loginfo("Opening Gripper...")
                self.send_gripper_command("open")
                self.current_phase = "waiting_gripper_open"
                self.wait_for_confirmation()

                if self.current_phase == "waiting_gripper_open":
                    rospy.loginfo("Gripper opened successfully. Returning Home...")
                    self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 2.0)
                    self.wait_for_confirmation()

                # Βήμα 8: Επιστροφή στο Home
                rospy.loginfo("Returning to Home Position...")
                self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "home"
                self.wait_for_confirmation()  # Περιμένουμε success ή error

                rospy.loginfo("Full Harvesting Sequence Completed.")
        else:
            rospy.logwarn(f"Unknown command received: {command_str}")

    def tomato_pose_callback(self, msg):
        """ Υπολογίζει τους ομογενείς μετασχηματισμούς όταν εντοπιστεί η ντομάτα, μόνο αν πατήθηκε το κουμπί 8. """
        if not hasattr(self, 'current_phase') or self.current_phase is None:
            rospy.logwarn("current_phase is None! Cannot process tomato pose.")
            return

        rospy.loginfo("Processing Tomato Pose...")
        p0T = np.array([msg.position.x, msg.position.y, msg.position.z])
        p0T.shape = (3,1)
        Q0T = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        R0T = quat2rot(Q0T)
        gtemp = np.hstack((R0T , p0T))
        gtemp = np.vstack((gtemp , np.array([0, 0, 0, 1])))

        self.g0T = SE3(gtemp)
        self.gPregrasp = self.get_pregrasp_pose()
        self.gGrasp = self.get_grasp_pose()
        self.gObservation = self.get_observation_pose()

        rospy.loginfo(f"Tomato pose: \n{self.g0T}")
        rospy.loginfo(f"Pre-grasp pose: \n{self.gPregrasp}")
        rospy.loginfo(f"Grasp pose: \n{self.gGrasp}")
        rospy.loginfo(f"Observation pose: \n{self.gObservation}")

        # Send->motion
        self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 2.0)
        self.current_phase = "pre_grasp"
        



    def get_grasp_pose(self):

        # read file of teaching 
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/odom_processor/scripts/graspPose.mat')
        ggrasp = f['ggrasp']

        gTE = SE3(ggrasp)

        g0E = self.g0T * gTE 
        #g0E=SE3(g0E.R,g0E.t)
        return g0E


    def get_pregrasp_pose(self):

        # read file of teaching 
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/odom_processor/scripts/graspPose.mat')
        ggrasp = f['ggrasp']

        ggrasp[0, 3] = ggrasp[0, 3] + 0.1

        gTE = SE3(ggrasp)
        

        g0E = self.g0T * gTE 
        
        #g0E=SE3(g0E.R,g0E.t)
        return g0E

    def get_observation_pose(self):
        """ (observation pose) """
        f = scipy.io.loadmat('/home/carpos/catkin_ws/src/odom_processor/scripts/graspPose.mat')
        ggrasp = f['ggrasp']
        ggrasp[0, 3] = ggrasp[0, 3] - 0.4  
        gTE = SE3(ggrasp)
        g0E = self.g0T * gTE 
        return g0E
    
    def process_harvest(self):
        """ Ξεκινά τη διαδικασία συγκομιδής """
        #if self.g0T is None:
            #rospy.logwarn("No tomato pose available! Aborting harvest sequence.")
           # return

        rospy.loginfo("Moving to Pregrasp Pose...")
        self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 2.0)
        self.current_phase = "pre_grasp"
        self.wait_for_confirmation()
    def process_go_home(self):
        """ Επαναφέρει το ρομπότ στη θέση εκκίνησης (home) και περιμένει επιβεβαίωση. """
        rospy.loginfo("Returning to home position...")
        home_pose = SE3(np.eye(4))  # Ταυτότητα 4x4
        self.send_motion_command("task", "home", home_pose, [0, 0, 0, 0, 0, 0], 2.0)
        self.current_phase = "home"
        self.wait_for_confirmation()  # Περιμένουμε success ή error

        rospy.loginfo("Home position reached. Computing pregrasp and grasp poses...")
        self.gPregrasp = self.get_pregrasp_pose()
        self.gGrasp = self.get_grasp_pose()

        rospy.loginfo("Moving to Pregrasp Pose...")
        self.send_motion_command("task", "poly", self.gPregrasp, [0, 0, 0, 0, 0, 0], 2.0)
        self.current_phase = "pre_grasp"
        self.wait_for_confirmation()  # Περιμένουμε success ή error

        rospy.loginfo("Moving to Grasp Pose...")
        self.send_motion_command("task", "poly", self.gGrasp, [0, 0, 0, 0, 0, 0], 2.0)
        self.current_phase = "grasp"
        self.wait_for_confirmation()  # Περιμένουμε success ή error

        rospy.loginfo("Closing gripper...")
        self.send_gripper_command("close")
        self.current_phase = "gripper_close"
        self.wait_for_confirmation()  # Περιμένουμε success ή error

        rospy.loginfo("Triggering Kinesthetic Correction...")
        self.send_kinesthetic_command("start")
        self.current_phase = "kinesthetic_correction"
        self.wait_for_confirmation()  # Περιμένουμε success ή error

        rospy.loginfo("Kinesthetic Correction completed.")


    def get_basket_pose(self):
        """ Ορίζει τη θέση του καλαθιού """
        return self.g0T * SE3.Tx(0.5)  # Προσαρμογή της θέσης καλαθιού
    
    def motion_status_callback(self, msg):
        """ Διαχείριση μηνυμάτων κατάστασης """
        rospy.loginfo(f"Received Motion Status: {msg.data}")
        if not hasattr(self, 'current_phase') or self.current_phase is None:
            rospy.logwarn("current_phase not initialized, setting to 'idle'")
            self.current_phase = "idle"
        if msg.data in ["success", "error"]:
            self.awaiting_confirmation = False  # Αποδεσμεύουμε την αναμονή

        if msg.data == "success":
            if self.current_phase == "observation":
                rospy.loginfo("Observation Pose completed. Returning Home...")
                self.process_go_home()

            elif self.current_phase == "pre_grasp":
                rospy.loginfo("Pre-Grasp completed. Moving to Grasp Pose...")
                self.send_motion_command("task", "poly", self.gGrasp, [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "grasp"

            elif self.current_phase == "grasp":
                rospy.loginfo("Grasp Pose completed. Closing gripper...")
                self.send_gripper_command("close")
                self.current_phase = "gripper_close"

            elif self.current_phase == "gripper_close":
                rospy.loginfo("Gripper closed. Executing DMP...")
                self.send_dmp_command("execute")
                self.current_phase = "dmp_executing"

            elif self.current_phase == "dmp_executing":
                rospy.loginfo("DMP Execution completed. Moving to basket...")
                self.send_motion_command("task", "basket", self.get_basket_pose(), [0, 0, 0, 0, 0, 0], 2.0)
                self.current_phase = "basket"

            elif self.current_phase == "basket":
                rospy.loginfo("Basket reached. Opening gripper...")
                self.gripper_pub.publish("open")
                self.current_phase = "waiting_gripper_open"

            elif self.current_phase == "waiting_gripper_open":
                rospy.loginfo("Gripper opened. Returning to home position...")
                self.process_go_home()
            elif self.current_phase == "home":
                rospy.loginfo("Returned to home position. Process complete.")
                rospy.loginfo("Press button 9 to continue...")
        elif msg.data == "error":
            rospy.logwarn("Motion command failed. Aborting sequence.")
            rospy.signal_shutdown("Motion failure.")
        
    def send_motion_command(self, space, motion_type, target_pose, target_config, duration):
        """ Στέλνει εντολή στο /amg_command και περιμένει επιβεβαίωση """
        if self.awaiting_confirmation:
            rospy.logwarn("Previous command not confirmed yet. Waiting...")
            self.wait_for_confirmation()  # Μπλοκάρουμε την εκτέλεση μέχρι να έρθει success

        """ Στέλνει εντολή κίνησης στο /amg_command """
        motion_msg = HarvestCommand()
        motion_msg.space = space
        motion_msg.motion_type = motion_type
        motion_msg.target_pose.position.x = target_pose.t[0]
        motion_msg.target_pose.position.y = target_pose.t[1]
        motion_msg.target_pose.position.z = target_pose.t[2]
        #motion_msg.target_pose.position.x = target_pose.t[0]
        #motion_msg.target_pose.position.y = target_pose.t[1]
        #motion_msg.target_pose.position.z = target_pose.t[2]

        #motion_msg.target_pose.orientation.x = target_pose.q[1]
        #motion_msg.target_pose.orientation.y = target_pose.q[2]
        #motion_msg.target_pose.orientation.z = target_pose.q[3]
        #motion_msg.target_pose.orientation.w = target_pose.q[0]  
        quaternion = r2q(target_pose.R)  # Μετατροπή από Rotation Matrix σε Quaternion
    
        motion_msg.target_pose.orientation.x = quaternion[0]
        motion_msg.target_pose.orientation.y = quaternion[1]
        motion_msg.target_pose.orientation.z = quaternion[2]
        motion_msg.target_pose.orientation.w = quaternion[3] 
       # motion_msg.target_pose.orientation.x, \
       # motion_msg.target_pose.orientation.y, \
       # motion_msg.target_pose.orientation.z = target_pose.rpy()
       # motion_msg.target_pose.orientation.w = 1.0 
        motion_msg.target_config = target_config
        motion_msg.duration = duration

        rospy.loginfo(f"Sending motion command: {motion_msg}")
        self.motion_pub.publish(motion_msg)
        self.awaiting_confirmation = True  # Ενεργοποίηση αναμονής επιβεβαίωσης
        #self.wait_for_confirmation()
    def send_gripper_command(self, command):
        """ Στέλνει εντολή στον Gripper και περιμένει επιβεβαίωση """
        if self.awaiting_confirmation:
            rospy.logwarn("Previous command not confirmed yet. Waiting...")
            self.wait_for_confirmation()

        rospy.loginfo(f"Sending gripper command: {command}")
        self.gripper_pub.publish(command)
        self.awaiting_confirmation = True
       # self.wait_for_confirmation()


    def send_dec_grasp_command(self, command):
        """ Στέλνει εντολή στο /decGRASP και περιμένει επιβεβαίωση """
        if self.awaiting_confirmation:
            rospy.logwarn("Previous command not confirmed yet. Waiting...")
            self.wait_for_confirmation()

        rospy.loginfo(f"Sending decGRASP command: {command}")
        self.detect_grasp_pub.publish(command)
        self.awaiting_confirmation = True
       # self.wait_for_confirmation()
    def send_kinesthetic_command(self, command):
        """ Στέλνει εντολή στο /kinesthetic_correction και περιμένει επιβεβαίωση """
        if self.awaiting_confirmation:
            rospy.logwarn("Previous command not confirmed yet. Waiting...")
            self.wait_for_confirmation()  # Περιμένει το προηγούμενο success

        rospy.loginfo(f"Sending kinesthetic correction command: {command}")
        self.kinesthetic_pub.publish(command)
        self.awaiting_confirmation = True  # Μπαίνει σε κατάσταση αναμονής
       # self.wait_for_confirmation()  # Περιμένει μέχρι να έρθει επιβεβαίωση


    def send_dmp_command(self, command):
        """ Στέλνει εντολή στο /active_node και περιμένει επιβεβαίωση """
        if self.awaiting_confirmation:
            rospy.logwarn("Previous command not confirmed yet. Waiting...")
            self.wait_for_confirmation()  # Περιμένει το προηγούμενο success

        rospy.loginfo(f"Sending DMP command: {command}")
        self.dmp_pub.publish(command)
        self.awaiting_confirmation = True  # Μπαίνει σε κατάσταση αναμονής
        #self.wait_for_confirmation()  # Περιμένει μέχρι να έρθει επιβεβαίωση
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    harvest_motion = HarvestMotionNode()
    harvest_motion.run()