import cv2
import mediapipe as mp
import numpy as np
import time 
import pathlib
import scipy
import os
import getch
import math
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Bool
import roboticstoolbox as rt
import rtde_receive
import rtde_control
from spatialmath import  SE3, SO3
from scipy.linalg import block_diag
from pynput import keyboard
from cmaes import CMAwM
import seaborn as sns
from grasp_active_perception import *

import pandas as pd
import os


import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_base')
from CSRL_math import * 
from CSRL_orientation import * 
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_dmpy')
from dmpSE3 import * 


knowledge_path = "/home/carpos/catkin_ws/src/task_knowledge/"

# Our status publisher 
status_pub = rospy.Publisher('active_perception_status', String, queue_size=10)


# Declare math pi
pi = math.pi

# this is the message that triggers the vision node
trigger_vision_pub = rospy.Publisher('/grasp_enable', Bool, queue_size=10)

# GLOBAL VARIABLES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
grasp_g = np.identity(4)
hand_g = np.identity(4)
tomato_g = np.identity(4)
iteration_counter = 0



# This is the callback function for the high level command of detecting and saving grasp
def detect_and_save_grasp(data):
    global grasp_g

    msg = Bool()
    msg.data = True 
    rospy.loginfo('[Active perception node] Triggering vision ... ')
    trigger_vision_pub.publish(msg)



    for i in range(30):
        time.sleep(0.03)

        z_vis_tom = tomato_g[0:3,2].T
        z_vis_tom.shape = (3,1)

        N = np.eye(3)- z_vis_tom @ z_vis_tom.T
        # print('z_vis_tom=',z_vis_tom)
        # print('N=',N)
        tomato_g[0:3,0] = N @ (hand_g[0:3,3] - tomato_g[0:3,3]) 
        tomato_g[0:3,0] = tomato_g[0:3,0] / np.linalg.norm(tomato_g[0:3,0])

        # print("hand_g[0:3,3] - tomato_g[0:3,3] = ", hand_g[0:3,3] - tomato_g[0:3,3])
        # print("tomato_g[0:3,0] = ", tomato_g[0:3,0])
        tomato_g[0:3,1] = np.cross(tomato_g[0:3,2],tomato_g[0:3,0])
        grasp_g = np.linalg.inv(tomato_g) @ hand_g


    print("tomato_g:", tomato_g) 
    print("hand_g:", hand_g) 

    
    mdic = {"ggrasp": grasp_g}
    scipy.io.savemat(str(knowledge_path) + "graspPose.mat", mdic)

    msg.data = False 
    rospy.loginfo('[Active perception node] Stopping vision ... ')
    # trigger_vision_pub.publish(msg)
  


# This is the callback for getting the hand pose from vision
def get_hand_pose_from_vision(data):
    global hand_g

    hand_g = np.identity(4)

    
    # Set position (origin)
    hand_pos = np.array([data.pose.position.x,  data.pose.position.y, data.pose.position.z]) 
    # Compute orientation (rotation matrix to quaternion)
    hand_quat = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])
    hand_R = quat2rot(hand_quat)

    
    hand_g[0:3, 0:3] = hand_R
    hand_g[0:3, 3] = hand_pos


# This is the callback for getting the tomato pose from vision
def get_tomato_pose_from_vision(data):
    global tomato_g

    tomato_g = np.identity(4)

    
    # Set position (origin)
    tomato_pos = np.array([data.pose.position.x,  data.pose.position.y, data.pose.position.z]) 
    # Compute orientation (rotation matrix to quaternion)
    tomato_quat = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])
    tomato_R = quat2rot(tomato_quat)

    
    tomato_g[0:3, 0:3] = tomato_R
    tomato_g[0:3, 3] = tomato_pos


    # print('tomato_quat=', tomato_quat)
    # print('tomato_R=', tomato_R)




# This is the callback function for the high level commands
def start_observation_callback(data):
    global zed, Kcamera, ph, image_width, image_height, iteration_counter, hand_g


    # Enable the vision node
    msg = Bool()
    msg.data = True 
    rospy.loginfo('[Active perception node] Triggering vision ... ')
    trigger_vision_pub.publish(msg)
    
    pcenter = np.zeros(3)
    pcenter[0] = data.data[0]
    pcenter[1] = data.data[1]
    pcenter[2] = data.data[2]


    if iteration_counter == 0:
        go_optimal_pose(p_center=pcenter)

    iteration_counter = iteration_counter + 1


    while True:
                  
        detect_and_save_grasp(data)
        
        
        if  input('Press: 0 -> re-detect,  1 -> Detection is ok .. ') != '0':
            break
    
    

    # init time
    t = time.time() 


    # set the FPS
    fps = 30
    dt = 1.0 / fps


    # initialize arrays
    Q_array = np.array([0, 0, 0, 0])
    Q_array.shape = (4,1)
    p_array = np.array([0, 0, 0])
    p_array.shape = (3,1)

    # Recording has started
    print('[Active perception node] Recording started ... ')

    # initialize time
    t0 = time.time()

    
    # Show the message picture
    image = cv2.imread("/home/carpos/catkin_ws/src/Active_perception_CARPOS_node/Message.jpg", 0)
    cv2.imshow('Message', image)

    vision_rate = rospy.Rate(fps) 
        
    while True:

        # Integrate time
        t = time.time() - t0

        # print time progress
        print("\r t=%.3f" % t , 's', end = '\r')
                
        # get hand pose from global variable (hand_g is set from the vision callback)
        p = np.copy(hand_g[0:3, 3])
        R = np.copy(hand_g[0:3, 0:3])

        # print pose
        # print('p=', p)
        # print('R=', R)

        # save pose to the array
        Q = rot2quat(R)
        p.shape = (3,1)
        Q.shape = (4,1)
        Q_array = np.hstack((Q_array,Q))
        p_array = np.hstack((p_array,p))


        # Break if esc is pressed
        # if (cv2.waitKey(5) & 0xFF == 27) or t > 6:
        if (cv2.waitKey(5) & 0xFF == 27):
            break

        vision_rate.sleep()


    cv2.destroyAllWindows()
    # cap.release()


    msg.data = False 
    rospy.loginfo('[Active perception node] Stopping vision ... ')
    trigger_vision_pub.publish(msg)


    print("[Active perception node] Recording ended. Duration = %.3f" % t)


    dt = t / p_array[0,:].size
    print("[Active perception node] Estimated mean dt = %.3f" % dt)

    # Get initial pose
    Q0 = np.copy(Q_array[:,1])
    p0 = np.copy(p_array[:,1])
    Rc0 = quat2rot(Q0)
    R0c = Rc0.T
    Nsamples = Q_array[0,:].size

    # get pose wrt the initial pose
    for i in range(Nsamples):
        # print(i)
        p_array[:,i] = R0c @ (p_array[:,i] - p0)
        Rch = quat2rot(Q_array[:,i])
        Q_array[:,i] = rot2quat(R0c @ Rch)

    # save dataset array
    Q_array[:,1:] = makeContinuous(Q_array[:,1:])
    mdic = {"p_array": p_array[:,1:], "Q_array": Q_array[:,1:], "dt": dt, "Rrobc": R0c}
    print(str(knowledge_path) + 'training_demo.mat')
    scipy.io.savemat(str(knowledge_path) + 'training_demo.mat', mdic)

    # Train the DMP model
    kernelType = 'Gaussian' # other option: sinc
    canonicalType = 'linear' # other option: exponential

    folderPath = pathlib.Path(__file__).parent.resolve()


    data = scipy.io.loadmat(str(knowledge_path) +'training_demo.mat')

    # x_train = data['x_data']

    p_train = np.array(data['p_array'])
    Q_train = np.array(data['Q_array'])

    dt = data['dt'][0][0]
    t = np.array(list(range(p_train[0,:].size))) * dt

    dmpTask = dmpSE3(20, t[-1])

    print("dmpTask.T=", dmpTask.T)
    dmpTask.train(dt, p_train, Q_train, False)

    # Save the DMP model weights
    mdic = {"W": dmpTask.get_weights(), "T": dmpTask.T}
    scipy.io.savemat(str(knowledge_path) + "dmp_model.mat", mdic)

    print("[Active perception node]  Training completed. The dmp_model is saved.")

    # publishing success
    status_str = "success" 
    rospy.loginfo('[Active perception node]  Status: ' + status_str)
    status_pub.publish(status_str)


    

##################################
############# MAIN ###############
##################################
if __name__ == '__main__':

    print('[Active perception node] Node started.')
    rospy.init_node('carpos_active_perception', anonymous=True)
    rate = rospy.Rate(100) 

    rospy.Subscriber("start_active_perception", Float32MultiArray, start_observation_callback)
    rospy.Subscriber("detect_and_save_grasp_pose", String, detect_and_save_grasp)
    rospy.Subscriber("/hand_pose", PoseStamped, get_hand_pose_from_vision)
    rospy.Subscriber("/tomato_pose", PoseStamped, get_tomato_pose_from_vision)
    rospy.Subscriber("/keypoints", Float32MultiArray, get_features_from_vision)

    print('[Active perception node] The subscribers are initialized.')
    print('[Active perception node] Waiting for commands.')
 
    while not rospy.is_shutdown():

        # motion_finished_error()
   
        rate.sleep()
