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
import os
import time


def beep_ubuntu_start():
    # Use 'play' command from sox package (if available)
    os.system(f'play -nq -t alsa synth 1 sine 432')
\

def beep_ubuntu_end():
    # Use 'play' command from sox package (if available)
    os.system(f'play -nq -t alsa synth 1 sine 100')

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

ok_redect_pressed = False
detection_decision = 0

actObs_enabled = False


def ok_redetect_callback(msg):
        global ok_redect_pressed, detection_decision, actObs_enabled


        if actObs_enabled:
            ok_redect_pressed = True


            detection_decision = 0
            if msg.data == 'ok':
                detection_decision = 1
        
        else:
            pass

        return
     



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

# Function required for the active perception
def calculate_dR_d(q, choice):
    if choice < 0 or choice > 3:
        print("Error in choice!")
        return
    q = np.array(q)
    q = q / np.linalg.norm(q)

    dr_d_ = np.zeros((3,3))

    if choice == 0:
        dr_d_[0,:] = [4 * q[0],    -2 * q[3],  2 * q[2]]
        dr_d_[1,:] = [2 * q[3],    4 * q[0],  -2 * q[1]]
        dr_d_[2,:] = [-2 * q[2],   2 * q[1],   4 * q[0]]

    elif choice == 1:
        dr_d_[0, :] = [4 * q[1],    2 * q[2],    2 * q[3]]
        dr_d_[1, :] = [2 * q[2],    0,          -2 * q[0]]
        dr_d_[2, :] = [2 * q[3],    2 * q[0],    0       ]

    elif choice == 2:
        dr_d_[0, :] = [0,           2 * q[1],   2 * q[0]]
        dr_d_[1, :] = [2 * q[1],    4 * q[2],   2 * q[3]]
        dr_d_[2, :] = [-2 * q[0],   2 * q[3],   0       ]
    else:
        dr_d_[0, :] = [0,           -2 * q[0],  2 * q[1]]
        dr_d_[1, :] = [2 * q[0],    0,          2 * q[2]]
        dr_d_[2, :] = [2 * q[1],    2 * q[2],   4 * q[3]]

    return dr_d_

def go_LAIF_pose(pc):

    global knowledge_path, hand_g

    

    print('[Active] Connecting to the robot ... ')

    rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")


    print('[Active] Connected. ')



    UR_robot = rt.DHRobot([
        rt.RevoluteDH(d = 0.1807, alpha = pi/2),
        rt.RevoluteDH(a = -0.6127),
        rt.RevoluteDH(a = -0.57155),
        rt.RevoluteDH(d = 0.17415, alpha = pi/2),
        rt.RevoluteDH(d = 0.11985, alpha = -pi/2),
        rt.RevoluteDH(d = 0.11655)
    ], name='UR10e')
    tcp_offset_position = [-0.04, -0.0675, 0.067] #D415

    # Create the SE(3) transformation
    TCP_offset = SE3.Trans(*tcp_offset_position) 

    # Set TCP on UR5e model
    UR_robot.tool = TCP_offset

    q0 = np.array(rtde_r.getActualQ())
    # rtde_c.moveJ(q0_rad, 0.5, 0.5)


    g0c= UR_robot.fkine(q0)
    p0c = g0c.t
    R0c = g0c.R

    g0h = g0c * SE3(hand_g)

    train_data = scipy.io.loadmat(str(knowledge_path) +'/training_demo.mat')


    Rrobc = np.array(train_data['Rrobc'])
   

    fx = 350
    fy = 625

    #uncertainties
    sigma_low = 0.1
    sigma_high = 1.0
    es_low = pi/20
    es_high = pi
    Ks = np.array([sigma_low,sigma_low,sigma_high,es_high,es_high,es_low])
    S_c = np.diag(Ks * Ks)

    # This is a loop to get the hand pose
    for i in range(10):
        time.sleep(0.1)
   
    R0h = g0h.R

    Qun = np.zeros((6,6))
    Qun[0:3, 0:3] = R0h @ Rrobc @ S_c[0:3, 0:3] @ Rrobc.T @ R0h.T
    Qun[3:6, 3:6] = R0h @ Rrobc @ S_c[3:6, 3:6] @ Rrobc.T @ R0h.T

    Q_inv = np.linalg.inv(Qun[0:3, 0:3])


    # Set the covariance matrix of the camera
    sigma_1 = 40.0
    sigma_2 = 40.0
    sigma_3 = 0.3

    Q_array = np.array([0, 0, 0, 0])
    p_array = np.array([0, 0, 0])



    # print('R0h=', R0h )
    # print('Rrobc=', Rrobc)
    # print('R0h @ Rrobc=', R0h @ Rrobc)
    # print(Qun[0:3, 0:3])

    # sadasdad
    dt = 0.002

    
    #  params --------------------------------------------------------------
    # ka = 30000.0
    ka = 5000.0
    LAIF_transient = 8
    

    print("[LAIF] Finding the next pose of the camera ... ")
    print("[LAIF] It will take ", LAIF_transient ," seconds ..." )

    t = 0
    while t<LAIF_transient:
        # Start control loop - synchronization with the UR
        t_start = rtde_c.initPeriod()
        
        # Integrate time
        t = t + dt

        # Get joint values
        q = np.array(rtde_r.getActualQ())

        # get robot pose
        g= UR_robot.fkine(q)
        p = g.t
        R = g.R
        Quat = rot2quat(R)

        # initialize v_p
        v_p = np.zeros(6)
        

        
        # get full jacobian
        J = np.array(UR_robot.jacob0(q))


        # compute Sigma now

        pcf_hat = hand_g[0:3, 3]

        sigma_x_sq = pow(sigma_1/fx,2)*pow(sigma_3,2)+(pow(pcf_hat[0]/pcf_hat[2],2))*pow(sigma_3,2)+pow(pcf_hat[2]*sigma_1/fx,2)
        sigma_y_sq = pow(sigma_2/fy,2)*pow(sigma_3,2)+(pow(pcf_hat[1]/pcf_hat[2],2))*pow(sigma_3,2)+pow(pcf_hat[2]*sigma_2/fy,2)
        sigma_d_sq = pow(sigma_3,2)
        Sigma_1 = np.diag(np.array([sigma_x_sq, sigma_y_sq, sigma_d_sq]))
     
        Sigma_now = R  @ Sigma_1  @ R.T

        Sigma_inv = np.linalg.inv(Sigma_now)

        P = np.linalg.inv( Q_inv + Sigma_inv )

        detP = np.linalg.det(P)
        invP = np.linalg.inv(P)

        Jq = getJq(Quat)

        Spp = skewSymmetric(pc-p)

        A = P @ P  @ Sigma_inv @ Sigma_inv

        ddet_dq = np.zeros(4)
        for j in range(4):

            dR_dqi = calculate_dR_d(Quat, j)
            dSigma_dqi = dR_dqi @ Sigma_1  @ np.transpose(R) + R  @ Sigma_1  @ np.transpose(dR_dqi)
            dP_dqi = A @ dSigma_dqi
            ddet_dq[j] = detP * np.trace( invP @ dP_dqi )
        # END FOR

        # print('Jq=', Jq)
        # print('ddet_dq=', ddet_dq)


        # time.sleep(10)

        v_p = np.zeros(6)
        if np.linalg.norm(p-p0c)<0.20:
            v_p[3:6] = - ka * np.transpose(Jq) @ ddet_dq
            v_p[0:3] = Spp @ v_p[3:6]
        else:
            print("[LAIF controller] The limit is reached !")
        # if p[2]<0.20:
        #     v_p[2]=0.0


        if v_p[2] < 0:
            v_p = np.zeros(6)

        # print("v_p= ", v_p)
        # Inverse kinematics mapping with singularity avoidance
        qdot = np.linalg.pinv(J, 0.1) @ ( v_p )


        Q_array = np.vstack((Q_array,Quat))
        p_array = np.vstack((p_array,p))
        

        # set joint speed with acceleration limits
        # qdot = np.zeros(6)
        # =========================================
        rtde_c.speedJ(qdot, 1.0, dt)

        # This is for synchronizing with the UR robot
        rtde_c.waitPeriod(t_start)

    print("[LAIF] Finish")
    #print(f"time: {ellapsed_time}")
    rtde_c.speedStop()

    print('[LAIF] Disconnecting from the robot ...')
    # rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()

    experiment_data = {
        'Qrobot':Q_array,
        'probot':p_array,
        'Qun' : Qun
    }

    # Save each experiment's data to a separate .mat file
    savemat('/home/carpos/catkin_ws/src/logging/LAIF_logging.mat', experiment_data)
    print('[LAIF] Logging data saved.')

    # we return sigma and Q with respect to the initial hand frame
    return  R0h.T @ Sigma_now @ R0h, Rrobc @ S_c[0:3, 0:3] @ Rrobc.T



# This is the callback function for the high level commands
def start_observation_callback(data):
    global zed, Kcamera, ph, image_width, image_height, iteration_counter, hand_g, ok_redect_pressed, detection_decision, actObs_enabled


    actObs_enabled = True

    # Enable the vision node
    msg = Bool()
    msg.data = True 
    rospy.loginfo('[Active perception node] Triggering vision ... ')
    trigger_vision_pub.publish(msg)
    
    pcenter = np.zeros(3)
    pcenter[0] = data.data[0]
    pcenter[1] = data.data[1]
    pcenter[2] = data.data[2]

    print('[Active Perception] pcenter=', pcenter)

    Sigma = np.zeros((3,3))
    Qkf = np.zeros((3,3))
    Sigma_inv = np.zeros((3,3))
    Q_inv = np.zeros((3,3))

    if iteration_counter == 0:
        go_optimal_pose(p_center=pcenter)
        # pass

    if iteration_counter == 1:
        Sigma, Qkf = go_LAIF_pose(pc=pcenter)
        Sigma_inv = np.linalg.inv(Sigma)
        Q_inv = np.linalg.inv(Qkf)

    

    


    continue_flag = False
    while not continue_flag:
                  
        detect_and_save_grasp(data)

        print('Is the initial pose correct?')
        
        
        while True:
            if ok_redect_pressed:

                print('Ok_redetect pressed !!!!   ', detection_decision)

                ok_redect_pressed = False

                if detection_decision != 0:
                    continue_flag = True

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

    
    N_prev = 10000000000000

    if iteration_counter == 1:

        print(str(knowledge_path) + 'training_demo.mat')

        data = scipy.io.loadmat(str(knowledge_path) +'training_demo.mat')
        p_train_prev = np.array(data['p_array'])
        Q_train_prev = np.array(data['Q_array'])

        dt_prev = data['dt'][0][0]
        Rrobc_prev = np.array(data['Rrobc'])

        N_prev = p_train_prev[0,:].size

        mdic_prev = {"p_array": p_train_prev, "Q_array": Q_train_prev, "dt": dt_prev, "Rrobc": Rrobc_prev}
        
        scipy.io.savemat(str(knowledge_path) + 'training_demo_prev.mat', mdic_prev)



   


    
    # Show the message picture
    image = cv2.imread("/home/carpos/catkin_ws/src/Active_perception_CARPOS_node/Message.jpg", 0)
    cv2.imshow('Message', image)

    vision_rate = rospy.Rate(fps) 

    n = 0

    beep_ubuntu_start()
        
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

        
        if iteration_counter == 1 and (p_array[0,:].size == p_train_prev[0,:].size):
            break

        n = n + 1

        vision_rate.sleep()


    cv2.destroyAllWindows()
    # cap.release()

    beep_ubuntu_end()


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

        if iteration_counter == 1:
            p_array[:,i] = np.linalg.inv( Q_inv + Sigma_inv ) @ ( (Q_inv @ p_train_prev[:,i]) + (Sigma_inv @ p_array[:,i]) )
            Q_array[:,i] = (Q_train_prev[:,i] + Q_array[:,i])/2
            Q_array[:,i] = Q_array[:,i] / math.sqrt( Q_array[0,i]*Q_array[0,i]  + Q_array[1,i]*Q_array[1,i] + Q_array[2,i]*Q_array[2,i] + Q_array[3,i]*Q_array[3,i] ) 



    if iteration_counter == 1:
        Q0c = (rot2quat(R0c) + rot2quat(Rrobc_prev))/2 
        Q0c = Q0c / math.sqrt( Q0c[0]*Q0c[0]  + Q0c[1]*Q0c[1] + Q0c[2]*Q0c[2] + Q0c[3]*Q0c[3] ) 
        R0c = quat2rot(Q0c)

    # save dataset array
    Q_array[:,1:] = makeContinuous(Q_array[:,1:])
    mdic = {"p_array": p_array[:,1:], "Q_array": Q_array[:,1:], "dt": dt, "Rrobc": R0c}
    
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

    ##################################  CHANGE COUNTER !!!!!!!!!!!!!!!!!!!!!
    iteration_counter = iteration_counter + 1

    actObs_enabled = False


    

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

    rospy.Subscriber('/ok_redetect', String, ok_redetect_callback, buff_size=1)


    print('[Active perception node] The subscribers are initialized.')
    print('[Active perception node] Waiting for commands.')
 
    while not rospy.is_shutdown():

        # motion_finished_error()
   
        rate.sleep()
