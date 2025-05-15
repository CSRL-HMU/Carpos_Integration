#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Bool
# from custom_msgs.msg import HarvestCommand
from CARPOS_amg.msg import amg_message
import numpy as np
import roboticstoolbox as rt
import scipy as sp
import spatialmath as sm
import matplotlib.pyplot as plt
import math
import time
from CSRL_control import *
from CSRL_trajectory import *
import rtde_receive
import rtde_control
import scipy.io 
from spatialmath import SE3, SO3
from geometry_msgs.msg import PoseStamped, Quaternion



import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_base')
from CSRL_orientation import * 
from CSRL_math import * 
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_dmpy')
from dmpSE3 import * 




# Our status publisher 
status_pub = rospy.Publisher('motion_status', String, queue_size=10)

gripper_pose_pub = rospy.Publisher('gripper_pose',PoseStamped , queue_size=10)
camera_pose_pub = rospy.Publisher('camera_pose', PoseStamped, queue_size=10)

rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")

# DMP params
kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential


# Declare math pi
pi = math.pi


# initial configuration
q0 = np.array(rtde_r.getActualQ())
q = q0

ur = rt.DHRobot([
    rt.RevoluteDH(d = 0.1807, alpha = pi/2),
    rt.RevoluteDH(a = -0.6127),
    rt.RevoluteDH(a = -0.57155),
    rt.RevoluteDH(d = 0.17415, alpha = pi/2),
    rt.RevoluteDH(d = 0.11985, alpha = -pi/2),
    rt.RevoluteDH(d = 0.11655)
], name='UR10e')

# # Define the tool
p_wrist_camera = SE3(-0.04, -0.0675, 0.067) #D415


rotation_matrix = SO3.AngVec(-pi/2, np.array([0,0,1]))
p_wrist_gripper = SE3(rotation_matrix) * SE3(0.0, 0.031, 0.04)  # Position of the tool (in meters)
# ur.tool = tool_position 


# Get initial end-eefector pose
g0 = ur.fkine(q0)
R0 = np.array(g0.R)
p0 = np.array(g0.t)



# Control cycle
dt = 0.002

# initialize qdot
qdot = np.zeros(6)

print('p0 = ', p0)
print('R0 = ', R0)


# The path in which the knowledge is stored
knowledge_path = "/home/carpos/catkin_ws/src/task_knowledge/"



# This function publishes success
def motion_finished_ok():
    status_str = "success" 
    rospy.loginfo('Automatic motion generation, motion status: ' + status_str)
    status_pub.publish(status_str)

# This function publishes error
def motion_finished_error():
    status_str = "error" 
    rospy.loginfo('Automatic motion generation, motion status: ' + status_str)
    status_pub.publish(status_str)


# This is the callback function for the high level commands
def amg_enable_robot(data):
    global rtde_c, rtde_r
    if data.data == True:
        print('Connecting to the robot!')

        if not rtde_c.isConnected():
            rtde_c.reconnect()
            rtde_r.reconnect()
    else:
        print('Disconnecting from the robot ...')
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()



# This is the callback function for the high level commands
def amg_command_callback(data):

    T = data.duration
    t = 0

    # default
    ur.tool = p_wrist_camera

    if data.end_effector == 'gripper':
        ur.tool = p_wrist_gripper
    elif data.end_effector == 'camera':
        ur.tool = p_wrist_camera
    else:
        print('Going with the default end-effector (camera)')


    if data.space == 'task':
        
        # get target
        pT = np.array( [data.target_pose.position.x, data.target_pose.position.y, data.target_pose.position.z] )
        QT = np.array( [data.target_pose.orientation.w, data.target_pose.orientation.x, data.target_pose.orientation.y, data.target_pose.orientation.z ] )
       
        print('[AMG] Received task target:')
        print('pT = ', pT)
        print('QT = ', QT)

        # Get initial end-eefector pose
        q0 = np.array(rtde_r.getActualQ())
        g0 = ur.fkine(q0)
        R0 = np.array(g0.R)
        p0 = np.array(g0.t)
        Q0 = rot2quat(R0)

        print('[AMG] Initial pose:')
        print('p0 = ', p0)
        print('Q0 = ', Q0)
        p = p0
        Q = Q0
        
       
        if data.motion_type == 'reach':

            print('[AMG] Reaching ... ')
            while t<T:
                
                t_start = rtde_c.initPeriod()

                #integrate time
                t = t + dt
                
                # get q
                q = np.array(rtde_r.getActualQ())

                # Get  end-eefector pose
                g = ur.fkine(q)
                R = np.array(g.R)
                p = np.array(g.t)
                Q = rot2quat(R)

                # get full jacobian
                J = np.array(ur.jacob0(q))

                # reaching control signal
                qdot = kinReaching_SE3(p=p, A=R, pT=pT, AT=QT, J=J, kr = 5.0/T)

                rtde_c.speedJ(qdot, 1.0, dt)

                rtde_c.waitPeriod(t_start)


        elif data.motion_type == 'poly':

            
            print('[AMG] Moving with poly ... ')
            while t<T:
                
                t_start = rtde_c.initPeriod()

                #integrate time
                t = t + dt
                
                # get q
                q = np.array(rtde_r.getActualQ())

                # Get  end-eefector pose
                g = ur.fkine(q)
                R = np.array(g.R)
                p = np.array(g.t)
                Q = rot2quat(R)

                # get full jacobian
                J = np.array(ur.jacob0(q))

                #generate trajectory
                pd, Rd, pd_dot, omegad = get5thorder_SE3(p0=p0, A0=R0, pT=pT, AT=QT, t=t, T=T)

                # tracking control signal
                qdot = kinTracking_SE3(p=p, A=R, pd=pd, Ad=Rd, pd_dot=pd_dot, omegad=omegad, J=J)

                rtde_c.speedJ(qdot, 1.0, dt)

                rtde_c.waitPeriod(t_start)


        elif data.motion_type == 'dmp':
            print('[AMG] Executing DMP fro mthe initial pose of the end-effector')

            # Import current dmp model
            data = scipy.io.loadmat(str(knowledge_path) +'/dmp_model.mat')

   

            # These are the weights of the DMP
            W = np.array(data['W']) 

            # The total duration 
            T = data['T'][0][0]
            print("T=",T)

            # Import training data 
            train_data = scipy.io.loadmat(str(knowledge_path) +'/training_demo.mat')


            # Rrobc = np.array(train_data['Rrobc'])

            p_train = np.array(train_data['p_array'])
            Q_train = np.array(train_data['Q_array'])


            # Set the initial and target values 
            # the DMP is xonstructed with respect to the initial pose of the end-effector
            space_scaling = 1.0
            p0 = p_train[:,0] 
            pT = space_scaling * p_train[:,-1] 

            Q0 = Q_train[:,0] 
            QT = Q_train[:,-1]

            # Create the DMP SE(3) object
            dmpTask = dmpSE3(N_in=W[0,:].size, T_in=T)

            # Loading the weights as taken from the saved model
            dmpTask.set_weights(W, T, Q0=Q0, Qtarget=QT)

            #  Setting the params of DMP
            dmpTask.set_goal(pT, QT)
            dmpTask.set_init_pose(p0, Q0)

            # Time scaling (if any ...)
            tau = 1
            dmpTask.set_tau(tau)

            q0 = np.array(rtde_r.getActualQ())

            # initialkizing the q_sim variable for the simuation
            q_sim = q0.copy()

            #get initial end effector position 
            g0rob = ur.fkine(q0)
            R0rob = np.array(g0rob.R)
            p0rob = np.array(g0rob.t)
            Q0rob = rot2quat(R0rob)


            #initialize DMP state

            # print('p0=', p0)
            pd = p0.copy()  # Initially, setting the desired position to the initial position p0
            dot_pd = np.zeros(3)

            ddp = np.zeros(3)
            dp = np.zeros(3)
            ddeo = np.zeros(3)
            deo = np.zeros(3)

            Q_desired = Q0.copy()  # Initially, setting the desired orientation to the initial orientation R0
            Q = Q0.copy()

            eo = logError(QT, Q0)  # Initial orientation error
            dot_eo = np.zeros(3)
            ddot_eo = np.zeros(3)
            omegad = np.zeros(3)
            dot_omegad = np.zeros(3)

            z = 0.0   #phase variable
            dz = 0.0

            t = 0


            while z < 1.2:

            
                t_start = rtde_c.initPeriod()


                # Integrate time
                t = t + dt

                # Euler integration to update the states
                z += dz * dt
                pd += dp * dt   
                eo += deo * dt
                dot_eo += ddeo * dt
                dot_pd += ddp * dt


            
                # Calculate DMP state derivatives (get z_dot, p_dot, pdot_dot)
                # dz, dp, ddp = model.get_state_dot(z, pd, dot_pd)
                # get state dot
                dz, dp, ddp, deo, ddeo = dmpTask.get_state_dot( z, 
                                                                pd,                                                                                      
                                                                dot_pd, 
                                                                eo,
                                                                dot_eo)
                
                
                Q_desired =  quatProduct( quatInv( quatExp( 0.5 * eo ) ) , QT )
                term3 = Jlog(quatProduct(QT,quatInv(Q_desired))) @ dot_eo
                Qdot = - 0.5 * quatProduct( quatProduct( quatProduct(Q_desired, quatInv(QT)) , term3), Q_desired) 
                QdotQinv = quatProduct(Qdot,quatInv(Q_desired))
                omegad = 2 * QdotQinv[1:4]
                # omegad = quat2rot(QT) @ omegad


                # translate everything to the world frame
                pdw = p0rob + R0rob @ pd
                Qdw = rot2quat(R0rob @ quat2rot(Q_desired))
                dot_pdw = R0rob @ dot_pd
                omegadw = R0rob @ omegad

                
                # Get the actual joint values 
                q = np.array(rtde_r.getActualQ())
                if simOn:
                    q = q_sim.copy()

                # Get  end-efector pose
                g = ur.fkine(q)
                R = np.array(g.R)
                p = np.array(g.t)

                
                

                # velocity array
                velocity_matrix = np.hstack((dot_pdw, omegadw))
                
                # error matrix 
                Q = rot2quatCont(R,Q)
                eo_robot = logError(Q, Qdw)
                error_matrix = np.hstack((p - pdw, eo_robot))

                # translate everything to the world frame
                pdw = p0rob + R0rob @ pd
                Qdw = rot2quat(R0rob @ quat2rot(Q_desired))
                dot_pdw = R0rob @ dot_pd
                omegadw = R0rob @ omegad

                # get full jacobian
                J = np.array(ur.jacob0(q))

                # commanded joint velocity
                qdot_command = kinTracking_SE3(p=p, A=Q, pd=pdw, Ad=Qdw, pd_dot=dot_pdw, omegad=omegadw, J=J)


                    
                # qdot = np.zeros(6)
                
                QT = Qdw.copy()
                pT = pdw.copy()

                # set joint speed
                rtde_c.speedJ(qdot_command, 1, dt) # comment for Kinesthetic only
                

                # synchronize
                rtde_c.waitPeriod(t_start)
                
            
            
            rtde_c.speedStop()                              
      

        ## Checking for Errors
        if np.linalg.norm(p-pT) < 0.05 and np.linalg.norm(logError(Q,QT)) < 5*pi/180:
            print('[AMG] Motion finished successfully')
            motion_finished_ok()
        else:
            print('[AMG] ERROR: Motion finished with an error greater than the set threshold!')
            print('[AMG] ||ep|| (m) = ', np.linalg.norm(p-pT) )
            print('[AMG] ||eo|| (rad) = ', np.linalg.norm(logError(Q,QT)))
            motion_finished_error()

    elif data.space == 'joint':

        # get target
        qT = np.array( data.target_config )
        

        # Get initial end-eefector pose
        q0 = np.array(rtde_r.getActualQ())

        q = q0
       

        print('[AMG] Initial configuration:')
        print('q0 = ', q0)


        if data.motion_type == 'reach':
            print('[AMG] Received joint target:')
            print('qT = ', qT)

            print('[AMG] Reaching ... ')
            while t<T:
                
                t_start = rtde_c.initPeriod()

                #integrate time
                t = t + dt
                
                # get q
                q = np.array(rtde_r.getActualQ())

                # reaching control signal
                qdot = kinReaching_Ndof(x=q, xT=qT, kr = 5.0/T)

                rtde_c.speedJ(qdot, 1.0, dt)

                rtde_c.waitPeriod(t_start)

            
        elif data.motion_type == 'poly':
            print('[AMG] Received joint target:')
            print('qT = ', qT)

            print('[AMG] Moving with poly ... ')
            while t<T:
                # Start of synch with UR
                t_start = rtde_c.initPeriod()

                #integrate time
                t = t + dt
                
                # get q
                q = np.array(rtde_r.getActualQ())

                #generate trajectory
                qd, qd_dot = get5thorder_Ndof(q0=q0, qT=qT, t=t, T=T)

                # tracking control signal
                qdot = kinTracking_Ndof(x=q, xd=qd, xd_dot=qd_dot)

                # command velocity
                rtde_c.speedJ(qdot, 1.0, dt)

                # synchronize with UR
                rtde_c.waitPeriod(t_start)

        elif data.motion_type == 'home':
            print('[AMG] Moving to home configuration: ')
            
            qT = np.array([-183, -90, 120, 140, -90, 0])
            qT = qT * pi / 180
            print('qT = ', qT)

            rtde_c.moveJ(qT, 0.5, 0.5)

            # Get initial end-eefector pose
            q = np.array(rtde_r.getActualQ())

        ## Checking for Errors
        if np.linalg.norm(q-qT) < 5*pi/180:
            print('[AMG] Motion finished successfully')
            motion_finished_ok()
        else:
            print('[AMG] ERROR: Motion finished with an error greater than the set threshold!')
            print('[AMG] ||eq|| (rad) = ', np.linalg.norm(q-qT) )
            motion_finished_error()
        
  
    # Stop robot 
    rtde_c.speedStop()


    

##################################
############# MAIN ###############
##################################
if __name__ == '__main__':
    rospy.init_node('carpos_amg', anonymous=True)
    rate = rospy.Rate(100) 

    rospy.Subscriber("amg_enable_robot", Bool, amg_enable_robot)

    rospy.Subscriber("amg_command", amg_message, amg_command_callback)
 
    while not rospy.is_shutdown():

        # Publish camera pose
        ur.tool = p_wrist_camera
        q = np.array(rtde_r.getActualQ())
        g = ur.fkine(q)
        R = np.array(g.R)
        p = np.array(g.t)

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera"  # Update with your camera frame
        # Set position (origin)
        msg.pose.position.x = p[0]
        msg.pose.position.y = p[1]
        msg.pose.position.z = p[2]
        # Compute orientation (rotation matrix to quaternion)
        
        quaternion = rot2quat(R)
        msg.pose.orientation = Quaternion(
            x=quaternion[1],
            y=quaternion[2],
            z=quaternion[3],
            w=quaternion[0]
        )
        camera_pose_pub.publish(msg)

        # Publish gripper pose
        ur.tool = p_wrist_gripper
        q = np.array(rtde_r.getActualQ())
        q = np.array(rtde_r.getActualQ())
        g = ur.fkine(q)
        R = np.array(g.R)
        p = np.array(g.t)

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "gripper"  # Update with your camera frame
        # Set position (origin)
        msg.pose.position.x = p[0]
        msg.pose.position.y = p[1]
        msg.pose.position.z = p[2]
        # Compute orientation (rotation matrix to quaternion)
        
        quaternion = rot2quat(R)
        msg.pose.orientation = Quaternion(
            x=quaternion[1],
            y=quaternion[2],
            z=quaternion[3],
            w=quaternion[0]
        )
        gripper_pose_pub.publish(msg)

        # motion_finished_error()
        
        rate.sleep()
