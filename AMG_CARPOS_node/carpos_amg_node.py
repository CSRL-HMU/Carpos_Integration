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
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovariance, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState



import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_base')
from CSRL_orientation import * 
from CSRL_math import * 
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_dmpy')
from dmpSE3 import * 


warning_flag = False

q_equil = np.array([0, -110, 130, 160, -90, 0])
q_equil = q_equil * math.pi / 180


null_gain = 2.5


def test_callback(data):

    print('Test callback !!!')

    for i in range(2000):
                
        t_start = rtde_c.initPeriod()
        

        q_dot = np.zeros(8)
        if i < 1000:
            V = np.array([0.10, 0.0, 0, 0, 0, -0.0])

            J = get_robot_Jacobian()

            q_dot = wpinv(J) @ V 
            print(q_dot)

        set_commanded_velocities(q_dot)

        rtde_c.waitPeriod(t_start)

  
    


def husky_pose_callback(data):

    global g_0husk


    p = np.array([data.pose.pose.position.x,  data.pose.pose.position.y, data.pose.pose.position.z]) 
    # Compute orientation (rotation matrix to quaternion)
    Q = np.array([data.pose.pose.orientation.w, data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z])

    g_0husk_odom = SE3()
    g_0husk_odom.R = quat2rot(Q)
    g_0husk_odom.t = p

    husky_top_center_offset = SE3(-0.035, 0 ,0.397) # This is for adding the offset from the husky's center to the UR base
    g_0husk = g_0husk_odom * husky_top_center_offset

    # print(g_0husk)


def get_ee_pose():
    global ur, rtde_c, rtde_r, g_0husk

    
    q_ur = np.array(rtde_r.getActualQ())
    # print('[get_ee_pose()]  q_ur=',q_ur)

    if q_ur.size<6:
        q_ur = np.zeros(6)

    g_huske = ur.fkine(q_ur)
    g_0e = g_0husk * g_huske


    R_0e = np.array(g_0e.R)
    p_0e  = np.array(g_0e.t)
    Q_0e  = rot2quat(R_0e)

    return p_0e , R_0e , Q_0e 


def get_cee_pose():
    global ur, rtde_c, rtde_r

    q_ur = np.array(rtde_r.getActualQ())

    if q_ur.size<6:
        q_ur = np.zeros(6)

    # print('[get_cee_pose()] q_ur=',q_ur)
    g_huske = ur.fkine(q_ur)
    


    R_huske = np.array(g_huske.R)
    p_huske  = np.array(g_huske.t)
    Q_huske  = rot2quat(R_huske)

    return p_huske , R_huske , Q_huske 
    
    


def get_robot_Jacobian():
    global ur, rtde_c, rtde_r, g_0husk

    q_ur = np.array(rtde_r.getActualQ())

    J_ur = ur.jacob0(q_ur)

    g_huske = ur.fkine(q_ur[0:6])
    
    p_huske  = np.array(g_huske.t)
    p_ehusk =  - p_huske

    R_0husk =  g_0husk.R

    A = np.zeros((6,2))

    A[0,0] = 1
    A[5,1] = 1
    A[0:3,1] = skewSymmetric(p_ehusk) @ np.array([0, 0, 1])

    RR = np.zeros((6,6))

    RR[0:3,0:3] = R_0husk
    RR[3:6,3:6] = R_0husk

    J = RR @ np.hstack((J_ur, A))

    return J


def set_commanded_velocities(q_dot_c):
    global ur, rtde_c, rtde_r, g_0husk

    rtde_c.speedJ(q_dot_c[0:6], 1.0, dt)
    

    v_husk = Twist()
    v_husk.linear.x = q_dot_c[6]  
    v_husk.angular.z = q_dot_c[7]  

    husky_pub.publish(v_husk)

    publish_joint_states(np.array(rtde_r.getActualQ()))

    return



mutex_flag = False

# Our status publisher 
status_pub = rospy.Publisher('motion_status', String, queue_size=10)

gripper_pose_pub = rospy.Publisher('gripper_pose',PoseStamped , queue_size=10)
camera_pose_pub = rospy.Publisher('camera_pose', PoseStamped, queue_size=10)

camera_pose_ur_pub = rospy.Publisher('camera_pose_ur', PoseStamped, queue_size=10)

husky_pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10)

joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)

rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")


foffset = np.array(rtde_r.getActualTCPForce())
# DMP params
kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential


# Declare math pi
pi = math.pi



d_adm = np.array([0,0,0])
ddot_adm = np.array([0,0,0])
dddot_adm = np.array([0,0,0])
M_adm = 2*np.eye(3)
D_adm = 10*np.eye(3)
K_adm = 40*np.eye(3)

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

ur_pforPub = ur.copy()


# # Define the tool
rotation_matrix = SO3.AngVec(0, np.array([0,0,1]))
p_wrist_camera = SE3(rotation_matrix) * SE3(-0.0325, -0.0675, 0.067) #D415

rotation_matrix = SO3.AngVec(-pi/2, np.array([0,0,1]))
p_wrist_gripper = SE3(rotation_matrix) * SE3(-0.07, 0,  0.14)  # Position of the tool (in meters)
# p_wrist_gripper = SE3(rotation_matrix, np.array([0.0, 0.031, 0.05])) 
ur.tool = p_wrist_camera 


# Get initial end-eefector pose
# g0 = ur.fkine(q0)
# R0 = np.array(g0.R)
# p0 = np.array(g0.t)


# Control cycle
dt = 0.002

# initialize qdot
qdot = np.zeros(6)



# The path in which the knowledge is stored
knowledge_path = "/home/carpos/catkin_ws/src/task_knowledge/"


g_0husk = SE3(np.eye(4))


def publish_joint_states(q):
    global joint_pub
    

    msg = JointState()
    msg.name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    msg.position = q  # Your custom function
    msg.header.stamp = rospy.Time.now()
    joint_pub.publish(msg)




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

            q_nothing = np.array(rtde_r.getActualQ())
            print(q_nothing)

            time.sleep(3)
    else:
        print('Disconnecting from the robot ...')
        # rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()



# This is the callback function for the high level commands
def amg_command_callback(data):

    global ur, warning_flag, d_adm, ddot_adm, dddot_adm, M_adm, D_adm, K_adm

    T = data.duration
    t = 0


    # default
    ur.tool = p_wrist_camera

    if data.end_effector == 'gripper':
        ur.tool = p_wrist_gripper
        print('Tool changed to gripper')
    elif data.end_effector == 'camera':
        ur.tool = p_wrist_camera
        print('Tool changed to camera')
    else:
        print('Going with the default end-effector (camera)')

    print('Tool pose wrt wrist:', ur.tool)


    

   

    if data.space == 'task':
        print('Tool pose wrt wrist (in the loop):', ur.tool)
        
        # get target
        pT = np.array( [data.target_pose.position.x, data.target_pose.position.y, data.target_pose.position.z] )
        QT = np.array( [data.target_pose.orientation.w, data.target_pose.orientation.x, data.target_pose.orientation.y, data.target_pose.orientation.z ] )
       
        print('[AMG] Received task target:')
        print('pT = ', pT)
        print('QT = ', QT)

        print('RT = ', quat2rot(QT))

        # Get initial end-eefector pose
        # q0 = np.array(rtde_r.getActualQ())
        # g0 = ur.fkine(q0)
        # R0 = np.array(g0.R)
        # p0 = np.array(g0.t)
        # Q0 = rot2quat(R0)

        p0, R0, Q0 = get_ee_pose()

        print('[AMG] Initial pose:')
        print('p0 = ', p0)
        print('Q0 = ', Q0)
        print('R0 = ', R0)
        p = p0
        Q = Q0


        # q0 = np.array(rtde_r.getActualQ()) 
       
        if data.motion_type == 'reach':

            print('[AMG] Reaching ... ')
            while t<T:
                
                t_start = rtde_c.initPeriod()

                #integrate time
                t = t + dt
                
                p, R, Q = get_ee_pose()
        
                # get full jacobian
                J = get_robot_Jacobian()

    
                # reaching control signal
                qdot = kinReaching_SE3(p=p, A=R, pT=pT, AT=QT, J=J, kr = 5.0/T)

                set_commanded_velocities(qdot)
                
    

                rtde_c.waitPeriod(t_start)


        elif data.motion_type == 'poly':

            print('Tool pose wrt wrist (in the loop):', ur.tool)

            
            print('[AMG] Moving with poly ... ')

            phriFlag = False

            # rtde_c.startContactDetection(); 

            if warning_flag:
                rtde_c.endTeachMode()

            warning_flag = False

            Qd_array = np.array([0, 0, 0, 0])
            Qd_array.shape = (4,1)
            Q_array = np.array([0, 0, 0, 0])
            Q_array.shape = (4,1)
            pd_array = np.array([0, 0, 0])
            pd_array.shape = (3,1)
            p_array = np.array([0, 0, 0])
            p_array.shape = (3,1)

            f_array = np.array([0, 0, 0, 0, 0, 0])
            f_array.shape = (6,1)

            vad_array = np.array([0, 0, 0, 0, 0, 0])
            vad_array.shape = (6,1)

            t_v = t

            while t<T*1.3:
                
                t_start = rtde_c.initPeriod()

                
                
                f = np.array(rtde_r.getActualTCPForce()) - foffset

                print('f=',f)

                # The force/torque with respect to the wrist of the leader robot 
                fp = f[:3]

                pd = np.zeros(0)
                Rd = np.zeros(0)
                
                # Dead-zone (thresholding) of the the measurement of the force
                fnorm = np.linalg.norm(fp)
                if fnorm > 0.001:
                    nF = fp / fnorm
                    if fnorm<30.0:
                        fp = np.zeros(3)
                    else:
                        fp = fp - 30.0 * nF

                # fp = np.zeros(3)
                f[:3] = fp
                

                fk = K_adm @ d_adm
                norm_fk = np.linalg.norm(fk)
                if norm_fk > 5:
                    fk = (5/norm_fk)*fk

            

                dddot_adm = np.linalg.inv(M_adm) @ (-D_adm @ ddot_adm - fk + fp) 
                d_adm = d_adm + dt * ddot_adm
                ddot_adm = ddot_adm + dt * dddot_adm

                p, R, Q = get_ee_pose()

                
                # get full jacobian
                J = get_robot_Jacobian()
                Jp = J[0:3,2:8]
                N = np.eye(8) - np.linalg.pinv(J)@J

                q = np.array(rtde_r.getActualQ()) 
                e_null = np.zeros(8)
                q_equil[0] = q[0]
                e_null[2:8] = q - q_equil
                # e_null[1] = 1*math.sin(0.8*2*math.pi*t)
                q_null = - null_gain * N @ e_null
                # q_null = - N @ e_null


                # mot_cur = rtde_r.getA

               
                t = t + dt
                #generate trajectory
                if not phriFlag:
                    #integrate time
                    t_v = t_v + dt

                # print(phriFlag)

                # print(t)

                pd, Rd, pd_dot, omegad = get5thorder_SE3(p0=p0, A0=R0, pT=pT, AT=QT, t=t_v, T=T)


                if phriFlag: 
                    pd_dot = np.zeros(3)
                    omegad = np.zeros(3)


                if np.linalg.norm(p-pd) > 0.1:

                    print('[AMG] RAISING PHRI FLAG !!!!!!!!!!!')
                    phriFlag = True

                
            

            

                # tracking control signal
                qdot = kinTracking_SE3(p=p, A=R, pd=pd+d_adm, Ad=Rd, pd_dot=pd_dot+ddot_adm, omegad=omegad, J=J)
                # qdot = kinTracking_SE3(p=p, A=R, pd=pd, Ad=Rd, pd_dot=pd_dot, omegad=omegad, J=J)

                

                set_commanded_velocities(qdot)
                test = np.array([1,0,0])

                # # COntact detection functionality
                # contact_detected = rtde_c.toolContact(test)
                # print("joint_toques=", np.linalg.norm(rtde_c.getJointTorques()))

                # contact_detected = rtde_c.toolContact()
                # Lambda = rtde_c.getMassMatrix()
                # q_ddot = rtde_r.getTargetJointAccelerations()
                # fx = np.linalg.pinv(Jp.T)@(rtde_c.getJointTorques() - Lambda @ q_ddot)
                # print("tool contact=", rtde_c.toolContact([1,1,1]))
           
                # if rtde_c.toolContact([1,1,1]):
                #     print('[Warning!] contact is detected ... switching to free drive mode ... ')
                #     warning_flag = True
                #     rtde_c.speedStop() 
                #     rtde_c.teachMode()

                #     break

                ########################### LOGGING!!!!!!!!!!!!!

                ptemp = p.copy()
                pdtemp = pd.copy()
                Qtemp = Q.copy()
                Qdtemp = Rd.copy()

                ftemp = f.copy()
                vtemp = np.concatenate((pd_dot+ddot_adm,omegad))
                
                ptemp.shape = (3,1)
                pdtemp.shape = (3,1)
                Qtemp.shape = (4,1)
                Qdtemp.shape = (4,1)
                ftemp.shape = (6,1)
                vtemp.shape = (6,1)

                Q_array = np.hstack((Q_array,Qtemp))
                p_array = np.hstack((p_array,ptemp))
                Qd_array = np.hstack((Qd_array,Qdtemp))
                pd_array = np.hstack((pd_array,pdtemp))
                f_array = np.hstack((f_array,ftemp))
                vad_array = np.hstack((vad_array,vtemp))


                


                rtde_c.waitPeriod(t_start)

            # rtde_c.stopContactDetection()
            # mdic = {"Q_array": Q_array, "Qd_array": Qd_array, "p_array": p_array, "pd_array": pd_array, "f_array": f_array, "vad_array": vad_array,}
        
            # scipy.io.savemat('/home/carpos/catkin_ws/src/logging/phri_logging.mat', mdic)


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
            # g0rob = ur.fkine(q0)
            # R0rob = np.array(g0rob.R)
            # p0rob = np.array(g0rob.t)
            # Q0rob = rot2quat(R0rob)
            p0rob, R0rob, Q0rob = get_ee_pose()

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


            while z < 1.5:

            
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

                
                # # Get the actual joint values 
                # q = np.array(rtde_r.getActualQ())

                # # Get  end-efector pose
                # g = ur.fkine(q)
                # R = np.array(g.R)
                # p = np.array(g.t)
                p, R, Q_nouse = get_ee_pose()
                
                

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
                # J = np.array(ur.jacob0(q))
                J = get_robot_Jacobian()

                
                qdot_command = np.zeros(8)
                # commanded joint velocity
                qdot_command[0:6] = kinTracking_SE3(p=p, A=Q, pd=pdw, Ad=Qdw, pd_dot=dot_pdw, omegad=omegadw, J=J[:,0:6], weightedEn = False)

            

                    
                # qdot = np.zeros(6)
                
                

                # print('Q=',Q)
                # print('Qdw=',Qdw)
                # print('||eo||=',np.linalg.norm(logError(Q,Qdw)))



                # set joint speed
                # rtde_c.speedJ(qdot_command, 1, dt) # comment for Kinesthetic only
                set_commanded_velocities(qdot_command)

                # synchronize
                rtde_c.waitPeriod(t_start)

            QT = Qdw.copy()
            pT = pdw.copy()
                
            
            if not warning_flag:
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
            
            # qT = np.array([-183, -90, 120, 140, -90, 0])
            # qT = np.array([-270, -150, 120, 213, -95, 0])
            qT = np.array([-270, -133, 96, 248, -92, 0])
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
    if not warning_flag:
        rtde_c.speedStop()


    

##################################
############# MAIN ###############
##################################
if __name__ == '__main__':
    rospy.init_node('carpos_amg', anonymous=True)
    rate = rospy.Rate(100) 

    rospy.Subscriber("amg_enable_robot", Bool, amg_enable_robot)

    rospy.Subscriber("amg_command", amg_message, amg_command_callback)

    rospy.Subscriber('/husky_velocity_controller/odom', Odometry, husky_pose_callback)



    rospy.Subscriber('/amg_test', String, test_callback)


    for i in range(1000):
        time.sleep(0.001)



    p0, R0, Q0 = get_ee_pose()

    print('p0 = ', p0 )
    print('R0 = ', R0 )
    print('Q0 = ', Q0 )
    
 
    while not rospy.is_shutdown():
        if rtde_c.isConnected() and rtde_r.isConnected():
            # Publish camera pose
            ur_pforPub.tool = p_wrist_camera
            # q = np.array(rtde_r.getActualQ())
            # g = ur_pforPub.fkine(q)
            # R = np.array(g.R)
            # p = np.array(g.t)
            
            p, R, Q = get_ee_pose()

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

    
            p, R, Q = get_cee_pose()

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


            camera_pose_ur_pub.publish(msg)






            # Publish gripper pose
            ur_pforPub.tool = p_wrist_gripper
            # q = np.array(rtde_r.getActualQ())
            # q = np.array(rtde_r.getActualQ())
            # g = ur_pforPub.fkine(q)
            # R = np.array(g.R)
            # p = np.array(g.t)
            p, R, Q = get_ee_pose()

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

            # p, R, Q = get_ee_pose()

            # print('[main] p=', p)
            # print('[main] R=', R)
            # print('[main] Q=', Q)
    

            # motion_finished_error()
        
        rate.sleep()
