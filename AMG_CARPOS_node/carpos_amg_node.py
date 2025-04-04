#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
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
from CSRL_orientation import *



# Our status publisher 
status_pub = rospy.Publisher('motion_status', String, queue_size=10)

rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")


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
def amg_command_callback(data):

    T = data.duration
    t = 0

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
            print('[AMG] Executing DMP towards xT ... ')
            print('[AMG] NOT FUNCTIONAL YET .... (TODO after the teaching phase)')
      

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
            
            qT = np.array([-180, -70, -120, 0, 90, -90 ])
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

    rospy.Subscriber("amg_command", amg_message, amg_command_callback)
 
    while not rospy.is_shutdown():

        # motion_finished_error()
        
        rate.sleep()
