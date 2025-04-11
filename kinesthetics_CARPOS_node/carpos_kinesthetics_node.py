import scipy.io
import pathlib
import os
import time
import numpy as np 
import rospy
from std_msgs.msg import String

import roboticstoolbox as rt
import rtde_receive
import rtde_control 
import math 
import scienceplots
import getch
import scipy.io as scio

from spatialmath import SE3



import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_base')
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_dmpy')

from CSRL_orientation import * 
from CSRL_math import * 
from dmpSE3 import * 


knowledge_path = "/home/carpos/catkin_ws/src/task_knowledge/"

# Our status publisher 
status_pub = rospy.Publisher('active_perception_status', String, queue_size=10)


# Declare math pi
pi = math.pi

#define UR3e
rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")



#define the robot with its DH parameters 
ur = rt.DHRobot([
    rt.RevoluteDH(d = 0.1807, alpha = pi/2),
    rt.RevoluteDH(a = -0.6127),
    rt.RevoluteDH(a = -0.57155),
    rt.RevoluteDH(d = 0.17415, alpha = pi/2),
    rt.RevoluteDH(d = 0.11985, alpha = -pi/2),
    rt.RevoluteDH(d = 0.11655)
], name='UR10e')

# # Define the tool
tool_position = SE3(0.0, 0.0, 0.16)  # Position of the tool (in meters)
ur.tool = tool_position 

kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential


dt = 0.002




# This is the callback function for the high level commands
def kinesthetic_correction_callback(data):
    global pi, status_pub, knowledge_path


    # Gains
    K = 4.0 * np.identity(6)
    K[-3:, -3:] = 5.0 * K[-3:, -3:]  # Orientation gains

  
    data = scipy.io.loadmat(str(knowledge_path) +'/dmp_model.mat')


    W = np.array(data['W'])
    T = data['T'][0][0]
    print("T=",T)

    # print(N)

    train_data = scipy.io.loadmat(str(knowledge_path) +'/training_demo.mat')


    Rrobc = np.array(train_data['Rrobc'])

    p_train = np.array(train_data['p_array'])
    Q_train = np.array(train_data['Q_array'])


    space_scaling = 0.8
    p0 = p_train[:,0] 
    pT = space_scaling * p_train[:,-1] 

    Q0 = Q_train[:,0] 
    QT = Q_train[:,-1]

    dmpTask = dmpSE3(N_in=W[0,:].size, T_in=T)
    dmpTask.set_weights(W, T, Q0=Q0, Qtarget=QT)

    dmpTask.set_goal(pT, QT)
    dmpTask.set_init_pose(p0, Q0)

    tau = 2
    dmpTask.set_tau(tau)

    ################ Read robot's initial pose

    q0 = np.array(rtde_r.getActualQ())

    q_sim = q0.copy()
    q_sim_log = q_sim.copy()

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

    #start logging 
    plog = np.copy(p0rob)
    pdlog = np.copy(p0rob)
    tlog = t
    Qlog = np.copy(Q0rob)
    Qd_log = np.copy(Q0rob)
    q_log = q0.copy()
    qdotc_log = np.zeros(6)
    f_log = np.zeros(6)
    K_log = np.zeros((6,6))


    #--------Admittance controler parameters-------#

    # Define the admittance inertia
    M = np.identity(6)
    l = 0.1
    M[0:3, 0:3] = 10.0 * M[0:3, 0:3]
    M[3:6, 3:6] = l * l * M[0, 0] * M[3:6, 3:6]

    # Compute inverse (constant)
    Minv = np.linalg.inv(M)

    # Define the admittance damping
    D = np.identity(6)
    D[0:3, 0:3] = 40.0 * D[0:3, 0:3]
    D[3:6, 3:6] = 1.5 * D[3:6, 3:6]

    # Define the admittance stiffness
    K = np.identity(6)
    K[0:3, 0:3] = 500.0 * K[0:3, 0:3]
    K[3:6, 3:6] =  8.0 * K[3:6, 3:6]

    K1 = K.copy()

    # K[0:3, 0:3] = 0.0 * K[0:3, 0:3]
    # K[3:6, 3:6] = 0.0 * K[3:6, 3:6]

    v_total = np.zeros(6)
    vdot_total = np.zeros(6)

    f_offset_tool = np.array(rtde_r.getActualTCPForce()) #- tool_mass * gAcc

    #----------------------------------------------#

    #uncertainties
    sigma_low = 0.001
    sigma_high = 1000
    es_low = pi/20
    es_high = pi
    Ks = np.array([sigma_low,sigma_low,sigma_high,es_high,es_high,es_low])
    S_c = np.diag(Ks * Ks)
    # Sinv = np.linalg.inv(S)
    lmax_p = 1/(sigma_low*sigma_low)
    lmax_o = 1/(es_low*es_low)

    kp = 200.0 
    ko = 20.0 

    S = np.zeros((6,6))
    S[0:3, 0:3] = R0rob @ Rrobc @ S_c[0:3, 0:3] @ Rrobc.T @ R0rob.T
    S[3:6, 3:6] = R0rob @ Rrobc @ S_c[3:6, 3:6] @ Rrobc.T @ R0rob.T
    Sinv = np.linalg.inv(S)

    K[0:3, 0:3] = (kp / lmax_p ) * Sinv[0:3, 0:3] 
    K[3:6, 3:6] = (ko / lmax_o) * Sinv[3:6, 3:6]
    K[3:6, 3:6] =  4.0 * K[3:6, 3:6]

    print(K)

    K2 = K.copy()

   
    # print(K[0:3, 0:3])
    # print(K[3:6, 3:6])

    t_stiff = 0

    # Kinesthetic only
    # rtde_c.teachMode()  


    i = 0
    while z < 1.2:
        i = i + 1


        tool_di = rtde_r.getActualDigitalInputBits()
        # tool_di = 2
        if tool_di > 0: 
            t_stiff = 1
            tau = 1
        else:
            t_stiff = t_stiff - dt
            tau = 1

        dmpTask.set_tau(tau)

        # saturate t_stiff between 0 and 1
        t_stiff = min(1, max(0 , t_stiff))

        s = simpleSigmoid(t=t_stiff)

        K = s * K2 + (1-s) * K1


        t_now = time.time()
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
        dz, dp, ddp, deo, ddeo = dmpTask.get_state_dot(   z, 
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
    

        # Get  end-efector pose
        g = ur.fkine(q)
        R = np.array(g.R)
        p = np.array(g.t)

        
        # get full jacobian
        J = np.array(ur.jacob0(q))



        # velocity array
        velocity_matrix = np.hstack((dot_pdw, omegadw))
        
        # error matrix 
        Q = rot2quatCont(R,Q)
        eo_robot = logError(Q, Qdw)
    

        # translate everything to the world frame
        pdw = p0rob + R0rob @ pd
        Qdw = rot2quat(R0rob @ quat2rot(Q_desired))
        dot_pdw = R0rob @ dot_pd
        omegadw = R0rob @ omegad





        # Admittance implementation

        # Get the force/torque measurement from the sensor
        f = np.array(rtde_r.getActualTCPForce()) - f_offset_tool

        # The force/torque with respect to the wrist of the leader robot 
        f[:3] = f[:3]
        f[-3:] = f[-3:]
        
        # Dead-zone (thresholding) of the the measurement of the force
        fnorm = np.linalg.norm(f[:3])
        if fnorm > 0.001:
            nF = f[:3] / fnorm
            if fnorm<4.0:
                f[:3] = np.zeros(3)
            else:
                f[:3] = f[:3] - 4.0 * nF

        # Dead-zone (thresholding) of the the measurement of the torqeu
        taunorm = np.linalg.norm(f[-3:])
        nTau = f[-3:] / taunorm
        if taunorm < 0.5:
            f[-3:] = np.zeros(3)
        else:
            f[-3:] = f[-3:] - 0.5 * nTau

        v_total = v_total + vdot_total * dt

        e_total = np.concatenate((p - pdw , eo_robot))

        vr_total = np.concatenate(( dot_pdw, omegadw))
        vrdot_total = np.concatenate(( np.zeros(3), np.zeros(3)))

        # f = np.zeros(6)
        print(f)
        # vdot_total = Minv @ (- D @ ( v_total) - K @ e_total + f)
        vdot_total = vrdot_total + Minv @ (- D @ ( v_total - vr_total) - K @ e_total + f)



        qdot_command = np.linalg.pinv(J, 0.01) @ v_total
        # qdot_command = np.zeros(6)
        
        # set joint speed
        rtde_c.speedJ(qdot_command, 10.0, dt) # comment for Kinesthetic only
        

        # log data
        tlog = np.vstack((tlog, t))
        plog = np.vstack((plog, p))
        pdlog = np.vstack((pdlog, pdw))
        
        Qlog = np.vstack((Qlog, rot2quat(R)))
        Qd_log = np.vstack((Qd_log, Qdw))

        
        q_log =  np.vstack((q_log, q))
        qdotc_log = np.vstack((qdotc_log, qdot_command))
        f_log = np.vstack((f_log, f))
        K_log = np.vstack((K_log, K))

        # synchronize
        rtde_c.waitPeriod(t_start)

    rtde_c.speedStop()


    # Write the data to a file 
    data_file = {'plog': plog, 'Qlog': Qlog, 'pdlog': pdlog, 'Qd_log': Qd_log  ,'tlog': tlog,'q_log': q_log ,'qdotc_log': qdotc_log , 'f_log': f_log , 'K_log': K_log }
    scio.savemat('Logging_kinesthetic.mat', data_file)


    
    p_train = plog.T
    Q_train = Qlog.T
    Q_train = makeContinuous(Q_train)

    t = np.array(list(range(p_train[1,:].size))) * dt

    print("T=", t[-1])

    Nsamples = p_train[1,:].size


    R0 = quat2rot(Q_train[:,0])
    p0 = np.copy(p_train[:,0])

    for i in range(Nsamples):
        # print(i)
        p_train[:,i] = R0.T @ (p_train[:,i] - p0)
        Rch = quat2rot(Q_train[:,i])
        Q_train[:,i] = rot2quat(R0.T @ Rch)



    dmpTask_corrected = dmpSE3(N_in=20, T_in=t[-1])


    dmpTask_corrected.train(dt, p_train, Q_train, True)


    print("Training completed. The new dmp_model is saved.")

    ################# UNCOMMENT THIS FOR SEVING THE weights FILE 
    mdic = {"W": dmpTask_corrected.get_weights(), "T": dmpTask_corrected.T}
    scipy.io.savemat(str(knowledge_path) +"dmp_model.mat", mdic)


    Q_array = makeContinuous(Q_train)
    p_array = p_train.copy()
    mdic = {"p_array": p_array[:,1:], "Q_array": Q_array[:,1:], "dt": dt}
    scipy.io.savemat(str(knowledge_path) +"training_demo.mat", mdic)


    status_str = "success" 
    rospy.loginfo('Kinesthetic corrections, status: ' + status_str)
    status_pub.publish(status_str)




##################################
############# MAIN ###############
##################################
if __name__ == '__main__':
    rospy.init_node('carpos_kinethetics', anonymous=True)
    rate = rospy.Rate(100) 

    rospy.Subscriber("start_kinesthetic_correction", String, kinesthetic_correction_callback)
    rospy.loginfo('Kinesthetics node started ... ')

    while not rospy.is_shutdown():

        # motion_finished_error()
   
        rate.sleep()
