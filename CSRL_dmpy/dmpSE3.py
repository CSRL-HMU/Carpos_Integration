from dmp import *
from CSRL_orientation import *
from CSRL_math import *
import numpy as np
from dmpR3 import *
from dmpSO3 import *

import scienceplots
import matplotlib.pyplot as plt

# latex style plots 
plt.style.use(["default","no-latex"])

class dmpSE3:
    def __init__(self, N_in, T_in, kernelType_in = 'Gaussian', xType_in = 'linear', a_x_in = 1.0, a_in=20.0, b_in=0.8, tau_in=1.0):
        self.kernelType = kernelType_in
        self.xType = xType_in
        self.a_x = a_x_in
        self.a = a_in
        self.b = b_in
        self.tau = tau_in
        self.N  = N_in
        self.T  = T_in

        # Create DMPs
        self.dmp_translation = dmpR3(N_in, T_in, kernelType_in = 'Gaussian', xType_in = 'linear', a_x_in = 1.0, a_in=20.0, b_in=0.8, tau_in=1.0)
        self.dmp_orientation = dmpSO3(N_in, T_in, kernelType_in = 'Gaussian', xType_in = 'linear', a_x_in = 1.0, a_in=20.0, b_in=0.8, tau_in=1.0)
       
    def get_weights(self):
        W = np.vstack((self.dmp_translation.dmp_array[0].W, 
                   self.dmp_translation.dmp_array[1].W, 
                   self.dmp_translation.dmp_array[2].W, 
                   self.dmp_orientation.dmp_array[0].W, 
                   self.dmp_orientation.dmp_array[1].W, 
                   self.dmp_orientation.dmp_array[2].W ))
        
        return W
    
    def set_weights(self, W, T, Q0, Qtarget):
        self.dmp_translation.dmp_array[0].set_weights( W[0,:] , T)
        self.dmp_translation.dmp_array[1].set_weights( W[1,:] , T)
        self.dmp_translation.dmp_array[2].set_weights( W[2,:] , T)
        self.dmp_orientation.dmp_array[0].set_weights( W[3,:] , T)
        self.dmp_orientation.dmp_array[1].set_weights( W[4,:] , T)
        self.dmp_orientation.dmp_array[2].set_weights( W[5,:] , T)
        self.Q0 = Q0
        self.Q_target = Qtarget
        self.dmp_orientation.Q0 = Q0
        self.dmp_orientation.Q_target = Qtarget
    

    # inputs p_array: 3xN (translation), Q_array: 4xN (orientation in quaternion) 
    def train(self, dt, p_array, Q_array, plotPerformance = False):

        # translation target 
        self.p_target = p_array[:,-1]

        # orientation target 
        self.Q_target = Q_array[:,-1]

        # computethe custom scaling factor in each axis 
        self.targetError = logError(Q_array[:,-1],Q_array[:,0])

        # train the DMPs
        self.dmp_translation.train(dt, p_array, plotPerformance = False)
        self.dmp_orientation.train(dt, Q_array, plotPerformance = False)

        # self.dmp_translation.dmp_array[0].kb.plot_t()

        # if plotPerformance flag is enabled ...
        if plotPerformance: self.plotPerformance(dt, p_array, Q_array)

    # returns the whole state of the system
    def get_state_dot(self, x, p, pdot, eOr, eOr_dot):
        
        p_out_dot = np.zeros(3)
        p_out_2dot = np.zeros(3)

        eOr_out_dot = np.zeros(3)
        eOr_out_2dot = np.zeros(3)

        # for position and orientation
        x_out_dot, p_out_dot, p_out_2dot = self.dmp_translation.get_state_dot( x, p, pdot) 
        x_out_dot, eOr_out_dot, eOr_out_2dot = self.dmp_orientation.get_state_dot( x, eOr, eOr_dot) 

        # return state dot
        return x_out_dot, p_out_dot, p_out_2dot, eOr_out_dot, eOr_out_2dot
      

    # sets the initial state p and A (can be quaternion or rotation matrix)
    def set_init_pose(self, p0_in, A0_in):

        # set the class variables
        self.p0 = p0_in

        # set the values in DMP objects 
        self.dmp_translation.set_init_state(self.p0)

        # set the class variables
        self.Q0 = enforceQuat(A0_in)

        # calculate initial error from the target 
        self.targetError = logError(self.Q_target, self.Q0)

        # set the values in DMP objects
        self.dmp_orientation.set_init_state(self.Q0)



    # sets the goal fro translation and orientation (can be quaternion or rotation matrix)
    def set_goal(self, gP_in, gOr_in):

        # set the class variables 
        self.p_target = gP_in

        # set the DMP object's variables
        self.dmp_translation.set_goal(self.p_target)

        # set the class variables
        self.Q_target = enforceQuat(gOr_in)

        # recalculate the initial orientation error 
        self.targetError = logError(self.Q_target, self.Q0)

        # set the DMP object's variables
        self.dmp_orientation.set_goal(self.Q_target)

        
    # sets time scaling parameter for all axes
    def set_tau(self, tau_in):

        # set the class variables 
        self.tau = tau_in

        # set the DMP object's variables
        self.dmp_translation.set_tau(self.tau)
        self.dmp_orientation.set_tau(self.tau)

    # plots the response of the DMP compared to the training dataset
    def plotPerformance(self, dt, p_array, Q_array):
        
        # ensure np array 
        p_array = np.array(p_array)
        Q_array = np.array(Q_array)

        # create time
        t = np.array(list(range(p_array[0,:].size))) * dt

        # initialize phase variable
        if self.xType == 'linear':
            state_x = 0
        else:
            state_x = 1

        # initialise position and orientation error 
        p0 = p_array[:,0]
        Qlog0 = Q_array[:,0]

        # this is the initial orientation error
        elog0 = self.targetError

        # initialise states
        state_p = p0 
        state_elog = elog0
        state_pdot = np.zeros(3)
        state_elogdot = np.zeros(3)

        # initialise states time derivative
        state_x_dot = 0
        state_p_dot = np.zeros(3)
        state_elog_dot = np.zeros(3)
        state_pdot_dot = np.zeros(3)
        state_elogdot_dot = np.zeros(3)

        # define logging variables
        pDMP = np.zeros((3,t.size))
        eOrDMP = np.zeros((3,t.size))
        QDMP = np.zeros((4,t.size))


        i = 0
        for ti in t:

            
            
            # Euler integration
            state_x = state_x + state_x_dot * dt
            state_p = state_p + state_p_dot * dt
            state_pdot = state_pdot + state_pdot_dot * dt
            state_elog = state_elog + state_elog_dot * dt
            state_elogdot = state_elogdot + state_elogdot_dot * dt


            # get state dot
            state_x_dot, state_p_dot, state_pdot_dot, state_elog_dot, state_elogdot_dot = self.get_state_dot(   state_x, 
                                                                                                                state_p,                                                                                      
                                                                                                                state_pdot, 
                                                                                                                state_elog,
                                                                                                                state_elogdot)
                        
            # log daata 
            pDMP[:,i] = state_p
            eOrDMP[:,i] = state_elog
            QDMP[:,i] = QDMP[:,i] = quatProduct( quatInv( quatExp(0.5 * state_elog) ) , self.Q_target )
            i = i + 1


        # plot results
        # figure 1 is for p and Q
        fig = plt.figure(figsize=(4, 7))


        for i in range(3):
            axs = fig.add_axes([0.21, ((5-i)/6)*0.8+0.2, 0.7, 0.11])
            axs.plot(t, pDMP[i, :], 'k', linewidth=1.0)
            axs.plot(t, p_array[i, :], 'k--', linewidth=1.0)
            axs.set_xlim([0, t[-1]])
            axs.set_ylabel('$p_' + str(i+1) + '(t)$ [m]',fontsize=14 )
            axs.set_xticks([])
        
        for i in range(4):
            axs = fig.add_axes([0.21, ((5-(i+3))/6)*0.8+0.2, 0.7, 0.11])
            axs.plot(t, QDMP[i, :], 'k', linewidth=1.0)
            axs.plot(t, Q_array[i, :], 'k--', linewidth=1.0)
            axs.set_xlim([0, t[-1]])
            axs.set_ylabel('$Q_' + str(i+1) + '(t)$ [rad]',fontsize=14 )
            
            if i==3:
                axs.set_xlabel('Time (s)',fontsize=14 )
            else:
                axs.set_xticks([])

        fig.show()

        # figure 2 is for e_log (orientation)
        fig2 = plt.figure(figsize=(4, 3))
        for i in range(3):
            axs = fig2.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
            axs.plot(t, eOrDMP[i, :], 'k', linewidth=1.0)
            axs.set_xlim([0, t[-1]])
            axs.set_ylabel('$e_{log,' + str(i+1) + '}(t)$ [rad]',fontsize=14 )
            
            if i==2:
                axs.set_xlabel('Time (s)',fontsize=14 )
            else:
                axs.set_xticks([])

        fig2.show()
        

    # plots the response of the DMP for a given initial positiob and orientation (assumes zero initial velocities)
    def plotResponse(self, dt, p0, Q0, iterations):
            
        
            # create time array
            t = np.array(list(range(iterations))) * dt

            # initialize phase variable
            if self.xType == 'linear':
                state_x = 0
            else:
                state_x = 1

            # initialize orientation error 
            elog0 = logError(self.Q_target, Q0)

            # initialize states
            state_p = p0 
            state_elog = elog0
            state_pdot = np.zeros(3)
            state_elogdot = np.zeros(3)

            # initialize states' time derivative 
            state_x_dot = 0
            state_p_dot = np.zeros(3)
            state_elog_dot = np.zeros(3)
            state_pdot_dot = np.zeros(3)
            state_elogdot_dot = np.zeros(3)

            # define logging variables
            pDMP = np.zeros((3,t.size))
            QDMP = np.zeros((4,t.size))
            elogDMP = np.zeros((3,t.size))

            i = 0
            for ti in t:
                
                # Euler integration
                state_x = state_x + state_x_dot * dt
                state_p = state_p + state_p_dot * dt
                state_pdot = state_pdot + state_pdot_dot * dt
                state_elog = state_elog + state_elog_dot * dt
                state_elogdot = state_elogdot + state_elogdot_dot * dt


                # get state dot
                state_x_dot, state_p_dot, state_pdot_dot, state_elog_dot, state_elogdot_dot = self.get_state_dot(   state_x, 
                                                                                                                    state_p,                                                                                      
                                                                                                                    state_pdot, 
                                                                                                                    state_elog,
                                                                                                                    state_elogdot)
                            
                # log data
                pDMP[:,i] = state_p
                QDMP[:,i] = quatProduct( quatInv( quatExp(0.5 * state_elog) ) , self.Q_target )
                elogDMP[:,i] =state_elog
                i = i + 1



            # plot results
            # figure 1 is for p and Q
            fig = plt.figure(figsize=(4, 7))


            for i in range(3):
                axs = fig.add_axes([0.21, ((5-i)/6)*0.8+0.2, 0.7, 0.11])
                axs.plot(t, pDMP[i, :], 'k', linewidth=1.0)
                axs.set_xlim([0, t[-1]])
                axs.set_ylabel('$p_' + str(i+1) + '(t)$ [m]',fontsize=14 )
                axs.set_xticks([])
            
            for i in range(4):
                axs = fig.add_axes([0.21, ((5-(i+3))/6)*0.8+0.2, 0.7, 0.11])
                axs.plot(t, QDMP[i, :], 'k', linewidth=1.0)
                axs.set_xlim([0, t[-1]])
                axs.set_ylabel('$Q_' + str(i+1) + '(t)$ [rad]',fontsize=14 )
                
                if i==3:
                    axs.set_xlabel('Time (s)',fontsize=14 )
                else:
                    axs.set_xticks([])

            # figure 2 is for e_log (orientation)
            fig2 = plt.figure(figsize=(4, 3))
            for i in range(3):
                axs = fig2.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
                axs.plot(t, elogDMP[i, :], 'k', linewidth=1.0)
                axs.set_xlim([0, t[-1]])
                axs.set_ylabel('$e_{log,' + str(i+1) + '}(t)$ [rad]',fontsize=14 )
                
                if i==2:
                    axs.set_xlabel('Time (s)',fontsize=14 )
                else:
                    axs.set_xticks([])

            plt.show(block = False)


        


        

  
