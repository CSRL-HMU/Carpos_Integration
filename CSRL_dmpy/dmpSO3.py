from dmp import *
from CSRL_orientation import *
from CSRL_math import *
import numpy as np

import scienceplots
import matplotlib.pyplot as plt

# latex style plots 
plt.style.use(["default","no-latex"])

class dmpSO3:
    def __init__(self, N_in, T_in, kernelType_in = 'Gaussian', xType_in = 'linear', a_x_in = 1.0, a_in=20.0, b_in=0.8, tau_in=1.0):
        self.kernelType = kernelType_in
        self.xType = xType_in
        self.a_x = a_x_in
        self.a = a_in
        self.b = b_in
        self.tau = tau_in

        self.dmp_array = []
        for i in range(3):

            # WARNING! 'append' command appends pointers!!!
            temp_dmp = dmp(N_in, T_in, kernelType_in, xType_in, a_x_in, a_in, b_in, tau_in)
            self.dmp_array.append(temp_dmp)

    # inputs Q_array: 4xN (orientation in quaternion) 
    def train(self, dt, Q_array, plotPerformance = False):

        # number of samples
        N = Q_array[0,:].size

        # orientation target 
        self.Q_target = Q_array[:,-1]

        # compute the logarithmic error for each time instance 
        e_log_array = np.zeros((3,N))

        for i in range(N):
            e_log_array[:,i] = logError(self.Q_target, Q_array[:,i])

        # computethe custom scaling factor in each axis 
        self.targetError = logError(Q_array[:,-1],Q_array[:,0])


        # train orientation DMP
        for i in range(3):
            self.dmp_array[i].train(dt, e_log_array[i,:], False, True, self.targetError[i])

        # if plotPerformance flag is enabled ...
        if plotPerformance: self.plotPerformance(dt, e_log_array)

    # returns the whole state of the system
    def get_state_dot(self, x, eOr, eOr_dot):
        
        eOr_out_dot = np.zeros(3)
        eOr_out_2dot = np.zeros(3)

        # for position and orientation
        for i in range(3):
            x_out_dot, eOr_out_dot[i], eOr_out_2dot[i] = self.dmp_array[i].get_state_dot( x, eOr[i], eOr_dot[i], True, self.targetError[i]) 

        # return state dot
        return x_out_dot, eOr_out_dot, eOr_out_2dot
      

    # sets the initial state A (can be quaternion or rotation matrix)
    def set_init_state(self, A0_in):


        # set the class variables
        self.Q0 = enforceQuat(A0_in)

        # calculate initial error from the target 
        self.targetError = logError(self.Q_target, self.Q0)

        # set the values in DMP objects
        for i in range(3):
            self.dmp_array[i].set_init_position(self.targetError[i])
            self.dmp_array[i].set_goal(0.0)



    # sets the goal for orientation (can be quaternion or rotation matrix)
    def set_goal(self, gOr_in):

        # set the class variables
        self.Q_target = enforceQuat(gOr_in)

        # recalculate the initial orientation error 
        self.targetError = logError(self.Q_target, self.Q0)

        # set the DMP object's variables
        for i in range(3):
            self.dmp_array[i].set_init_position(self.targetError[i])
            self.dmp_array[i].set_goal(0.0)

        
    # sets time scaling parameter for all axes
    def set_tau(self, tau_in):

        # set the class variables 
        self.tau = tau_in

        # set the DMP object's variables
        for i in range(3):
            self.dmp_array[i].set_tau(self.tau)

    # plots the response of the DMP compared to the training dataset
    def plotPerformance(self, dt, e_log_array):
        
        # ensure np array 
        e_log_array = np.array(e_log_array)

        # create time
        t = np.array(list(range(e_log_array[0,:].size))) * dt

        # initialize phase variable
        if self.xType == 'linear':
            state_x = 0
        else:
            state_x = 1

        # initialise position and orientation error 
        elog0 = e_log_array[:,0]

        # initialise states
        state_elog = elog0
        state_elogdot = np.zeros(3)

        # initialise states time derivative
        state_x_dot = 0
        state_elog_dot = np.zeros(3)
        state_elogdot_dot = np.zeros(3)

        # define logging variables
        eOrDMP = np.zeros((3,t.size))

        i = 0
        for ti in t:
            
            # Euler integration
            state_x = state_x + state_x_dot * dt
            state_elog = state_elog + state_elog_dot * dt
            state_elogdot = state_elogdot + state_elogdot_dot * dt


            # get state dot
            state_x_dot, state_elog_dot, state_elogdot_dot = self.get_state_dot(    state_x, 
                                                                                    state_elog,
                                                                                    state_elogdot)
                        
            # log daata 
            eOrDMP[:,i] = state_elog
            i = i + 1


        # plot results
        fig = plt.figure(figsize=(4, 3))

        for i in range(3):
            axs = fig.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
            axs.plot(t, eOrDMP[i, :], 'k', linewidth=1.0)
            axs.plot(t, e_log_array[i, :], 'k--', linewidth=2.0)
            axs.set_xlim([0, t[-1]])
            axs.set_ylabel('$e_{log,' + str(i+1) + '}(t)$ [rad]',fontsize=14 )
            
            if i==2:
                axs.set_xlabel('Time (s)',fontsize=14 )
                lgnd = axs.legend(['$DMP$','$demo$'],fontsize=11,ncol=2,loc="lower right")
                lgnd.get_frame().set_alpha(None)
            else:
                axs.set_xticks([])


        plt.show()

    # plots the response of the DMP for a given initial orientation (assumes zero initial velocity)
    def plotResponse(self, dt, Q0, iterations):
            
        
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
            state_elog = elog0
            state_elogdot = np.zeros(3)

            # initialize states' time derivative 
            state_x_dot = 0
            state_elog_dot = np.zeros(3)
            state_elogdot_dot = np.zeros(3)

            # define logging variables
            QDMP = np.zeros((4,t.size))
            elogDMP = np.zeros((3,t.size))

            i = 0
            for ti in t:
                
                # Euler integration
                state_x = state_x + state_x_dot * dt
                state_elog = state_elog + state_elog_dot * dt
                state_elogdot = state_elogdot + state_elogdot_dot * dt


                # get state dot
                state_x_dot, state_elog_dot, state_elogdot_dot = self.get_state_dot(    state_x, 
                                                                                        state_elog,
                                                                                        state_elogdot)
                            
                # log data
                QDMP[:,i] = quatProduct( quatInv( quatExp(0.5 * state_elog) ) , self.Q_target )
                elogDMP[:,i] =state_elog
                i = i + 1



            # plot results
            # figure 1 is for p and Q
            fig = plt.figure(figsize=(4, 4))


            for i in range(4):
                axs = fig.add_axes([0.21, ((5-i)/4)*0.8-0.3, 0.7, 0.18])
                axs.plot(t, QDMP[i, :], 'k', linewidth=1.0)
                axs.set_xlim([0, t[-1]])
                axs.set_ylabel('$Q_' + str(i+1) + '(t)$ [m]',fontsize=14 )
                axs.set_xticks([])
            
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

            plt.show()


        


        

  
