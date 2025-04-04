from dmp import *
from CSRL_math import *
import numpy as np

import scienceplots
import matplotlib.pyplot as plt

# latex style plots 
plt.style.use(["default","no-latex"])

class dmpR3:
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

    # inputs p_array: 3xN (translation)
    def train(self, dt, p_array, plotPerformance = False):

        # number of samples
        N = p_array[0,:].size

        # translation target 
        self.p_target = p_array[:,-1]

        # train the DMP in each axis
        for i in range(3):
            self.dmp_array[i].train(dt, p_array[i,:])

        # if plotPerformance flag is enabled ...
        if plotPerformance: self.plotPerformance(dt, p_array)

    # returns the whole state of the system
    def get_state_dot(self, x, p, pdot):
        
        p_out_dot = np.zeros(3)
        p_out_2dot = np.zeros(3)

        # for position and orientation
        for i in range(3):
            x_out_dot, p_out_dot[i], p_out_2dot[i] = self.dmp_array[i].get_state_dot( x, p[i], pdot[i]) 
           
        # return state dot
        return x_out_dot, p_out_dot, p_out_2dot
      

    # sets the initial state p 
    def set_init_state(self, p0_in):

        # set the class variables
        self.p0 = p0_in

        # set the values in DMP objects 
        for i in range(3):
            self.dmp_array[i].set_init_position(self.p0[i])

       
    # sets the goal 
    def set_goal(self, gP_in):

        # set the class variables 
        self.p_target = gP_in

        # set the DMP object's variables
        for i in range(3):
            self.dmp_array[i].set_goal(self.p_target[i])
 
    # sets time scaling parameter for all axes
    def set_tau(self, tau_in):

        # set the class variables 
        self.tau = tau_in

        # set the DMP object's variables
        for i in range(3):
            self.dmp_array[i].set_tau(self.tau)
            

    # plots the response of the DMP compared to the training dataset
    def plotPerformance(self, dt, p_array):
        
        # ensure np array 
        p_array = np.array(p_array)

        # create time
        t = np.array(list(range(p_array[0,:].size))) * dt

        # initialize phase variable
        if self.xType == 'linear':
            state_x = 0
        else:
            state_x = 1

        # initialise position
        p0 = p_array[:,0]

        # initialise states
        state_p = p0 
        state_pdot = np.zeros(3)


        # initialise states time derivative
        state_x_dot = 0
        state_p_dot = np.zeros(3)
        state_pdot_dot = np.zeros(3)

        # define logging variables
        pDMP = np.zeros((3,t.size))

        i = 0
        for ti in t:
            
            # Euler integration
            state_x = state_x + state_x_dot * dt
            state_p = state_p + state_p_dot * dt
            state_pdot = state_pdot + state_pdot_dot * dt

            # get state dot
            state_x_dot, state_p_dot, state_pdot_dot = self.get_state_dot(  state_x, 
                                                                            state_p,                                                                                      
                                                                            state_pdot)
                        
            # log daata 
            pDMP[:,i] = state_p
            i = i + 1

        # plot results
        fig = plt.figure(figsize=(4, 3))

        for i in range(3):
            axs = fig.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
            axs.plot(t, pDMP[i, :], 'k', linewidth=1.0)
            axs.plot(t, p_array[i, :], 'k--', linewidth=2.0)
            axs.set_xlim([0, t[-1]])
            axs.set_ylabel('$p_' + str(i+1) + '(t)$ [m]',fontsize=14 )
            axs.set_xticks([])

            if i==2:
                axs.set_xlabel('Time (s)',fontsize=14 )
                lgnd = axs.legend(['$DMP$','$demo$'],fontsize=11,ncol=2,loc="lower right")
                lgnd.get_frame().set_alpha(None)
            else:
                axs.set_xticks([])


        plt.show()

    # plots the response of the DMP for a given initial position (assumes zero initial velocity)
    def plotResponse(self, dt, p0, iterations):
            
        
            # create time array
            t = np.array(list(range(iterations))) * dt

            # initialize phase variable
            if self.xType == 'linear':
                state_x = 0
            else:
                state_x = 1

            # initialize states
            state_p = p0 
            state_pdot = np.zeros(3)
   

            # initialize states' time derivative 
            state_x_dot = 0
            state_p_dot = np.zeros(3)
            state_pdot_dot = np.zeros(3)

            # define logging variables
            pDMP = np.zeros((3,t.size))

            i = 0
            for ti in t:
                
                # Euler integration
                state_x = state_x + state_x_dot * dt
                state_p = state_p + state_p_dot * dt
                state_pdot = state_pdot + state_pdot_dot * dt
               


                # get state dot
                state_x_dot, state_p_dot, state_pdot_dot = self.get_state_dot(  state_x, 
                                                                                state_p,                                                                                      
                                                                                state_pdot)
                            
                # log data
                pDMP[:,i] = state_p
                i = i + 1



            # plot results
            # figure 1 is for p and Q
            fig = plt.figure(figsize=(4, 3))
            for i in range(3):
                axs = fig.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
                axs.plot(t, pDMP[i, :], 'k', linewidth=1.0)
                axs.set_xlim([0, t[-1]])
                axs.set_ylabel('$p_,' + str(i+1) + '(t)$ [rad]',fontsize=14 )
                
                if i==2:
                    axs.set_xlabel('Time (s)',fontsize=14 )
                else:
                    axs.set_xticks([])

            plt.show()


        


        

  
