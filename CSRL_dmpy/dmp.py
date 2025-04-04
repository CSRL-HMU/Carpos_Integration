from kernelBase import *
from CSRL_math import *
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

# latex style plots 
plt.style.use(["default","no-latex"])

class dmp:

    def __init__(self, N_in, T_in, kernelType_in, xType_in, a_x_in = 1.0, a_in=20.0, b_in=0.8, tau_in=1.0):
        self.N = N_in
        self.T = T_in
        self.kernelType = kernelType_in
        self.xType = xType_in
        self.a_x = a_x_in
        self.a = a_in
        self.b = b_in
        self.W = np.zeros(N_in)

        # initial state of the canonical system
        if xType_in == 'linear':
            self.x0 = 0.0
        else: # canonical system is exponential
            self.x0 = 1.0
            
        self.tau = tau_in

    # sets the initial state y
    def set_init_position(self, y0_in):
        self.y0 = y0_in

    # sets the goal
    def set_goal(self, g_in):
        self.g = g_in

    # sets time scaling parameter
    def set_tau(self, tau_in):
        self.tau = tau_in

    def set_weights(self, W, T):
        self.W = W
        self.T = T
        self.kb = kernelBase(self.N,  self.T, self.kernelType, self.xType, self.a_x)


    def get_state_dot(self, x, y, z, customScale = False, scalingTerm = 1):
        
        # get the derivative of te canonical system
        if self.xType == 'linear':
            xdot =  1.0 / self.T  
        else:   # canonical system is exponential 
            xdot = - self.a_x * x 
        
        # the derivative of the y state 
        ydot = z

        # get the values of kenrel base for this x
        Psi = self.kb.get_psi_vals_x(x)

     
        # the forcing term value
        if self.kernelType == 'sinc':
            f = self.W @ Psi 
        else:
            f = self.W @ Psi / (np.sum(Psi) + 0.0000001)

        # the derivative of the z state 
        s = 1 - sigmoid(self.kb.ksi_inv(x), 1.0*self.T , 0.05*self.T) 
        if not customScale:
            scalingTerm = (self.g - self.y0)

        zdot = self.a * ( self.b * ( self.g - y) - z ) + s * scalingTerm * f


        # print(xdot)
        # return the time scaled derivative of the state 
        return xdot/self.tau, ydot/self.tau, zdot/self.tau
    
    def train(self, dt, y_array, plotEn = False, customScale = False, scalingTerm = 1):

        # make sure that this is numpy array
        y_array = np.array(y_array)

        # get the number of points
        Npoints = y_array.size


        # time array
        t_array = np.array(list(range(Npoints)))*dt

        # set total time
        self.T = t_array[-1]

        # set time scaling to 1
        self.tau = 1.0

        # set initial state and goal
        self.y0 = y_array[0]
        self.g = y_array[-1]
  
        
        # initialize the kernelBase
        self.kb = kernelBase(self.N,  self.T, self.kernelType, self.xType, self.a_x)

        # as z = ydot, we compute z_array  
        z_array = maFilter(np.gradient(y_array, dt) , 5)

        # compute the zdot array
        zdot_array = maFilter(np.gradient(z_array, dt) , 5)

        # initialize the phase cariable array
        x_array = np.zeros(Npoints)

        i = 0
        # compute phase variable array
        for ti in t_array:
            x_array[i] =  self.kb.ksi(ti)
            i = i + 1

        if not customScale:
            scalingTerm = (self.g - self.y0)

        #compute the demonstrated forcing term
        fd_array = ( zdot_array - self.a * ( self.b * ( self.g - y_array) - z_array ) ) / scalingTerm

        # train and set the weights W 
        if self.kernelType == 'sinc':
            self.approximate_sincs(t_array, fd_array, Npoints, plotEn)
        else:   # kernel type is 'Gaussian'
            self.approximate_LS_gaussians(t_array, fd_array, Npoints, plotEn)
        
    # compute the weigths W for sinc interpolation
    def approximate_sincs(self, t, fd, Npoints, plotEn = False):

        c = self.kb.c_t
        j = 0
        for i in range(Npoints):
            if t[i] > c[j]:
                self.W[j] = ( fd[i] + fd[i-1] )/2.0 
                j = j + 1

        
        if plotEn:
            PsiPsi = np.zeros((Npoints, self.N))
            for i in range(Npoints):
                Psi = self.kb.get_psi_vals_t(t[i])
                PsiPsi[i,:] = Psi 

            f = PsiPsi @ self.W
            plt.plot(t,fd)
            plt.plot(t,f)
            plt.xlabel('$t$',fontsize=14 )
            plt.ylabel('$f(x(t))$',fontsize=14 )
            plt.title('Function approximation with sinc base. $N$=' + str(self.N),fontsize=14 )
            plt.xlim(0,self.T)
            plt.grid()
            plt.show()
        

    # compute the weigths W for Gaussian interpolation
    def approximate_LS_gaussians(self, t, fd, Npoints, plotEn = False):


        PsiPsi = np.zeros((Npoints, self.N))

       
        for i in range(Npoints):
            Psi = self.kb.get_psi_vals_t(t[i])
            PsiPsi[i,:] = Psi / np.sum(Psi) 

        self.W = np.linalg.pinv(PsiPsi) @ fd

        if plotEn:
            f = PsiPsi @ self.W
            plt.plot(t,fd)
            plt.plot(t,f)
            plt.xlabel('$t$',fontsize=14 )
            plt.ylabel('$f(x(t))$',fontsize=14 )
            plt.title('Function approximation with Gaussian base. $N$=' + str(self.N),fontsize=14 )
            plt.xlim(0,self.T)
            plt.grid()
            plt.show()




        
                



