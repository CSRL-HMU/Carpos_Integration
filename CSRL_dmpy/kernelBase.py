from CSRL_math import *
from kernel import *
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

# latex style plots 
plt.style.use(["default","no-latex"])

class kernelBase:

    def __init__(self, N_in, T_in, kernelType_in, xType_in, a_x_in = 1.0):
        
        # Number of kernels
        self.N = N_in

        # Canonical system type. Options:
        #       'linear' 
        #       'exponential'  (default)
        self.xType = xType_in

        # Pole of the exponential canonical system
        self.a_x = a_x_in

        # Total duration of motion 
        self.totalTime = T_in

        #  Kernel types. Options: 
        #      'Gaussian' (default)
        #      'sinc'
        self.kernelType = kernelType_in

       
        # The kenrel centers are set in time
        self.c_t = np.linspace(0, self.totalTime, self.N)

        # The kenrel widths are set in times
        if self.kernelType == 'sinc':
            self.h_t = 1 / (self.c_t[2] - self.c_t[1]) 
        else: # the kernel type is 'Gaussian'
            self.h_t = 1 / math.pow( self.c_t[2] - self.c_t[1] , 2) 

        self.kernelArray = []
        # Create the kernel array
        for i in range(self.N):  
            k = kernel(self.kernelType, self.h_t , self.c_t[i])
            self.kernelArray.append(k)


    # mapping time -> phase variable
    def ksi(self, t_in):
        if self.xType == 'linear':
            if t_in < self.totalTime:
                return t_in / self.totalTime
            else: 
                return 1
        else: # # canonical system is exponential 
            return math.exp(-  self.a_x  * t_in )
        
    # inverse mapping phase variable -> time
    def ksi_inv(self, x_in):
        if self.xType == 'linear':
            return x_in * self.totalTime
        else: # # canonical system is exponential 
            return - math.log( x_in ) / self.a_x
        
    # Reurns the base values in an array for specific phase variable value (x)
    def get_psi_vals_x(self, x_in):

        # phase var -> time
        t = self.ksi_inv(x_in)

        # init psi_vals
        psi_vals = self.get_psi_vals_t(t)

        # return array
        return psi_vals
    
    # Reurns the base values in an array for specific time instance (t)
    def get_psi_vals_t(self, t_in):

        # init psi_vals
        psi_vals = np.zeros(self.N)
        for i in range(self.N):
            # get kernel value for this instance
            psi_vals[i] = self.kernelArray[i].psi(t_in)

        # return array
        return psi_vals
    
    # Plots the kernel base in time
    def plot_t(self):

        # plot points
        N_points = 1000

        # time array
        t_array = np.linspace(0, 1.2 * self.totalTime , N_points)

        # kernel value array
        y_array = np.zeros((self.N , N_points))
        i = 0
        
        # find the values in time 
        for ti in t_array:  
            y_array[:,i] = self.get_psi_vals_t(ti)
            i = i + 1

        # plot all kernels
        for i in range(self.N):
            plt.plot(t_array,y_array[i,:])

        # aesthetics and labeling 
        plt.xlabel('$t$ (s)',fontsize=14 )
        plt.ylabel('$\psi_i(t)$',fontsize=14 )
        plt.xlim(0 , 1.2* self.totalTime)
        plt.title('Kernel bases in $t$. $N$=' + str(self.N),fontsize=14 )
        plt.grid()
        plt.show()


    # Plots the kernel base as a function of phase variable x
    def plot_x(self):

        # plot points
        N_points = 1000

        # the min value of x if linear or exponential 
        if self.xType == 'linear':
            # x is 0 at the start of the motion 
            xmin = 0
        else:
            # find x at the end of the motion 
            xmin = self.ksi(self.totalTime)

        # create the x array 
        x_array = np.linspace(xmin, 1 , N_points)

        # array of the kernel values
        y_array = np.zeros((self.N , N_points))
        i = 0
        
        # compute the kernel values
        for xi in x_array:
            y_array[:,i] = self.get_psi_vals_x(xi)
            i = i + 1

        # plot results 
        for i in range(self.N):
            plt.plot(x_array,y_array[i,:])

        # aesthetics and labeling
        
        plt.xlabel('$x$',fontsize=14 )
        plt.ylabel('$\psi_i(x)$',fontsize=14 )
        plt.title('Kernel bases in $x$. $N$=' + str(self.N),fontsize=14 )
        plt.xlim(xmin,1)
        plt.grid()
        plt.show()
    
   



    
