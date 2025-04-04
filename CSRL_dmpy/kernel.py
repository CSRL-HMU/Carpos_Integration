import math
from CSRL_math import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# latex style plots 
plt.style.use(["default","no-latex"])

class kernel:

    def __init__(self, type_in, h_in, c_in):

        # The kernel type. Options:
        #    - 'Gaussian' (default)
        #    - 'sinc'
        self.type = type_in

        # the width of the kernel
        self.h = h_in

        # the center of the kernel
        self.c = c_in

    # get the value of the kernel at x
    def psi(self, x):

        if self.type == 'sinc':
            return sinc( self.h * pi * (x - self.c) )
        else:
            return math.exp(- self.h * ( x - self.c ) * ( x - self.c ) )

    # plots the kernel 
    def plot(self):

        # points of plot
        N_points = 1000

        # th array of variable x
        x_array = np.linspace(self.c - 4*self.h , self.c + 4*self.h, N_points)

        # the array of kernel value
        y_array = np.zeros(N_points)
        i = 0
        
        # compute the value for each x
        for xi in x_array:
            y_array[i] = self.psi(xi)
            i = i + 1

        # plot the results 
        plt.plot(x_array,y_array)

        # aesthetics and labeling
        plt.xlabel('$x$',fontsize=14 )
        plt.ylabel('$\psi(x)$',fontsize=14 )
        plt.title(self.type + ' kernel function. $c$=' + str(self.c) + ', $h$=' + str(self.h) , fontsize=14 )
        plt.grid()
        plt.show()
        
         

