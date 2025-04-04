import roboticstoolbox as rt
import numpy as np
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

from matplotlib.pyplot import figure



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

# Init time
t = 0.0

# get time now
t_now = time.time()

# initialize qdot
qdot = np.zeros(6)


# ------------------ For the controller

# Target in joint space
RT = R0 @ rotY(pi/6)
pT = p0 + [0.0, 0, 0.3]


print('p0 = ', p0)
print('R0 = ', R0)
print('pT = ', pT)
print('RT = ', RT)

# -------------------------------------


for i in range(10 * 500):

    t = t + dt
    t_start = rtde_c.initPeriod()

    q = np.array(rtde_r.getActualQ())

    # Get  end-eefector pose
    g = ur.fkine(q)
    R = np.array(g.R)
    p = np.array(g.t)

    # get full jacobian
    J = np.array(ur.jacob0(q))

    pd, Rd, pd_dot, omegad = get5thorder_SE3(p0=p0, A0=R0, pT=pT, AT=RT, t=t, T=10)

    # reaching control signal
    qdot = kinTracking_SE3(p=p, A=R, pd=pd, Ad=Rd, pd_dot=pd_dot, omegad=omegad, J=J)

    rtde_c.speedJ(qdot, 1.0, dt)

    rtde_c.waitPeriod(t_start)

# Stop robot 
rtde_c.speedStop()



