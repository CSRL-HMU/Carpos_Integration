import numpy as np
import sys

sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_base')

from CSRL_orientation import *

# gains for reaching and tracking respectively
kt = 4.0


# kinametic controller for reaching in Ndof
def kinReaching_Ndof(x, xT, kr):
    qdot = - kr * (x - xT)
    return qdot

# kinametic controller for trajetctory tracking in Ndof
def kinTracking_Ndof(x, xd, xd_dot):
    qdot = xd_dot - kt * (x - xd)
    return qdot

# kinametic controller for reaching in end-effector position space
def kinReaching_R3(p, pT, Jp, kr):
    qdot = np.linalg.pinv(Jp , 0.01) @ ( - kr * (p - pT) )
    return qdot

# kinametic controller for reaching in end-effector position space
def kinTracking_R3(p, pd, pd_dot, Jp):
    qdot = np.linalg.pinv(Jp , 0.01) @ ( pd_dot - kt * (p - pd) )
    return qdot

# kinametic controller for reaching in end-effector orientation space
def kinReaching_SO3(A, AT, Jo, kr):
    eo = logError(A, AT)
    qdot = np.linalg.pinv(Jo , 0.01) @ ( - kr * eo )
    return qdot

# kinametic controller for reaching in end-effector orientation space
def kinTracking_SO3(A, Ad, omegad, Jo):
    eo = logError(A, Ad)
    qdot = np.linalg.pinv(Jo , 0.01) @ ( omegad - kt * eo )
    return qdot

# kinametic controller for reaching in end-effector generalized SE3 space
def kinReaching_SE3(p, A, pT, AT, J, kr):
    ep = p - pT
    eo = logError(A, AT)
    e = np.concatenate((ep,eo))
    qdot = np.linalg.pinv(J , 0.01) @ ( - kr * e )
    return qdot

# kinametic controller for reaching in end-effector generalized SE3 space
def kinTracking_SE3(p, A, pd, Ad, pd_dot, omegad, J):
    ep = p - pd
    eo = logError(A, Ad)
    e = np.concatenate((ep,eo))
    v = np.concatenate((pd_dot,omegad))
    qdot = np.linalg.pinv(J , 0.01) @ ( v - kt * e )
    return qdot


