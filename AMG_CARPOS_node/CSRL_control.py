import numpy as np
import sys

sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_base')
from CSRL_orientation import *

# gains for reaching and tracking respectively
kt = 4.0



def wpinv(J):

    W = np.eye(8)
    W[6,6] = 10
    W[7,7] = 10
    invW = np.linalg.inv(W)
    return invW @ J.T @ np.linalg.inv(J @ invW @ J.T)

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
    qdot = wpinv(Jp) @ ( - kr * (p - pT) )
    return qdot

# kinametic controller for reaching in end-effector position space
def kinTracking_R3(p, pd, pd_dot, Jp):
    qdot = wpinv(Jp) @ ( pd_dot - kt * (p - pd) )
    return qdot

# kinametic controller for reaching in end-effector orientation space
def kinReaching_SO3(A, AT, Jo, kr):
    eo = logError(A, AT)
    qdot = wpinv(Jo) @ ( - kr * eo )
    return qdot

# kinametic controller for reaching in end-effector orientation space
def kinTracking_SO3(A, Ad, omegad, Jo):
    eo = logError(A, Ad)
    qdot = wpinv(Jo) @ ( omegad - kt * eo )
    return qdot

# kinametic controller for reaching in end-effector generalized SE3 space
def kinReaching_SE3(p, A, pT, AT, J, kr):
    ep = p - pT
    eo = logError(A, AT)
    e = np.concatenate((ep,eo))
    qdot = wpinv(J) @ ( - kr * e )
    return qdot

# kinametic controller for reaching in end-effector generalized SE3 space
def kinTracking_SE3(p, A, pd, Ad, pd_dot, omegad, J, weightedEn = True):
    ep = p - pd
    eo = logError(A, Ad)
    e = np.concatenate((ep,eo))
    v = np.concatenate((pd_dot,omegad))
    if weightedEn:
        qdot = wpinv(J) @ ( v - kt * e )
    else:
        qdot = np.linalg.pinv(J) @ ( v - kt * e )

        
    return qdot


