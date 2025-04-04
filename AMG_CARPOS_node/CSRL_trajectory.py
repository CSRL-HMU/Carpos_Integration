from CSRL_math import *
from CSRL_orientation import *


# This function implements the 5th order polynomal trajectory for an array on N elements
def get5thorder_Ndof(q0, qT, t, T):
    N = len(q0)

    k0 = q0
    k1 = np.zeros( N )
    k2 = np.zeros( N )
    k3 = 10 / pow( T , 3 ) * ( qT - q0 )    
    k4 = -15 / pow( T , 4 ) * ( qT - q0 )
    k5 = 6 / pow( T , 5 ) * ( qT - q0 )

    if t<0:
        q = q0  
        qdot = np.zeros( N )
    elif t<T:
        q = k0 + k1 * t + k2 * pow( t , 2 ) + k3 * pow( t , 3 ) + k4 * pow( t , 4 ) + k5 * pow( t , 5 )
        qdot =   k1     + 2*k2*t            + 3*k3*pow( t , 2 ) + 4*k4*pow( t , 3 ) + 5*k5*pow( t , 4 )
    else:
        q = qT
        qdot = np.zeros( N )

    return q, qdot    


# This function implements the 5th order polynomal trajectory in R3 Euclidean space
def get5thorder_R3(p0, pT, t, T):
    return get5thorder_Ndof(p0, pT, t, T)

# This function implements the 5th order polynomal trajectory in SO3 rotation space
# A# can either be quaternion or rotation matrices
def get5thorder_SO3(A0, AT, t, T):
    
    isRot = False
    if AT.size > 4:
        isRot = True

    eT = logError( A0, AT )
    
    elog, elogdot = get5thorder_Ndof(eT, np.zeros(3) , t, T)

    Aout = quatProduct(  quatExp( 0.5 * elog ), enforceQuat(AT))

    if isRot:
        omega = AT @ logDerivative2_AngleOmega(elogdot, Aout)
        Aout = quat2rot(Aout)
    else:
        omega = quat2rot(AT) @ logDerivative2_AngleOmega(elogdot, Aout)


    return Aout, omega


# This function implements the 5th order polynomal trajectory in SE3 Task space
# A# can either be quaternion or rotation matrices
def get5thorder_SE3(p0, pT, A0, AT, t, T):
    p, pdot = get5thorder_R3(p0, pT, t , T)
    Aout, omega = get5thorder_SO3(A0, AT, t , T)

    return p, Aout, pdot, omega