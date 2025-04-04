from kernel import *
from kernelBase import *
from CSRL_math import *
from CSRL_orientation import *
from dmpSE3 import *
from dmpR3 import *
from dmpSO3 import *
import scipy.io
import pathlib
import os
import time



from dmp import *

kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential



########### Test kernel
# k = kernel(kernelType, 1, 2)
# k.plot()

########### Test kernel base
# kb = kernelBase(10, 5, kernelType, canonicalType, 0.5)

# kb.plot_t() 
# kb.plot_x()

########### Test 5th order
# t = np.array(list(range(5000))) * 0.002
# p_dp_ddp = np.zeros((3,t.size))
# p0 = 1
# pT = 5

# i = 0
# for ti in t:
#     p_dp_ddp[:,i] = get5thOrder(ti, p0, pT, t[-1])
#     i = i + 1


# plt.plot(t,p_dp_ddp[0,:],'k')
# plt.plot(t,p_dp_ddp[1,:],'r')
# plt.plot(t,p_dp_ddp[2,:],'b')


# plt.xlabel('$t$(s)',fontsize=14 )
# plt.ylabel('$p(t)$',fontsize=14 )
# plt.title('5th order polynomial')
# plt.legend(('$p$', '$\dot{p}$', '$\ddot{p}$'))
# plt.xlim(0,t[-1])
# plt.grid()
# plt.show()


########### Test sigmoid
# t = np.array(list(range(5000))) * 0.002
# y = np.zeros(t.size)

# i = 0
# for ti in t:
#     y[i] = sigmoid(ti, 4, 0.5)
#     i = i + 1


# plt.plot(t,y,'k')

# plt.xlabel('$t$(s)',fontsize=14 )
# plt.ylabel('$y(t)$',fontsize=14 )
# plt.title('Sigmoid')
# plt.xlim(0,t[-1])
# plt.grid()
# plt.show()
    

########### Test orientation transitions 
# R = np.array([[ 0, 1, 0],
#               [-1, 0, 0],
#               [ 0, 0, 1]])

# print('R=',R)
# print('Q=',rot2quat(R))
# print('R=',quat2rot(rot2quat(R)))


########### Test quaternion functions 
# R = np.identity(3)
# Rd = rotZ(pi-0.001) 

# # Q = rot2quat(R)
# # Qd = rot2quat(Rd)

# print('[log] e=',logError(R,Rd))
# print('[vect] e=',vectorError(R,Rd))





########### Test single axis dmp
# folderPath = pathlib.Path(__file__).parent.resolve()

# if os.name == 'nt': # the OS is Windows
#     data = scipy.io.loadmat(str(folderPath) +'\\example_data.mat')
# else:   # the OS is Linux
#     data = scipy.io.loadmat(str(folderPath) +'/example_data.mat')

# pd = data['p1']

# yd = pd[:,1]
# y = np.zeros(yd.size)

# dt = 0.001
# t = np.array(list(range(yd.size))) * dt

# dmpx = dmp(40, t[-1], kernelType, canonicalType)
# dmpx.train(dt, yd)

# # y_offset = 0.1
# # goal_offset = 0.1
# # tau = 0.7
# y_offset = 0.0
# goal_offset = 0.0
# tau = 1


# state = np.array([0, yd[0]+y_offset, 0])
# state_dot = np.zeros(3)

# dmpx.set_init_position(yd[0]+y_offset)
# dmpx.set_goal(yd[-1]+goal_offset)
# dmpx.set_tau(tau)

# i = 0
# for ti in t:
#     state = state + state_dot * dt
#     state_dot[0], state_dot[1], state_dot[2] = dmpx.get_state_dot( state[0], state[1], state[2])
#     y[i] = state[1]
#     i = i + 1

# plt.plot(t,yd,'r--')
# plt.plot(t,y,'k-')

# plt.xlabel('$t$(s)',fontsize=14 )
# plt.ylabel('$y(t)$',fontsize=14 )
# plt.title('DMP evolution')
# plt.legend(('demonstrated', 'DMP'))
# plt.xlim(0,t[-1])
# plt.grid()
# plt.show()




########### Test SE(3) dmp
folderPath = pathlib.Path(__file__).parent.resolve()

if os.name == 'nt': # the OS is Windows
    data = scipy.io.loadmat(str(folderPath) +'\\example_SE3.mat')
else:   # the OS is Linux
    data = scipy.io.loadmat(str(folderPath) +'/training_demo.mat')

# x_train = data['x_data']

p_train = np.array(data['p_array'])
Q_train = np.array(data['Q_array'])

dt = data['dt'][0][0]
t = np.array(list(range(p_train[1,:].size))) * dt

dmpTask = dmpSE3(20, t[-1])

print(dt)

dmpTask.train(dt, p_train, Q_train, True)

plt.show()
# p_0_offset = np.array([0.1, -0.2, 0.05])
# p_goal_offset = np.array([0.2, -0.4, 0.15])
# Q_0_offset = rot2quat(rotX(pi/3))
# Q_goal_offset = rot2quat(rotZ(pi/3))

p_0_offset = np.zeros(3)
p_goal_offset = np.zeros(3)
Q_0_offset = np.array([1, 0, 0, 0])
Q_goal_offset = np.array([1, 0, 0, 0])



p0 = p_train[:,0] + p_0_offset
pT = p_train[:,-1] + p_goal_offset

Q0 = quatProduct(  Q_0_offset, Q_train[:,0] )
QT = quatProduct(  Q_goal_offset, Q_train[:,-1])

dmpTask.set_init_pose(p0, Q0)
dmpTask.set_goal(pT, QT)

dmpTask.set_tau(1.0)

dmpTask.plotResponse(dt,p0,Q0,2500)


# plot results
fig = plt.figure(figsize=(4, 7))

fig.suptitle('Training dataset')

for i in range(3):
    axs = fig.add_axes([0.21, ((5-i)/6)*0.8+0.2, 0.7, 0.11])
    axs.plot(t, p_train[i, :], 'k', linewidth=1.0)
    axs.set_xlim([0, t[-1]])
    axs.set_ylabel('$p_' + str(i+1) + '(t)$ [m]',fontsize=14 )
    axs.set_xticks([])

for i in range(4):
    axs = fig.add_axes([0.21, ((5-(i+3))/6)*0.8+0.2, 0.7, 0.11])
    axs.plot(t, Q_train[i, :], 'k', linewidth=1.0)
    axs.set_xlim([0, t[-1]])
    axs.set_ylabel('$Q_' + str(i+1) + '(t)$ [rad]',fontsize=14 )
    
    if i==3:
        axs.set_xlabel('Time (s)',fontsize=14 )
    else:
        axs.set_xticks([])

plt.show()


################# Test R3 DMP
# folderPath = pathlib.Path(__file__).parent.resolve()

# if os.name == 'nt': # the OS is Windows
#     data = scipy.io.loadmat(str(folderPath) +'\\example_SE3.mat')
# else:   # the OS is Linux
#     data = scipy.io.loadmat(str(folderPath) +'/example_SE3.mat')

# x_train = data['x_data']

# p_train = np.array(x_train[:3,:])

# dt = 0.001
# t = np.array(list(range(p_train[1,:].size))) * dt

# dmpPos = dmpR3(20, t[-1])
# dmpPos.train(dt, p_train, True)


# # p_0_offset = np.array([0.1, -0.2, 0.05])
# # p_goal_offset = np.array([0.2, -0.4, 0.15])


# p_0_offset = np.zeros(3)
# p_goal_offset = np.zeros(3)

# p0 = p_train[:,0] + p_0_offset
# pT = p_train[:,-1] + p_goal_offset

# dmpPos.set_init_state(p0)
# dmpPos.set_goal(pT)

# dmpPos.set_tau(1)

# dmpPos.plotResponse(dt,p0,2500)


############# Test DMP SO3
# folderPath = pathlib.Path(__file__).parent.resolve()

# if os.name == 'nt': # the OS is Windows
#     data = scipy.io.loadmat(str(folderPath) +'\\example_SE3.mat')
# else:   # the OS is Linux
#     data = scipy.io.loadmat(str(folderPath) +'/example_SE3.mat')

# x_train = data['x_data']

# Q_train = np.array(x_train[-4:,:])

# dt = 0.001
# t = np.array(list(range(Q_train[1,:].size))) * dt

# dmpOri = dmpSO3(20, t[-1])
# dmpOri.train(dt, Q_train, True)

# Q_0_offset = rot2quat(rotX(pi/3))
# # Q_goal_offset = rot2quat(rotZ(pi/3))

# # Q_0_offset = np.array([1, 0, 0, 0])
# Q_goal_offset = np.array([1, 0, 0, 0])


# Q0 = quatProduct(  Q_0_offset, Q_train[:,0] )
# QT = quatProduct(  Q_goal_offset, Q_train[:,-1])


# dmpOri.set_init_state(Q0)
# dmpOri.set_goal(QT)

# dmpOri.set_tau(1)

# dmpOri.plotResponse(dt,Q0,2500)
