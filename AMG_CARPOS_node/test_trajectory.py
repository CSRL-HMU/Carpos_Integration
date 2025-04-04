from CSRL_trajectory import *
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["default","no-latex"])

########################### TEST N DOF once

# p0 = np.array([1, 3, 5, 7, -2, 6])
# pT = np.array([-9, 2, 0, -6, 5, 6])

# p = get5thorder_R3(p0, pT, 3 , 3)
# print(p)


########################### TEST N DOF all times

# p0 = np.array([1, 3, 5, 7, -2, 6])
# pT = np.array([-9, 2, 0, -6, 5, 6])
# dt = 0.02
# t = np.array(range(100))
# t = t*dt

# p = np.zeros((100, 6))
# pdot = np.zeros((100, 6))

# for i in range(100):

#     p[i, :], pdot[i, :] = get5thorder_Ndof(p0, pT, t[i] , 1)

# # plot results
# fig = plt.figure(figsize=(4, 6))


# for i in range(6):
#     axs = fig.add_axes([0.21, ((5-i)/6)*0.9+0.1, 0.7, 0.12])
#     axs.plot(t, p[:, i], 'k', linewidth=1.0)
#     axs.plot(t, pdot[:, i], 'k--', linewidth=2.0)

    
# plt.show()




########################### TEST R3 all

# p0 = np.array([1, 3, 5])
# pT = np.array([-9, 2, 0])
# dt = 0.02
# t = np.array(range(100))
# t = t*dt

# p = np.zeros((100, 3))
# pdot = np.zeros((100, 3))

# for i in range(100):

#     p[i, :], pdot[i, :] = get5thorder_R3(p0, pT, t[i] , 1)

# # plot results
# fig = plt.figure(figsize=(4, 4))


# for i in range(3):
#     axs = fig.add_axes([0.21, ((2-i)/3)*0.82+0.15, 0.7, 0.25])
#     axs.plot(t, p[:, i], 'k', linewidth=1.0)
#     axs.plot(t, pdot[:, i], 'k--', linewidth=2.0)

    
# plt.show()

########################### TEST SO(3) once

R0 = np.identity(3)
print('R0=', R0)

RT = rotZ(-pi/2)
print('RT=', RT)

R, omega = get5thorder_SO3(R0, RT, 1.5 , 3)
print('R=', R)
print('omega=', omega)


