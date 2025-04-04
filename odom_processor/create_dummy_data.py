from scipy.io import savemat
import numpy as np

ggrasp = np.array([[ 0 , 0 , -1 , 0.1 ],[ 0 , 1 , 0 , 0 ],[ 1 , 0 , 0 , 0 ],[ 0 , 0 , 0 , 1 ]])


mdic = {"ggrasp": ggrasp}
savemat("graspPose.mat", mdic)
