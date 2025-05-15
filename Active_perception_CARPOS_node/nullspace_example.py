import numpy as np



z = np.array([1 ,0 ,0])

zT = z.T
zT.shape = (3,1)

N = np.eye(3) - zT @ zT.T

print(N)