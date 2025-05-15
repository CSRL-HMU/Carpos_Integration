
import numpy as np
import scipy

knowledge_path = "/home/carpos/catkin_ws/src/task_knowledge/"
grasp_g = np.array([[0,	0,	-1,	-0.34103    ], [-1,	0	,0	,0], [0	,1	,0	,0], [0	,0	,0	,1]])

mdic = {"ggrasp": grasp_g}
scipy.io.savemat(str(knowledge_path) + "graspPose.mat", mdic)
