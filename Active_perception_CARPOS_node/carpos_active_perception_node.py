import cv2
import mediapipe as mp
import numpy as np
import time 
from vision_aux import *
import pathlib
import scipy
import os
import getch
import math
import rospy
from std_msgs.msg import String
import pyzed.sl as sl
from pinhole import *


import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_base')
sys.path.insert(1, '/home/carpos/catkin_ws/src/CSRL_dmpy')

from CSRL_orientation import * 
from CSRL_math import * 
from dmpSE3 import * 


knowledge_path = "/home/carpos/catkin_ws/src/task_knowledge/"

# Our status publisher 
status_pub = rospy.Publisher('active_perception_status', String, queue_size=10)


# Declare math pi
pi = math.pi

# ZED 2
image_width = 1280
image_height = 720



# Camera intrinsic parameters of ZED2 for 1280 X 720 resolution
fx = 720  # Focal length in x+ 0.14
fy = 720  # Focal length in y
cx = 640  # Principal point x (center of the image)
cy = 360  # Principal point y (center of the image)

# # Camera intrinsic parameters of D435 Realsense
# fx = 870.00
# fy = 900.00
# cx = 640.886
# cy = 363.087

# Camera intrinsic matrix - K
Kcamera = np.array([[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])

ph = PinholeCamera(fx, fy, cx, cy, image_width, image_height)


# Initialize the ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE 



status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Error: {status}")
    exit(1)

# Set camera to manual exposure mode
zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 10)

# Prepare containers
image_zed = sl.Mat()

depth_map = sl.Mat()




# This is the callback function for the high level commands
def start_observation_callback(data):
    global zed, Kcamera, ph, image_width, image_height


    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # init time
    t = time.time() 

    # For webcam input:
    cap = cv2.VideoCapture("/dev/video0")

    

    # set the FPS
    fps = 30
    dt = 1.0 / fps

    # cap.set(cv2.CAP_PROP_FPS, fps)

    # get width and height
    c_width = image_width
    c_height = image_height 

    # initialize arrays
    Q_array = np.array([0, 0, 0, 0])
    Q_array.shape = (4,1)
    p_array = np.array([0, 0, 0])
    p_array.shape = (3,1)

    # get the logging name
    # print('Press any key...')
    # char = getch.getch() 

    # Recording has started
    print('Recording started ... ')

    # time.sleep(3)


    # initialize time
    t0 = time.time()



    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1) as hands:
    
        # printProgressBar(0, 20, prefix = 'Progress:', suffix = 'Complete', length = 50)
        while zed.grab() == sl.ERROR_CODE.SUCCESS:
    
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            image = image_zed.get_data()
            d_image = depth_map.get_data()
         



            t = time.time() - t0
            print("\r t=%.3f" % t , 's', end = '\r')

            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('LbD through MediaPipe Hands (press Esc to stop recording)', cv2.flip(image, 1))
            # cv2.waitKey(0)
        
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            

            

            p, R = get_hand_pose(hand_landmarks = hand_landmarks, image = image, width=c_width, height=c_height, ph_instance = ph, depth_image = d_image)

            print('p=', p)
            print('R=', R)

            Q = rot2quat(R)

            p.shape = (3,1)
            Q.shape = (4,1)

        
            Q_array = np.hstack((Q_array,Q))
            p_array = np.hstack((p_array,p))


            
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('LbD through MediaPipe Hands (press Esc to stop recording)', cv2.flip(image, 1))
            # if (cv2.waitKey(5) & 0xFF == 27) or t > 6:
            if (cv2.waitKey(5) & 0xFF == 27):
                break


    cv2.destroyAllWindows()
    cap.release()


    print("Recording ended. Duration = %.3f" % t)


    dt = t / p_array[0,:].size
    print("Estimated mean dt = %.3f" % dt)

    Q0 = np.copy(Q_array[:,1])
    p0 = np.copy(p_array[:,1])
    Rc0 = quat2rot(Q0)
    R0c = Rc0.T
    Nsamples = Q_array[0,:].size

    for i in range(Nsamples):
        # print(i)
        p_array[:,i] = R0c @ (p_array[:,i] - p0)
        Rch = quat2rot(Q_array[:,i])
        Q_array[:,i] = rot2quat(R0c @ Rch)

    Q_array[:,1:] = makeContinuous(Q_array[:,1:])
    mdic = {"p_array": p_array[:,1:], "Q_array": Q_array[:,1:], "dt": dt, "Rrobc": R0c}
    print(str(knowledge_path) + 'training_demo.mat')
    scipy.io.savemat(str(knowledge_path) + 'training_demo.mat', mdic)



    kernelType = 'Gaussian' # other option: sinc
    canonicalType = 'linear' # other option: exponential

    folderPath = pathlib.Path(__file__).parent.resolve()


    data = scipy.io.loadmat(str(knowledge_path) +'training_demo.mat')

    # x_train = data['x_data']

    p_train = np.array(data['p_array'])
    Q_train = np.array(data['Q_array'])

    dt = data['dt'][0][0]
    t = np.array(list(range(p_train[0,:].size))) * dt

    dmpTask = dmpSE3(20, t[-1])

    print("dmpTask.T=", dmpTask.T)
    dmpTask.train(dt, p_train, Q_train, False)

    mdic = {"W": dmpTask.get_weights(), "T": dmpTask.T}
    scipy.io.savemat(str(knowledge_path) + "dmp_model.mat", mdic)

    print("Training completed. The dmp_model is saved.")

   


    

##################################
############# MAIN ###############
##################################
if __name__ == '__main__':
    rospy.init_node('carpos_active_perception', anonymous=True)
    rate = rospy.Rate(100) 

    rospy.Subscriber("start_active_perception", String, start_observation_callback)
 
    while not rospy.is_shutdown():

        # motion_finished_error()
   
        rate.sleep()
