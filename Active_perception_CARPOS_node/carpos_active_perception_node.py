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






# This is the callback function for the high level commands
def start_observation_callback(data):



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

    cap.set(cv2.CAP_PROP_FPS, fps)

    # get width and height
    c_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    c_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  

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
        while cap.isOpened():


            t = time.time() - t0
            print("\r t=%.3f" % t , 's', end = '\r')

            
            success, image = cap.read()
            if not success: 
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

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
            

            

            p, R = get_hand_pose(hand_landmarks, image, width=c_width, height=c_height)

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
