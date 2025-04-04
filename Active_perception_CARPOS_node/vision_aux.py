import cv2
import mediapipe as mp
import numpy as np
import pinhole


zed_depth_enable = False

def get_hand_pose(hand_landmarks, image, width, height, ph_instance, depth_image):

    # get the cordinates of the specific landmarks 0, 1, 5 and 17
    j = 0 
    pw = np.array( [ hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])

    j = 1 
    pbth = np.array( [ hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])

    j = 5 
    pbi = np.array( [ hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])

    j = 17 
    pbs = np.array( [ hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y, hand_landmarks.landmark[j].z])

    # compute the vectors to the finger bases
    p1 = (pbth - pw)/np.linalg.norm(pbth - pw)
    p2 = (pbi - pw)/np.linalg.norm(pbi - pw)
    p3 = (pbs - pw)/np.linalg.norm(pbs - pw)

    # z will be towards the mid vector from those vectors (giving more weight to the thumb) 
    z = (0.5*p1 + 0.25*p2 + 0.25*p3)
    z = z / np.linalg.norm(z)
    z.shape = (3,1)


    # x will be towards the thumb
    x = (np.identity(3) - z @ z.T) @ p1 
    x = x / np.linalg.norm(x)
    x.shape = (3,1)

    # y will be according to the right hand rule
    y = np.cross( z.T , x.T)
    y = y / np.linalg.norm(y)
    y.shape = (3,1)

    # create the rotation matrix
    R = np.hstack(( x , y , z))

    # print the lines of the frame to the image
    w_pixel = np.array([int(pw[0]*width), int(pw[1]*height)])
    x_pixel = np.array([int(x[0]*100), int(x[1]*100)])
    y_pixel = np.array([int(y[0]*100), int(y[1]*100)])
    z_pixel = np.array([int(z[0]*100), int(z[1]*100)])
    cv2.line(image, w_pixel , w_pixel + x_pixel, color=(0,0,200), thickness=10) 
    cv2.line(image, w_pixel , w_pixel + y_pixel, color=(0,200,0), thickness=10)
    cv2.line(image, w_pixel , w_pixel + z_pixel, color=(200,0,0), thickness=10)

    depth_val = depth_image[w_pixel[0],w_pixel[1]]

    if zed_depth_enable:
        pw = ph_instance.back_project(w_pixel, depth_val)

    return pw, R


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()