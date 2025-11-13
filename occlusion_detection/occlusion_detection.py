#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import mediapipe as mp
import torch
from scipy.io import savemat
import time


# --- Initialize detectors ---
print("YOLO running on:", "GPU" if torch.cuda.is_available() else "CPU")
yolo = YOLO('models/keypoints_new.pt')
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Setup RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
cv2.namedWindow("Tomato & Hand Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tomato & Hand Detection", 1280, 960)
palm_conf_list = []
iou_list = []
time_list = []
start_time = time.time()


def compute_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

while True:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        continue
    frame = np.asanyarray(color_frame.get_data())

    # --- YOLO tomato detection ---
    detections = yolo.predict(source=frame, conf=0.7, verbose=False, show=False)[0]
    tomato_box = None
    if len(detections.boxes.xyxy) > 0:
        # Take first detection (you can modify to pick highest confidence)
        x1, y1, x2, y2 = map(int, detections.boxes.xyxy[0].cpu().numpy())
        tomato_box = [x1, y1, x2, y2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Tomato", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # --- MediaPipe hand detection ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)
    palm_conf = 0.0
    hand_box = None
    if results.multi_handedness:
        palm_conf = results.multi_handedness[0].classification[0].score
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        hand_box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        cv2.rectangle(frame, (hand_box[0], hand_box[1]), (hand_box[2], hand_box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"PalmConf: {palm_conf:.2f}", (hand_box[0], hand_box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # --- IoU between tomato and hand ---
    if tomato_box and hand_box:
        iou = compute_iou(tomato_box, hand_box)
        cv2.putText(frame, f"IoU: {iou:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # print(f"Palm conf={palm_conf:.3f}, IoU={iou:.3f}")
        timestamp = time.time() - start_time
        palm_conf_list.append(palm_conf)
        iou_list.append(iou)
        time_list.append(timestamp)
        print(f"[{timestamp:.2f}s] Palm={palm_conf:.3f}, IoU={iou:.3f}")

    cv2.imshow("Tomato & Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

data = {
    'time': np.array(time_list),
    'palm_conf': np.array(palm_conf_list),
    'iou': np.array(iou_list)
}
pre_stamp = str(time.time())
savemat(f'logs/{pre_stamp}.mat', data)
print("Saved to hand_tomato_kpi.mat")


pipeline.stop()
cv2.destroyAllWindows()
