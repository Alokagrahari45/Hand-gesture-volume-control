# Contributor: SABARINATH KR - Hand Detection & Landmarks

# Contributor: Gokul - Video Capture

# Contributor: Alok - Setup and Initialization Module

# Contributor: Gauri - MediaPipe Configuration

# Contributor: Shubh Sahu - Gesture Logic & Exit Control

import cv2
import mediapipe as mp
import math
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles  

hands = mp_hands.Hands(
    static_image_mode=False,          # Video stream (real-time)
    max_num_hands=1,                  # Single hand for volume control
    model_complexity=1,               # Better accuracy than 0 (balanced performance)
    min_detection_confidence=0.7,     # Higher threshold = fewer false detections
    min_tracking_confidence=0.7       # More stable tracking
)

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
prev_volume=0 

while True:
    ok,frame=cap.read()
    if not ok:
        break

    frame=cv2.flip(frame,1)

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            h,w,_=frame.shape
            thumb=hand_landmarks.landmark[4]
            index=hand_landmarks.landmark[8]

            x1,y1=int(thumb.x*w),int(thumb.y*h)
            x2,y2=int(index.x*w),int(index.y*h)
            distance=math.hypot(x2-x1,y2-y1)

            #Distance Calculation
            if distance<30:
                gesture="SELECT"
            elif distance < 80:
                gesture="HOLD"
            else: gesture="RELEASE"

            #Volume Mapping
            min_dist=30
            max_dist=200
            distance =np.clip(distance,min_dist,max_dist)
            volume_percent =np.interp(distance, [min_dist,max_dist],[0,100])

            #Volume Control
            volume_step=int(volume_percent/10)
            if volume_step >prev_volume:
                pyautogui.press("volumeup")
            elif volume_step < prev_volume:
                pyautogui.press("volumedown")

            prev_volume=volume_step




            #Interpolation of distance to volume range(0-100)


            cv2.circle(frame,(x1,y1),8,(0,255,0),-1)
            cv2.circle(frame,(x2,y2),8,(0,255,0),-1)
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

            cv2.putText(frame,f"Distance: {int(distance)}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.putText(frame,f"Gesture: {gesture}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

            #volume bar UI
            bar_height=int(np.interp(volume_percent, [0,100], [400,150]))
            cv2.rectangle(frame,(550,150), (580,400), (255,255,255), 2)
            cv2.rectangle(frame,(550, bar_height), (580,400), (0,255,255), -1)
    cv2.imshow("Hand Detection",frame)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
cap.release()
cv2.destroyAllWindows()


