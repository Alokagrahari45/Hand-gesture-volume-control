# Contributor: Gouri - MediaPipe Configuration
import cv2
import mediapipe as mp
import math

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

while True:
    ok,frame=cap.read()
    if not ok:break
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
            if distance<30:
                gesture="SELECT"
            elif distance < 80:
                gesture="HOLD"
            else: gesture="RELEASE"

            cv2.circle(frame,(x1,y1),8,(0,255,0),-1)
            cv2.circle(frame,(x2,y2),8,(0,255,0),-1)
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

            cv2.putText(frame,f"Distance: {int(distance)}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.putText(frame,f"Gestures: {gesture}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.imshow("Gesture Detection",frame)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
cap.release()
cv2.destroyAllWindows()


