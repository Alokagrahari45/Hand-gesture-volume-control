from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math
import pyautogui
from pycaw.pycaw import AudioUtilities

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = None

endpoint_volume = AudioUtilities.GetSpeakers().EndpointVolume


def generate_frames():

    global cap

    while True:

        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                h, w, _ = frame.shape

                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                distance = math.hypot(x2 - x1, y2 - y1)

                if distance < 45:
                    pyautogui.press("volumedown")

                elif distance > 80:
                    pyautogui.press("volumeup")

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start")
def start():

    global cap
    cap = cv2.VideoCapture(0)

    return jsonify({"status": "started"})


@app.route("/stop")
def stop():

    global cap

    if cap:
        cap.release()

    return jsonify({"status": "stopped"})


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/volume")
def volume():

    vol = int(endpoint_volume.GetMasterVolumeLevelScalar() * 100)

    return jsonify({"volume": vol})


if __name__ == "__main__":
    app.run(debug=True)