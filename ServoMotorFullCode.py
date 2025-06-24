# Your Setup
# Function	            Connected to PWM Channel
# Horizontal (Pan)	    Channel 0
# Vertical (Tilt)	    Channel 1
# Trigger Mechanism	    Channel 2

# sudo apt update
# sudo apt install -y python3-smbus i2c-tools
# pip3 install adafruit-circuitpython-pca9685

# sudo raspi-config
# Go to Interface Options → I2C → Enable
# sudo reboot


# Wiring Summary
# PCA9685 Pin	Raspberry Pi 3B+ GPIO
# VCC	        3.3V (Pin 1)
# GND	        GND (Pin 6)
# SDA	        GPIO 2 (Pin 3)
# SCL	        GPIO 3 (Pin 5)
# V+	        External 5V (for servo power)
# GND	        Same GND as Pi and power supply

from datetime import datetime
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import time
import threading
import queue

# ---------- TEST MODE SWITCH ----------
TEST_MODE = False
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
except (ImportError, NotImplementedError) as e:
    print("PCA9685 module not available, switching to TEST_MODE.")
    TEST_MODE = True

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
CSV_PATH = os.path.join(BASE_DIR, "labels.csv")
TOLERANCE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

PAN_CHANNEL = 0     # Horizontal (left-right)
TILT_CHANNEL = 1    # Vertical (up-down)
TRIGGER_CHANNEL = 2 # Trigger
FIRE_DELAY = 0.3    # seconds
FIRE_ANGLE = 30     # fire position
REST_ANGLE = 90     # resting servo angle
STABILITY_THRESHOLD = 5  # number of consistent frames before triggering fire

if not TEST_MODE:
    print("Running in REAL hardware mode (PCA9685 active)")
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 50  # 50 Hz for servo motors

    def angle_to_duty(angle):
        return int((angle / 180.0 * 2.0 + 1.0) / 20.0 * 65535)

    def set_servo_angle(channel, angle):
        angle = max(0, min(180, angle))
        duty = angle_to_duty(angle)
        print(f"[REAL] Setting servo on channel {channel} to angle {angle} → duty {duty}")
        pca.channels[channel].duty_cycle = duty
else:
    print("Running in TEST_MODE (Simulation only)")
    def set_servo_angle(channel, angle):
        print(f"[SIM] Channel {channel}: angle = {angle:.1f}°")

def point_servo_to_face(face_center_x, face_center_y, frame_width, frame_height, name):
    norm_x = face_center_x / frame_width
    norm_y = face_center_y / frame_height
    pan_angle = 180 * norm_x
    tilt_angle = 180 * norm_y
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"[TRACKING] {timestamp} | name: {name} | X: {face_center_x}, Y: {face_center_y} | Pan: {pan_angle:.2f}°, Tilt: {tilt_angle:.2f}°")

    set_servo_angle(PAN_CHANNEL, pan_angle)
    set_servo_angle(TILT_CHANNEL, tilt_angle)

def fire_trigger():
    print("Trigger activated")
    set_servo_angle(TRIGGER_CHANNEL, FIRE_ANGLE)
    time.sleep(FIRE_DELAY)
    set_servo_angle(TRIGGER_CHANNEL, REST_ANGLE)
    print("Trigger reset")

# ---------- Load Known Faces ----------
def load_known_faces():
    print("[INFO] Loading known face encodings...")
    df = pd.read_csv(CSV_PATH)
    known_encodings = []
    known_names = []
    for _, row in df.iterrows():
        image_path = os.path.join(KNOWN_FACES_DIR, row['filename'])
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(row['name'])
            print(f"[LOADED] {row['name']} from {row['filename']}")
        else:
            print(f"[WARNING] No face found in {row['filename']}")
    return known_encodings, known_names

# ---------- Video Stream Class ----------
class VideoStream:
    def __init__(self, src=0):
        print("[INFO] Initializing video stream...")
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# ---------- Webcam Face Recognition ----------
def process_webcam_mt(known_encodings, known_names):
    print("[INFO] Starting face detection pipeline...")
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    q = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    video_stream = VideoStream(0).start()
    time.sleep(1.0)

    def process_thread():
        frame_count = 0
        FRAME_SKIP = 2
        last_faces = []
        unknown_counter = 0

        while not stop_event.is_set():
            ret, frame = video_stream.read()
            if not ret or frame is None:
                print("[ERROR] Failed to read frame")
                stop_event.set()
                break

            if frame_count % FRAME_SKIP == 0:
                faces_this_frame = []
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]

                for (x, y, w, h) in faces:
                    top = int(y * scale_y)
                    right = int((x + w) * scale_x)
                    bottom = int((y + h) * scale_y)
                    left = int(x * scale_x)
                    face_image = frame[top:bottom, left:right]
                    if face_image.size == 0:
                        continue
                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(face_rgb)
                    name = "Unknown"
                    color = (0, 0, 255)
                    if encodings:
                        matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=TOLERANCE)
                        if any(matches):
                            best_index = np.argmin(face_recognition.face_distance(known_encodings, encodings[0]))
                            name = known_names[best_index]
                            color = (0, 255, 0)

                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2
                    point_servo_to_face(face_center_x, face_center_y, frame.shape[1], frame.shape[0], name)

                    if name == "Unknown":
                        unknown_counter += 1
                        print(f"[DEBUG] Unknown detected - counter = {unknown_counter}")
                        if unknown_counter >= STABILITY_THRESHOLD:
                            print("[ACTION] Stable unknown detected, firing trigger")
                            fire_trigger()
                            unknown_counter = 0
                    else:
                        unknown_counter = 0  # reset if known face is found

                    faces_this_frame.append((left, top, right, bottom, name, color, face_center_x, face_center_y))
                last_faces = faces_this_frame

            annotated = frame.copy()
            for (left, top, right, bottom, name, color, face_center_x, face_center_y) in last_faces:
                cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
                cv2.circle(annotated, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
                cv2.putText(annotated, name, (left, top - 10), FONT, 0.6, color, 2)
            try:
                if not q.full():
                    q.put(annotated, timeout=0.05)
            except queue.Full:
                pass
            frame_count += 1

        video_stream.stop()

    def display_thread():
        while not stop_event.is_set():
            try:
                frame = q.get(timeout=0.1)
                cv2.imshow("Face Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            except queue.Empty:
                continue
        cv2.destroyAllWindows()

    threading.Thread(target=process_thread).start()
    threading.Thread(target=display_thread).start()

def process_webcam_mt_v2(known_encodings, known_names):
    print("[INFO] Starting face detection pipeline...")
    q = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    video_stream = VideoStream(0).start()
    time.sleep(1.0)

    def process_thread():
        frame_count = 0
        FRAME_SKIP = 2
        last_faces = []
        unknown_counter = 0

        while not stop_event.is_set():
            ret, frame = video_stream.read()
            if not ret or frame is None:
                print("[ERROR] Failed to read frame")
                stop_event.set()
                break

            if frame_count % FRAME_SKIP == 0:
                faces_this_frame = []
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"
                    color = (0, 0, 255)
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    if any(matches):
                        best_index = np.argmin(face_distances)
                        name = known_names[best_index]
                        color = (0, 255, 0)

                    top *= scale_y
                    right *= scale_x
                    bottom *= scale_y
                    left *= scale_x

                    face_center_x = int((left + right) // 2)
                    face_center_y = int((top + bottom) // 2)
                    point_servo_to_face(face_center_x, face_center_y, frame.shape[1], frame.shape[0], name)

                    if name == "Unknown":
                        unknown_counter += 1
                        print(f"[DEBUG] Unknown detected - counter = {unknown_counter}")
                        if unknown_counter >= STABILITY_THRESHOLD:
                            print("[ACTION] Stable unknown detected, firing trigger")
                            fire_trigger()
                            unknown_counter = 0
                    else:
                        unknown_counter = 0  # reset if known face is found

                    faces_this_frame.append((int(left), int(top), int(right), int(bottom), name, color, face_center_x, face_center_y))
                last_faces = faces_this_frame

            annotated = frame.copy()
            for (left, top, right, bottom, name, color, face_center_x, face_center_y) in last_faces:
                cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
                cv2.circle(annotated, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
                cv2.putText(annotated, name, (left, top - 10), FONT, 0.6, color, 2)
            try:
                if not q.full():
                    q.put(annotated, timeout=0.05)
            except queue.Full:
                pass
            frame_count += 1

        video_stream.stop()

    def display_thread():
        while not stop_event.is_set():
            try:
                frame = q.get(timeout=0.1)
                cv2.imshow("Face Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            except queue.Empty:
                continue
        cv2.destroyAllWindows()

    threading.Thread(target=process_thread).start()
    threading.Thread(target=display_thread).start()

# ---------- MAIN ----------
if __name__ == "__main__":
    print("[MAIN] Loading known faces...")
    known_encodings, known_names = load_known_faces()
    print("[MAIN] Launching face tracking system (Press 'q' to quit)...")
    process_webcam_mt_v2(known_encodings, known_names)
    #process_webcam_mt(known_encodings, known_names)
