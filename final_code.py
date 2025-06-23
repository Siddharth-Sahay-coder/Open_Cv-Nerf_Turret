import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import time
import threading
import queue
from threading import Lock, Thread
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# ---------- CONFIG ----------
KNOWN_FACES_DIR = "known_faces"
CSV_PATH = "labels.csv"
TOLERANCE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FRAME_SKIP = 2
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: Known faces directory not found at {KNOWN_FACES_DIR}")
    exit(1)
if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}")
    exit(1)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------- Servo Setup ----------
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50

FIRE_CHANNEL = 2
TILT_CHANNEL = 1
PAN_CHANNEL = 3

last_shot_time = 0
FIRE_COOLDOWN = 2.0  # seconds

def set_servo_angle(channel, angle):
    angle = max(0, min(180, angle))  # Clamp
    min_pulse = 0.5 / 20 * 65535
    max_pulse = 2.5 / 20 * 65535
    pulse = int(min_pulse + (angle / 180.0) * (max_pulse - min_pulse))
    pca.channels[channel].duty_cycle = pulse

def point_servo_to_face(face_center_x, face_center_y, frame_width, frame_height):
    norm_x = face_center_x / frame_width
    norm_y = face_center_y / frame_height
    pan_angle = 180 * norm_x
    tilt_angle = 180 * norm_y
    set_servo_angle(PAN_CHANNEL, pan_angle)
    set_servo_angle(TILT_CHANNEL, tilt_angle)

def fire_nerf_gun():
    print("FIRING!")
    set_servo_angle(FIRE_CHANNEL, 90)
    time.sleep(0.4)
    set_servo_angle(FIRE_CHANNEL, 0)

def is_target_centered(face_center_x, face_center_y, frame_width, frame_height):
    cx = frame_width // 2
    cy = frame_height // 2
    return abs(face_center_x - cx) < 40 and abs(face_center_y - cy) < 40

# ---------- Video Stream Class ----------
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=()).start()
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

# ---------- Load Known Faces ----------
def load_known_faces():
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
        else:
            print(f"No face found in {row['filename']}")
    return known_encodings, known_names

# ---------- Main Face Recognition Thread ----------
def process_webcam_mt(known_encodings, known_names, target_fps=12):
    q = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    video_stream = VideoStream(0).start()
    time.sleep(1.0)

    def process_thread():
        nonlocal last_shot_time
        frame_count = 0
        fps_counter = 0
        fps = 0
        last_time = time.time()
        FRAME_SKIP = 2
        min_skip, max_skip = 1, 8
        last_faces = []

        while not stop_event.is_set():
            ret, frame = video_stream.read()
            if not ret or frame is None:
                stop_event.set()
                break

            process_this_frame = (frame_count % FRAME_SKIP == 0)
            if process_this_frame:
                faces_this_frame = []
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]

                for (x, y, w, h) in faces:
                    top = int(y * scale_y)
                    right = int((x + w) * scale_x)
                    bottom = int((y + h) * scale_y)
                    left = int(x * scale_x)

                    face_image = frame[top:bottom, left:right]
                    if face_image.shape[0] == 0 or face_image.shape[1] == 0:
                        continue
                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(face_rgb)
                    name = "Unknown"
                    color = (0, 0, 255)
                    if encodings:
                        face_encoding = encodings[0]
                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        if any(matches):
                            best_index = np.argmin(face_distances)
                            name = known_names[best_index]
                            color = (0, 255, 0)

                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2
                    point_servo_to_face(face_center_x, face_center_y, frame.shape[1], frame.shape[0])

                    if name == "Unknown" and is_target_centered(face_center_x, face_center_y, frame.shape[1], frame.shape[0]):
                        if time.time() - last_shot_time > FIRE_COOLDOWN:
                            fire_nerf_gun()
                            last_shot_time = time.time()

                    faces_this_frame.append((left, top, right, bottom, name, color))
                last_faces = faces_this_frame

            annotated_frame = frame.copy()
            for (left, top, right, bottom, name, color) in last_faces:
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(annotated_frame, name, (left, top - 10), FONT, 0.6, color, 2)

            fps_counter += 1
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = fps_counter
                if fps < target_fps and FRAME_SKIP < max_skip:
                    FRAME_SKIP += 1
                elif fps > target_fps and FRAME_SKIP > min_skip:
                    FRAME_SKIP -= 1
                fps_counter = 0
                last_time = now

            cv2.putText(annotated_frame, f"FPS: {fps} (Skip:{FRAME_SKIP})", (10, 30), FONT, 0.7, (0, 255, 0), 2)
            try:
                if not q.full():
                    q.put(annotated_frame, timeout=0.05)
            except queue.Full:
                pass

        video_stream.stop()

    def display_thread():
        while not stop_event.is_set():
            try:
                frame = q.get(timeout=0.1)
                cv2.imshow("Face Recognition - Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            except queue.Empty:
                continue
        cv2.destroyAllWindows()

    t1 = threading.Thread(target=process_thread)
    t2 = threading.Thread(target=display_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# ---------- Main ----------
if __name__ == "__main__":
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces()
    print(f"Loaded {len(known_names)} known faces")
    process_webcam_mt(known_encodings, known_names)
    cv2.destroyAllWindows()

