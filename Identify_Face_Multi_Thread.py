import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import time
from threading import Thread, Lock
import threading
import queue

# ---------- CONFIG ----------
KNOWN_FACES_DIR = "known_faces"
CSV_PATH = "labels.csv"
TOLERANCE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FRAME_SKIP = 2  # Process every 3rd frame
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
print(cv2.__version__)
print(face_recognition.__version__)
print(np.__version__)

if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: Known faces directory not found at {KNOWN_FACES_DIR}")
    exit(1)
if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}")
    exit(1)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------- Thread-safe Video Capture ----------
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
        self.stream.release()  # Add this to properly release the camera

def load_known_faces():
    df = pd.read_csv(CSV_PATH)
    known_encodings = []
    known_names = []

    for _, row in df.iterrows():
        image_path = os.path.join(KNOWN_FACES_DIR, row['filename'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(row['name'])
        else:
            print(f"No face found in {row['filename']}")

    return known_encodings, known_names

# ---------- Optimized Face Recognition ----------
def recognize_faces_in_frame(frame, known_encodings, known_names):
    # Resize frame for faster processing (tune as needed)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 1. Use Haar cascade to detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    scale_x = frame.shape[1] / small_frame.shape[1]
    scale_y = frame.shape[0] / small_frame.shape[0]

    for (x, y, w, h) in faces:
        # Scale coords back to original frame
        top = int(y * scale_y)
        right = int((x + w) * scale_x)
        bottom = int((y + h) * scale_y)
        left = int(x * scale_x)

        # Crop face region (from original frame, RGB)
        face_image = frame[top:bottom, left:right]
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # 2. Get face encoding for the detected face
        encodings = face_recognition.face_encodings(face_rgb)
        name = "Unknown"
        color = (0, 0, 255)  # Red by default

        if encodings:
            face_encoding = encodings[0]
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if any(matches):
                best_index = np.argmin(face_distances)
                name = known_names[best_index]
                color = (0, 255, 0)  # Green

        # Draw rectangle and name on the original frame
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), FONT, 0.6, color, 2)

    return frame


def process_image(image_path, known_encodings, known_names):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Resize for display if too large
    h, w = image.shape[:2]
    if w > DISPLAY_WIDTH or h > DISPLAY_HEIGHT:
        scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Process image
    result = recognize_faces_in_frame(image, known_encodings, known_names)

    # Display
    cv2.namedWindow("Face Recognition - Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Recognition - Image", result)
    print("Press any key to close the image window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- Optimized Webcam Processing ----------
def process_webcam(known_encodings, known_names, target_fps=12):
    from collections import deque
    video_stream = VideoStream(0).start()
    time.sleep(1.0)  # Allow camera to warm up

    frame_count = 0
    fps_counter = 0
    fps = 0
    last_time = time.time()
    FRAME_SKIP = 2   # initial skip, adaptive
    min_skip, max_skip = 1, 8

    last_faces = []  # To persist face box/name between frames

    while True:
        ret, frame = video_stream.read()
        if not ret or frame is None:
            break

        process_this_frame = (frame_count % FRAME_SKIP == 0)
        if process_this_frame:
            # Do full recognition
            faces_this_frame = []

            # ---- Use Haar cascade for detection as in previous reply ----
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]

            for (x, y, w, h) in faces:
                # Scale coords back to original frame
                top = int(y * scale_y)
                right = int((x + w) * scale_x)
                bottom = int((y + h) * scale_y)
                left = int(x * scale_x)

                face_image = frame[top:bottom, left:right]
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

                faces_this_frame.append((left, top, right, bottom, name, color))

            last_faces = faces_this_frame  # update for skipped frames

        # Draw last known boxes & names for this frame
        for (left, top, right, bottom, name, color) in last_faces:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), FONT, 0.6, color, 2)

        # FPS calculation and adaptive skip logic
        fps_counter += 1
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = fps_counter
            # ADAPT SKIP: If FPS too low, skip more. If FPS high, skip less.
            if fps < target_fps and FRAME_SKIP < max_skip:
                FRAME_SKIP += 1
            elif fps > target_fps and FRAME_SKIP > min_skip:
                FRAME_SKIP -= 1
            fps_counter = 0
            last_time = now

        cv2.putText(frame, f"FPS: {fps} (Skip:{FRAME_SKIP})", (10, 30), FONT, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Recognition - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()

# ---------- Optimized Video File Processing ----------
def process_video(video_path, known_encodings, known_names, target_fps=12):
    import time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    frame_count = 0
    fps_counter = 0
    fps = 0
    last_time = time.time()
    FRAME_SKIP = 2   # Initial skip, will be adapted
    min_skip, max_skip = 1, 8

    last_faces = []  # Persist face boxes/names for skipped frames

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        process_this_frame = (frame_count % FRAME_SKIP == 0)
        if process_this_frame:
            # ---- Use Haar cascade for detection ----
            faces_this_frame = []
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]

            for (x, y, w, h) in faces:
                # Scale coordinates to original frame
                top = int(y * scale_y)
                right = int((x + w) * scale_x)
                bottom = int((y + h) * scale_y)
                left = int(x * scale_x)

                face_image = frame[top:bottom, left:right]
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(face_rgb)
                name = "Unknown"
                color = (0, 0, 255)  # Red

                if encodings:
                    face_encoding = encodings[0]
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    if any(matches):
                        best_index = np.argmin(face_distances)
                        name = known_names[best_index]
                        color = (0, 255, 0)  # Green

                faces_this_frame.append((left, top, right, bottom, name, color))

            last_faces = faces_this_frame  # Save for skipped frames

        # Draw last detected boxes & names
        for (left, top, right, bottom, name, color) in last_faces:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), FONT, 0.6, color, 2)

        # FPS calculation and adaptive skip logic
        fps_counter += 1
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = fps_counter
            # Adapt skip: If FPS too low, skip more. If high, skip less.
            if fps < target_fps and FRAME_SKIP < max_skip:
                FRAME_SKIP += 1
            elif fps > target_fps and FRAME_SKIP > min_skip:
                FRAME_SKIP -= 1
            fps_counter = 0
            last_time = now

        cv2.putText(frame, f"FPS: {fps} (Skip:{FRAME_SKIP})", (10, 30), FONT, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Recognition - Video", frame)

        # Adjust wait time to match video speed
        elapsed = time.time() - start_time
        remaining = max(1, int((1. / original_fps - elapsed) * 1000))
        if cv2.waitKey(remaining) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def process_webcam_mt(known_encodings, known_names, target_fps=12):
    q = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    video_stream = VideoStream(0).start()
    time.sleep(1.0)  # Allow camera to warm up

    def process_thread():
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

def process_video_mt(video_path, known_encodings, known_names, target_fps=12):
    q = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    def process_thread():
        frame_count = 0
        fps_counter = 0
        fps = 0
        last_time = time.time()
        FRAME_SKIP = 2
        min_skip, max_skip = 1, 8
        last_faces = []

        while not stop_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()
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

            elapsed = time.time() - start_time
            remaining = max(1, int((1. / original_fps - elapsed) * 1000))
            cv2.waitKey(remaining)

        cap.release()

    def display_thread():
        while not stop_event.is_set():
            try:
                frame = q.get(timeout=0.1)
                cv2.imshow("Face Recognition - Video", frame)
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


# ---------- STEP 4: MODE SELECTION ----------
if __name__ == "__main__":
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces()
    print(f"Loaded {len(known_names)} known faces")

    while True:
        print("\nSelect input source:")
        print("1 - Image")
        print("2 - Video File")
        print("3 - Webcam")
        print("4 - Exit")
        choice = input("Enter choice (1/2/3/4): ").strip()

        if choice == "1":
            image_path = input("Enter image path: ").strip('"')
            if not os.path.exists(image_path):
                print(f"Error: File not found at {image_path}")
                continue
            process_image(image_path, known_encodings, known_names)

        elif choice == "2":
            video_path = input("Enter video file path: ").strip('"')
            if not os.path.exists(video_path):
                print(f"Error: File not found at {video_path}")
                continue
            process_video_mt(video_path, known_encodings, known_names)

        elif choice == "3":
            print("Starting webcam... Press 'q' to quit")
            process_webcam_mt(known_encodings, known_names)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

        # Clear any remaining OpenCV windows
        cv2.destroyAllWindows()