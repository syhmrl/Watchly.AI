import os
import random
import cv2
import time
import threading
import queue
import tkinter as tk
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import torch

# Video testing setup
VIDEO_NAME = 'test_clash.mp4'
VIDEO_PATH = os.path.join('.', 'video', f'{VIDEO_NAME}')
VIDEO_OUT_PATH = os.path.join('.', 'video', f'predicted_{VIDEO_NAME.split(".")[0]}.mp4')

# Camera Settings
CAM_USERNAME = "admin"
CAM_PASSWORD = "Abcdefghi1"
CAM_IP = "192.168.1.64"
RTSP_URL = f"rtsp://{CAM_USERNAME}:{CAM_PASSWORD}@{CAM_IP}:554/Streaming/Channels/101?transport=tcp"

# Source of the Video/Stream
VIDEO_SOURCE = RTSP_URL

# Model used
MODEL_NAME = "yolo11s.pt"

# Setup frame size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

# Setup the line coordinate for crossing, enter and exit count
line_x = FRAME_WIDTH // 2  # Vertical line for counting
enter_count = 0
exit_count = 0

# Setup framerate variable
fps_avg_len = 200

# Connect to SQLite database
conn = sqlite3.connect('watchly_ai.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS crossing_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        track_id INTEGER,
        direction TEXT,
        timestamp TEXT           
    )
''')
conn.commit()

# Check for GPU availability and load model on GPU if available
if torch.cuda.is_available():
    model = YOLO(MODEL_NAME).to('cuda')
    print("Using GPU")
else:
    model = YOLO(MODEL_NAME)
    print("Using CPU")

# Stop event for threads
stop_event = threading.Event()

# Threaded video capture
frame_queue = queue.Queue(maxsize=1)

def capture_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Discard old frame to keep only the latest
    cap.release()

# Threaded database inserts
pending_inserts = []

def insert_to_db():
    while not stop_event.is_set():
        time.sleep(1)  # Process every second
        if pending_inserts:
            with conn:
                cursor.executemany("INSERT INTO crossing_events (source, track_id, direction, timestamp) VALUES (?, ?, ?, ?)", pending_inserts)
                conn.commit()
            pending_inserts.clear()

# Video processing function
def video_processing():
    global enter_count, exit_count
    previous_centroids = {}
    frame_rate_buffer = []
    avg_frame_rate = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
    
    while not stop_event.is_set():
        t_start = time.perf_counter()
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        frame = cv2.resize(frame, FRAME_SIZE)
        results = model.track(
            frame,
            verbose=False,
            classes=[0],  # Track people only
            conf=0.5,
            stream=True,
            stream_buffer=True,
            persist=True,
            tracker="bytetrack.yaml"
        )
        for result in results:
            for box in result.boxes:
                track_id = int(box.id.item()) if box.id is not None else None

                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) / 2

                if track_id in previous_centroids:
                    prev_cx = previous_centroids[track_id]
                    if prev_cx < line_x and cx >= line_x:
                        direction = 'enter'
                        enter_count += 1
                    elif prev_cx >= line_x and cx < line_x:
                        direction = 'exit'
                        exit_count += 1
                    else:
                        direction = None
                else:
                    direction = None
                if direction:
                    timestamp = datetime.now().isoformat()
                    pending_inserts.append((VIDEO_SOURCE, track_id, direction, timestamp))

                previous_centroids[track_id] = cx
                color = colors[track_id % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(frame, (line_x, 0), (line_x, FRAME_HEIGHT), (0, 255, 0), 2)
        cv2.putText(frame, f"Enter: {enter_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exit: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        t_stop = time.perf_counter()
        frame_rate_calc = float(1 / (t_stop - t_start))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)

        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)

        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.set()
            break

# Start threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

db_thread = threading.Thread(target=insert_to_db, daemon=True)
db_thread.start()

video_thread = threading.Thread(target=video_processing, daemon=True)
video_thread.start()

# Tkinter setup
root = tk.Tk()
root.title("People Enter Count")
label = tk.Label(root, text="Enter: 0", font=("Helvetica", 48))
label.pack(expand=True)

def update_label():
    label.config(text=f"Enter: {enter_count}")
    root.after(100, update_label)

def on_resize(event):
    new_size = max(10, event.height // 10)
    label.config(font=("Helvetica", new_size))

root.bind("<Configure>", on_resize)

def on_close():
    stop_event.set()
    conn.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

def check_stop():
    if stop_event.is_set():
        on_close()
    else:
        root.after(100, check_stop)

# Optional: Full-screen toggle
def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))

root.bind("<F>", toggle_fullscreen)
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

# Start updating label and checking stop
update_label()
check_stop()

# Start Tkinter mainloop
root.mainloop()