import os
import random
import cv2
import time
import threading
import tkinter as tk
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import torch
from tracker import Tracker

# Video testing setup
VIDEO_NAME = 'test_clash.mp4'
VIDEO_PATH = os.path.join('.', 'video', f'{VIDEO_NAME}')
VIDEO_OUT_PATH = os.path.join('.', 'video', f'predicted_{VIDEO_NAME.split(".")[0]}.mp4')

# Source of the Video/Stream
VIDEO_SOURCE = VIDEO_PATH

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
def video_processing_v1():
    print("Testing video processing version 1")
    global enter_count, exit_count
    previous_centroids = {}
    frame_rate_buffer = []
    avg_frame_rate = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    while not stop_event.is_set():
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        results = model.track(
            frame,
            verbose=False,
            classes=[0],  # Track people only
            conf=0.5,
            persist=True,
            tracker="custom_tracker.yaml"
        )

        for result in results:
            for box in result.boxes:
                track_id = int(box.id.item()) if box.id is not None else None

                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                direction = None
                if track_id in previous_centroids:
                    prev_cx = previous_centroids[track_id]
                    if prev_cx < line_x and cx >= line_x:
                        direction = 'enter'
                        enter_count += 1
                    elif prev_cx >= line_x and cx < line_x:
                        direction = 'exit'
                        exit_count += 1

                previous_centroids[track_id] = cx

                if direction:
                    timestamp = datetime.now().isoformat()
                    pending_inserts.append((VIDEO_SOURCE, track_id, direction, timestamp))
                
                color = colors[track_id % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.putText(frame, f"{track_id}", (cx + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.line(frame, (line_x, 0), (line_x, FRAME_HEIGHT), (0, 255, 0), 2)
        cv2.putText(frame, f"Enter: {enter_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exit: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate and display FPS
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)
        cv2.putText(frame, f"FPS: {avg_frame_rate:.1f}", (10, FRAME_HEIGHT - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("People Counter", frame)
        if cv2.waitKey(0) & 0xFF == 27:
            stop_event.set()
            cap.release()
            break

# Video processing function
def video_processing_v2():
    print("Testing video processing version 2")
    global enter_count, exit_count
    track_states = {}  
    previous_centroids = {}
    last_seen = {}
    frame_idx = 0
    MAX_MISSING = 10  # Number of frames before considering a track lost
    frame_rate_buffer = []
    avg_frame_rate = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
    
    # Make OpenCV window a normal window that can be closed
    cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    while not stop_event.is_set():
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Process the frame
        frame = cv2.resize(frame, FRAME_SIZE)
        results = model.track(
            frame,
            verbose=False,
            classes=[0],  # Track people only
            conf=0.5,
            persist=True,
            tracker="custom_tracker.yaml"
        )

        seen_ids = set()

        for result in results:
            for box in result.boxes:
                track_id = int(box.id.item()) if box.id is not None else None

                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Track the ID
                seen_ids.add(track_id)
                last_seen[track_id] = frame_idx
                state = track_states.get(track_id, 0)
                prev_cx = previous_centroids.get(track_id, cx)

                direction = None
                # State machine transitions
                if state == 0 and prev_cx < line_x <= cx:
                    enter_count += 1
                    direction = 'enter'
                    track_states[track_id] = 1   # ENTERED
                elif state == 0 and prev_cx >= line_x > cx:
                    exit_count += 1
                    direction = 'exit'
                    track_states[track_id] = 2   # EXITED
                
                # Update previous position
                previous_centroids[track_id] = cx

                if direction:
                    timestamp = datetime.now().isoformat()
                    pending_inserts.append((VIDEO_SOURCE, track_id, direction, timestamp))

                # Draw bounding box and ID
                color = colors[track_id % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.putText(frame, f"{track_id}", (cx + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Clean up tracks that haven't been seen recently
        for tid in list(track_states):
            if frame_idx - last_seen.get(tid, frame_idx) > MAX_MISSING:
                if tid in track_states:
                    del track_states[tid]
                if tid in last_seen:
                    del last_seen[tid]
                if tid in previous_centroids:
                    del previous_centroids[tid]

        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, FRAME_HEIGHT), (0, 255, 0), 2)

        # Display counters
        cv2.putText(frame, f"Enter: {enter_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exit: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate and display FPS
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)
        cv2.putText(frame, f"FPS: {avg_frame_rate:.1f}", (10, FRAME_HEIGHT - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("People Counter", frame)
        key = cv2.waitKey(0) & 0xFF
        
        # Handle window close or ESC key
        if key == 27 or cv2.getWindowProperty("People Counter", cv2.WND_PROP_VISIBLE) < 1:
            stop_event.set()
            cap.release()
            break
    
    # Clean up
    cv2.destroyAllWindows()
    print("Video processing thread stopped")


db_thread = threading.Thread(target=insert_to_db, daemon=True)
db_thread.start()

video_thread = threading.Thread(target=video_processing_v2, daemon=True)
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