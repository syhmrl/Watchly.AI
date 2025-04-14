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


VIDEO_NAME = 'test_clash.mp4'
VIDEO_PATH = os.path.join('.', 'video', f'{VIDEO_NAME}')
VIDEO_OUT_PATH = os.path.join('.', 'video', f'predicted_{VIDEO_NAME.split(".")[0]}.mp4')

# Camera Settings (unused if using video_path)
CAM_USERNAME = "admin"
CAM_PASSWORD = "Abcdefghi1"
CAM_IP = "192.168.1.64"
RTSP_URL = f"rtsp://{CAM_USERNAME}:{CAM_PASSWORD}@{CAM_IP}:554/Streaming/Channels/101"

# Source of the Video/Stream
VIDEO_SOURCE = VIDEO_PATH

# Model used
MODEL_NAME = "yolo11s.pt"

# Initialize opencv VideoCapture with the source
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Setup frame size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

# Setup VideoWriter
FPS = cap.get(cv2.CAP_PROP_FPS)
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT_PATH, FOURCC, FPS, FRAME_SIZE)

# Setup the line coordinate for crossing, enter and exit count
line_x = FRAME_WIDTH // 2  # Vertical line for counting
enter_count = 0
exit_count = 0
previous_centroids = {}

# Setup framerate variable
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

# lock = threading.Lock()

# counter_window = tk.Tk()
# counter_window.title("Entry Counter")
# counter_window.geometry("200x100")

# # Create a label to display the enter count
# enter_label = tk.Label(counter_window, text="Enter: 0", font=("Arial", 16))
# enter_label.pack(pady=20)
# counter_window.update()

# start_time = time.time()
# frame_count = 0

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

# Initialize model
model = YOLO(MODEL_NAME)
# Initialize random colors generator for bounding box
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# def video_processing():
#     global enter_count, exit_count, previous_centroids, avg_frame_rate

while True:
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, FRAME_SIZE)

    # model to perform detection and tracking
    results = model.track(
        frame,
        verbose=False,
        classes=[0],  # Track people only
        conf=0.5,
        # imgsz=320,
        # stream=True,
        # stream_buffer=True,
        persist=True,  # Maintain track state between frames
        tracker="bytetrack.yaml"  # Use ByteTrack tracker
    )



    # Process tracking results
    for result in results:
        for box in result.boxes:
            # Extract track ID
            track_id = int(box.id.item()) if box.id is not None else None
            if track_id is None:
                continue  # Skip if no track ID

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2  # Current centroid

            # Check line crossing
            if track_id in previous_centroids:
                prev_cx = previous_centroids[track_id]
                if prev_cx < line_x and cx >= line_x:
                    direction = 'enter'
                    enter_count += 1
                    # enter_label.config(text=f"Enter: {enter_count}")
                    # counter_window.update()
                elif prev_cx >= line_x and cx < line_x:
                    direction = 'exit'
                    exit_count += 1
                else:
                    direction = None
            else:
                direction = None

            
            if direction:
                timestamp = datetime.now().isoformat()
                cursor.execute("INSERT INTO crossing_events (source, track_id, direction, timestamp) VALUES (?, ?, ?, ?)", (VIDEO_SOURCE, track_id, direction, timestamp))
                conn.commit()

            previous_centroids[track_id] = cx

            # # Draw bounding box and ID
            color = colors[track_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            

    # Draw counting line and counts
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
    cv2.line(frame, (line_x, 0), (line_x, FRAME_HEIGHT), (0, 255, 0), 2)
    cv2.putText(frame, f"Enter: {enter_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # frame_count += 1

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)

    out.write(frame)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break

# fps = frame_count / (time.time() - start_time)
# print(f"FPS: {fps:.2f}")

cap.release()
out.release()
cv2.destroyAllWindows()
# counter_window.destroy()
conn.close()

# def run_tkinter_gui():
#     global enter_count
#     root = tk.Tk()
#     root.title("Entry Counter")
#     root.geometry("200x100")

#     # Use StringVar to store the counter text
#     count_var = tk.StringVar()
#     count_var.set("Enter: 0")
#     label = tk.Label(root, textvariable=count_var, font=("Arial", 16))
#     label.pack(pady=20)

#     # Update the label periodically using the after() method
#     def update_label():
#         with lock:
#             current_count = enter_count
#         count_var.set(f"Enter: {current_count}")
#         root.after(500, update_label)  # Update every 500 ms

#     update_label()
#     root.mainloop()

# Create and start the video processing thread.
# video_thread = threading.Thread(target=video_processing)
# video_thread.start()

# # Run the Tkinter GUI in the main thread.
# run_tkinter_gui()

# # Wait for the video processing thread to complete before exiting.
# video_thread.join()
