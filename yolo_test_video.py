import os
import random
import cv2
import time
from ultralytics import YOLO
import sqlite3
from datetime import datetime

video_name = 'test_people.mp4'
video_path = os.path.join('.', 'video', f'{video_name}')
video_out_path = os.path.join('.', 'video', f'predicted_{video_name.split(".")[0]}.mp4')

# Camera Settings (unused if using video_path)
username = "admin"
password = "Abcdefghi1"
camera_ip = "169.254.66.46"
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/Streaming/Channels/101"

source = 0

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_width = 1280
frame_height = 720
frame_size = (frame_width, frame_height)

line_x = frame_width // 2  # Vertical line for counting
enter_count = 0
exit_count = 0
previous_centroids = {}

# start_time = time.time()
# frame_count = 0

# Connect to SQLite database
conn = sqlite3.connect('watchly_ai.db')
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

model = YOLO("yolo11s.pt")
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, frame_size)

    # Use model.track() for tracking
    results = model.track(
        frame,
        verbose=False,
        classes=[0],  # Track people only
        conf=0.5,
        # imgsz=320,
        stream=True,
        stream_buffer=True,
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
                elif prev_cx >= line_x and cx < line_x:
                    direction = 'exit'
                    exit_count += 1
                else:
                    direction = None
            else:
                direction = None

            
            if direction:
                timestamp = datetime.now().isoformat()
                cursor.execute("INSERT INTO crossing_events (source, track_id, direction, timestamp) VALUES (?, ?, ?, ?)", (source, track_id, direction, timestamp))
                conn.commit()

            previous_centroids[track_id] = cx

            # # Draw bounding box and ID
            color = colors[track_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            

    # Draw counting line and counts
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 255, 0), 2)
    cv2.putText(frame, f"Enter: {enter_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # frame_count += 1

    

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# fps = frame_count / (time.time() - start_time)
# print(f"FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()
conn.close()