import os
import random
import cv2
from ultralytics import YOLO

from tracker import Tracker
# from stream import VideoStream

video_name = 'test_room.mp4'
video_path = os.path.join('.', 'video', f'{video_name}')
video_out_path = os.path.join('.', 'video', f'predicted_{video_name.split(".")[0]}.mp4')

# Camera Settings
username = "admin"
password = "Abcdefghi1"
camera_ip = "169.254.66.46"
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/Streaming/Channels/101"


cap = cv2.VideoCapture(video_path)

# Check if the capture opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_width = 1280
frame_height = 720

frame_size = (frame_width, frame_height)
# vs.start_recording(video_out_path, frame_size)

# Initialize counting variables
line_x = frame_width // 2  # Vertical line at the door (adjust as needed)
enter_count = 0
exit_count = 0
previous_centroids = {}


# Get mouse coordinate
# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y] # x,y current coordinates
#         print(colorsBGR)
    
# cv2.namedWindow('People Counter')
# cv2.setMouseCallback('People Counter', RGB)


model = YOLO("yolo11n.pt")
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

while True:
    
    ret, frame = cap.read()

    if not ret:
        break
    
    frame = cv2.resize(frame, frame_size)

    results = model.predict(
            frame,
            # verbose=False,
            classes=[0],
            conf=0.5,
            imgsz=320,
            stream=True,
            # tracker="bytetrack.yaml"
        )

    # Post-processing
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            detections.append([x1, y1, x2, y2, conf])

    # Tracking
    tracker.update(frame, detections)

    for track in tracker.tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        track_id = track.track_id

        cx = (x1 + x2) / 2  # Current x-coordinate of centroid

        # Check for line crossing
        if track_id in previous_centroids:
            prev_cx = previous_centroids[track_id]
            if prev_cx < line_x and cx >= line_x:  # Left to right (entering)
                enter_count += 1
            elif prev_cx >= line_x and cx < line_x:  # Right to left (exiting)
                exit_count += 1

        previous_centroids[track_id] = cx


        track_color = colors[track_id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 1)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)


    # Draw the counting line and display counts
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 255, 0), 2)
    cv2.putText(frame, f"Enter: {enter_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # vs.write_frame(frame)
    cv2.imshow("People Counter", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break


# vs.stop_recording()
cap.release()
# print("Your predicted video is successfully saved in the video directory.")
cv2.destroyAllWindows()
