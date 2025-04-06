import cv2
import os
import time
import random
import imutils
from threading import Thread
from ultralytics import YOLO
from tracker import Tracker

class VideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not self.stream.isOpened():
            raise ValueError("Unable to open video source")
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False
        self.thread = None

    def start(self):
        # start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
    
    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stopped = True  # Stop on read failure
                break
            time.sleep(.01)
        # Release stream when the thread exits
        self.stream.release()

    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        # Wait for the thread to finish
        if self.thread is not None:
            self.thread.join()

# Camera Settings
username = "admin"
password = "Abcdefghi1"
camera_ip = "169.254.66.46"
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/Streaming/Channels/101"

# Tracking Settings
TRACKER_CONFIG = {
    "max_cosine_distance": 0.3,
    "nn_budget": 50,
    "max_age": 30,
    "n_init": 3
}

# Random colors generator
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

vs = VideoStream(src=rtsp_url).start()

video_out_path = os.path.join('.', 'video', f'predicted_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20  # Adjust based on your needs
writer = None

model = YOLO('yolo11s.pt')
tracker = Tracker(**TRACKER_CONFIG)

# loop over some frames...this time using the threaded stream
while True:
	# grab the frame from the threaded video stream and resize it
    frame = vs.read()

    # Check if the frame is None (end of video or read error)
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)

    # Initialize video writer if not initialized
    # if writer is None:
    #     height, width = frame.shape[:2]
    #     writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

	
    results = model.predict(
            frame,
            verbose=False,
            classes=[0],
            conf=0.5,
            imgsz=320,
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
        track_color = colors[track_id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)
        
    # writer.write(frame)
	
    cv2.imshow("Frame", frame)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
# if writer is not None:
#     writer.release()
vs.stop()
