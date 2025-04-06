import cv2
import time
from ultralytics import YOLO
import numpy as np

# Camera Settings
username = "admin"
password = "Abcdefghi1"
camera_ip = "169.254.66.46"
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:554/Streaming/Channels/101"

MODEL_PATH = "yolo11n.pt"

def main():
    # Initialize YOLO model
    model = YOLO(MODEL_PATH)

    # Connect to camera
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Failed to open RTSP stream")
        return
    
    # Create a window that can be resized.
    cv2.namedWindow("Visitor Counting", cv2.WINDOW_NORMAL)

    # Set a specific size (width x height)
    cv2.resizeWindow("Visitor Counting", 640, 480)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Connection lost - attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue

            # Run YOLO inference
            results = model.predict(frame, conf=0.7, verbose=False)

            #Filter for person detections (class 0 in COCO dataset)
            person_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls) == 0: # Class 0 = person
                        person_detections.append(box.xyxy[0].cpu().numpy())


            # Draw bounding boxes and count
            count = len(person_detections)
            for box in person_detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2)

            # Display count
            cv2.putText(frame,
                        f"Persons: {count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

            # Show output
            cv2.imshow("Visitor Counting", frame)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


