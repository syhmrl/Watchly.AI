import os
import random
import cv2
import time
import queue
import numpy as np
import config

from thread_manager import thread_controller
from datetime import datetime

CAMERA_SOURCES = config.get_camera_sources()
FRAME_WIDTH, FRAME_HEIGHT = config.get_frame_size()
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

# Setup the line coordinate for crossing, enter and exit count
# line_x = FRAME_WIDTH // 2  # Vertical line for counting
line_positions = [FRAME_WIDTH // 2, FRAME_WIDTH // 2]  # Line position for each camera
enter_count      = [0 for _ in CAMERA_SOURCES]
exit_count       = [0 for _ in CAMERA_SOURCES]
crowd_count      = [0 for _ in CAMERA_SOURCES]
total_enter_count = 0
total_exit_count = 0
total_crowd_count = 0  # New total for crowd counting

# Global variables for ROI drawing
drawing = False
roi_points = []
temp_roi = []
roi_set = False
current_mouse_pos = (0, 0)  # Store current mouse position

random_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

# Region of Interest settings
# Define ROI polygons for each camera - customize these coordinates for your setup
ROI_POINTS = []

# Frame capture function
def capture_frames(source_index):
    source = CAMERA_SOURCES[source_index]
    source_name = f"Camera {source_index + 1}"

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Failed to open video source for {source_name}: {source}")
        return
    else:
        print(f"Successfully opened video source for {source_name}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
    
    while not thread_controller.stop_event.is_set():
        ret, frame = cap.read()

        # frame = cv2.flip(frame, 1)

        if not ret:
            print(f"Failed to get frame from {source_name}")
            time.sleep(1)  # Wait before retrying
            # Try to reconnect
            cap.release()
            cap = cv2.VideoCapture(source)
            continue
            
        
        try:
            # Put frame in queue, replace if full
            if thread_controller.frame_queue[source_index].full():
                try:
                    thread_controller.frame_queue[source_index].get_nowait()
                except queue.Empty:
                    pass
            thread_controller.frame_queue[source_index].put(frame, block=False)
        except queue.Full:
            pass  # Skip frame if queue is full
    
    # Clean up resources
    cap.release()
    print(f"Frame capture thread for {source_name} stopped")

# Mouse callback function for drawing ROI
def draw_roi(event, x, y, flags, param):
    global drawing, roi_points, temp_roi, roi_set, current_mouse_pos

    # Update current mouse position regardless of event
    current_mouse_pos = (x, y)
    
    source_index = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_set:
            drawing = True
            roi_points.append((x, y))
            temp_roi = roi_points.copy()
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and len(roi_points) > 0:  # Only update if we have points
            temp_roi = roi_points.copy()
            temp_roi.append((x, y))
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to finish drawing if we have at least 3 points
        if len(roi_points) >= 3:
            roi_set = True
            # Convert to numpy array and update the ROI_POINTS
            points_array = np.array(roi_points, np.int32).reshape((-1, 1, 2))
            if len(ROI_POINTS) > source_index:
                ROI_POINTS[source_index] = points_array
            
            # Print the coordinates for future use
            print(f"\nROI coordinates for Camera {source_index + 1}:")
            print(f"np.array({roi_points}, np.int32),")
            
            # Keep the points for display but mark as completed
            drawing = False
            print(f"ROI drawing completed with {len(roi_points)} points")
    elif event == cv2.EVENT_LBUTTONUP:
        # We just continue drawing, no need to finalize ROI here
        pass

# Function to reset ROI for a specific camera
def reset_roi(source_index):
    global roi_points, temp_roi, roi_set
    roi_points = []
    temp_roi = []
    roi_set = False
    print(f"ROI for Camera {source_index + 1} has been reset. Please draw a new ROI.")

# Add this function to check if a detection is in your ROI
def is_in_roi(box, roi_points):
    """Check if the detection is inside the region of interest"""
    # Get center point of the bottom of the bounding box (person's feet)
    x1, y1, x2, y2 = map(int, box)
    foot_point = (int((x1 + x2) / 2), y2)
    
    # Check if point is inside polygon
    return cv2.pointPolygonTest(roi_points, foot_point, False) >= 0

def draw_roi_overlay(frame, temp_roi, drawing=False, current_mouse_pos=None):
    """Draw ROI points and instructions on the frame."""
    # Draw the current ROI points
    if len(temp_roi) > 0:
        points = np.array(temp_roi, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], len(temp_roi) > 2, (0, 255, 0), 2)
        
        # Draw the points with numbers
        for i, point in enumerate(temp_roi):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (point[0]+5, point[1]+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Drawing instructions
    instruction_text = [
        "Draw ROI: Left-click to add points",
        "Press ENTER to finish (min 3 points)",
        f"Points: {len(temp_roi)}/3+"
    ]
    for i, text in enumerate(instruction_text):
        cv2.putText(frame, text, (10, 30 + 30*i), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show active cursor position if drawing
    if drawing and current_mouse_pos is not None:
        cv2.circle(frame, current_mouse_pos, 3, (0, 255, 255), -1)
    
    return frame

def model_frame(model, frame):

    results = model.track(
        frame,
        verbose=False,
        classes=[0],  # Track people only
        conf=config.get_model_conf(),
        iou=config.get_model_iou(),
        stream=True,
        stream_buffer=True,
        persist=True,
        tracker="custom_tracker.yaml"
    )

    return results

def model_video(model, frame):

    results = model.track(
        frame,
        verbose=False,
        classes=[0],  # Track people only
        conf=config.get_model_conf(),
        iou=config.get_model_iou(),
        persist=True,
        tracker="custom_tracker.yaml"
    )

    return results

def init_record(source_name, recording_fps = 15):
    os.makedirs("video/processed", exist_ok=True)
    filename = f"video/processed/processed_{source_name}_crowd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, recording_fps, (FRAME_WIDTH, FRAME_HEIGHT))
    print("Recording started:", source_name)

    return out

def init_video_record(source_name, recording_fps = 15, width = FRAME_WIDTH, height = FRAME_HEIGHT):
    os.makedirs("video/testing", exist_ok=True)
    filename = f"video/testing/test_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, recording_fps, (width, height))
    print("Recording started:", source_name)

    return out

def display_fps(fps, frame):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def display_crowd_count(crowd_count, frame):
    cv2.putText(frame, f"Crowd Count: {crowd_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

def display_inframe_count(inframe_count, frame):
    cv2.putText(frame, f"In-frame Count: {inframe_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

def count_to_db(source_name, tid, direction, mode):
    # Record in database
    timestamp = datetime.now().isoformat()
    thread_controller.pending_inserts.put((source_name, tid, direction, timestamp, mode))
    