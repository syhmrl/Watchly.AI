import os
import random
import cv2
import time
import threading
import queue
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, date
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry
from tkinter import messagebox
from database_utils import Database, insert_to_db
from helpers import (
    load_model, calculate_fps, cleanup_stale
)

# Camera Settings
CAM_USERNAME = "admin"
CAM_PASSWORD = "Abcdefghi1"
CAM_IP = ["192.168.1.64", "192.168.1.65"]
RTSP_URL = [
    f"rtsp://{CAM_USERNAME}:{CAM_PASSWORD}@{CAM_IP[0]}:554/Streaming/Channels/101",
    f"rtsp://{CAM_USERNAME}:{CAM_PASSWORD}@{CAM_IP[1]}:554/Streaming/Channels/101"
]

# Source of the Video/Stream
# VIDEO_SOURCE = 0
CAMERA_SOURCES = [
    0,#RTSP_URL[0],#
    None
]

# Model used
MODEL_NAME = "headv1.pt"

# Setup frame size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

# Setup the line coordinate for crossing, enter and exit count
# line_x = FRAME_WIDTH // 2  # Vertical line for counting
line_positions = [FRAME_WIDTH // 2, FRAME_WIDTH // 2]  # Line position for each camera
enter_count = [0,0]
exit_count = [0,0]
crowd_count = [0, 0]  # New counter for crowd counting mode
total_enter_count = 0
total_exit_count = 0
total_crowd_count = 0  # New total for crowd counting

# Setup framerate variable
fps_avg_len = 200

# Count mode
COUNT_MODE = "LINE"  # "line" or "crowd"

# Recording settings
ENABLE_RAW_RECORDING = False
ENABLE_PREDICTED_RECORDING = False

# Add these constants at the beginning of your file where other constants are defined
INITIAL_ZOOM = 1.0  # No zoom by default
MAX_ZOOM = 5.0      # Maximum zoom level
ZOOM_STEP = 0.1     # How much to change zoom per key press

# Global variables for ROI drawing
drawing = False
roi_points = []
temp_roi = []
roi_set = False
current_mouse_pos = (0, 0)  # Store current mouse position

# Globals for line drawing
line_points   = []   # Will hold lists of two endpoints per line
temp_line_pts = []   # Current line’s temporary endpoints (updates as mouse moves)
line_roles    = ["ENTER", "COUNT", "EXIT"]
line_index    = 0    # Which of the 3 lines we’re currently drawing (0, 1, or 2)
orientation   = None # 'H' for horizontal or 'V' for vertical
lines_final   = [None, None, None]  # To store [(x1,y1),(x2,y2)] for each of the 3 lines
drawing_line  = False
editing_mode  = False
selected_line = None
line_colors   = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]  # Green, Orange, Red


# Region of Interest settings
ENABLE_ROI = True  # Set to True to only count people in a specific region
ENABLE_ROI_DRAWING = False  # Set to True to enable drawing ROI at start
# Define ROI polygons for each camera - customize these coordinates for your setup
ROI_POINTS = [
    np.array([(3, 70), (1279, 91), (1277, 715), (8, 716)], np.int32),
    np.array([[], [], []], np.int32),
]

for i in range(len(ROI_POINTS)):
    ROI_POINTS[i] = ROI_POINTS[i].reshape((-1, 1, 2))

# Add this class to handle zoom parameters
class ZoomController:
    def __init__(self):
        self.zoom_factor = INITIAL_ZOOM  # Current zoom level
        self.zoom_center_x = 0.5         # Center point of zoom (normalized 0-1)
        self.zoom_center_y = 0.5         # Center point of zoom (normalized 0-1)
        self.pan_step = 0.02             # How much to pan per key press

    def increase_zoom(self):
        self.zoom_factor = min(self.zoom_factor + ZOOM_STEP, MAX_ZOOM)
        return self.zoom_factor
        
    def decrease_zoom(self):
        self.zoom_factor = max(self.zoom_factor - ZOOM_STEP, 1.0)
        return self.zoom_factor
    
    def apply_zoom(self, frame):
        """Apply digital zoom to the frame"""
        if self.zoom_factor <= 1.0:
            return frame  # No zoom needed
            
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate the region of interest based on zoom factor and center point
        # The higher the zoom, the smaller the ROI
        roi_size = 1.0 / self.zoom_factor
        
        # Calculate the top-left corner of the ROI
        x1 = int(w * (self.zoom_center_x - roi_size/2))
        y1 = int(h * (self.zoom_center_y - roi_size/2))
        
        # Calculate the bottom-right corner of the ROI
        x2 = int(w * (self.zoom_center_x + roi_size/2))
        y2 = int(h * (self.zoom_center_y + roi_size/2))
        
        # Ensure ROI is within the frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resize the ROI to the original frame size
        if roi.size > 0:  # Check if ROI is not empty
            return cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            return frame  # Return original frame if ROI is invalid

# Thread control
class ThreadControl:
    def __init__(self):
        self.stop_event = threading.Event()
        self.threads = []
        self.frame_queue = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
        self.pending_inserts = []
        self.video_window_open = set()
        self.reset()
        # Add zoom controllers for each camera
        self.zoom_controllers = [ZoomController() for _ in range(len(CAMERA_SOURCES))]
        self.enable_visual = True
        self.enable_processed_frame_recording = False
        self.enable_roi_drawing_mode = ENABLE_ROI_DRAWING
    
    def reset(self):
        """Reset all thread states and counters"""
        self.stop_event.clear()
        self.frame_queue = [queue.Queue(maxsize=1) for _ in range(len(CAMERA_SOURCES))]
        self.pending_inserts = []
        self.threads = []
        self.video_window_open = set()

# Global thread controller
thread_controller = ThreadControl()

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
        if drawing:
            temp_roi = roi_points.copy()
            temp_roi.append((x, y))
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to finish drawing if we have at least 3 points
        if len(roi_points) >= 3:
            roi_set = True
            # Convert to numpy array and update the ROI_POINTS
            points_array = np.array(roi_points, np.int32).reshape((-1, 1, 2))
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
        "Right-click to finish (min 3 points)",
        f"Points: {len(temp_roi)}/3+"
    ]
    for i, text in enumerate(instruction_text):
        cv2.putText(frame, text, (10, 30 + 30*i), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show active cursor position if drawing
    if drawing and current_mouse_pos is not None:
        cv2.circle(frame, current_mouse_pos, 3, (0, 255, 255), -1)
    
    return frame

def get_clicked_line(x, y, threshold=10):
    """
    Determine which line (if any) was clicked based on mouse position.
    Returns line index (0, 1, 2) or None if no line was clicked.
    """
    for i, line in enumerate(lines_final):
        if line is None:
            continue
            
        (x1, y1), (x2, y2) = line
        
        # Calculate distance from point to line
        if orientation == 'H':
            # For horizontal lines, check if y is close to line's y and x is within line's x range
            if abs(y - y1) <= threshold and min(x1, x2) <= x <= max(x1, x2):
                return i
        else:  # orientation == 'V'
            # For vertical lines, check if x is close to line's x and y is within line's y range
            if abs(x - x1) <= threshold and min(y1, y2) <= y <= max(y1, y2):
                return i
    
    return None

def draw_line_mode(event, x, y, flags, param):
    """
    Mouse callback for drawing one of the three lines.
    User clicks two points to define a line (either horizontal or vertical).
    """
    global drawing_line, temp_line_pts, line_index, lines_final, current_mouse_pos, editing_mode, selected_line

    current_mouse_pos = (x, y)

    if editing_mode:
        # In editing mode, handle line selection and modification
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if user clicked on an existing line to select it
            selected_line = get_clicked_line(x, y)
            if selected_line is not None:
                print(f"Selected {line_roles[selected_line]} line for editing")
                # Start drawing the selected line
                drawing_line = True
                temp_line_pts = [(x, y)]
            else:
                selected_line = None
                
        elif event == cv2.EVENT_MOUSEMOVE and drawing_line and selected_line is not None:
            # While dragging, update second endpoint temporarily
            temp_line_pts = [temp_line_pts[0], (x, y)]
            
        elif event == cv2.EVENT_LBUTTONUP and drawing_line and selected_line is not None:
            # On release, finalize the second endpoint
            temp_line_pts = [temp_line_pts[0], (x, y)]
            drawing_line = False

            # Apply orientation constraint
            (x1, y1), (x2, y2) = temp_line_pts
            if orientation == 'H' and y1 != y2:
                temp_line_pts[1] = (x2, y1)
            if orientation == 'V' and x1 != x2:
                temp_line_pts[1] = (x1, y2)

            # Update the selected line
            lines_final[selected_line] = temp_line_pts.copy()
            print(f"{line_roles[selected_line]} line updated to: {lines_final[selected_line]}")
            
            # Validate the new configuration
            if not validate_three_lines():
                print(f"Warning: Line configuration is invalid! {line_roles[selected_line]} line violates ordering rules.")
            
            selected_line = None
    else:
        # Original drawing mode logic
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_line = True
            temp_line_pts = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing_line:
            temp_line_pts = [temp_line_pts[0], (x, y)]
        elif event == cv2.EVENT_LBUTTONUP and drawing_line:
            temp_line_pts = [temp_line_pts[0], (x, y)]
            drawing_line = False

            # Apply orientation constraint
            (x1, y1), (x2, y2) = temp_line_pts
            if orientation == 'H' and y1 != y2:
                temp_line_pts[1] = (x2, y1)
            if orientation == 'V' and x1 != x2:
                temp_line_pts[1] = (x1, y2)

            # Store into lines_final for this role
            lines_final[line_index] = temp_line_pts.copy()
            line_index_plus = line_index

            print(f"{line_roles[line_index_plus]} line defined at: {lines_final[line_index_plus]}")
            line_index += 1

def validate_three_lines():
    """
    Enhanced validation that ensures proper ordering and provides detailed feedback.
    """
    if any(l is None for l in lines_final):
        return False

    # Extract coordinates for comparison
    coords = []
    for (x1, y1), (x2, y2) in lines_final:
        if orientation == 'H':
            coords.append(y1)
        else:
            coords.append(x1)

    enter_coord, count_coord, exit_coord = coords
    
    # Check ordering based on orientation
    if orientation == 'H':
        # For horizontal lines, enter should be above count, count above exit (smaller y values)
        return enter_coord < count_coord < exit_coord
    else:  # orientation == 'V'
        # For vertical lines, enter should be left of count, count left of exit (smaller x values)
        return enter_coord < count_coord < exit_coord

def get_validation_message():
    """
    Provide detailed validation feedback to user.
    """
    if any(l is None for l in lines_final):
        return "Not all lines are defined"
    
    coords = []
    for (x1, y1), (x2, y2) in lines_final:
        if orientation == 'H':
            coords.append(y1)
        else:
            coords.append(x1)
    
    enter_coord, count_coord, exit_coord = coords
    
    if orientation == 'H':
        if not (enter_coord < count_coord < exit_coord):
            return f"Invalid: ENTER({enter_coord}) must be above COUNT({count_coord}) must be above EXIT({exit_coord})"
    else:
        if not (enter_coord < count_coord < exit_coord):
            return f"Invalid: ENTER({enter_coord}) must be left of COUNT({count_coord}) must be left of EXIT({exit_coord})"
    
    return "Valid configuration"

def setup_predefined_lines(source_index, frame_width, frame_height):
    """
    Set up predefined lines that user can edit when system launches.
    """
    global lines_final, orientation
    
    if orientation == 'H':
        # Predefined horizontal lines (evenly spaced vertically)
        enter_y = frame_height // 4
        count_y = frame_height // 2
        exit_y = (frame_height * 3) // 4
        
        lines_final[0] = [(50, enter_y), (frame_width - 50, enter_y)]  # ENTER
        lines_final[1] = [(50, count_y), (frame_width - 50, count_y)]  # COUNT
        lines_final[2] = [(50, exit_y), (frame_width - 50, exit_y)]   # EXIT
    else:  # orientation == 'V'
        # Predefined vertical lines (evenly spaced horizontally)
        enter_x = frame_width // 4
        count_x = frame_width // 2
        exit_x = (frame_width * 3) // 4
        
        lines_final[0] = [(enter_x, 50), (enter_x, frame_height - 50)]  # ENTER
        lines_final[1] = [(count_x, 50), (count_x, frame_height - 50)]  # COUNT
        lines_final[2] = [(exit_x, 50), (exit_x, frame_height - 50)]   # EXIT
    
    print("Predefined lines set up:")
    for i, role in enumerate(line_roles):
        print(f"  {role}: {lines_final[i]}")

def draw_lines_with_labels(frame):
    """
    Draw all lines with proper colors and labels.
    """
    for i, line in enumerate(lines_final):
        if line is None:
            continue
            
        color = line_colors[i]
        (x1, y1), (x2, y2) = line
        
        # Draw the line
        thickness = 3 if (editing_mode and selected_line == i) else 2
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label_pos = (x1 + 10, y1 - 10) if y1 > 20 else (x1 + 10, y1 + 20)
        cv2.putText(frame, line_roles[i], label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def video_processing_line(source_index, on_close=None):  
    global enter_count, exit_count, total_enter_count, total_exit_count, orientation, line_index, lines_final, drawing_line, line_points, temp_line_pts, line_roles, editing_mode, selected_line

    source_name = f"Camera {source_index + 1}"
    window_name = f"People Counter - {source_name} - LINE MODE"

    model = load_model(MODEL_NAME)

    # Dictionary to track people who have been counted
    tracked_ids = {}
    previous_centroids = {}
    last_seen = {}
    frame_idx = 0
    MAX_MISSING = 400  # Number of frames before considering a track lost
    frame_rate_buffer = []
    avg_frame_rate = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

    # Get zoom controller for this source
    zoom_controller = thread_controller.zoom_controllers[source_index]

    processed_out = None

    # Make OpenCV window a normal window that can be closed
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 1) Ask user for orientation
    orientation = None
    print("Press 'H' for Horizontal lines, 'V' for Vertical lines.")
    
    # Create window early for key input
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    empty_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(empty_frame, "Press 'H' for Horizontal, 'V' for Vertical", 
                (50, FRAME_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, empty_frame)
    
    while orientation not in ('H', 'V'):
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('H'), ord('h')):
            orientation = 'H'
        elif key in (ord('V'), ord('v')):
            orientation = 'V'
        elif key == 27:  # ESC to exit
            cv2.destroyWindow(window_name)
            return
    
    print(f"Orientation set to {'Horizontal' if orientation=='H' else 'Vertical'}.")
    
    # 2) Set up predefined lines
    setup_predefined_lines(source_index, FRAME_WIDTH, FRAME_HEIGHT)
    
    # 3) Set mouse callback for editing
    cv2.setMouseCallback(window_name, draw_line_mode, source_index)
    editing_mode = True  # Start in editing mode
    
    print("\n=== LINE EDITING MODE ===")
    print("Instructions:")
    print("- Click on any line to select and redraw it")
    print("- Right-click to cancel selection")
    print("- Press 'E' to toggle between editing and view mode")
    print("- Press 'R' to reset to predefined lines")
    print("- Press 'V' to validate current configuration")
    print("- Press ENTER to confirm and start processing")
    print("- Press ESC to exit")
    
    while True:
        # Create display frame
        display_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
        # Draw all defined lines
        draw_lines_with_labels(display_frame)
        
        # Draw temporary line if currently drawing
        if drawing_line and len(temp_line_pts) == 2:
            cv2.line(display_frame, temp_line_pts[0], temp_line_pts[1], (255, 255, 255), 1)
        
        # Add status information
        mode_text = "EDITING MODE" if editing_mode else "VIEW MODE"
        cv2.putText(display_frame, mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if selected_line is not None:
            cv2.putText(display_frame, f"Selected: {line_roles[selected_line]}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show validation status
        validation_msg = get_validation_message()
        color = (0, 255, 0) if "Valid" in validation_msg else (0, 0, 255)
        cv2.putText(display_frame, validation_msg, (10, FRAME_HEIGHT - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Show controls
        controls = [
            "E: Toggle Edit Mode | R: Reset | V: Validate",
            "ENTER: Confirm | ESC: Exit"
        ]
        for i, control in enumerate(controls):
            cv2.putText(display_frame, control, (10, FRAME_HEIGHT - 60 + i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            print("Exiting line configuration.")
            cv2.destroyWindow(window_name)
            return
        elif key == 13:  # ENTER
            if validate_three_lines():
                print("Line configuration confirmed. Starting processing...")
                break
            else:
                print("Cannot confirm: Invalid line configuration!")
        elif key in (ord('E'), ord('e')):
            editing_mode = not editing_mode
            selected_line = None
            mode_text = "EDITING" if editing_mode else "VIEW"
            print(f"Switched to {mode_text} mode")
        elif key in (ord('R'), ord('r')):
            setup_predefined_lines(source_index, FRAME_WIDTH, FRAME_HEIGHT)
            selected_line = None
            print("Lines reset to predefined positions")
        elif key in (ord('V'), ord('v')):
            print(f"Validation: {get_validation_message()}")
    
    # 4) Extract line positions for processing
    if orientation == 'V':
        enter_x = lines_final[0][0][0]
        count_x = lines_final[1][0][0]
        exit_x = lines_final[2][0][0]
        line_positions[source_index] = (enter_x, count_x, exit_x)
        enter_line, count_line, exit_line = line_positions[source_index]
    else:
        enter_y = lines_final[0][0][1]
        count_y = lines_final[1][0][1]
        exit_y = lines_final[2][0][1]
        line_positions[source_index] = (enter_y, count_y, exit_y)
        enter_line, count_line, exit_line = line_positions[source_index]
    
    print(f"Final line positions: {line_positions[source_index]}")
    
    # Now continue with the rest of your video processing logic...
    # (The rest of your video processing code would go here)
    
    cv2.destroyWindow(window_name)

    while not thread_controller.stop_event.is_set():
        t_start = time.perf_counter()

        try:
            frame = thread_controller.frame_queue[source_index].get(timeout=0.1)
        except queue.Empty:
            continue

        frame_idx += 1

        # Apply digital zoom before any processing
        frame = zoom_controller.apply_zoom(frame)

        # Process the frame
        frame = cv2.resize(frame, FRAME_SIZE)

        
        results = model.track(
            frame,
            verbose=False,
            classes=[0],  # Track people only
            conf=0.3,
            iou=0.5,
            stream=True,
            stream_buffer=True,
            persist=True,
            tracker="custom_tracker.yaml"
        )

        # frame = cv2.flip(frame, 1)

        # Create a copy of the frame for visualization and recording
        visualization_frame = frame.copy()

        seen_ids = set()

        for result in results:

            for box in result.boxes:
                
                track_id = int(box.id.item()) if box.id is not None else None

                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Track the ID
                seen_ids.add(track_id)
                last_seen[track_id] = frame_idx
                cx = (x1 + x2)//2
                cy = (y1+y2)//2
                state = tracked_ids.get(track_id, 0)

                # If vertical, compare cx to vertical lines; if horizontal, compare cy to horizontal lines.
                if orientation == 'V':
                    prev = previous_centroids.get(track_id, cx)
                    direction = None
                    state = tracked_ids.get(track_id, 0)
                    if state==0 and prev < enter_line <= cx:
                        enter_count[source_index] += 1
                        direction = 'enter'
                        tracked_ids[track_id] = 1
                    elif state==0 and prev >= exit_line > cx:
                        exit_count[source_index] += 1
                        direction = 'exit'
                        tracked_ids[track_id] = 2
                    elif state==1 and prev < count_line <= cx:
                        # If already entered before, but then passes count line: reaffirm as entering
                        pass
                    elif state==2 and prev >= count_line > cx:
                        # If already marked exit before, but passes count line: reaffirm as exiting
                        pass

                    previous_centroids[track_id] = cx

                else:  # orientation == 'H'
                    prev = previous_centroids.get(track_id, cy)
                    direction = None
                    state = tracked_ids.get(track_id, 0)
                    if state==0 and prev < enter_line <= cy:
                        enter_count[source_index] += 1
                        direction = 'enter'
                        tracked_ids[track_id] = 1
                    elif state==0 and prev >= exit_line > cy:
                        exit_count[source_index] += 1
                        direction = 'exit'
                        tracked_ids[track_id] = 2
                    elif state==1 and prev < count_line <= cy:
                        pass
                    elif state==2 and prev >= count_line > cy:
                        pass

                    previous_centroids[track_id] = cy

                if direction:
                    timestamp = datetime.now().isoformat()
                    source_identifier = f"camera_{source_index}" 
                    thread_controller.pending_inserts.append((source_identifier, track_id, direction, timestamp, 'line'))
        
                # Draw bounding box and ID
                if thread_controller.enable_visual:
                    color = colors[track_id % len(colors)]
                    cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(visualization_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Clean up tracks that haven't been seen recently
        for tid in list(tracked_ids):
            if frame_idx - last_seen.get(tid, frame_idx) > MAX_MISSING:
                if tid in tracked_ids:
                    del tracked_ids[tid]
                if tid in last_seen:
                    del last_seen[tid]
                if tid in previous_centroids:
                    del previous_centroids[tid]

        # Calculate and display FPS
        avg_frame_rate = calculate_fps(frame_rate_buffer, t_start, fps_avg_len)

        if thread_controller.enable_visual:
            cv2.putText(visualization_frame, f"FPS: {avg_frame_rate:.1f}", (10, FRAME_HEIGHT - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw the three lines on visualization
            if orientation=='V':
                cv2.line(visualization_frame, (enter_line, 0), (enter_line, FRAME_HEIGHT), (0,255,0), 2)
                cv2.line(visualization_frame, (count_line,  0), (count_line,  FRAME_HEIGHT), (0,165,255), 2)
                cv2.line(visualization_frame, (exit_line,   0), (exit_line,   FRAME_HEIGHT), (0,0,255), 2)
            else:
                cv2.line(visualization_frame, (0, enter_line), (FRAME_WIDTH, enter_line), (0,255,0), 2)
                cv2.line(visualization_frame, (0, count_line), (FRAME_WIDTH, count_line), (0,165,255), 2)
                cv2.line(visualization_frame, (0, exit_line),  (FRAME_WIDTH, exit_line),  (0,0,255), 2)

            # # Draw counting line
            # cv2.line(visualization_frame, (line_positions[source_index], 0), (line_positions[source_index], FRAME_HEIGHT), (0, 255, 0), 2)

            # Display counters
            cv2.putText(visualization_frame, f"Enter: {enter_count[source_index]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization_frame, f"Exit: {exit_count[source_index]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write processed frame to video
        if thread_controller.enable_processed_frame_recording and processed_out is None:
            recording_fps = 15.0
            os.makedirs("video/processed", exist_ok=True)
            processed_filename = f"video/processed/processed_{source_name}_crowd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            processed_out = cv2.VideoWriter(processed_filename, fourcc, recording_fps, (FRAME_WIDTH, FRAME_HEIGHT))

        # Write frame to video if recording is active
        if thread_controller.enable_processed_frame_recording and processed_out is not None:
            if processed_out.isOpened():
                processed_out.write(visualization_frame)

        cv2.imshow(window_name, visualization_frame)
        key = cv2.waitKey(1) & 0xFF

        # Handle zoom controls
        if key == ord('+') or key == ord('='):  # Zoom in with + or = key
            zoom_controller.increase_zoom()
            print(f"initial zoom: {zoom_controller.zoom_factor}")
        elif key == ord('-') or key == ord('_'):  # Zoom out with - or _ key
            zoom_controller.decrease_zoom()
        elif key == ord('r') or key == ord('R'):  # Reset zoom with R key
            zoom_controller.zoom_factor = INITIAL_ZOOM
            zoom_controller.zoom_center_x = 0.5
            zoom_controller.zoom_center_y = 0.5
        elif key == ord('v') or key == ord('V'):
            thread_controller.enable_visual = not thread_controller.enable_visual
            print(f"Visualization {'enabled' if thread_controller.enable_visual else 'disabled'}")
        elif key == ord('q') or key == ord('Q'):
            thread_controller.enable_processed_frame_recording = not thread_controller.enable_processed_frame_recording
            # If we're turning off recording, release the video writer
            if not thread_controller.enable_processed_frame_recording and processed_out is not None:
                processed_out.release()
                processed_out = None
            
            print(f"Recording {'started' if thread_controller.enable_processed_frame_recording else 'ended'} {datetime.now().strftime('%Y%m%d_%H%M%S')}")

        def handle_close():
            thread_controller.stop_event.set()
            
            # Wait for all threads to finish, but skip the current thread
            current_thread = threading.current_thread()
            for thread in thread_controller.threads:
                if thread.is_alive() and thread != current_thread:
                    thread.join(timeout=2.0)
            
            # # If video window is still open, close it
            # if thread_controller.video_window_open:
            cv2.destroyAllWindows()
            
            # Call the provided callback
            if on_close:
                on_close()
            
        # Handle window close or ESC key
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            thread_controller.stop_event.set()
            handle_close()
            break
    
    # Clean up
    if processed_out is not None and processed_out.isOpened():
        processed_out.release()

    # Safer window destruction
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)  # Destroy specific window instead of all
    except cv2.error:
        pass  # Window was already closed by user

    thread_controller.video_window_open = False
    print(f"Video processing thread for {source_name} stopped")
        
def video_processing_crowd(source_index, on_close=None):
    global crowd_count, total_crowd_count, drawing, roi_points, temp_roi, roi_set

    source_name = f"Camera {source_index + 1}"
    window_name = f"People Counter - {source_name} - CROWD MODE"

    # Create a separate model instance for each thread
    model = load_model(MODEL_NAME)

    # Dictionary to track people who have been counted
    tracked_ids = {}
    last_seen = {}
    frame_idx = 0
    MAX_MISSING = 400  # Number of frames before considering a track lost
    frame_rate_buffer = []
    avg_frame_rate = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

    # Add minimum detection threshold
    MIN_DETECTIONS = 30  # Require this many consecutive detections before counting
    
    # Add to your tracking data structures
    detection_count = {}  # track_id -> consecutive detection count

    # Get zoom controller for this source
    zoom_controller = thread_controller.zoom_controllers[source_index]

    processed_out = None
    
    # Make OpenCV window a normal window that can be closed
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set up mouse callback for drawing ROI
    cv2.setMouseCallback(window_name, draw_roi, source_index)

    thread_controller.video_window_open = True

    # Print zoom controls instruction
    print(f"Zoom Controls for {window_name}:")
    print("  + : Zoom in")
    print("  - : Zoom out")
    print("  Arrow keys: Pan the view")
    print("  R : Reset zoom")
    print(f"Other Controls:")
    print("  Q : Record Video")
    print("  V : Visualization")
    print("  T : Toggle ROI display")
    print("  C : Clear/Reset ROI")

    # NEW: ROI visualization toggle
    show_roi = True

    # If ROI drawing is enabled, wait for user to draw ROI before starting
    if ENABLE_ROI and thread_controller.enable_roi_drawing_mode:
        roi_set = False
        roi_points = []
        temp_roi = []
        print(f"Please draw Region of Interest for {source_name}.")
        print("Left-click to add points (minimum 3 needed), right-click to finish. Press D to cancel")

        # Initialize drawing frame with a black background
        drawing_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Wait for ROI to be drawn
        while not roi_set and not thread_controller.stop_event.is_set():
            # Initialize drawing frame
            try:
                drawing_frame = thread_controller.frame_queue[source_index].get(timeout=0.5)
                drawing_frame = zoom_controller.apply_zoom(drawing_frame)
                drawing_frame = cv2.resize(drawing_frame, FRAME_SIZE)
            except queue.Empty:
                continue

            drawing_display = drawing_frame.copy()
            drawing_display = draw_roi_overlay(drawing_display, temp_roi, drawing, current_mouse_pos)
                
            cv2.imshow(window_name, drawing_display)
            key = cv2.waitKey(1) & 0xFF

            ## Note to add cancelation and using the previous ROI

    while not thread_controller.stop_event.is_set():
        t_start = time.perf_counter()

        try:
            frame = thread_controller.frame_queue[source_index].get(timeout=0.1)
        except queue.Empty:
            continue

        frame_idx += 1

        # Apply digital zoom before any processing
        frame = zoom_controller.apply_zoom(frame)

        # Process the frame
        frame = cv2.resize(frame, FRAME_SIZE)

        # Only process frames for detection if not in drawing mode
        if thread_controller.enable_roi_drawing_mode and not roi_set and ENABLE_ROI:
            # In drawing mode, just show the current frame with drawing overlay
            visualization_frame = frame.copy()
            
            # Draw the current ROI points
            visualization_frame = draw_roi_overlay(visualization_frame, temp_roi, drawing, current_mouse_pos)
        else:
            results = model.track(
                frame,
                verbose=False,
                classes=[0],  # Track people only
                conf=0.3,
                iou=0.5,
                stream=True,
                stream_buffer=True,
                persist=True,
                tracker="custom_tracker.yaml"
            )

            # frame = cv2.flip(frame, 1)

            # Create a copy of the frame for visualization and recording
            visualization_frame = frame.copy()

            # Draw ROI if enabled
            if ENABLE_ROI and show_roi:
                cv2.polylines(visualization_frame, [ROI_POINTS[source_index]], True, (0, 255, 0), 2)
                cv2.putText(visualization_frame, "Target Area", 
                            tuple(ROI_POINTS[source_index][0][0]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            seen_ids = set()

            for result in results:

                for box in result.boxes:
                    
                    track_id = int(box.id.item()) if box.id is not None else None

                    if track_id is None:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Track the ID
                    seen_ids.add(track_id)
                    last_seen[track_id] = frame_idx
                    
                    # Increment detection counter for this ID
                    detection_count[track_id] = detection_count.get(track_id, 0) + 1

                    # Apply ROI check if enabled
                    in_roi = not ENABLE_ROI or is_in_roi(box.xyxy[0].tolist(), ROI_POINTS[source_index])
                    
                    # If this is a countable person and in ROI (if enabled), count them
                    if  in_roi and track_id not in tracked_ids and detection_count[track_id] >= MIN_DETECTIONS:
                        tracked_ids[track_id] = True
                        crowd_count[source_index] += 1
                        total_crowd_count += 1
                        
                        # Record in database
                        timestamp = datetime.now().isoformat()
                        source_identifier = f"camera_{source_index}"
                        thread_controller.pending_inserts.append((source_identifier, track_id, 'enter', timestamp, 'crowd'))

                    # Draw bounding box and ID
                    if thread_controller.enable_visual and in_roi:
                        color = colors[track_id % len(colors)]
                        cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(visualization_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Clean up tracks that haven't been seen recently
            cleanup_stale(last_seen, frame_idx, MAX_MISSING, detection_count)

            # Calculate and display FPS
            avg_frame_rate = calculate_fps(frame_rate_buffer, t_start, fps_avg_len)

            if thread_controller.enable_visual:
                cv2.putText(visualization_frame, f"FPS: {avg_frame_rate:.1f}", (10, FRAME_HEIGHT - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Display counters
                cv2.putText(visualization_frame, f"Crowd Count: {crowd_count[source_index]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

                # Display ROI drawing mode status
                roi_mode_text = "ROI Drawing Mode: ON" if thread_controller.enable_roi_drawing_mode else "ROI Drawing Mode: OFF"
                cv2.putText(visualization_frame, roi_mode_text, (10, FRAME_HEIGHT - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                

        # Write processed frame to video
        if thread_controller.enable_processed_frame_recording and processed_out is None:
            recording_fps = 15.0
            os.makedirs("video/processed", exist_ok=True)
            processed_filename = f"video/processed/processed_{source_name}_crowd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            processed_out = cv2.VideoWriter(processed_filename, fourcc, recording_fps, (FRAME_WIDTH, FRAME_HEIGHT))

        # Write frame to video if recording is active
        if thread_controller.enable_processed_frame_recording and processed_out is not None:
            if processed_out.isOpened():
                processed_out.write(visualization_frame)

        cv2.imshow(window_name, visualization_frame)
        key = cv2.waitKey(1) & 0xFF

        # Handle zoom controls
        if key == ord('+') or key == ord('='):  # Zoom in with + or = key
            zoom_controller.increase_zoom()
            print(f"initial zoom: {zoom_controller.zoom_factor}")
        elif key == ord('-') or key == ord('_'):  # Zoom out with - or _ key
            zoom_controller.decrease_zoom()
        elif key == ord('r') or key == ord('R'):  # Reset zoom with R key
            zoom_controller.zoom_factor = INITIAL_ZOOM
            zoom_controller.zoom_center_x = 0.5
            zoom_controller.zoom_center_y = 0.5
        elif key == ord('v') or key == ord('V'):
            thread_controller.enable_visual = not thread_controller.enable_visual
            print(f"Visualization {'enabled' if thread_controller.enable_visual else 'disabled'}")
        elif key == ord('q') or key == ord('Q'):
            thread_controller.enable_processed_frame_recording = not thread_controller.enable_processed_frame_recording
            # If we're turning off recording, release the video writer
            if not thread_controller.enable_processed_frame_recording and processed_out is not None:
                processed_out.release()
                processed_out = None
            
            print(f"Recording {'started' if thread_controller.enable_processed_frame_recording else 'ended'} {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        elif key == ord('t') or key == ord('T'):  # Toggle ROI display
            show_roi = not show_roi
        elif key == ord('c') or key == ord('C'):  # Clear/Reset ROI
            print("ROI Cleared. ROI drawing mode is now enabled. Draw a new ROI.")
            reset_roi(source_index)
            thread_controller.enable_roi_drawing_mode = True

        def handle_close():
            thread_controller.stop_event.set()
            
            # Wait for all threads to finish, but skip the current thread
            current_thread = threading.current_thread()
            for thread in thread_controller.threads:
                if thread.is_alive() and thread != current_thread:
                    thread.join(timeout=2.0)
            
            # # If video window is still open, close it
            # if thread_controller.video_window_open:
            cv2.destroyAllWindows()
            
            # Call the provided callback
            if on_close:
                on_close()
            
        # Handle window close or ESC key
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            thread_controller.stop_event.set()
            handle_close()
            break
    
    # Clean up
    if processed_out is not None and processed_out.isOpened():
        processed_out.release()

    # Safer window destruction
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)  # Destroy specific window instead of all
    except cv2.error:
        pass  # Window was already closed by user

    thread_controller.video_window_open = False
    print(f"Video processing thread for {source_name} stopped")

def video_processing_dispatcher(mode, source_index, on_close=None):
    if mode == "LINE":
        video_processing_line(source_index, on_close)
    elif mode == "CROWD":
        video_processing_crowd(source_index, on_close)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# Start the tracking threads
def start_threads(on_session_end=None):
    # Reset the thread controller
    thread_controller.reset()
    
    # Create and start capture threads for each camera
    for i, source in enumerate(CAMERA_SOURCES):
        if source is not None:  # Only start threads for defined sources
            # Capture thread
            capture_thread = threading.Thread(target=capture_frames, args=(i,), daemon=True)
            capture_thread.start()
            thread_controller.threads.append(capture_thread)
            
            # Processing thread for each camera - use correct mode
            if COUNT_MODE == "LINE":
                video_thread = threading.Thread(target=video_processing_dispatcher, args=("LINE", i, on_session_end,), daemon=True)
            elif COUNT_MODE == "CROWD":
                video_thread = threading.Thread(target=video_processing_dispatcher, args=("CROWD", i, on_session_end,), daemon=True)

            video_thread.start()
            thread_controller.threads.append(video_thread)

    db_thread = threading.Thread(target=insert_to_db,args=(thread_controller,), daemon=True)
    db_thread.start()
    thread_controller.threads.append(db_thread)

# Selection window function
def show_selection_window():
    """
    Creates and runs the selection dialog. When 'Start' is clicked,
    this window is destroyed and start_threads() is called.
    When the counter window later closes, it will re-invoke this function.
    """
    global COUNT_MODE

    # Ensure any previous stop event is set
    thread_controller.stop_event.set()
    
    # Create the main window
    sel = tk.Tk()
    sel.title("People Counter - Select Counting Mode")
    sel.geometry("500x500")
    
    # Add padding and styling
    content_frame = tk.Frame(sel, padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)

    # Count type selection
    count_type_frame = tk.LabelFrame(content_frame, text="Counting Type", padx=10, pady=10)
    count_type_frame.pack(fill=tk.X, pady=(0, 10))
    
    count_type_var = tk.StringVar(value="LINE")
    
    tk.Radiobutton(count_type_frame, text="Line Crossing (Standard)", variable=count_type_var, 
                  value="LINE", command=lambda: on_count_type_change("LINE")).pack(anchor='w')
    tk.Radiobutton(count_type_frame, text="Crowd Count (Person in Frame)", variable=count_type_var, 
                  value="CROWD", command=lambda: on_count_type_change("CROWD")).pack(anchor='w')
    
    # Mode selection
    mode_frame = tk.LabelFrame(content_frame, text="Select Mode", padx=10, pady=10)
    mode_frame.pack(fill=tk.X, pady=(0, 10))
    
    mode_var = tk.IntVar(value=0)

    tk.Radiobutton(mode_frame, text="Start Fresh", variable=mode_var, value=0).pack(anchor='w')
    tk.Radiobutton(mode_frame, text="Use Today's Data", variable=mode_var, value=1).pack(anchor='w')
    tk.Radiobutton(mode_frame, text="Custom Date Range", variable=mode_var, value=2).pack(anchor='w')

    # Date and time selection for custom range
    date_time_frame = tk.Frame(content_frame)
    date_time_frame.pack(fill=tk.X, pady=10)

    # Start date/time frame
    start_frame = tk.LabelFrame(date_time_frame, text="Start", padx=5, pady=5)
    start_frame.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    
    # Start date
    start_date_entry = DateEntry(start_frame, date_pattern='yyyy-MM-dd')
    start_date_entry.grid(row=0, column=0, padx=5, pady=5)
    
    # Start time
    start_time_frame = tk.Frame(start_frame)
    start_time_frame.grid(row=0, column=1, padx=5, pady=5)
    
    start_hour = tk.Spinbox(start_time_frame, from_=0, to=23, width=2, format="%02.0f")
    start_hour.grid(row=0, column=0)
    start_hour.delete(0, tk.END)
    start_hour.insert(0, "00")
    
    tk.Label(start_time_frame, text=":").grid(row=0, column=1)
    
    start_min = tk.Spinbox(start_time_frame, from_=0, to=59, width=2, format="%02.0f")
    start_min.grid(row=0, column=2)
    start_min.delete(0, tk.END)
    start_min.insert(0, "00")
    
    tk.Label(start_time_frame, text=":").grid(row=0, column=3)
    
    start_sec = tk.Spinbox(start_time_frame, from_=0, to=59, width=2, format="%02.0f")
    start_sec.grid(row=0, column=4)
    start_sec.delete(0, tk.END)
    start_sec.insert(0, "00")
    
    # End date/time frame
    end_frame = tk.LabelFrame(date_time_frame, text="End", padx=5, pady=5)
    end_frame.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    
    # End date
    end_date_entry = DateEntry(end_frame, date_pattern='yyyy-MM-dd')
    end_date_entry.grid(row=0, column=0, padx=5, pady=5)
    
    # End time
    end_time_frame = tk.Frame(end_frame)
    end_time_frame.grid(row=0, column=1, padx=5, pady=5)
    
    end_hour = tk.Spinbox(end_time_frame, from_=0, to=23, width=2, format="%02.0f")
    end_hour.grid(row=0, column=0)
    end_hour.delete(0, tk.END)
    end_hour.insert(0, "23")
    
    tk.Label(end_time_frame, text=":").grid(row=0, column=1)
    
    end_min = tk.Spinbox(end_time_frame, from_=0, to=59, width=2, format="%02.0f")
    end_min.grid(row=0, column=2)
    end_min.delete(0, tk.END)
    end_min.insert(0, "59")
    
    tk.Label(end_time_frame, text=":").grid(row=0, column=3)
    
    end_sec = tk.Spinbox(end_time_frame, from_=0, to=59, width=2, format="%02.0f")
    end_sec.grid(row=0, column=4)
    end_sec.delete(0, tk.END)
    end_sec.insert(0, "59")
    
    # Initially hide the date entries
    date_time_frame.pack_forget()
    
    # Show/hide date entries based on mode selection
    def on_mode_change(*_):
        if mode_var.get() == 2:
            date_time_frame.pack(fill=tk.X, pady=10)
        else:
            date_time_frame.pack_forget()

    mode_var.trace_add('write', on_mode_change)

    # Handle count type change
    def on_count_type_change(mode):
        global COUNT_MODE
        COUNT_MODE = mode

    # Initialize count for today's data
    def init_today_counts():
        global enter_count, exit_count, total_enter_count, total_exit_count
        db = Database()
        _, cursor = db.get_connection()

        today = date.today()
        start_dt = datetime.combine(today, datetime.min.time()).isoformat()

        # Reset counts
        enter_count = [0, 0]
        exit_count = [0, 0]
        crowd_count = [0, 0]
        total_enter_count = 0
        total_exit_count = 0
        total_crowd_count = 0

        if COUNT_MODE == "LINE":
            cursor.execute("""SELECT direction, COUNT(*) FROM crossing_events 
                        WHERE timestamp >= ? AND mode_type = 'line' GROUP BY direction""",
                        (start_dt,))
            
            for d, c in cursor.fetchall():
                if d == 'enter':
                    total_enter_count = c
                else:
                    total_exit_count = c
            
            # Set the first camera's count to the total
            enter_count[0] = total_enter_count
            exit_count[0] = total_exit_count
        elif COUNT_MODE == "CROWD":
            cursor.execute("""SELECT COUNT(*) FROM crossing_events 
                          WHERE timestamp >= ? AND mode_type = 'crowd'""",
                          (start_dt,))
            
            total_crowd_count = cursor.fetchone()[0] or 0
            
            # Set the first camera's count to the total
            crowd_count[0] = total_crowd_count

    # Initialize count for custom date range
    def init_custom_counts():
        global enter_count, exit_count, total_enter_count, total_exit_count, crowd_count, total_crowd_count
        db = Database()
        _, cursor = db.get_connection()

        

        try:
            # Get start date and time
            start_date = start_date_entry.get_date()
            start_time = f"{start_hour.get().zfill(2)}:{start_min.get().zfill(2)}:{start_sec.get().zfill(2)}"
            
            # Get end date and time
            end_date = end_date_entry.get_date()
            end_time = f"{end_hour.get().zfill(2)}:{end_min.get().zfill(2)}:{end_sec.get().zfill(2)}"
            
            # Format the complete timestamps
            s = f"{start_date.isoformat()}T{start_time}"
            e = f"{end_date.isoformat()}T{end_time}"

            # Reset counts
            enter_count = [0, 0]
            exit_count = [0, 0]
            crowd_count = [0, 0]
            total_enter_count = 0
            total_exit_count = 0
            total_crowd_count = 0

            if COUNT_MODE == "LINE":
                cursor.execute("""SELECT direction, COUNT(*) FROM crossing_events 
                                WHERE timestamp BETWEEN ? AND ? AND mode_type = 'line' GROUP BY direction""",
                                (s, e))
                
                for d, c in cursor.fetchall():
                    if d == 'enter':
                        total_enter_count = c
                    else:
                        total_exit_count = c

                # Set the first camera's count to the total
                enter_count[0] = total_enter_count
                exit_count[0] = total_exit_count
            elif COUNT_MODE == "CROWD":
                cursor.execute("""SELECT COUNT(*) FROM crossing_events 
                              WHERE timestamp BETWEEN ? AND ? AND mode_type = 'crowd'""",
                              (s, e))
                
                total_crowd_count = cursor.fetchone()[0] or 0
                
                # Set the first camera's count to the total
                crowd_count[0] = total_crowd_count
            
        except Exception as ex:
            messagebox.showerror("Error", f"Failed to get data: {str(ex)}")
            total_enter_count = total_exit_count = total_crowd_count = 0
            enter_count = [0, 0]
            exit_count = [0, 0]
            crowd_count = [0, 0]

    # Create and show the query window
    def open_query_window():
        query_win = tk.Toplevel(sel)
        query_win.title("Query People Entering")
        query_win.geometry("900x700")
        query_win.grab_set()  # Make window modal
        
        # Create frames for better organization
        control_frame = tk.Frame(query_win, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        result_frame = tk.Frame(query_win, padx=10)
        result_frame.pack(fill=tk.X)
        
        graph_frame = tk.Frame(query_win, padx=10, pady=10)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Date and time selection
        date_time_frame = tk.Frame(control_frame)
        date_time_frame.pack(fill=tk.X)
        
        # Start date/time
        start_frame = tk.LabelFrame(date_time_frame, text="Start", padx=5, pady=5)
        start_frame.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        start_date_entry = DateEntry(start_frame, date_pattern='yyyy-MM-dd')
        start_date_entry.grid(row=0, column=0, padx=5, pady=5)

        time_frame = tk.Frame(start_frame)
        time_frame.grid(row=0, column=1, padx=5, pady=5)
        
        start_hour = tk.Spinbox(time_frame, from_=0, to=23, width=2, format="%02.0f")
        start_hour.grid(row=0, column=0)
        tk.Label(time_frame, text=":").grid(row=0, column=1)
        start_min = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        start_min.grid(row=0, column=2)
        tk.Label(time_frame, text=":").grid(row=0, column=3)
        start_sec = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        start_sec.grid(row=0, column=4)
        
        # End date/time
        end_frame = tk.LabelFrame(date_time_frame, text="End", padx=5, pady=5)
        end_frame.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        end_date_entry = DateEntry(end_frame, date_pattern='yyyy-MM-dd')
        end_date_entry.grid(row=0, column=0, padx=5, pady=5)

        time_frame = tk.Frame(end_frame)
        time_frame.grid(row=0, column=1, padx=5, pady=5)
        
        end_hour = tk.Spinbox(time_frame, from_=0, to=23, width=2, format="%02.0f")
        end_hour.grid(row=0, column=0)
        tk.Label(time_frame, text=":").grid(row=0, column=1)
        end_min = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        end_min.grid(row=0, column=2)
        tk.Label(time_frame, text=":").grid(row=0, column=3)
        end_sec = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        end_sec.grid(row=0, column=4)
        
        # Resolution selection
        resolution_frame = tk.LabelFrame(control_frame, text="Time Resolution", padx=5, pady=5)
        resolution_frame.pack(fill=tk.X, pady=10)
        
        resolution_var = tk.StringVar(value="hour")
        
        tk.Radiobutton(resolution_frame, text="Second", variable=resolution_var, value="second").grid(row=0, column=0, padx=10)
        tk.Radiobutton(resolution_frame, text="Minute", variable=resolution_var, value="minute").grid(row=0, column=1, padx=10)
        tk.Radiobutton(resolution_frame, text="Hour", variable=resolution_var, value="hour").grid(row=0, column=2, padx=10)
        tk.Radiobutton(resolution_frame, text="Day", variable=resolution_var, value="day").grid(row=0, column=3, padx=10)
        
        # Visualization options
        visual_frame = tk.LabelFrame(control_frame, text="Visualization", padx=5, pady=5)
        visual_frame.pack(fill=tk.X, pady=10)
        
        visual_var = tk.StringVar(value="bar")
        
        tk.Radiobutton(visual_frame, text="Bar Chart", variable=visual_var, value="bar").grid(row=0, column=0, padx=10)
        tk.Radiobutton(visual_frame, text="Line Chart", variable=visual_var, value="line").grid(row=0, column=1, padx=10)
        tk.Radiobutton(visual_frame, text="Area Chart", variable=visual_var, value="area").grid(row=0, column=2, padx=10)
        
        # Fetch button
        fetch_button = tk.Button(
            control_frame, 
            text="Fetch Data", 
            command=lambda: fetch_data(
                start_date_entry, start_hour, start_min, start_sec,
                end_date_entry, end_hour, end_min, end_sec,
                resolution_var.get(), visual_var.get(),
                result_label, graph_frame
            ),
            bg="#4CAF50", fg="white", padx=10, pady=5
        )
        fetch_button.pack(pady=10)
        
        # Results display
        result_label = tk.Label(result_frame, text="Total Entries: 0", font=("Helvetica", 12))
        result_label.pack(pady=5)
        
        # Default values - today
        today = date.today()
        start_date_entry.set_date(today)
        end_date_entry.set_date(today)
        start_hour.delete(0, tk.END)
        start_hour.insert(0, "00")
        start_min.delete(0, tk.END)
        start_min.insert(0, "00")
        start_sec.delete(0, tk.END)
        start_sec.insert(0, "00")
        end_hour.delete(0, tk.END)
        end_hour.insert(0, "23")
        end_min.delete(0, tk.END)
        end_min.insert(0, "59")
        end_sec.delete(0, tk.END)
        end_sec.insert(0, "59")
        
        def on_close():
            query_win.grab_release()
            query_win.destroy()
    
        query_win.protocol("WM_DELETE_WINDOW", on_close)

    # Fetch and display data for the query window
    def fetch_data(start_date_entry, start_hour, start_min, start_sec, 
                   end_date_entry, end_hour, end_min, end_sec,
                   resolution, visualization,
                   result_label, graph_frame):
        # Get date and time values
        start_date = start_date_entry.get_date()
        start_time = f"{start_hour.get().zfill(2)}:{start_min.get().zfill(2)}:{start_sec.get().zfill(2)}"

        end_date = end_date_entry.get_date()
        end_time = f"{end_hour.get().zfill(2)}:{end_min.get().zfill(2)}:{end_sec.get().zfill(2)}"

        start_timestamp = f"{start_date}T{start_time}"
        end_timestamp = f"{end_date}T{end_time}"

        db = Database()
        _, cursor = db.get_connection()

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM crossing_events WHERE direction = 'enter' AND timestamp BETWEEN ? AND ?", 
                      (start_timestamp, end_timestamp))
        count = cursor.fetchone()[0]
        result_label.config(text=f"Total Entries: {count}")

        # Clear previous graph
        for widget in graph_frame.winfo_children():
            widget.destroy()
            
        # Exit if no data
        if count == 0:
            tk.Label(graph_frame, text="No data for the selected period", font=("Helvetica", 12)).pack()
            return

        # Query based on resolution
        title = ""
        groupby = ""
        
        if resolution == "second":
            title = "Entries by Second"
            groupby = "strftime('%Y-%m-%d %H:%M:%S', timestamp)"
        elif resolution == "minute":
            title = "Entries by Minute"
            groupby = "strftime('%Y-%m-%d %H:%M', timestamp)"
        elif resolution == "hour":
            title = "Hourly Entries"
            groupby = "strftime('%Y-%m-%d %H:00:00', timestamp)"
        else:  # day
            title = "Daily Entries"
            groupby = "DATE(timestamp)"

        # Query data
        query = f"""
            SELECT {groupby} as timeperiod, COUNT(*) 
            FROM crossing_events 
            WHERE direction = 'enter' AND timestamp BETWEEN ? AND ? 
            GROUP BY timeperiod
            ORDER BY timeperiod
        """
        
        cursor.execute(query, (start_timestamp, end_timestamp))
        data = cursor.fetchall()
        
        # Convert timestamps to datetime objects for better plotting
        time_periods = []
        counts = []
        
        for row in data:
            try:
                if resolution == "day":
                    dt = datetime.strptime(row[0], "%Y-%m-%d")
                elif resolution == "hour":
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:00:00")
                elif resolution == "minute":
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M")
                else:  # second
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                print(f"Error parsing datetime '{row[0]}': {e}")
                continue
            time_periods.append(dt)
            counts.append(row[1])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if visualization == "bar":
            ax.bar(time_periods, counts, width=0.8)
        elif visualization == "line":
            ax.plot(time_periods, counts, marker='o', linestyle='-', linewidth=2)
        elif visualization == "area":
            ax.fill_between(time_periods, counts, alpha=0.4)
            ax.plot(time_periods, counts, marker='o', linestyle='-', linewidth=2)
        
        # Format x-axis based on resolution
        if resolution == "second" or resolution == "minute":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)
        elif resolution == "hour":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45)
        else:  # day
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Entries')
        ax.set_title(title)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Close the matplotlib figure to prevent memory leaks
        plt.close(fig)

    # Handle start button click
    def on_start():
        global enter_count, exit_count, total_enter_count, total_exit_count, crowd_count, total_crowd_count
        m = mode_var.get()
        if m == 0:
            # Start fresh
            enter_count = [0,0]
            exit_count = [0,0]
            crowd_count = [0, 0]
            total_enter_count = 0
            total_exit_count = 0
            total_crowd_count = 0

        elif m == 1:
            # Use today's data
            init_today_counts()
        elif m == 2:
            # Custom date range
            try:
                init_custom_counts()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading custom date range: {e}")
                return
        
        # Hide selection window and start threads
        sel.withdraw()  # Hide instead of destroy
        
        # Define callback for when counter window closes
        def on_counting_close():
            try: sel.deiconify()
            except tk.TclError: show_selection_window()
        # Start the threads
        start_threads(on_session_end=on_counting_close)

    # Add buttons to selection window
    button_frame = tk.Frame(content_frame)
    button_frame.pack(fill=tk.X, pady=10)

    query_button = tk.Button(
        button_frame, 
        text="Query Past Data", 
        command=open_query_window,
        width=15
    )
    query_button.pack(side=tk.LEFT, padx=5)

    start_button = tk.Button(
        button_frame, 
        text="Start", 
        command=on_start,
        width=15
    )
    start_button.pack(side=tk.RIGHT, padx=5)

    # Handle window close properly
    def on_close():
        # Clean up database connection
        db = Database()
        db.close()
        # Destroy the window
        sel.destroy()

    sel.protocol("WM_DELETE_WINDOW", on_close)
    
    # Start the main loop for selection window
    sel.mainloop()

if __name__ == "__main__":
    show_selection_window()