import os
import random
import cv2
import time
import threading
import queue
import numpy as np
import config

from thread_manager import thread_controller
from datetime import datetime
from helpers import (
    load_model, calculate_fps, cleanup_stale
)

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

# Recording settings
ENABLE_RAW_RECORDING = False
ENABLE_PREDICTED_RECORDING = False

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

random_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

# Region of Interest settings
ENABLE_ROI = True  # Set to True to only count people in a specific region
# Define ROI polygons for each camera - customize these coordinates for your setup
ROI_POINTS = [
    np.array([(3, 70), (1279, 91), (1277, 715), (8, 716)], np.int32),
    np.array([[], [], []], np.int32),
]

for i in range(len(ROI_POINTS)):
    ROI_POINTS[i] = ROI_POINTS[i].reshape((-1, 1, 2))

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

    model = load_model(config.get_model_name())

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
                    thread_controller.pending_inserts.put((source_identifier, track_id, direction, timestamp, 'line'))
        
                # Draw bounding box and ID
                if thread_controller.enable_visual:
                    color = colors[track_id % len(colors)]
                    cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(visualization_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Clean up tracks that haven't been seen recently
        cleanup_stale(last_seen, frame_idx, MAX_MISSING, tracked_ids, previous_centroids)

        # Calculate and display FPS
        avg_frame_rate = calculate_fps(frame_rate_buffer, t_start)

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

        # # Handle zoom controls
        # if key == ord('+') or key == ord('='):  # Zoom in with + or = key
        #     zoom_controller.increase_zoom()
        #     print(f"initial zoom: {zoom_controller.zoom_factor}")
        # elif key == ord('-') or key == ord('_'):  # Zoom out with - or _ key
        #     zoom_controller.decrease_zoom()
        # elif key == ord('r') or key == ord('R'):  # Reset zoom with R key
        #     zoom_controller.zoom_factor = INITIAL_ZOOM
        #     zoom_controller.zoom_center_x = 0.5
        #     zoom_controller.zoom_center_y = 0.5
        # elif key == ord('v') or key == ord('V'):
        #     thread_controller.enable_visual = not thread_controller.enable_visual
        #     print(f"Visualization {'enabled' if thread_controller.enable_visual else 'disabled'}")
        # elif key == ord('q') or key == ord('Q'):
        #     thread_controller.enable_processed_frame_recording = not thread_controller.enable_processed_frame_recording
        #     # If we're turning off recording, release the video writer
        #     if not thread_controller.enable_processed_frame_recording and processed_out is not None:
        #         processed_out.release()
        #         processed_out = None
            
        #     print(f"Recording {'started' if thread_controller.enable_processed_frame_recording else 'ended'} {datetime.now().strftime('%Y%m%d_%H%M%S')}")

        def handle_close():
            thread_controller.stop_event.set()
            
            # Wait for all threads to finish, but skip the current thread
            current_thread = threading.current_thread()
            for thread in thread_controller.threads:
                if thread.is_alive() and thread != current_thread:
                    thread.join(timeout=2.0)
            
            # Safer window destruction
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow(window_name)  # Destroy specific window instead of all
            except cv2.error:
                pass  # Window was already closed by user
            
            # Call the provided callback
            if on_close:
                on_close()
            
        # Handle window close or ESC key
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            handle_close()
            break
    
    # Clean up
    if processed_out is not None and processed_out.isOpened():
        processed_out.release()

    thread_controller.video_window_open = False
    print(f"Video processing thread for {source_name} stopped")
        
def video_processing_crowd(source_index, on_close=None):
    global crowd_count, total_crowd_count, drawing, roi_points, temp_roi, roi_set

    source_name = f"Camera {source_index + 1}"
    window_name = f"People Counter - {source_name} - CROWD MODE"

    # Create a separate model instance for each thread
    model = load_model(config.get_model_name())

    # Dictionary to track people who have been counted
    tracked_ids = {}
    last_seen = {}
    frame_idx = 0
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
                        thread_controller.pending_inserts.put((source_identifier, track_id, 'enter', timestamp, 'crowd'))

                    # Draw bounding box and ID
                    if thread_controller.enable_visual and in_roi:
                        color = colors[track_id % len(colors)]
                        cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(visualization_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Clean up tracks that haven't been seen recently
            cleanup_stale(last_seen, frame_idx, detection_count)

            # Calculate and display FPS
            avg_frame_rate = calculate_fps(frame_rate_buffer, t_start)

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

        # # Handle zoom controls
        # if key == ord('+') or key == ord('='):  # Zoom in with + or = key
        #     zoom_controller.increase_zoom()
        #     print(f"initial zoom: {zoom_controller.zoom_factor}")
        # elif key == ord('-') or key == ord('_'):  # Zoom out with - or _ key
        #     zoom_controller.decrease_zoom()
        # elif key == ord('r') or key == ord('R'):  # Reset zoom with R key
        #     zoom_controller.zoom_factor = INITIAL_ZOOM
        #     zoom_controller.zoom_center_x = 0.5
        #     zoom_controller.zoom_center_y = 0.5
        # elif key == ord('v') or key == ord('V'):
        #     thread_controller.enable_visual = not thread_controller.enable_visual
        #     print(f"Visualization {'enabled' if thread_controller.enable_visual else 'disabled'}")
        # elif key == ord('q') or key == ord('Q'):
        #     thread_controller.enable_processed_frame_recording = not thread_controller.enable_processed_frame_recording
        #     # If we're turning off recording, release the video writer
        #     if not thread_controller.enable_processed_frame_recording and processed_out is not None:
        #         processed_out.release()
        #         processed_out = None
            
        #     print(f"Recording {'started' if thread_controller.enable_processed_frame_recording else 'ended'} {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # elif key == ord('t') or key == ord('T'):  # Toggle ROI display
        #     show_roi = not show_roi
        # elif key == ord('c') or key == ord('C'):  # Clear/Reset ROI
        #     print("ROI Cleared. ROI drawing mode is now enabled. Draw a new ROI.")
        #     reset_roi(source_index)
        #     thread_controller.enable_roi_drawing_mode = True

        def handle_close():
            thread_controller.stop_event.set()
            
            # Wait for all threads to finish, but skip the current thread
            current_thread = threading.current_thread()
            for thread in thread_controller.threads:
                if thread.is_alive() and thread != current_thread:
                    thread.join(timeout=2.0)
            
            # Safer window destruction
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow(window_name)  # Destroy specific window instead of all
            except cv2.error:
                pass  # Window was already closed by user
            
            # Call the provided callback
            if on_close:
                on_close()
            
        # Handle window close or ESC key
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            handle_close()
            break
    
    # Clean up
    if processed_out is not None and processed_out.isOpened():
        processed_out.release()

    thread_controller.video_window_open = False
    print(f"Video processing thread for {source_name} stopped")

def video_processing_dispatcher(mode, source_index, on_close=None):
    if mode == "LINE":
        video_processing_line(source_index, on_close)
    elif mode == "CROWD":
        video_processing_crowd(source_index, on_close)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
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
    cv2.putText(frame, f"Crowd Count: {crowd_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

def display_inframe_count(inframe_count, frame):
    cv2.putText(frame, f"In-frame Count: {inframe_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

def count_to_db(source_name, tid, direction, mode):
    # Record in database
    timestamp = datetime.now().isoformat()
    thread_controller.pending_inserts.put((source_name, tid, direction, timestamp, mode))
    