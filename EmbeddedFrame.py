import tkinter as tk
import cv2
import threading
import time

from PIL import Image, ImageTk

# existing modules
import config
import thread_manager
import VideoProcessor

from helpers import *

class EmbeddedFrame:
    def __init__(self, root, source_index, mode="CROWD", on_close=None):
        self.on_close = on_close
        self.root = root
        self.mode = mode

        self.source_index = source_index
        self.source_name = f"camera_{source_index}"
        self.tc = thread_manager.thread_controller

        self.zc = self.tc.zoom_controllers[source_index]
        
        #ROI State
        self.current_roi_name = None
        self.roi_points = None
        self.custom_roi_set = False
        self.roi_window_open = False # Track if ROI window is open
        
        # Line Crossing State (for LINE mode)
        self.crossing_line = None
        self.line_orientation = "vertical"  # "horizontal" or "vertical"
        self.line_y = VideoProcessor.FRAME_HEIGHT // 2  # Default line in middle
        self.line_x = VideoProcessor.FRAME_WIDTH // 2   # Default vertical line in middle
        self.track_positions = {}  # track_id -> list of recent positions
        self.position_history_size = 5
        
        # Load model
        self.model = load_model(config.get_model_name())

        # Visual flags
        self.enable_visual = True
        self.enable_recording = False
        self.enable_roi = False
        self.writer = None
        
        # Counting states
        self.temp_count = set()
        self.counted_ids     = set()    # track_id → has contributed to total
        self.last_seen       = {}    # track_id -> last frame_idx seen
        self.detection_count = {}    # track_id -> consecutive frames seen
        self.frame_idx       = 0
        
        # Line crossing specific counters
        self.entries = 0
        self.exits = 0
        self.crossed_ids = set()  # Track IDs that have already crossed to prevent double counting


        self.min_detection = 20

        # Build UI
        self._build_ui()

        # 1) Reset and start the capture thread(s)
        self.tc.reset()
        self._start_capture_threads()

        self.running = True
        self._update_loop()

    def _build_ui(self):
        # This parent is expected to be a Toplevel
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        # Video display
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Controls
        ctrl = tk.Frame(self.root)
        ctrl.pack(fill=tk.X, pady=5)

        self.btn_stop = tk.Button(ctrl, text="Close", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, pady=5)

        self.btn_visual = tk.Button(ctrl, text="Toggle Visual", command=self._toggle_visual)
        self.btn_visual.pack(side=tk.LEFT, padx=2)

        self.btn_zoom_in = tk.Button(ctrl, text="Zoom +", command=self._toggle_zoom_in)
        self.btn_zoom_in.pack(side=tk.LEFT, pady=5)

        self.btn_zoom_out = tk.Button(ctrl, text="Zoom -", command=self._toggle_zoom_out)
        self.btn_zoom_out.pack(side=tk.LEFT, pady=5)

        self.btn_reset_zoom = tk.Button(ctrl, text="Reset Zoom", command=self._toggle_reset_zoom)
        self.btn_reset_zoom.pack(side=tk.LEFT, pady=5)

        self.btn_record = tk.Button(ctrl, text="Start Rec", command=self._toggle_record)
        self.btn_record.pack(side=tk.LEFT, padx=2)
        
        # Mode-specific controls
        if self.mode == "CROWD":
            # ROI controls for crowd mode
            self.btn_select_roi = tk.Button(ctrl, text="Select ROI", command=self._open_select_roi)
            self.btn_select_roi.pack(side=tk.LEFT, padx=2)
            
            self.btn_clear_roi = tk.Button(ctrl, text="Clear ROI", command=self._clear_roi)
            self.btn_clear_roi.config(state=tk.DISABLED)
            self.btn_clear_roi.pack(side=tk.LEFT, padx=2)
            
            self.btn_new_roi = tk.Button(ctrl, text="Create New ROI", command=self._create_new_roi)
            self.btn_new_roi.pack(side=tk.LEFT, padx=2)
        
        elif self.mode == "LINE":
            # Line orientation toggle
            self.btn_toggle_orientation = tk.Button(ctrl, text="Toggle Orientation", command=self._toggle_line_orientation)
            self.btn_toggle_orientation.pack(side=tk.LEFT, padx=2)
            
            # Line adjustment controls for line crossing mode
            self.btn_adjust_line = tk.Button(ctrl, text="Adjust Line", command=self._adjust_crossing_line)
            self.btn_adjust_line.pack(side=tk.LEFT, padx=2)

    def _start_capture_threads(self):
        # Start capture_frames for each defined source
        for i, src in enumerate(VideoProcessor.CAMERA_SOURCES):
            if src is not None:
                t = threading.Thread(
                    target=VideoProcessor.capture_frames,
                    args=(i,),
                    daemon=True
                )
                t.start()
                self.tc.threads.append(t)
                
    def _update_loop(self):        
        frame_rate_buffer = []
        avg_frame_rate = 0
        t_start = time.perf_counter()

        # window_name = f"People Counter - {source_name} - CROWD MODE"

        if not self.running: return

        frame = None
        try:
            frame = self.tc.frame_queue[self.source_index].get_nowait()
        except Exception:
            pass

        self.frame_idx += 1
        seen_this_frame = set()

        if frame is not None:
            frame = self.zc.apply_zoom(frame)

            frame = cv2.resize(frame, (VideoProcessor.FRAME_WIDTH, VideoProcessor.FRAME_HEIGHT))
            
            # YOLO inference
            results = VideoProcessor.model_frame(self.model, frame)
            
            visual_frame = frame.copy()
            
            # Draw mode-specific overlays
            if self.mode == "CROWD":
                if self.enable_roi and self.enable_visual and self.roi_points is not None:
                    cv2.polylines(visual_frame, [self.roi_points], True, (0, 255, 0), 2)
            elif self.mode == "LINE":
                if self.enable_visual:
                    self._draw_crossing_line(visual_frame)
            
            for result in results:

                for box in result.boxes:
                    
                    tid = int(box.id.item()) if box.id is not None else None

                    if tid is None:
                        continue
                    
                    coords = box.xyxy[0].tolist()
                    
                    # Mode-specific processing
                    if self.mode == "CROWD":
                        self._process_crowd_detection(tid, coords, seen_this_frame, visual_frame, box)
                    elif self.mode == "LINE":
                        self._process_line_crossing(tid, coords, seen_this_frame, visual_frame, box)
                        
            # Clean up tracks that haven't been seen recently
            cleanup_stale(self.last_seen, self.frame_idx, self.detection_count)

            # Remove the temp count if the person is out of frame for 10 frames
            
            for track_id in list(self.temp_count):
                if self.frame_idx - self.last_seen[track_id] > 10:
                    self.temp_count.remove(track_id)

            # Calculate and display FPS
            avg_frame_rate = calculate_fps(frame_rate_buffer, t_start)

            if self.enable_visual:
                VideoProcessor.display_fps(avg_frame_rate, visual_frame)
                
                if self.mode == "CROWD":
                    VideoProcessor.display_crowd_count(VideoProcessor.crowd_count[self.source_index], visual_frame)
                    VideoProcessor.display_inframe_count(len(self.temp_count), visual_frame)
                elif self.mode == "LINE":
                    self._display_line_crossing_counts(visual_frame)

            # if recording, write the raw processed frame
            if self.enable_recording and self.writer:
                self.writer.write(visual_frame)

            # Convert to Tk image and display
            rgb = cv2.cvtColor(visual_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        # Schedule next frame update
        self.root.after(30, self._update_loop)
        
    
        
    def _process_crowd_detection(self, tid, coords, seen_this_frame, visual_frame, box):
        """Process detection for crowd monitoring mode"""
        # ROI filtering for crowd mode
        if self.enable_roi and self.roi_points is not None:
            if not VideoProcessor.is_in_roi(coords, self.roi_points):
                return  # Skip detections outside ROI
                
        self.last_seen[tid] = self.frame_idx
        seen_this_frame.add(tid)
        
        # Increment detection counter for this ID
        self.detection_count[tid] = self.detection_count.get(tid, 0) + 1
        
        if tid not in self.counted_ids and self.detection_count[tid] >= self.min_detection:
            self.counted_ids.add(tid)
            VideoProcessor.crowd_count[self.source_index] += 1
            VideoProcessor.count_to_db(self.source_name, tid, 'enter', 'crowd')

        # Add to temp count if met the min detection threshold
        if self.detection_count[tid] >= self.min_detection:
            self.temp_count.add(tid)

        # Draw bounding box and ID
        self._draw_detection(visual_frame, box, tid)
        
    def _draw_crossing_line(self, visual_frame):
        """Draw the crossing line based on current orientation"""
        if self.line_orientation == "horizontal":
            # Draw horizontal line
            cv2.line(visual_frame, (0, self.line_y), (VideoProcessor.FRAME_WIDTH, self.line_y), (0, 255, 255), 3)
            cv2.putText(visual_frame, "CROSSING LINE (HORIZONTAL)", (10, self.line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:  # vertical
            # Draw vertical line
            cv2.line(visual_frame, (self.line_x, 0), (self.line_x, VideoProcessor.FRAME_HEIGHT), (0, 255, 255), 3)
            cv2.putText(visual_frame, "CROSSING LINE (VERTICAL)", (self.line_x + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        
    def _process_line_crossing(self, tid, coords, seen_this_frame, visual_frame, box):
        """Process detection for line crossing mode"""
        self.last_seen[tid] = self.frame_idx
        seen_this_frame.add(tid)
        
        # Get center bottom point of bounding box (person's feet)
        x1, y1, x2, y2 = coords
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)  # Bottom of bounding box
        
        # Store position history
        if tid not in self.track_positions:
            self.track_positions[tid] = []
        
        self.track_positions[tid].append((center_x, center_y, self.frame_idx))
        
        # Keep only recent positions
        if len(self.track_positions[tid]) > self.position_history_size:
            self.track_positions[tid].pop(0)
        
        # Check for line crossing
        if len(self.track_positions[tid]) >= 2 and tid not in self.crossed_ids:
            self._check_line_crossing(tid)
        
        # Draw bounding box and ID
        self._draw_detection(visual_frame, box, tid)
        
        # Draw trajectory
        if self.enable_visual and len(self.track_positions[tid]) > 1:
            points = [(pos[0], pos[1]) for pos in self.track_positions[tid]]
            for i in range(len(points) - 1):
                cv2.line(visual_frame, points[i], points[i+1], (255, 0, 255), 2)
                
    def _check_line_crossing(self, tid):
        """Check if a track has crossed the line and determine direction"""
        positions = self.track_positions[tid]
        
        # Need at least 2 positions to determine crossing
        if len(positions) < 2:
            return
            
        # Check if the track crossed the line
        prev_pos = positions[-2]
        curr_pos = positions[-1]
        
        if self.line_orientation == "horizontal":
            prev_coord = prev_pos[1]  # Y coordinate
            curr_coord = curr_pos[1]  # Y coordinate
            line_coord = self.line_y
            
            # Check if horizontal line was crossed
            if (prev_coord < line_coord < curr_coord) or (prev_coord > line_coord > curr_coord):
                if prev_coord < line_coord and curr_coord > line_coord:
                    # Crossed from top to bottom (entering)
                    direction = 'enter'
                    self.entries += 1
                else:
                    # Crossed from bottom to top (exiting)
                    direction = 'exit'  
                    self.exits += 1
                
                self._handle_crossing(tid, direction)
                
        else:  # vertical orientation
            prev_coord = prev_pos[0]  # X coordinate
            curr_coord = curr_pos[0]  # X coordinate
            line_coord = self.line_x
            
            # Check if vertical line was crossed
            if (prev_coord < line_coord < curr_coord) or (prev_coord > line_coord > curr_coord):
                if prev_coord < line_coord and curr_coord > line_coord:
                    # Crossed from left to right (entering)
                    direction = 'enter'
                    self.entries += 1
                else:
                    # Crossed from right to left (exiting)
                    direction = 'exit'
                    self.exits += 1
                
                self._handle_crossing(tid, direction)
                
    def _handle_crossing(self, tid, direction):
        """Handle the crossing event"""
        # Mark as crossed to prevent double counting
        self.crossed_ids.add(tid)
        
        # Log to database
        VideoProcessor.count_to_db(self.source_name, tid, direction, 'line')
        
        orientation_str = self.line_orientation.upper()
        print(f"Track {tid} crossed {orientation_str} line: {direction} (Entries: {self.entries}, Exits: {self.exits})")


    def _draw_detection(self, visual_frame, box, tid):
        """Draw bounding box and ID for a detection"""
        if not self.enable_visual:
            return
            
        color = VideoProcessor.random_colors[tid % len(VideoProcessor.random_colors)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(visual_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(visual_frame, f"ID: {tid}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _display_line_crossing_counts(self, frame):
        """Display line crossing statistics on frame"""
        # Current count (entries - exits)
        current_count = self.entries - self.exits
        
        # Display orientation
        orientation_text = f"Mode: {self.line_orientation.upper()}"
        cv2.putText(frame, orientation_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        
        # Display statistics
        cv2.putText(frame, f"Entries: {self.entries}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {self.exits}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Current: {current_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    def _toggle_line_orientation(self):
        """Toggle between horizontal and vertical line orientation"""
        if self.line_orientation == "horizontal":
            self.line_orientation = "vertical"
        else:
            self.line_orientation = "horizontal"
        
        print(f"Line orientation changed to: {self.line_orientation}")


    def stop(self):
        # Signal threads to stop
        self.running = False
        self.tc.stop_event.set()

        # Join all threads
        current = threading.current_thread()
        for t in self.tc.threads:
            if t.is_alive() and t is not current:
                t.join(timeout=1.0)

        # Destroy window
        self.root.destroy()

        if callable(self.on_close):
            self.on_close()

    def _toggle_visual(self):
        self.enable_visual = not self.enable_visual

    def _toggle_zoom_in(self):
        self.zc.increase_zoom()
    
    def _toggle_zoom_out(self):
        self.zc.decrease_zoom()

    def _toggle_reset_zoom(self):
        self.zc.reset_zoom()
    
    def _toggle_record(self):
        self.enable_recording = not self.enable_recording
        # Write processed frame to video
        if self.enable_recording:
            self.writer = VideoProcessor.init_record(self.source_name)
            self.btn_record.config(text="Stop Recording")
        else:
            # Stop recording
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.btn_record.config(text="Record")
            print("Recording stopped", self.source_name)
            
    def _adjust_crossing_line(self):
        """Allow user to adjust the crossing line position"""
        # Create adjustment window
        adjust_win = tk.Toplevel(self.root)
        adjust_win.title("Adjust Crossing Line")
        adjust_win.geometry("400x200")
        adjust_win.transient(self.root)
        
        tk.Label(adjust_win, text=f"Adjust {self.line_orientation} line position:").pack(pady=10)
        
        if self.line_orientation == "horizontal":
            # Adjust Y coordinate for horizontal line
            line_var = tk.IntVar(value=self.line_y)
            scale = tk.Scale(adjust_win, from_=50, to=VideoProcessor.FRAME_HEIGHT-50, 
                            orient=tk.HORIZONTAL, variable=line_var, length=350,
                            label="Y Position")
            scale.pack(pady=10)
            
            def apply_changes():
                self.line_y = line_var.get()
                adjust_win.destroy()
                
        else:  # vertical
            # Adjust X coordinate for vertical line
            line_var = tk.IntVar(value=self.line_x)
            scale = tk.Scale(adjust_win, from_=50, to=VideoProcessor.FRAME_WIDTH-50, 
                            orient=tk.HORIZONTAL, variable=line_var, length=350,
                            label="X Position")
            scale.pack(pady=10)
            
            def apply_changes():
                self.line_x = line_var.get()
                adjust_win.destroy()
        
        tk.Button(adjust_win, text="Apply", command=apply_changes).pack(pady=10)

    def _open_select_roi(self):
        """Open ROI selection window (crowd mode only)"""
        if self.mode != "CROWD":
            return
        
        roi_defs = config.get_roi_values()
        names = list(roi_defs.keys())
        
        if not names:
            tk.messagebox.showwarning("No ROIs", "No saved ROIs found. Create a new ROI first.")
            return
        
        win = tk.Toplevel(self.root)
        win.title("Select ROI")
        win.geometry("200x150")
        win.transient(self.root)
        win.grab_set()
        
        # Set window open flag and disable button
        self.roi_window_open = True
        self.btn_select_roi.config(state=tk.DISABLED)

        tk.Label(win, text="Choose ROI:").pack()
        var = tk.StringVar(value=self.current_roi_name or names[0])
        tk.OptionMenu(win, var, *names).pack(padx=10, pady=10)
        
        def on_window_close():
            # Re-enable button and reset flag when window closes
            self.roi_window_open = False
            self.btn_select_roi.config(state=tk.NORMAL)
            win.destroy()
        
        def apply():
            name = var.get()
            coords = roi_defs[name]['value']
            arr = np.array(coords, np.int32).reshape((-1,1,2))
            self.roi_points = arr
            self.current_roi_name = name; 
            self.custom_roi_set = False
            self.enable_roi = True
            self.btn_clear_roi.config(state=tk.NORMAL)
            on_window_close()
            
        tk.Button(win, text="Apply", command=apply).pack()
        
        win.protocol("WM_DELETE_WINDOW", on_window_close)
        
    def _clear_roi(self):
        """Clear ROI (crowd mode only)"""
        if self.mode != "CROWD":
            return
        
        VideoProcessor.reset_roi(self.source_index)
        self.current_roi_name = None
        self.enable_roi = False
        self.btn_clear_roi.config(state=tk.DISABLED)
        self.btn_new_roi.config(state=tk.NORMAL)
        
    def _create_new_roi(self):
        """Create a new ROI by drawing on the current frame"""
        
        if self.mode != "CROWD":
            return
        
        # Reset any existing temp ROI points
        VideoProcessor.temp_roi = []
        VideoProcessor.roi_points = []
        VideoProcessor.drawing = False
        VideoProcessor.current_mouse_pos = None
        
        # Launch a cv2 window for drawing
        win_name = f"ROI Draw Cam {self.source_index}"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, VideoProcessor.draw_roi, self.source_index)
        
        self.btn_new_roi.config(state=tk.DISABLED)
        
        # Instructions
        print(f"Drawing ROI for Camera {self.source_index}")
        print("Click to add points, press ENTER when finished, ESC to cancel, C to clear points")
        
        roi_creation_active = True
        
        try:
            while roi_creation_active:
                # Get current frame
                try:
                    frame = self.tc.frame_queue[self.source_index].get(timeout=0.1)
                except:
                    # If no frame available, use a black frame as fallback
                    frame = np.zeros((VideoProcessor.FRAME_HEIGHT, VideoProcessor.FRAME_WIDTH, 3), dtype=np.uint8)
                
                # Apply zoom if needed
                if hasattr(self, 'zc'):
                    frame = self.zc.apply_zoom(frame)
                frame = cv2.resize(frame, (VideoProcessor.FRAME_WIDTH, VideoProcessor.FRAME_HEIGHT))
                
                # Draw ROI overlay
                disp = VideoProcessor.draw_roi_overlay(
                    frame.copy(), 
                    VideoProcessor.temp_roi, 
                    VideoProcessor.drawing, 
                    VideoProcessor.current_mouse_pos
                )
                
                cv2.imshow(win_name, disp)
                
                # Check if window was closed by user (X button)
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("ROI creation window closed by user")
                    roi_creation_active = False
                    break
                
                key = cv2.waitKey(1) & 0xFF
                
                # Check for key presses
                if key == 13:  # ENTER key
                    if len(VideoProcessor.temp_roi) >= 3:  # Need at least 3 points for a polygon
                        roi_creation_active = False
                        cv2.destroyWindow(win_name)
                        self._save_new_roi()
                        return
                    else:
                        tk.messagebox.showwarning("Insufficient ROIs", "Need at least 3 points to create ROI. Continue drawing...")
                        print("Need at least 3 points to create ROI. Continue drawing...")
                elif key == 27 or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:  # ESC key
                    print("ROI creation cancelled")
                    roi_creation_active = False
                    break
                elif key == ord('c') or key == ord('C'):  # Clear current drawing
                    VideoProcessor.temp_roi = []
                    VideoProcessor.roi_points = []  # Also clear this to sync
                    VideoProcessor.drawing = False
                    VideoProcessor.current_mouse_pos = None
                    print("ROI points cleared. Start drawing again...")
                    
        except Exception as e:
            print(f"Error during ROI creation: {e}")
        finally:
            # Cleanup
            try:
                cv2.destroyWindow(win_name)
            except:
                pass
            # VideoProcessor.temp_roi = []
            VideoProcessor.roi_points = []
            VideoProcessor.drawing = False
            self.btn_new_roi.config(state=tk.NORMAL)
            
    def _save_new_roi(self):
        """Pop up window to name and save the new ROI"""
        if not VideoProcessor.temp_roi:
            print("No ROI points to save")
            self.btn_new_roi.config(state=tk.NORMAL)    
            return
        
        # Create naming window
        name_window = tk.Toplevel(self.root)
        name_window.title("Save New ROI")
        name_window.geometry("300x150")
        name_window.transient(self.root)
        name_window.grab_set()
        
        # Center the window
        name_window.update_idletasks()
        x = (name_window.winfo_screenwidth() // 2) - (name_window.winfo_width() // 2)
        y = (name_window.winfo_screenheight() // 2) - (name_window.winfo_height() // 2)
        name_window.geometry(f"+{x}+{y}")
        
        # UI elements
        tk.Label(name_window, text="Enter a unique name for this ROI:", font=("Arial", 10)).pack(pady=10)
        
        name_var = tk.StringVar()
        name_entry = tk.Entry(name_window, textvariable=name_var, width=30)
        name_entry.pack(pady=5)
        name_entry.focus_set()
        
        error_label = tk.Label(name_window, text="", fg="red", font=("Arial", 8))
        error_label.pack()
        
        def save_roi():
            roi_name = name_var.get().strip()
            
            # Validation
            if not roi_name:
                error_label.config(text="Please enter a name")
                return
            
            # Check for duplicate names
            existing_rois = config.get_roi_values().keys()
            if roi_name in existing_rois:
                error_label.config(text="Name already exists. Please choose a different name.")
                return
            
            # Convert temp_roi points to the format expected by config
            roi_coords = [[int(point[0]), int(point[1])] for point in VideoProcessor.temp_roi]
            
            roi_data = {
                "width": VideoProcessor.FRAME_WIDTH,
                "height": VideoProcessor.FRAME_HEIGHT,
                "value": roi_coords
            }
            
            try:
                # Save to config
                config.set_roi_values(roi_name, roi_data)
                
                # Apply the new ROI immediately
                self.roi_points = np.array(roi_coords, np.int32).reshape((-1, 1, 2))
                self.current_roi_name = roi_name
                self.custom_roi_set = True
                self.enable_roi = True
                self.btn_clear_roi.config(state=tk.NORMAL)
                
                print(f"ROI '{roi_name}' saved successfully")
                
                # Clear temp ROI
                VideoProcessor.temp_roi = []
                VideoProcessor.roi_points = []
                
                # Close the naming window
                name_window.destroy()
                self.btn_new_roi.config(state=tk.NORMAL)
                
            except Exception as e:
                error_label.config(text=f"Error saving ROI: {str(e)}")
                print(f"Error saving ROI: {e}")
                
        
        def cancel_save():
            VideoProcessor.temp_roi = []
            VideoProcessor.roi_points = []
            self.btn_new_roi.config(state=tk.NORMAL)
            name_window.destroy()
        
        name_entry.bind('<Return>', lambda e: save_roi())
        
        button_frame = tk.Frame(name_window)
        button_frame.pack(pady=10)
        
        # Buttons
        tk.Button(button_frame, text="Save", command=save_roi, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_save, width=10).pack(side=tk.LEFT, padx=5)
        
        # Handle window close
        name_window.protocol("WM_DELETE_WINDOW", cancel_save)
    
    