import tkinter as tk
import cv2
import threading
import os
import datetime

from PIL import Image, ImageTk

# Your existing modules
import config
import thread_manager
import VideoProcessor
import tracker_config

from database_utils import insert_video_analysis
from helpers import *

class VideoAnalysisFrame:
    def __init__(self, root, video_path, on_close=None, ground_truth_count=None, run_index=1):
        self.on_close = on_close
        self.root = root
        
        self.video_path = os.path.join('.', "video", video_path)
        self.video_name = video_path
        
        # Performance tracking variables
        self.ground_truth_count = ground_truth_count  # Expected count for precision/recall
        self.run_index = run_index
        self.processing_start_time = None
        self.total_processing_time_ms = 0
        self.frame_count = 0
        
        self.start_timestamp = datetime.datetime.now().isoformat()
        self.end_timestamp = None
        
        
        self.tc = thread_manager.thread_controller
        
        # Open video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            tk.messagebox.showerror("Error", f"Cannot open video: {self.video_path}")
            self.root.destroy()
            if self.on_close:
                self.on_close()
            return

        self.model = load_model(config.get_model_name())

        self.enable_visual = True
        self.enable_recording = False
        self.writer = None
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.total_count = 0
        self.last_tracked_id = None
        self.temp_count = set()
        self.counted_ids     = set()    # track_id → has contributed to total
        self.last_seen       = {}    # track_id -> last frame_idx seen
        self.detection_count = {}    # track_id -> consecutive frames seen
        self.frame_idx       = 0

        self.min_detection = 30

        # Build UI
        self._build_ui()

        # 1) Reset and start the capture thread(s)
        self.tc.reset()

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

        self.btn_record = tk.Button(ctrl, text="Start Rec", command=self._toggle_record)

        self.btn_record.pack(side=tk.LEFT, padx=2)

    def _update_loop(self):        
        if not self.running:
            return
        
        # Start timing for this frame
        frame_start_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        self.frame_idx += 1
        seen_this_frame = set()
        tid = None
           
        frame = cv2.resize(frame, (VideoProcessor.FRAME_WIDTH, VideoProcessor.FRAME_HEIGHT))

        # YOLO inference
        results = VideoProcessor.model_video(self.model, frame)
                
        for result in results:

            for box in result.boxes:
                
                tid = int(box.id.item()) if box.id is not None else None

                if tid is None:
                    continue
                
                self.last_seen[tid] = self.frame_idx
                seen_this_frame.add(tid)
                
                # Increment detection counter for this ID
                self.detection_count[tid] = self.detection_count.get(tid, 0) + 1
                
                if  tid not in self.counted_ids and self.detection_count[tid] >= self.min_detection:
                    self.counted_ids.add(tid)
                    self.total_count += 1

                    VideoProcessor.count_to_db(self.video_name, tid, 'enter', 'video')

                # Add met the min detection threshold
                if self.detection_count[tid] >= self.min_detection:
                    self.temp_count.add(tid)

                # Draw bounding box and ID
                color = VideoProcessor.random_colors[tid % len(VideoProcessor.random_colors)]
                if self.enable_visual:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Clean up tracks that haven't been seen recently
        cleanup_stale(self.last_seen, self.frame_idx, self.detection_count)

        # Remove the temp count if the person is out of frame for 20 frames
        for track_id in list(self.temp_count):
            if self.frame_idx - self.last_seen[track_id] > 20:
                self.temp_count.remove(track_id)

        if self.enable_visual:
            VideoProcessor.display_crowd_count(self.total_count, frame)
            VideoProcessor.display_inframe_count(len(self.temp_count), frame)

        # if recording, write the raw processed frame
        if self.enable_recording and self.writer:
            self.writer.write(frame)
            
        self.last_tracked_id = tid

        # Convert to Tk image and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        
        # Calculate processing time for this frame
        frame_end_time = time.time()
        frame_processing_time_ms = (frame_end_time - frame_start_time) * 1000
        self.total_processing_time_ms += frame_processing_time_ms

        # Schedule next frame update
        self.root.after(30, self._update_loop)

    def stop(self):
        # Store end timestamp when video analysis finishes
        self.end_timestamp = datetime.datetime.now().isoformat()
        
        # Calculate performance metrics
        precision, recall, f1_score = self._calculate_performance_metrics(
            self.total_count, 
            self.ground_truth_count
        )
        
        # Calculate average processing time per frame
        avg_processing_time_ms = (
            self.total_processing_time_ms / self.frame_idx 
            if self.frame_idx > 0 else 0
        )
        
        # Gather video and tracker parameters
        w, h = VideoProcessor.FRAME_SIZE
        fps  = self.video_fps
        ts   = tracker_config._load_yaml()  # or get_tracker_settings()

        insert_video_analysis(
            video_name=self.video_name,
            video_width=w,
            video_height=h,
            video_fps=fps,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            total_count=self.total_count,
            model_name=config.get_model_name(),
            confidence=config.get_model_conf(),
            iou=config.get_model_iou(),
            last_tracked_id=self.last_tracked_id,
            tracker_settings=ts,
            run_index=self.run_index,
            ground_truth_count=self.ground_truth_count,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            processing_time_ms=avg_processing_time_ms,
            frame_count=self.frame_count
        )
        
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
            
    def _calculate_performance_metrics(self, predicted_count, ground_truth_count):
        """
        Calculate precision, recall, and F1-score based on counting accuracy.
        
        For object counting, we can treat it as:
        - True Positives (TP): min(predicted, ground_truth)
        - False Positives (FP): max(0, predicted - ground_truth)
        - False Negatives (FN): max(0, ground_truth - predicted)
        """
        if ground_truth_count is None or ground_truth_count == 0:
            return None, None, None
        
        tp = min(predicted_count, ground_truth_count)
        fp = max(0, predicted_count - ground_truth_count)
        fn = max(0, ground_truth_count - predicted_count)
        
        # Calculate precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Calculate recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate F1-score: 2 * (precision * recall) / (precision + recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score

    def _toggle_visual(self):
        self.enable_visual = not self.enable_visual
    
    def _toggle_record(self):
        self.enable_recording = not self.enable_recording
        # Write processed frame to video
        if self.enable_recording:
            self.writer = VideoProcessor.init_video_record(self.video_name, self.video_fps)
            self.btn_record.config(text="Stop Recording")
        else:
            # Stop recording
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.btn_record.config(text="Record")
            print("Recording stopped", self.video_path)
        