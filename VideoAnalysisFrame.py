import tkinter as tk
import cv2
import threading
import os
import datetime
import matplotlib.pyplot as plt

from PIL import Image, ImageTk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# existing modules
import config
import thread_manager
import VideoProcessor
import tracker_config

from database_utils import insert_video_analysis
from ReIdentification import ReIdentificationManager
from helpers import *

class VideoAnalysisFrame:
    def __init__(self, root, video_path, on_close=None, ground_truth_count=None, run_index=1, start_recording=False):
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
        
        # Load re-identification settings from config
        reid_settings = config.get_reid_settings()
        
        # Initialize Re-ID Manager with config settings
        self.reid_manager = ReIdentificationManager(
            enable_reidentification=reid_settings.get("enabled", True),
            similarity_threshold=reid_settings.get("similarity_threshold", 0.4),
            reid_model_name=reid_settings.get("model", "osnet_x0_25")
        )

        self.enable_visual = True
        self.enable_recording = start_recording
        self.writer = None
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.total_count = 0
        self.last_tracked_id = None
        self.temp_count = set()
        self.counted_ids     = set()    # track_id → has contributed to total
        self.last_seen       = {}    # track_id -> last frame_idx seen
        self.detection_count = {}    # track_id -> consecutive frames seen
        self.frame_idx       = 0

        self.min_detection = 15
        
        # Mapping from BoTSORT's temporary ID to our persistent ID
        self.tracker_id_map = {}

        # Build UI
        self._build_ui()

        # 1) Reset and start the capture thread(s)
        self.tc.reset()

        self.running = True
        
        # Start recording if enabled
        if self.enable_recording:
            self._start_recording()
        
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

        # Update button text based on initial recording state
        record_text = "Stop Recording" if self.enable_recording else "Start Rec"
        self.btn_record = tk.Button(ctrl, text=record_text, command=self._toggle_record)
        self.btn_record.pack(side=tk.LEFT, padx=2)
        
    def _start_recording(self):
        """Start recording if not already started"""
        if self.enable_recording and self.writer is None:
            self.writer = VideoProcessor.init_video_record(self.video_name, self.video_fps)
            self.enable_recording = True
            self.btn_record.config(text="Stop Recording")
            print("Recording started automatically")
            
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
        
        frame = cv2.resize(frame, (VideoProcessor.FRAME_WIDTH, VideoProcessor.FRAME_HEIGHT))
        results = VideoProcessor.model_video(self.model, frame)  # Your model_video function
        
        current_tracker_ids = set()

        for result in results:
            for box in result.boxes:
                tracker_id = int(box.id.item()) if box.id is not None else None
                if tracker_id is None:
                    continue
                
                current_tracker_ids.add(tracker_id)
                xyxy = box.xyxy[0].tolist()

                # Use the ReIdentificationManager to get persistent ID
                persistent_id = self.reid_manager.process_detection(
                    tracker_id, frame, xyxy, self.frame_idx
                )
                
                # --- Counting and Drawing Logic (using persistent_id) ---
                if persistent_id is not None:
                    seen_this_frame.add(persistent_id)
                    
                    if persistent_id not in self.counted_ids:
                        self.counted_ids.add(persistent_id)
                        self.total_count += 1
                        VideoProcessor.count_to_db(self.video_name, persistent_id, 'enter', 'video')

                    self.temp_count.add(persistent_id)

                    # Draw bounding box with the persistent ID
                    color = VideoProcessor.random_colors[persistent_id % len(VideoProcessor.random_colors)]
                    if self.enable_visual:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {persistent_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                print(f"tracker id: {tracker_id}\npersitent_id: {persistent_id}")

        # Handle lost tracks using the ReIdentificationManager
        self.reid_manager.handle_lost_tracks(current_tracker_ids, self.temp_count)
        
        # Clean up tracks that haven't been seen recently
        cleanup_stale(self.last_seen, self.frame_idx, self.detection_count)

        # Remove the temp count if the person is out of frame for 10 frames
        for track_id in list(self.temp_count):
            if track_id in self.last_seen and (self.frame_idx - self.last_seen[track_id]) > 10:
                self.temp_count.remove(track_id)

        if self.enable_visual:
            VideoProcessor.display_crowd_count(self.total_count, frame)
            VideoProcessor.display_inframe_count(len(self.temp_count), frame)

        # if recording, write the raw processed frame
        if self.enable_recording and self.writer:
            self.writer.write(frame)
            
        self.last_tracked_id = tracker_id if 'tracker_id' in locals() else None
        
        

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
        self.root.after(10, self._update_loop)

    def stop(self):
        print("Stopping video analysis...")
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

        # Get ReID settings from the reid_manager
        with_torchreid = self.reid_manager.enable_reidentification
        reid_model = self.reid_manager.reid_model_name
        similarity_threshold = self.reid_manager.similarity_threshold
        
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
            frame_count=self.frame_idx,
            with_torchreid=with_torchreid,
            reid_model=reid_model,
            similarity_threshold=similarity_threshold
        )
        
        # Stop recording if active
        if self.enable_recording and self.writer:
            self.writer.release()
            self.writer = None
        
        # Signal threads to stop
        self.running = False
        
        # Release video capture
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        # Destroy window
        self.root.destroy()
                    
        # Show statistics window before closing
        self._show_statistics_window(precision, recall, f1_score, avg_processing_time_ms)

        if callable(self.on_close):
            self.on_close()
            
    def _show_statistics_window(self, precision, recall, f1_score, avg_processing_time_ms):
        
        """Show a statistics window with analysis results and charts"""
        stats_window = tk.Toplevel()
        stats_window.title(f"Analysis Results - {self.video_name} (Run #{self.run_index})")
        
        # Get the screen height
        screen_height = stats_window.winfo_screenheight()
        
        stats_window.geometry(f"900x{screen_height}")
        stats_window.grab_set()  # Make it modal
        
        # Create main container with fixed close button at bottom
        main_container = tk.Frame(stats_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics Tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        # Create scrollable frame for statistics
        stats_canvas = tk.Canvas(stats_frame)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=stats_canvas.yview)
        stats_scrollable_frame = ttk.Frame(stats_canvas)
        
        stats_scrollable_frame.bind(
            "<Configure>",
            lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))
        )
        
        stats_canvas.create_window((0, 0), window=stats_scrollable_frame, anchor="nw")
        stats_canvas.configure(yscrollcommand=stats_scrollbar.set)
        
        # Create 2x2 grid layout for statistics frames
        grid_container = tk.Frame(stats_scrollable_frame)
        grid_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Configure grid weights for centered layout
        grid_container.grid_columnconfigure(0, weight=1)
        grid_container.grid_columnconfigure(1, weight=1)
        grid_container.grid_rowconfigure(0, weight=1)
        grid_container.grid_rowconfigure(1, weight=1)
        
        # Basic Statistics
        basic_stats = ttk.LabelFrame(grid_container, text="Basic Statistics", padding=15)
        basic_stats.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        tk.Label(basic_stats, text=f"Video: {self.video_name}", font=("Arial", 10, "bold")).pack(anchor="w", pady=2)
        tk.Label(basic_stats, text=f"Run Index: {self.run_index}").pack(anchor="w", pady=1)
        tk.Label(basic_stats, text=f"Total Objects Counted: {self.total_count}").pack(anchor="w", pady=1)
        tk.Label(basic_stats, text=f"Frames Processed: {self.frame_idx}").pack(anchor="w", pady=1)
        tk.Label(basic_stats, text=f"Video FPS: {self.video_fps:.2f}").pack(anchor="w", pady=1)
        tk.Label(basic_stats, text=f"Processing Time/Frame: {avg_processing_time_ms:.2f} ms").pack(anchor="w", pady=1)
        
        # Duration calculation
        if self.start_timestamp and self.end_timestamp:
            start_time = datetime.datetime.fromisoformat(self.start_timestamp)
            end_time = datetime.datetime.fromisoformat(self.end_timestamp)
            duration = end_time - start_time
            tk.Label(basic_stats, text=f"Analysis Duration: {duration}").pack(anchor="w", pady=1)
            
        # Performance Metrics (Top-Right)
        perf_stats = ttk.LabelFrame(grid_container, text="Performance Metrics", padding=15)
        perf_stats.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Performance Metrics (if ground truth provided)
        if self.ground_truth_count is not None and precision is not None:
            tk.Label(perf_stats, text=f"Ground Truth Count: {self.ground_truth_count}").pack(anchor="w", pady=1)
            tk.Label(perf_stats, text=f"Predicted Count: {self.total_count}").pack(anchor="w", pady=1)
            tk.Label(perf_stats, text=f"Precision: {precision:.3f}").pack(anchor="w", pady=1)
            tk.Label(perf_stats, text=f"Recall: {recall:.3f}").pack(anchor="w", pady=1)
            tk.Label(perf_stats, text=f"F1-Score: {f1_score:.3f}").pack(anchor="w", pady=1)
            
            # Error analysis
            error = abs(self.total_count - self.ground_truth_count)
            error_rate = (error / self.ground_truth_count) * 100 if self.ground_truth_count > 0 else 0
            tk.Label(perf_stats, text=f"Absolute Error: {error}").pack(anchor="w", pady=1)
            tk.Label(perf_stats, text=f"Error Rate: {error_rate:.2f}%").pack(anchor="w", pady=1)
        else:
            tk.Label(perf_stats, text="No ground truth data available", fg="gray", font=("Arial", 9, "italic")).pack(anchor="w", pady=20)
        
        # Technical Details
        tech_stats = ttk.LabelFrame(grid_container, text="Technical Details", padding=15)
        tech_stats.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        tk.Label(tech_stats, text=f"Model: {config.get_model_name()}").pack(anchor="w", pady=1)
        tk.Label(tech_stats, text=f"Confidence Threshold: {config.get_model_conf()}").pack(anchor="w", pady=1)
        tk.Label(tech_stats, text=f"IoU Threshold: {config.get_model_iou()}").pack(anchor="w", pady=1)
        tk.Label(tech_stats, text=f"Min Detection Frames: {self.min_detection}").pack(anchor="w", pady=1)
        tk.Label(tech_stats, text=f"Last Tracked ID: {self.last_tracked_id}").pack(anchor="w", pady=1)
        
        # ReID Details Section
        reid_stats = ttk.LabelFrame(grid_container, text="Re-Identification Settings", padding=15)
        reid_stats.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        reid_enabled_text = "Enabled" if self.reid_manager.enable_reidentification else "Disabled"
        reid_color = "green" if self.reid_manager.enable_reidentification else "red"
        
        reid_label = tk.Label(reid_stats, text=f"TorchReID: {reid_enabled_text}", 
                             fg=reid_color, font=("Arial", 9, "bold"))
        reid_label.pack(anchor="w", pady=2)
        
        if self.reid_manager.enable_reidentification:
            tk.Label(reid_stats, text=f"ReID Model: {self.reid_manager.reid_model_name}").pack(anchor="w", pady=1)
            tk.Label(reid_stats, text=f"Similarity Threshold: {self.reid_manager.similarity_threshold:.3f}").pack(anchor="w", pady=1)
            
            # Additional ReID statistics if available
            active_tracks = len(self.reid_manager.active_track_features)
            lost_tracks = len(self.reid_manager.lost_track_features)
            total_mappings = len(self.reid_manager.tracker_id_map)
            
            tk.Label(reid_stats, text=f"Active Tracks: {active_tracks}").pack(anchor="w", pady=1)
            tk.Label(reid_stats, text=f"Lost Tracks in Memory: {lost_tracks}").pack(anchor="w", pady=1)
            tk.Label(reid_stats, text=f"Current ID Mappings: {total_mappings}").pack(anchor="w", pady=1)
            tk.Label(reid_stats, text=f"Next Person ID: {self.reid_manager.next_person_id}").pack(anchor="w", pady=1)
        else:
            tk.Label(reid_stats, text="ReID Model: N/A (Disabled)", fg="gray").pack(anchor="w", pady=1)
            tk.Label(reid_stats, text="Similarity Threshold: N/A (Disabled)", fg="gray").pack(anchor="w", pady=1)
        
        stats_canvas.pack(side="left", fill="both", expand=True)
        stats_scrollbar.pack(side="right", fill="y")
        
        # Chart Tab (if performance metrics available)
        if self.ground_truth_count is not None and precision is not None:
            chart_frame = ttk.Frame(notebook)
            notebook.add(chart_frame, text="Performance Chart")
            
            # Create matplotlib figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
            fig.suptitle(f'Analysis Results - {self.video_name} (Run #{self.run_index})', fontsize=14, y=0.98)
            
            # Performance metrics bar chart
            metrics = ['Precision', 'Recall', 'F1-Score']
            values = [precision, recall, f1_score]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
            ax1.set_title('Performance Metrics', pad=30)
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1.2)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Count comparison
            counts = ['Ground Truth', 'Predicted']
            count_values = [self.ground_truth_count, self.total_count]
            count_colors = ['#95A5A6', '#E74C3C' if self.total_count != self.ground_truth_count else '#27AE60']
            
            bars2 = ax2.bar(counts, count_values, color=count_colors, alpha=0.7)
            ax2.set_title('Count Comparison', pad=30)
            ax2.set_ylabel('Count')
            
            # Calculate appropriate ylim to prevent overlap
            max_count = max(count_values)
            ax2.set_ylim(0, max_count * 1.2)
            
            # Add value labels
            for bar, value in zip(bars2, count_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                        f'{value}', ha='center', va='bottom', fontsize=10)
            
            # Error analysis pie chart
            if self.ground_truth_count > 0:
                tp = min(self.total_count, self.ground_truth_count)
                fp = max(0, self.total_count - self.ground_truth_count)
                fn = max(0, self.ground_truth_count - self.total_count)
                
                if fp > 0 or fn > 0:
                    error_labels = []
                    error_values = []
                    error_colors = []
                    
                    if tp > 0:
                        error_labels.append(f'Correct ({tp})')
                        error_values.append(tp)
                        error_colors.append('#27AE60')
                    
                    if fp > 0:
                        error_labels.append(f'False Positives ({fp})')
                        error_values.append(fp)
                        error_colors.append('#E74C3C')
                    
                    if fn > 0:
                        error_labels.append(f'False Negatives ({fn})')
                        error_values.append(fn)
                        error_colors.append('#F39C12')
                    
                    ax3.pie(error_values, labels=error_labels, colors=error_colors, autopct='%1.1f%%')
                    ax3.set_title('Error Analysis', pad=20)
                else:
                    ax3.text(0.5, 0.5, 'Perfect\nPrediction!', ha='center', va='center', 
                            transform=ax3.transAxes, fontsize=16, color='green', weight='bold')
                    ax3.set_title('Error Analysis', pad=20)
            
            # Processing time visualization
            if self.frame_idx > 0:
                ax4.axhline(y=avg_processing_time_ms, color='#3498DB', linestyle='-', linewidth=2, 
                           label=f'Avg: {avg_processing_time_ms:.1f}ms')
                ax4.axhline(y=1000/30, color='#E74C3C', linestyle='--', linewidth=2, 
                           label='30 FPS Target (33.3ms)')
                ax4.fill_between([0, 1], 0, avg_processing_time_ms, alpha=0.3, color='#3498DB')
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, max(avg_processing_time_ms * 1.5, 50))
                ax4.set_title('Processing Performance', pad=20)
                ax4.set_ylabel('Time (ms)')
                ax4.legend()
                ax4.set_xticks([])
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlapping
            
            # Embed plot in tkinter
            canvas = FigureCanvasTkAgg(fig, chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Fixed Close button at the bottom (always visible)
        close_frame = tk.Frame(main_container)
        close_frame.pack(fill=tk.X, pady=(5, 0))
        
        def close_stats():
            self.tc.stop_event.set()
            # Join all threads
            print("Waiting for threads to finish...")
            current = threading.current_thread()
            for t in self.tc.threads:
                if t.is_alive() and t is not current:
                    print(f"Joining thread: {t.name}")
                    t.join(timeout=1.0)
                    if t.is_alive():
                        print(f"Warning: Thread {t.name} did not stop gracefully")
                        
            plt.close('all')  # Clean up matplotlib figures
            stats_window.destroy()
            
        # Create a separator line above the button
        separator = tk.Frame(close_frame, height=1, bg="lightgray")
        separator.pack(fill=tk.X, pady=(5, 10))
        
        tk.Button(close_frame, text="Close", command=close_stats, width=20, 
             font=("Arial", 10), relief="raised", bd=2).pack()
        
        stats_window.protocol("WM_DELETE_WINDOW", close_stats)
        
        # Enable mouse wheel scrolling for the canvas
        def on_mousewheel(event):
            stats_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        stats_canvas.bind("<MouseWheel>", on_mousewheel)  # Windows
        stats_canvas.bind("<Button-4>", lambda e: stats_canvas.yview_scroll(-1, "units"))  # Linux
        stats_canvas.bind("<Button-5>", lambda e: stats_canvas.yview_scroll(1, "units"))   # Linux
        
        # Wait for window to close before continuing
        stats_window.wait_window()
            
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
            print("Recording stopped", self.video_name)
        