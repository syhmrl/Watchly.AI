import tkinter as tk
import cv2
import threading
import time

from PIL import Image, ImageTk

# Your existing modules
import config
import thread_manager
import VideoProcessor

from helpers import *

class EmbeddedFrame:
    def __init__(self, root, source_index, on_close=None):
        self.on_close = on_close
        self.root = root

        self.source_index = source_index
        self.source_name = f"camera_{source_index}"
        self.tc = thread_manager.thread_controller

        self.zc = self.tc.zoom_controllers[source_index]

        self.model = load_model(config.get_model_name())

        self.enable_visual = True
        self.enable_recording = False
        self.writer = None
        
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

        if not self.running:
            return

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
                        VideoProcessor.crowd_count[self.source_index] += 1

                        VideoProcessor.count_to_db(self.source_name, tid, 'enter', 'crowd')

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

            # Calculate and display FPS
            avg_frame_rate = calculate_fps(frame_rate_buffer, t_start)

            if self.enable_visual:
                VideoProcessor.display_fps(avg_frame_rate, frame)
                VideoProcessor.display_crowd_count(VideoProcessor.crowd_count[self.source_index], frame)
                VideoProcessor.display_inframe_count(len(self.temp_count), frame)

            # if recording, write the raw processed frame
            if self.enable_recording and self.writer:
                self.writer.write(frame)

            # Convert to Tk image and display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        # Schedule next frame update
        self.root.after(30, self._update_loop)

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
        