import os
import random
import cv2
import time
import threading
import queue
import tkinter as tk
from tkinter import messagebox
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime, date, timedelta
import torch
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# Camera Settings
CAM_USERNAME = "admin"
CAM_PASSWORD = "Abcdefghi1"
CAM_IP = "192.168.1.64"
RTSP_URL = f"rtsp://{CAM_USERNAME}:{CAM_PASSWORD}@{CAM_IP}:554/Streaming/Channels/101?transport=tcp"

# Source of the Video/Stream
VIDEO_SOURCE = 0

# Model used
MODEL_NAME = "yolo11s.pt"

# Setup frame size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

# Setup the line coordinate for crossing, enter and exit count
line_x = FRAME_WIDTH // 2  # Vertical line for counting
enter_count = 0
exit_count = 0

# Setup framerate variable
fps_avg_len = 200

# Check for GPU availability and load model on GPU if available
if torch.cuda.is_available():
    model = YOLO(MODEL_NAME).to('cuda')
    print("Using GPU")
else:
    model = YOLO(MODEL_NAME)
    print("Using CPU")

# Thread control
class ThreadControl:
    def __init__(self):
        self.stop_event = threading.Event()
        self.threads = []
        self.frame_queue = queue.Queue(maxsize=1)
        self.pending_inserts = []
        self.video_window_open = False
        self.reset()
    
    def reset(self):
        """Reset all thread states and counters"""
        self.stop_event.clear()
        self.frame_queue = queue.Queue(maxsize=1)
        self.pending_inserts = []
        self.threads = []
        self.video_window_open = False

# Global thread controller
thread_controller = ThreadControl()

# Database Connection
class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._connection = None
            cls._instance._cursor = None
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        self._connection = sqlite3.connect('watchly_ai.db', check_same_thread=False)
        self._cursor = self._connection.cursor()
        
        # Initialize schema
        self._cursor.execute('''
            CREATE TABLE IF NOT EXISTS crossing_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                source    TEXT,
                track_id  INTEGER,
                direction TEXT,
                timestamp TEXT
            )
        ''')
        self._connection.commit()
    
    def get_connection(self):
        return self._connection, self._cursor
    
    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            self._cursor = None
            Database._instance = None

# Frame capture function
def capture_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
    
    # Setup video writer if recording is enabled
    output_filename = f"video/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 15.0, FRAME_SIZE)
    
    while not thread_controller.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from camera")
            time.sleep(1)  # Wait before retrying
            # Try to reconnect
            cap.release()
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            continue
            
        # Save frame to video if recording
        out.write(frame)
        
        try:
            # Put frame in queue, replace if full
            if thread_controller.frame_queue.full():
                try:
                    thread_controller.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            thread_controller.frame_queue.put(frame, block=False)
        except queue.Full:
            pass  # Skip frame if queue is full
    
    # Clean up resources
    cap.release()
    out.release()
    print("Frame capture thread stopped")

# Database insert function
def insert_to_db():
    db = Database()
    conn, cursor = db.get_connection()
    
    while not thread_controller.stop_event.is_set():
        time.sleep(0.5)  # Process every half second
        if thread_controller.pending_inserts:
            # Copy the current pending inserts and clear the list
            inserts_to_process = thread_controller.pending_inserts.copy()
            thread_controller.pending_inserts.clear()
            
            try:
                with conn:
                    cursor.executemany(
                        "INSERT INTO crossing_events (source, track_id, direction, timestamp) VALUES (?, ?, ?, ?)", 
                        inserts_to_process
                    )
            except sqlite3.Error as e:
                print(f"Database error: {e}")
    
    print("Database thread stopped")

# Video processing function
def video_processing():
    global enter_count, exit_count
    track_states = {}  
    previous_centroids = {}
    last_seen = {}
    frame_idx = 0
    MAX_MISSING = 10  # Number of frames before considering a track lost
    frame_rate_buffer = []
    avg_frame_rate = 0
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
    
    # Make OpenCV window a normal window that can be closed
    cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
    thread_controller.video_window_open = True
    
    while not thread_controller.stop_event.is_set():
        t_start = time.perf_counter()

        try:
            frame = thread_controller.frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_idx += 1

        # Process the frame
        frame = cv2.resize(frame, FRAME_SIZE)
        results = model.track(
            frame,
            verbose=False,
            classes=[0],  # Track people only
            conf=0.5,
            stream=True,
            stream_buffer=True,
            persist=True,
            tracker="custom_tracker.yaml"
        )

        seen_ids = set()

        for result in results:
            for box in result.boxes:
                track_id = int(box.id.item()) if box.id is not None else None

                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) / 2

                # Track the ID
                seen_ids.add(track_id)
                last_seen[track_id] = frame_idx
                state = track_states.get(track_id, 0)
                prev_cx = previous_centroids.get(track_id, cx)

                direction = None
                # State machine transitions
                if state == 0 and prev_cx < line_x <= cx:
                    enter_count += 1
                    direction = 'enter'
                    track_states[track_id] = 1   # ENTERED
                elif state == 0 and prev_cx >= line_x > cx:
                    exit_count += 1
                    direction = 'exit'
                    track_states[track_id] = 2   # EXITED
                
                # Update previous position
                previous_centroids[track_id] = cx

                if direction:
                    timestamp = datetime.now().isoformat()
                    thread_controller.pending_inserts.append((VIDEO_SOURCE, track_id, direction, timestamp))

                # Draw bounding box and ID
                color = colors[track_id % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Clean up tracks that haven't been seen recently
        for tid in list(track_states):
            if frame_idx - last_seen.get(tid, frame_idx) > MAX_MISSING:
                if tid in track_states:
                    del track_states[tid]
                if tid in last_seen:
                    del last_seen[tid]
                if tid in previous_centroids:
                    del previous_centroids[tid]

        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, FRAME_HEIGHT), (0, 255, 0), 2)

        # Display counters
        cv2.putText(frame, f"Enter: {enter_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exit: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate and display FPS
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)
        cv2.putText(frame, f"FPS: {avg_frame_rate:.1f}", (10, FRAME_HEIGHT - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("People Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Handle window close or ESC key
        if key == 27 or cv2.getWindowProperty("People Counter", cv2.WND_PROP_VISIBLE) < 1:
            thread_controller.stop_event.set()
            break

# Start threads
# capture_thread = threading.Thread(target=capture_frames, daemon=True)
# capture_thread.start()

# db_thread = threading.Thread(target=insert_to_db, daemon=True)
# db_thread.start()

# video_thread = threading.Thread(target=video_processing, daemon=True)
# video_thread.start()

# Tkinter setup
BG_COLOR = 'cadet blue'
FG_COLOR = 'white'

    root = tk.Tk()
    root.title("People Enter Count")
    root.configure(bg=BG_COLOR)
    
    # Store the update timer ID
    label_update_id = None

    # Configure grid
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Count label
    label = tk.Label(root, text="People Counter\n0", font=("Helvetica", 54), bg=BG_COLOR, fg=FG_COLOR)
    label.grid(row=0, column=0, sticky='ew', padx=0, pady=0)

    def update_label():
        nonlocal label_update_id
        if thread_controller.stop_event.is_set():
            return
        label.config(text=f"People Counter\n{enter_count}")
        label_update_id = root.after(100, update_label)

    def handle_close():
        nonlocal label_update_id
        if label_update_id is not None:
            root.after_cancel(label_update_id)
        
        thread_controller.stop_event.set()
        
        # Wait for all threads to finish
        for thread in thread_controller.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        root.destroy()
        
        # If video window is still open, close it
        if thread_controller.video_window_open:
            cv2.destroyAllWindows()
        
        # Call the provided callback
        if on_close:
            on_close()

    # Handle window close properly
    root.protocol("WM_DELETE_WINDOW", handle_close)

    # Fullscreen toggle
    root.bind("<F11>", lambda event: root.attributes("-fullscreen",
                                        not root.attributes("-fullscreen")))
    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

    # Start updating label
    update_label()

    # Start Tkinter mainloop
    root.mainloop()

# Start the tracking threads
def start_threads(on_session_end=None):
    # Reset the thread controller
    thread_controller.reset()
    
    # Create and start the threads
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    thread_controller.threads.append(capture_thread)

    db_thread = threading.Thread(target=insert_to_db, daemon=True)
    db_thread.start()
    thread_controller.threads.append(db_thread)

    video_thread = threading.Thread(target=video_processing, daemon=True)
    video_thread.start()
    thread_controller.threads.append(video_thread)

    # Start the counter window
    counter_window(on_close=on_session_end)

# Selection window function
def show_selection_window():
    """
    Creates and runs the selection dialog. When 'Start' is clicked,
    this window is destroyed and start_threads() is called.
    When the counter window later closes, it will re-invoke this function.
    """
    # Ensure any previous stop event is set
    thread_controller.stop_event.set()
    
    # Create the main window
    sel = tk.Tk()
    sel.title("Select Counting Mode")
    sel.geometry("400x350")
    
    # Add padding and styling
    content_frame = tk.Frame(sel, padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    # Mode selection
    mode_frame = tk.LabelFrame(content_frame, text="Select Mode", padx=10, pady=10)
    mode_frame.pack(fill=tk.X, pady=(0, 10))
    
    mode_var = tk.IntVar(value=0)

    tk.Radiobutton(mode_frame, text="Start Fresh", variable=mode_var, value=0).pack(anchor='w')
    tk.Radiobutton(mode_frame, text="Use Today's Data", variable=mode_var, value=1).pack(anchor='w')
    tk.Radiobutton(mode_frame, text="Custom Date Range", variable=mode_var, value=2).pack(anchor='w')

    # Date selection for custom range
    date_frame = tk.Frame(content_frame)
    date_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(date_frame, text="Start Date:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    start_entry = DateEntry(date_frame, date_pattern='yyyy-MM-dd')
    start_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
    
    tk.Label(date_frame, text="End Date:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    end_entry = DateEntry(date_frame, date_pattern='yyyy-MM-dd')
    end_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
    
    # Initially hide the date entries
    date_frame.pack_forget()
    
    # Show/hide date entries based on mode selection
    def on_mode_change(*_):
        if mode_var.get() == 2:
            date_frame.pack(fill=tk.X, pady=10)
        else:
            date_frame.pack_forget()

    mode_var.trace_add('write', on_mode_change)

    # Initialize count for today's data
    def init_today_counts():
        global enter_count, exit_count
        db = Database()
        _, cursor = db.get_connection()

        today = date.today()
        start_dt = datetime.combine(today, datetime.min.time()).isoformat()

        cursor.execute("""SELECT direction, COUNT(*) FROM crossing_events 
                        WHERE timestamp >= ? GROUP BY direction""",
                        (start_dt,))
        ec = xc = 0
        for d, c in cursor.fetchall():
            if d=='enter': ec=c
            else: xc=c
        enter_count, exit_count = ec, xc

    # Initialize count for custom date range
    def init_custom_counts():
        global enter_count, exit_count
        db = Database()
        _, cursor = db.get_connection()

        try:
            s = start_entry.get_date().isoformat() + "T00:00:00"
            e = end_entry.get_date().isoformat() + "T23:59:59"

            cursor.execute("""SELECT direction, COUNT(*) FROM crossing_events 
                            WHERE timestamp BETWEEN ? AND ? GROUP BY direction""",
                            (s, e))
            ec = xc = 0
            for d, c in cursor.fetchall():
                if d=='enter': ec=c
                else: xc=c
            enter_count, exit_count = ec, xc
            
        except Exception as ex:
            messagebox.showerror("Error", f"Failed to get data: {str(ex)}")
            enter_count, exit_count = 0, 0

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
        format_string = ""
        title = ""
        groupby = ""
        
        if resolution == "second":
            format_string = "%Y-%m-%d %H:%M:%S"
            title = "Entries by Second"
            groupby = "strftime('%Y-%m-%d %H:%M:%S', timestamp)"
        elif resolution == "minute":
            format_string = "%Y-%m-%d %H:%M"
            title = "Entries by Minute"
            groupby = "strftime('%Y-%m-%d %H:%M', timestamp)"
        elif resolution == "hour":
            format_string = "%Y-%m-%d %H:00"
            title = "Hourly Entries"
            groupby = "strftime('%Y-%m-%d %H:00:00', timestamp)"
        else:  # day
            format_string = "%Y-%m-%d"
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
            # if resolution == "day":
            #     dt = datetime.strptime(row[0], "%Y-%m-%d")
            # else:
            #     dt = datetime.strptime(row[0], format_string)
            try:
                if resolution == "day":
                    dt = datetime.strptime(row[0], "%Y-%m-%d")
                elif resolution == "hour":
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H")
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
        global enter_count, exit_count
        m = mode_var.get()
        if m == 0:
            # Start fresh
            enter_count = 0
            exit_count = 0
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
        def on_counter_close():
            try: sel.deiconify()
            except tk.TclError: show_selection_window()
        # Start the threads and counter window
        start_threads(on_session_end=on_counter_close)

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