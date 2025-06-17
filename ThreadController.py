import threading
import queue
import config
from ZoomController import ZoomController 

CAMERA_SOURCES = config.get_camera_sources()

# stop_event = threading.Event()

class ThreadController:
    stop_event = threading.Event()

    def __init__(self):
        self.stop_event = threading.Event()
        self.threads = []
        self.frame_queue = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
        self.pending_inserts = queue.Queue()
        self.video_window_open = set()
        self.reset()
        # Add zoom controllers for each camera
        self.zoom_controllers = [ZoomController() for _ in range(len(CAMERA_SOURCES))]
        self.enable_visual = True
        self.enable_processed_frame_recording = False
        self.enable_roi_drawing_mode = False
    
    def reset(self):
        """Reset all thread states and counters"""
        self.stop_event.clear()
        self.frame_queue = [queue.Queue(maxsize=1) for _ in range(len(CAMERA_SOURCES))]
        self.pending_inserts = queue.Queue()
        self.threads = []
        self.video_window_open = set()