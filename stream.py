import cv2
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not self.stream.isOpened():
            raise ValueError("Unable to open video source")
        (self.grabbed, self.frame) = self.stream.read()

         # Video writer properties
        self.writer = None
        self.recording = False
        self.output_file = ""
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)

        self.stopped = False
        self.thread = None

    def start(self):
        # start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
    
    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True  # Stop on read failure
                break
            self.grabbed, self.frame = grabbed, frame
        # Release stream when the thread exits
        self.stream.release()
        self.frame = None

    def start_recording(self, output_file, frame_size):
        """Call this after getting first valid frame"""
        self.output_file = output_file
        w, h = frame_size
        self.writer = cv2.VideoWriter(
            self.output_file, 
            self.fourcc, 
            self.fps, 
            (w, h)
        )
        self.recording = True

    def write_frame(self, frame):
        """Call this with processed frame"""
        if self.recording and self.writer is not None:
            self.writer.write(frame)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
        self.recording = False

    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()