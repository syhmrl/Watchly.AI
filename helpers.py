# helpers.py
import time, numpy as np
from ultralytics import YOLO
import torch

def load_model(model_name):
    """Load YOLO model on GPU if available, else CPU."""
    model = YOLO(model_name)
    if torch.cuda.is_available():
        print("Using GPU")
        return model.to('cuda')
    return model

def calculate_fps(frame_rate_buffer, t_start, fps_avg_len = 200):
    """Update fps buffer and return average FPS."""
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    return np.mean(frame_rate_buffer)

def cleanup_stale(last_seen, frame_idx, *dicts_to_clean):
    max_missing = 400
    for tid in list(last_seen):
        if frame_idx - last_seen[tid] > max_missing:
            del last_seen[tid]
            for d in dicts_to_clean:
                d.pop(tid, None)
