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

def calculate_fps(frame_rate_buffer, t_start, fps_avg_len):
    """Update fps buffer and return average FPS."""
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    return np.mean(frame_rate_buffer)

def cleanup_stale(last_seen, frame_idx, max_missing, detection_count):
    """Remove IDs not seen in the last max_missing frames."""
    for tid in list(last_seen):
        if frame_idx - last_seen.get(tid, frame_idx) > max_missing:
            if tid in last_seen:
                del last_seen[tid]
            if tid in detection_count:
                del detection_count[tid]
            # Note: We don't remove from tracked_ids to avoid recounting
