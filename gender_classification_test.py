# video_yolo_neutral_color.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# ---------- CONFIG ----------
# MODEL_PATH = "training/yolov8_people_detection/run1/weights/last.pt"
MODEL_PATH = "headv3.pt"
VIDEO_PATH = "video/masuk_u_test.mp4"
OUTPUT_PATH = "output_neutral_color.mp4"         # set to None to disable saving
IMG_SIZE = 640
CONF_THRESH = 0.1
IOU_THRESH = 0.1
DISPLAY = True
SAVE_OUTPUT = True if OUTPUT_PATH else False
# ----------------------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_half = torch.cuda.is_available()
print(f"Device: {device} | FP16: {use_half}")

# open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Unable to open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {VIDEO_PATH}  size=({W},{H}) fps={fps}")

# prepare writer
writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))
    print("Saving annotated video to:", OUTPUT_PATH)

# load model
model = YOLO(MODEL_PATH)

frame_idx = 0
t0 = time.time()
try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Optionally resize for speed (keeps colour since BGR)
        # frame_proc = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
        frame_proc = frame_bgr  # use original resolution

        # Ultralytics accepts numpy arrays as input (BGR or RGB). We'll pass BGR,
        # but it handles various formats. We request predictions (non-stream).
        results = model.predict(
            source=frame_proc,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            verbose=False
        )

        # results is a list; take first result
        res = results[0]

        # get boxes (defensive conversion: could be torch or numpy)
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # convert tensors to numpy arrays (works for both torch/numpy)
            def to_numpy(x):
                try:
                    return x.cpu().numpy()
                except Exception:
                    return np.array(x)

            xyxy = to_numpy(boxes.xyxy)    # shape (N,4)
            conf = to_numpy(boxes.conf)    # shape (N,)
            cls  = to_numpy(boxes.cls).astype(int) if hasattr(boxes, "cls") else np.zeros((len(xyxy),), int)

            # draw boxes on original BGR frame (neutral colour preserved)
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box.tolist())
                c = int(cls[i])
                conf_i = float(conf[i])
                label = f"{c} {conf_i:.2f}"

                # choose color by class (BGR tuples)
                color = (0, 255, 0) if c == 0 else (0, 165, 255)  # example: class 0 green, else orange

                # rectangle and filled label background
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(frame_bgr, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # show FPS
        if frame_idx % 5 == 0:
            t_now = time.time()
            fps_now = (frame_idx + 1) / (t_now - t0 + 1e-9)
        cv2.putText(frame_bgr, f"Frame: {frame_idx} FPS_est: {fps_now:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # display & save (both operate on frame_bgr so colours stay neutral)
        if DISPLAY:
            cv2.imshow("YOLO Neutral Color", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if SAVE_OUTPUT:
            writer.write(frame_bgr)

        frame_idx += 1

finally:
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done. Total frames:", frame_idx)
