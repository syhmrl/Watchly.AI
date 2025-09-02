import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import argparse
import os

# ---------- CONFIG ----------
MODEL_PATH = "training/yolov8_people_detection/run1/weights/last.pt"    # Replace with your model path
INPUT_SOURCE = 0                               # 0 for webcam, or path to video file
OUTPUT_PATH = "output_classification.mp4"      # Output file if saving
IMG_SIZE = 640
CONF_THRESH = 0.2
IOU_THRESH = 0.45
USE_TRACKING = True                           # Set True to use model.track() instead of model.predict()
DISPLAY = True
SAVE_OUTPUT = False
FLIP_WEBCAM = True                             # Mirror effect for webcam only
# ----------------------------

def setup_input_source(source):
    """Setup video capture from webcam or video file"""
    if isinstance(source, int):
        # Webcam
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Unable to open webcam with index: {source}")
        
        # Set webcam resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        source_type = "webcam"
        print(f"Using webcam index: {source}")
        
    else:
        # Video file
        if not os.path.exists(source):
            raise SystemExit(f"Video file not found: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Unable to open video file: {source}")
        
        source_type = "video"
        print(f"Using video file: {source}")
    
    return cap, source_type

def get_video_info(cap):
    """Get video properties"""
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return W, H, fps, total_frames

def setup_video_writer(output_path, fps, width, height, save_output):
    """Setup video writer if saving is enabled"""
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
        return writer
    return None

def run_inference(model, frame, conf_thresh, iou_thresh, img_size, use_tracking):
    """Run inference using either predict or track method"""
    if use_tracking:
        # Using tracking - maintains object IDs across frames
        results = model.track(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
            imgsz=img_size,
            persist=True,  # Maintain tracker state
        )
    else:
        # Using prediction - independent detection per frame
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
            imgsz=img_size,
        )
    
    return results

def draw_detections(frame, results, use_tracking, gender_labels, gender_colors):
    """Draw bounding boxes and labels on frame"""
    res = results[0]
    boxes = getattr(res, "boxes", None)
    
    detection_count = 0
    
    if boxes is not None and len(boxes) > 0:
        # Convert to numpy arrays
        def to_numpy(x):
            try:
                return x.cpu().numpy()
            except Exception:
                return np.array(x)

        xyxy = to_numpy(boxes.xyxy)    # Bounding boxes
        conf = to_numpy(boxes.conf)    # Confidence scores
        cls = to_numpy(boxes.cls).astype(int) if hasattr(boxes, "cls") else np.zeros((len(xyxy),), int)
        
        # Get tracking IDs if using tracking
        track_ids = None
        if use_tracking and hasattr(boxes, "id") and boxes.id is not None:
            track_ids = to_numpy(boxes.id).astype(int)

        detection_count = len(xyxy)

        # Draw detections
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_id = int(cls[i])
            confidence = float(conf[i])
            
            # Get gender label and color
            gender_label = gender_labels.get(class_id, f"Class_{class_id}")
            color = gender_colors.get(class_id, (0, 255, 255))  # Default yellow if unknown
            
            # Create label with confidence and tracking ID
            if use_tracking and track_ids is not None:
                track_id = track_ids[i]
                label = f"ID:{track_id} {gender_label}: {confidence:.2f}"
            else:
                label = f"{gender_label}: {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - baseline - 4), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
    
    return detection_count

def main():
    # Parse command line arguments (optional)
    parser = argparse.ArgumentParser(description='Gender Classification with Webcam or Video')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model file')
    parser.add_argument('--source', default=INPUT_SOURCE, help='Input source: webcam index (int) or video path')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH, help='Output video path')
    parser.add_argument('--conf', type=float, default=CONF_THRESH, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=IOU_THRESH, help='IoU threshold')
    parser.add_argument('--track', action='store_true', default=USE_TRACKING, help='Use tracking instead of detection')
    parser.add_argument('--save', action='store_true', default=SAVE_OUTPUT, help='Save output video')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (webcam index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | FP16: {torch.cuda.is_available()}")
    
    # Load model
    try:
        model = YOLO(args.model)
        print(f"Model loaded: {args.model}")
        print(f"Model classes: {model.names}")
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")
    
    # Setup input source
    cap, source_type = setup_input_source(source)
    W, H, fps, total_frames = get_video_info(cap)
    
    print(f"Resolution: {W}x{H} @ {fps:.1f} FPS")
    if source_type == "video":
        print(f"Total frames: {total_frames}")
    
    # Setup output writer
    writer = setup_video_writer(args.output, fps, W, H, args.save)
    
    # Class labels and colors - update based on your model
    GENDER_LABELS = {0: "Male", 1: "Female"}
    GENDER_COLORS = {0: (255, 0, 0), 1: (255, 0, 255)}  # Blue for Male, Magenta for Female
    
    # Processing variables
    frame_idx = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    method_text = "TRACKING" if args.track else "DETECTION"
    print(f"\nStarting {method_text} on {source_type}...")
    print("Press 'q' to quit, 's' to save frame, 'p' to pause/resume")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame_bgr = cap.read()
                if not ret:
                    if source_type == "video":
                        print("End of video reached")
                    else:
                        print("Failed to read from webcam")
                    break

                # Flip frame for webcam mirror effect
                if source_type == "webcam" and FLIP_WEBCAM:
                    frame_bgr = cv2.flip(frame_bgr, 1)
                
                # Run inference
                results = run_inference(
                    model, frame_bgr, args.conf, args.iou, IMG_SIZE, args.track
                )
                
                # Draw detections
                detection_count = draw_detections(
                    frame_bgr, results, args.track, GENDER_LABELS, GENDER_COLORS
                )
                
                frame_idx += 1
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 10 == 0:
                current_time = time.time()
                current_fps = 10 / (current_time - fps_start_time)
                fps_start_time = current_time
            
            # Add info overlay
            method_info = f"Method: {method_text}"
            frame_info = f"Frame: {frame_idx}"
            if source_type == "video" and total_frames > 0:
                progress = (frame_idx / total_frames) * 100
                frame_info += f"/{total_frames} ({progress:.1f}%)"
            
            fps_info = f"FPS: {current_fps:.1f}"
            detection_info = f"Detections: {detection_count if 'detection_count' in locals() else 0}"
            
            info_lines = [method_info, frame_info, fps_info, detection_info]
            
            for i, info in enumerate(info_lines):
                y_pos = 25 + (i * 25)
                cv2.putText(frame_bgr, info, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add controls info
            controls = "Controls: 'q'=quit, 's'=save, 'p'=pause"
            if paused:
                controls += " [PAUSED]"
            cv2.putText(frame_bgr, controls, (10, H - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            if not args.no_display:
                window_name = f"Gender Classification - {source_type.title()} ({method_text})"
                cv2.imshow(window_name, frame_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quitting...")
                    break
                elif key == ord("s"):
                    save_filename = f"gender_frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(save_filename, frame_bgr)
                    print(f"Saved frame: {save_filename}")
                elif key == ord("p"):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
            
            # Save to output video
            if writer and not paused:
                writer.write(frame_bgr)
            
            # Auto-quit for video files when done
            if source_type == "video" and not args.no_display:
                continue
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nSession completed!")
        print(f"Total frames processed: {frame_idx}")
        if args.save:
            print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()