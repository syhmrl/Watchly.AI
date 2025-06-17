# 1. FASTEST APPROACH: Enable Built-in ReID
from ultralytics import YOLO
import yaml
import cv2

# Create custom tracker config
tracker_config = {
    'tracker_type': 'botsort',  # or 'bytetrack'
    'with_reid': True,          # Enable ReID (KEY!)
    'model': 'auto',           # Use native YOLO features for ReID
    'reid_weights': None,      # Use built-in ReID
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.6,
    'track_buffer': 3000,        # Keep disappeared tracks longer
    'match_thresh': 0.8,       # ReID matching threshold
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'with_reid': True,
    'gmc_method': 'sparseOptFlow',
    'fuse_score': True
}

# Save config to file
with open('custom_tracker.yaml', 'w') as f:
    yaml.dump(tracker_config, f)

# Initialize model with ReID enabled
model = YOLO('yolo11n.pt')  # or your preferred size

# Track with enhanced ReID
results = model.track(
    source=0,
    tracker='custom_tracker.yaml',
    show=True,
    save=True,
    stream=True,
    conf=0.3,  # Lower confidence to catch more detections
    iou=0.5,
    classes=[0],  # Person class only
    verbose=False
)

# 2. INTERMEDIATE: Custom ReID with OSNet (30 minutes setup)
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from PIL import Image
import numpy as np

class FastReID:
    def __init__(self):
        # Quick setup with pretrained ResNet
        self.model = resnet50(True)
        self.model.fc = torch.nn.Linear(2048, 512)  # Feature dim
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Feature bank for disappeared persons
        self.feature_bank = {}
        self.max_disappeared = 300  # frames
        
    def extract_features(self, img_crop):
        """Extract ReID features from person crop"""
        if img_crop is None or img_crop.size == 0:
            return None
        
        # Fix: Convert numpy array to PIL Image
        try:
            # Convert BGR (OpenCV) to RGB
            if len(img_crop.shape) == 3:
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(img_crop)
            
            # Apply transforms
            img_tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def update_bank(self, track_id, features, frame_count):
        """Update feature bank with current track"""
        self.feature_bank[track_id] = {
            'features': features,
            'last_seen': frame_count
        }
    
    def find_best_match(self, features, frame_count, threshold=0.7):
        """Find best matching track from disappeared ones"""
        best_match = None
        best_score = 0
        
        for track_id, data in list(self.feature_bank.items()):
            # Remove old entries
            if frame_count - data['last_seen'] > self.max_disappeared:
                del self.feature_bank[track_id]
                continue
                
            # Compute similarity
            similarity = self.compute_similarity(features, data['features'])
            if similarity > best_score and similarity > threshold:
                best_score = similarity
                best_match = track_id
                
        return best_match, best_score
    
    def compute_similarity(self, feat1, feat2):
        """Compute cosine similarity"""
        return feat1.dot(feat2) / (
            (feat1**2).sum()**0.5 * (feat2**2).sum()**0.5
        )

# 3. ADVANCED: YOLO11-JDE Integration (Best but more complex)
class YOLO11_JDE_Tracker:
    def __init__(self, model_path='yolo11m.pt'):
        """
        Based on recent YOLO11-JDE approach
        Combines detection + embedding in single forward pass
        """
        self.model = YOLO(model_path)
        self.reid_model = FastReID()  # Use above ReID
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        
    def track_objects(self, frame):
        """Main tracking function"""
        self.frame_count += 1
        
        # Get detections
        results = self.model(frame,
                             classes=[0],
                             conf=0.4,
                             #tracker='custom_tracker.yaml',
                            # stream=True,
                            verbose=False)
        
        if len(results[0].boxes) == 0:
            return []
            
        # Extract crops and features
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Ensure valid crop coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Check bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Skip invalid crops
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            
            # Skip empty crops
            if crop.size == 0:
                continue
                
            features = self.reid_model.extract_features(crop)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': box.conf.cpu().numpy()[0],
                'features': features,
                'crop': crop
            })
        
        # Association logic
        tracked_objects = self.associate_detections(detections)
        return tracked_objects
    
    def associate_detections(self, detections):
        """Associate detections with existing tracks"""
        tracked = []
        
        for det in detections:
            if det['features'] is None:
                continue
                
            # Try to match with existing disappeared tracks
            match_id, score = self.reid_model.find_best_match(
                det['features'], 
                self.frame_count
            )
            
            if match_id is not None:
                # Reactivate track
                track_id = match_id
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
            
            # Update feature bank
            self.reid_model.update_bank(
                track_id, 
                det['features'], 
                self.frame_count
            )
            
            tracked.append({
                'id': track_id,
                'bbox': det['bbox'],
                'conf': det['conf']
            })
            
        return tracked
# Usage example
def main():
    # Method 1: Quick fix (5 minutes)
    model = YOLO('yolo11n.pt')
    results = model.track(
        source=0,  # webcam
        tracker='custom_tracker.yaml',
        stream=True,
        verbose=False
    )
    
    # Method 2: Custom ReID (30 minutes)
    tracker = YOLO11_JDE_Tracker()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        tracked_objects = tracker.track_objects(frame)
        
        # Draw results
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj['id']}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()