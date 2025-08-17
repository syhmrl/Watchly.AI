import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from sklearn.metrics.pairwise import cosine_similarity

class Tracker:
    def __init__(self, max_disappeared=60, max_distance=100, feature_buffer_size=10):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.feature_buffer_size = feature_buffer_size
        
        # Core tracking data
        self.next_id = 1
        self.active_tracks = {}  # id -> track_info
        self.disappeared = {}    # id -> frames_disappeared
        self.feature_buffers = defaultdict(lambda: deque(maxlen=feature_buffer_size))
        
        # Re-identification features
        self.lost_tracks = {}    # Store tracks that disappeared for potential re-identification
        self.reid_threshold = 0.7
        
    def extract_features(self, frame, bbox):
        """Extract simple appearance features from bounding box region"""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(128)  # Return zero vector for invalid bbox
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(128)
            
        # Resize to fixed size and extract color histogram features
        roi_resized = cv2.resize(roi, (64, 128))
        
        # Color histogram in HSV space
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize and concatenate
        features = np.concatenate([
            hist_h.flatten() / (hist_h.sum() + 1e-6),
            hist_s.flatten() / (hist_s.sum() + 1e-6),
            hist_v.flatten() / (hist_v.sum() + 1e-6),
            [roi_resized.mean()]  # Overall brightness
        ])
        
        return features / (np.linalg.norm(features) + 1e-6)
    
    def compute_similarity(self, features1, features2):
        """Compute cosine similarity between feature vectors"""
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        return cosine_similarity([features1], [features2])[0][0]
    
    def update(self, frame, detections):
        """
        Update tracker with new detections
        detections: list of (bbox, confidence) tuples
        """
        current_centroids = []
        current_features = []
        
        # Extract centroids and features from detections
        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            features = self.extract_features(frame, bbox)
            current_centroids.append(centroid)
            current_features.append(features)
        
        # If no existing tracks, create new ones
        if len(self.active_tracks) == 0:
            for i, (centroid, features) in enumerate(zip(current_centroids, current_features)):
                self.active_tracks[self.next_id] = {
                    'centroid': centroid,
                    'bbox': detections[i][0],
                    'confidence': detections[i][1]
                }
                self.feature_buffers[self.next_id].append(features)
                self.next_id += 1
            return self.active_tracks
        
        # Compute distance matrix between existing tracks and new detections
        track_ids = list(self.active_tracks.keys())
        D = np.zeros((len(track_ids), len(current_centroids)))
        
        for i, track_id in enumerate(track_ids):
            track_centroid = self.active_tracks[track_id]['centroid']
            track_features = list(self.feature_buffers[track_id])
            
            for j, (det_centroid, det_features) in enumerate(zip(current_centroids, current_features)):
                # Euclidean distance
                euclidean_dist = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
                
                # Feature similarity (convert to distance)
                if len(track_features) > 0:
                    avg_track_features = np.mean(track_features, axis=0)
                    similarity = self.compute_similarity(avg_track_features, det_features)
                    feature_dist = 1 - similarity
                else:
                    feature_dist = 1.0
                
                # Combined distance (weighted)
                D[i][j] = 0.7 * euclidean_dist + 0.3 * feature_dist * 100
        
        # Hungarian algorithm for assignment (simplified greedy approach)
        used_detection_indices = set()
        used_track_indices = set()
        
        # Sort by distance and assign
        assignments = []
        if D.size > 0:
            for _ in range(min(len(track_ids), len(current_centroids))):
                min_idx = np.unravel_index(np.argmin(D), D.shape)
                i, j = min_idx
                
                if i not in used_track_indices and j not in used_detection_indices:
                    if D[i][j] < self.max_distance:
                        assignments.append((track_ids[i], j))
                        used_track_indices.add(i)
                        used_detection_indices.add(j)
                
                D[min_idx] = float('inf')  # Mark as used
        
        # Update assigned tracks
        for track_id, detection_idx in assignments:
            self.active_tracks[track_id] = {
                'centroid': current_centroids[detection_idx],
                'bbox': detections[detection_idx][0],
                'confidence': detections[detection_idx][1]
            }
            self.feature_buffers[track_id].append(current_features[detection_idx])
            
            # Remove from disappeared if it was there
            if track_id in self.disappeared:
                del self.disappeared[track_id]
        
        # Handle unassigned detections - try re-identification first
        unassigned_detections = [i for i in range(len(current_centroids)) if i not in used_detection_indices]
        
        for detection_idx in unassigned_detections:
            det_features = current_features[detection_idx]
            det_centroid = current_centroids[detection_idx]
            
            # Try to match with lost tracks
            best_match_id = None
            best_similarity = 0
            
            for lost_id, lost_info in self.lost_tracks.items():
                if len(lost_info['features']) > 0:
                    avg_lost_features = np.mean(lost_info['features'], axis=0)
                    similarity = self.compute_similarity(avg_lost_features, det_features)
                    
                    # Also check spatial proximity
                    spatial_dist = np.linalg.norm(np.array(lost_info['last_centroid']) - np.array(det_centroid))
                    
                    if similarity > best_similarity and similarity > self.reid_threshold and spatial_dist < self.max_distance * 2:
                        best_similarity = similarity
                        best_match_id = lost_id
            
            if best_match_id is not None:
                # Re-identify lost track
                self.active_tracks[best_match_id] = {
                    'centroid': det_centroid,
                    'bbox': detections[detection_idx][0],
                    'confidence': detections[detection_idx][1]
                }
                self.feature_buffers[best_match_id] = deque(self.lost_tracks[best_match_id]['features'], 
                                                          maxlen=self.feature_buffer_size)
                self.feature_buffers[best_match_id].append(det_features)
                del self.lost_tracks[best_match_id]
            else:
                # Create new track
                self.active_tracks[self.next_id] = {
                    'centroid': det_centroid,
                    'bbox': detections[detection_idx][0],
                    'confidence': detections[detection_idx][1]
                }
                self.feature_buffers[self.next_id].append(det_features)
                self.next_id += 1
        
        # Handle disappeared tracks
        unassigned_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in used_track_indices]
        
        for track_id in unassigned_tracks:
            if track_id in self.disappeared:
                self.disappeared[track_id] += 1
            else:
                self.disappeared[track_id] = 1
            
            # Move to lost_tracks for potential re-identification
            if self.disappeared[track_id] > self.max_disappeared:
                if track_id in self.active_tracks:
                    self.lost_tracks[track_id] = {
                        'features': list(self.feature_buffers[track_id]),
                        'last_centroid': self.active_tracks[track_id]['centroid'],
                        'timestamp': time.time()
                    }
                    del self.active_tracks[track_id]
                    del self.feature_buffers[track_id]
                    del self.disappeared[track_id]
        
        # Clean up old lost tracks (after 5 minutes)
        current_time = time.time()
        expired_tracks = [tid for tid, info in self.lost_tracks.items() 
                         if current_time - info['timestamp'] > 300]
        for tid in expired_tracks:
            del self.lost_tracks[tid]
        
        return self.active_tracks