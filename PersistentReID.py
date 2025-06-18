
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class PersistentReID:
    """
    Fixed Optimized Persistent ReID system with consistent feature dimensions
    """
    def __init__(self, feature_dim=256, max_gallery_size=50, similarity_threshold=0.65, 
                 min_features_per_person=3, max_features_per_person=5):
        self.feature_dim = feature_dim
        self.max_gallery_size = max_gallery_size
        self.similarity_threshold = similarity_threshold
        self.min_features_per_person = min_features_per_person
        self.max_features_per_person = max_features_per_person
        
        # Gallery of known persons
        self.person_gallery = {}
        self.next_person_id = 1
        
        # Track ID to Person ID mapping
        self.track_to_person = {}
        self.person_to_tracks = defaultdict(set)
        
        # Feature extraction optimization
        self.feature_extractor = self._load_optimized_feature_extractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.feature_extractor:
            self.feature_extractor.to(self.device)
        
        # Preprocessing optimized for speed
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Smaller size for speed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Performance optimization
        self.feature_cache = {}
        self.cache_size_limit = 100
        
        # Enhanced discrimination
        self.person_creation_threshold = 0.4  # Lower threshold for creating new persons
        self.stable_match_threshold = 0.75    # Higher threshold for stable matches
        self.temporal_consistency_weight = 0.3
        
        # Version control for feature compatibility
        self.feature_version = "1.0"
        
    def _load_optimized_feature_extractor(self):
        """Load optimized feature extractor with consistent output dimension"""
        try:
            import torchvision.models as models
            # Use MobileNetV3 for better speed/accuracy trade-off
            model = models.ResNet50_Weights(weights='DEFAULT')
            # Ensure consistent feature dimension output
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(model.classifier[0].in_features, self.feature_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self.feature_dim, self.feature_dim)  # Ensure exact dimension
            )
            model.eval()
            return model
        except Exception as e:
            print(f"Could not load optimized model: {e}")
            return None
    
    def extract_features_fast(self, person_crop):
        """Fast feature extraction with caching and consistent dimensions"""
        if self.feature_extractor is None:
            # Use fast handcrafted features as fallback
            return self._extract_handcrafted_features(person_crop)
        
        # Create cache key from crop
        cache_key = hash(person_crop.tobytes())
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        try:
            # Resize crop to reduce computation
            if person_crop.shape[0] > 128 or person_crop.shape[1] > 64:
                person_crop = cv2.resize(person_crop, (64, 128))
            
            input_tensor = self.preprocess(person_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().cpu().numpy()
                
                # Ensure exact feature dimension
                if len(features) != self.feature_dim:
                    if len(features) > self.feature_dim:
                        features = features[:self.feature_dim]
                    else:
                        features = np.pad(features, (0, self.feature_dim - len(features)))
                
                # Normalize features
                features = features / (np.linalg.norm(features) + 1e-8)
                
                # Ensure features are exactly the right dimension
                assert len(features) == self.feature_dim, f"Feature dimension mismatch: {len(features)} != {self.feature_dim}"
            
            # Cache the result
            if len(self.feature_cache) < self.cache_size_limit:
                self.feature_cache[cache_key] = features
            
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self._extract_handcrafted_features(person_crop)
    
    def _extract_handcrafted_features(self, person_crop):
        """Fast handcrafted features as fallback with exact dimension control"""
        if person_crop.size == 0:
            return np.random.rand(self.feature_dim) * 0.01
        
        # Resize for consistency
        crop_resized = cv2.resize(person_crop, (64, 128))
        
        # Color histogram features
        hist_b = cv2.calcHist([crop_resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([crop_resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([crop_resized], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-8)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-8)
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-8)
        
        # Simple texture features (LBP-like)
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Additional features to reach target dimension
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Combine features
        features = np.concatenate([hist_b, hist_g, hist_r, [texture, edge_density]])
        
        # Ensure exact dimension
        if len(features) < self.feature_dim:
            # Pad with small random values
            padding = np.random.normal(0, 0.01, self.feature_dim - len(features))
            features = np.concatenate([features, padding])
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        
        # Final dimension check
        assert len(features) == self.feature_dim, f"Handcrafted feature dimension mismatch: {len(features)} != {self.feature_dim}"
        
        return features / (np.linalg.norm(features) + 1e-8)
    
    def _validate_feature_dimensions(self, features):
        """Validate and fix feature dimensions if needed"""
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        if len(features) != self.feature_dim:
            print(f"Warning: Feature dimension mismatch {len(features)} != {self.feature_dim}, fixing...")
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            else:
                features = np.pad(features, (0, self.feature_dim - len(features)))
        
        return features
    
    def find_best_match(self, features, exclude_recent=True):
        """Find best matching person with improved discrimination and dimension checking"""
        if not self.person_gallery:
            return None, 0.0
        
        # Validate input features
        features = self._validate_feature_dimensions(features)
        
        best_person_id = None
        best_similarity = 0.0
        
        for person_id, person_data in self.person_gallery.items():
            if len(person_data['features']) < self.min_features_per_person:
                continue
            
            try:
                # Validate stored features dimensions
                valid_features = []
                for stored_feature in person_data['features']:
                    validated_feature = self._validate_feature_dimensions(stored_feature)
                    valid_features.append(validated_feature)
                
                if not valid_features:
                    continue
                
                # Calculate similarity with stored features
                person_features = np.array(valid_features)
                
                # Ensure all features have same dimension
                if person_features.shape[1] != self.feature_dim:
                    print(f"Warning: Stored features dimension mismatch for person {person_id}")
                    continue
                
                similarities = cosine_similarity([features], person_features)[0]
                
                # Use median similarity for robustness
                median_similarity = np.median(similarities)
                max_similarity = np.max(similarities)
                
                # Weighted combination
                combined_similarity = 0.7 * max_similarity + 0.3 * median_similarity
                
                if combined_similarity > best_similarity:
                    best_similarity = combined_similarity
                    best_person_id = person_id
                    
            except Exception as e:
                print(f"Error matching person {person_id}: {e}")
                continue
        
        return best_person_id, best_similarity
    
    def should_create_new_person(self, features):
        """Determine if a new person should be created"""
        if not self.person_gallery:
            return True
        
        best_person_id, best_similarity = self.find_best_match(features)
        
        # More strict criteria for creating new persons
        return best_similarity < self.person_creation_threshold
    
    def update_person_gallery(self, person_id, features, frame_idx):
        """Update person gallery with smart feature management and dimension validation"""
        # Validate features before storing
        features = self._validate_feature_dimensions(features)
        
        if person_id not in self.person_gallery:
            self.person_gallery[person_id] = {
                'features': [],
                'last_seen': frame_idx,
                'creation_frame': frame_idx,
                'feature_version': self.feature_version
            }
        
        person_data = self.person_gallery[person_id]
        person_data['features'].append(features)
        person_data['last_seen'] = frame_idx
        
        # Keep only the most recent and diverse features
        if len(person_data['features']) > self.max_features_per_person:
            # Remove oldest feature
            person_data['features'] = person_data['features'][-self.max_features_per_person:]
    
    def process_detection(self, track_id, person_crop, frame_idx):
        """Process detection with improved logic and dimension checking"""
        # Skip processing if crop is too small
        if person_crop.shape[0] < 30 or person_crop.shape[1] < 15:
            if track_id in self.track_to_person:
                return self.track_to_person[track_id]
            return None
        
        # Extract features
        features = self.extract_features_fast(person_crop)
        
        # Validate features
        features = self._validate_feature_dimensions(features)
        
        # Check if track is already mapped
        if track_id in self.track_to_person:
            person_id = self.track_to_person[track_id]
            self.update_person_gallery(person_id, features, frame_idx)
            return person_id
        
        # Find best match
        best_person_id, best_similarity = self.find_best_match(features)
        
        if best_person_id and best_similarity > self.similarity_threshold:
            # Map to existing person
            person_id = best_person_id
            self.track_to_person[track_id] = person_id
            self.person_to_tracks[person_id].add(track_id)
            self.update_person_gallery(person_id, features, frame_idx)
        else:
            # Create new person only if sufficiently different
            if self.should_create_new_person(features):
                person_id = self.next_person_id
                self.next_person_id += 1
                self.track_to_person[track_id] = person_id
                self.person_to_tracks[person_id].add(track_id)
                self.update_person_gallery(person_id, features, frame_idx)
            else:
                # Assign to closest match with lower threshold
                if best_person_id:
                    person_id = best_person_id
                    self.track_to_person[track_id] = person_id
                    self.person_to_tracks[person_id].add(track_id)
                    self.update_person_gallery(person_id, features, frame_idx)
                else:
                    return None
        
        return person_id
    
    def cleanup_old_tracks(self, active_track_ids):
        """Clean up inactive tracks"""
        inactive_tracks = set(self.track_to_person.keys()) - set(active_track_ids)
        
        for track_id in inactive_tracks:
            if track_id in self.track_to_person:
                person_id = self.track_to_person[track_id]
                self.person_to_tracks[person_id].discard(track_id)
                del self.track_to_person[track_id]
    
    def cleanup_old_persons(self, frame_idx, max_age=3000):  # 100 seconds at 30fps
        """Remove very old persons to prevent memory bloat"""
        old_persons = [
            pid for pid, data in self.person_gallery.items()
            if frame_idx - data['last_seen'] > max_age
        ]
        
        for person_id in old_persons:
            if person_id in self.person_gallery:
                del self.person_gallery[person_id]
            if person_id in self.person_to_tracks:
                del self.person_to_tracks[person_id]
    
    def clean_incompatible_gallery(self):
        """Clean gallery entries with incompatible feature dimensions"""
        cleaned_persons = []
        
        for person_id, person_data in list(self.person_gallery.items()):
            try:
                # Check if features have correct dimensions
                valid_features = []
                for feature in person_data['features']:
                    if len(feature) == self.feature_dim:
                        valid_features.append(feature)
                
                if valid_features:
                    # Keep only valid features
                    person_data['features'] = valid_features
                else:
                    # Remove person with no valid features
                    del self.person_gallery[person_id]
                    cleaned_persons.append(person_id)
                    
            except Exception as e:
                print(f"Error cleaning person {person_id}: {e}")
                del self.person_gallery[person_id]
                cleaned_persons.append(person_id)
        
        if cleaned_persons:
            print(f"Cleaned {len(cleaned_persons)} persons with incompatible features")
        
        return cleaned_persons
    
    def save_gallery(self, filepath):
        """Save person gallery with dimension validation"""
        try:
            # Clean incompatible entries before saving
            self.clean_incompatible_gallery()
            
            gallery_data = {
                'person_gallery': self.person_gallery,
                'feature_dim': self.feature_dim,
                'feature_version': self.feature_version
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(gallery_data, f)
        except Exception as e:
            print(f"Failed to save gallery: {e}")
    
    def load_gallery(self, filepath):
        """Load person gallery with dimension compatibility checking"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                # Handle legacy format
                if isinstance(data, dict) and 'person_gallery' in data:
                    saved_feature_dim = data.get('feature_dim', None)
                    
                    # Check feature dimension compatibility
                    if saved_feature_dim and saved_feature_dim != self.feature_dim:
                        print(f"Warning: Saved feature dimension {saved_feature_dim} != current {self.feature_dim}")
                        print("Creating new gallery to avoid dimension mismatch")
                        return
                    
                    self.person_gallery = data['person_gallery']
                else:
                    # Legacy format - might have dimension issues
                    print("Warning: Loading legacy gallery format")
                    self.person_gallery = data
                
                # Clean any incompatible entries
                self.clean_incompatible_gallery()
                
                if self.person_gallery:
                    self.next_person_id = max(self.person_gallery.keys()) + 1
                    
        except Exception as e:
            print(f"Failed to load gallery: {e}")
            # Reset gallery on load failure
            self.person_gallery = {}
            self.next_person_id = 1