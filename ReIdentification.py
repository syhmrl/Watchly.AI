import torch
import torchreid
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine

import config


def cosine_similarity(embedding1, embedding2):
    """A simple helper function to calculate similarity."""
    # The 'cosine' function from scipy calculates distance, so 1 - distance is similarity
    return 1 - cosine(embedding1, embedding2)

class ReIdentificationManager:
    """Handles person re-identification logic and state management."""
    
    def __init__(self, enable_reidentification=None, similarity_threshold=None, reid_model_name=None):
        # Load settings from config
        reid_settings = config.get_reid_settings()
        
        self.enable_reidentification = enable_reidentification if enable_reidentification is not None else reid_settings.get("enabled", True)
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else reid_settings.get("similarity_threshold", 0.4)
        self.reid_model_name = reid_model_name if reid_model_name is not None else reid_settings.get("model", "osnet_x0_25")
        
        self.next_person_id = 1
        
        # Data structures for Re-ID
        self.active_track_features = {}  # track_id -> list of embeddings
        self.lost_track_features = {}    # track_id -> final aggregated embedding
        self.tracker_id_map = {}         # Mapping from BoTSORT's temporary ID to our persistent ID
        
        # Initialize Re-ID model
        self.reid_model = None
        self.reid_transform = None
        
        if self.enable_reidentification:
            self._initialize_reid_model()
        else:
            print("Re-identification disabled.")
    
    def _initialize_reid_model(self):
        """Initialize the re-identification model."""
        print(f"Loading Re-ID model: {self.reid_model_name}...")
        try:
            self.reid_model = torchreid.models.build_model(
                name=self.reid_model_name,      # A lightweight but effective model
                num_classes=1,          # A placeholder, not used for feature extraction
                pretrained=True
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            self.reid_model.eval()
            self.reid_transform = torchreid.data.transforms.build_transforms(
                height=256, width=128, is_train=False,
            )[0]
            print(f"Re-ID model {self.reid_model_name} loaded successfully.")
        except Exception as e:
            print(f"Failed to load Re-ID model {self.reid_model_name}: {e}")
            self.enable_reidentification = False
            self.reid_model = None
            self.reid_transform = None
            
    def update_settings(self, enable_reidentification=None, similarity_threshold=None, reid_model_name=None):
        """Update re-identification settings at runtime."""
        settings_changed = False
        
        if enable_reidentification is not None and enable_reidentification != self.enable_reidentification:
            self.enable_reidentification = enable_reidentification
            settings_changed = True
            
        if similarity_threshold is not None and similarity_threshold != self.similarity_threshold:
            self.similarity_threshold = similarity_threshold
            settings_changed = True
            
        if reid_model_name is not None and reid_model_name != self.reid_model_name:
            self.reid_model_name = reid_model_name
            settings_changed = True
            # Reinitialize model with new name
            if self.enable_reidentification:
                self._initialize_reid_model()
        
        if settings_changed:
            print(f"Re-ID settings updated: enabled={self.enable_reidentification}, "
                  f"threshold={self.similarity_threshold}, model={self.reid_model_name}")
            
            # If re-identification was disabled, clear existing data
            if not self.enable_reidentification:
                self.active_track_features.clear()
                self.lost_track_features.clear()
                self.tracker_id_map.clear()
                print("Re-identification disabled. Cleared existing track data.")
    
    @torch.no_grad()
    def get_embedding(self, frame, box):
        """Extracts a feature embedding from a person's bounding box."""
        if not self.enable_reidentification or self.reid_model is None:
            return None
            
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None

        try:
            # Convert crop to RGB (PIL format for transforms) and apply transforms
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image = self.reid_transform(crop_pil).unsqueeze(0)
            image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get the feature embedding
            features = self.reid_model(image)
            return features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def process_detection(self, tracker_id, frame, coords, frame_idx):
        """
        Process a detection and return the persistent ID.
        
        Args:
            tracker_id: The tracker ID from BoTSORT
            frame: The current frame
            coords: Bounding box coordinates [x1, y1, x2, y2]
            frame_idx: Current frame index
            
        Returns:
            persistent_id: The persistent person ID, or tracker_id if re-ID is disabled
        """
        if not self.enable_reidentification:
            # Re-ID disabled: use tracker_id directly as persistent_id
            return tracker_id
        
        if tracker_id not in self.tracker_id_map:
            # This is a new track according to BoTSORT
            embedding = self.get_embedding(frame, coords)
            if embedding is None:
                # If embedding extraction fails, assign new ID directly
                persistent_id = self.next_person_id
                self.tracker_id_map[tracker_id] = persistent_id
                self.next_person_id += 1
                # print(f"New person detected (no embedding)! BoTSORT ID {tracker_id} -> Persistent ID {persistent_id}")
                return persistent_id
            
            best_match_id = -1
            best_match_score = self.similarity_threshold
            
            # Compare with lost tracks
            # print(f"--- Frame {frame_idx}: New BoTSORT ID {tracker_id} appeared. Checking {len(self.lost_track_features)} lost tracks. ---")
            for lost_id, lost_embedding in self.lost_track_features.items():
                similarity = cosine_similarity(embedding, lost_embedding)
                
                # print(f"Comparing to Persistent ID {lost_id}. Similarity: {similarity:.4f}")
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_id = lost_id
            
            if best_match_id != -1:
                # It's a match! Re-assign the old ID
                persistent_id = best_match_id
                self.tracker_id_map[tracker_id] = persistent_id
                # Move features from lost back to active
                self.active_track_features[persistent_id] = [self.lost_track_features.pop(best_match_id)]
                # print(f"Re-identified person! BoTSORT ID {tracker_id} -> Persistent ID {persistent_id}")
            else:
                # It's a genuinely new person
                persistent_id = self.next_person_id
                self.tracker_id_map[tracker_id] = persistent_id
                self.active_track_features[persistent_id] = [embedding]
                self.next_person_id += 1
                # print(f"New person detected! BoTSORT ID {tracker_id} -> Persistent ID {persistent_id}")
        else:
            # This is an existing track
            persistent_id = self.tracker_id_map[tracker_id]
            # Update features periodically for robustness (e.g., every 10 frames)
            if self.enable_reidentification and frame_idx % 10 == 0:
                embedding = self.get_embedding(frame, coords)
                if embedding is not None and persistent_id in self.active_track_features:
                    self.active_track_features[persistent_id].append(embedding)
                    # Keep only the last N features to save memory
                    self.active_track_features[persistent_id] = self.active_track_features[persistent_id][-20:]
        
        return persistent_id
    
    def handle_lost_tracks(self, current_tracker_ids, temp_count=None):
        """
        Handle tracks that are no longer detected in the current frame.
        
        Args:
            current_tracker_ids: Set of tracker IDs detected in current frame
            temp_count: Optional temp_count set to update
        """
        if not self.enable_reidentification:
            return
        
        lost_tracker_ids = set(self.tracker_id_map.keys()) - current_tracker_ids
        for tracker_id in lost_tracker_ids:
            persistent_id = self.tracker_id_map.pop(tracker_id)
            
            if persistent_id in self.active_track_features:
                # Aggregate features to get a stable representation
                features = np.mean(self.active_track_features.pop(persistent_id), axis=0)
                self.lost_track_features[persistent_id] = features
                # print(f"Track for persistent ID {persistent_id} lost. Storing features.")

            if temp_count is not None and persistent_id in temp_count:
                temp_count.remove(persistent_id)