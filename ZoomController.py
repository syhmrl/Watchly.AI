import cv2

INITIAL_ZOOM = 1.0  # No zoom by default
MAX_ZOOM = 5.0      # Maximum zoom level
ZOOM_STEP = 0.1     # How much to change zoom per key press

class ZoomController:
    

    def __init__(self):
        self.zoom_factor = INITIAL_ZOOM  # Current zoom level
        self.zoom_center_x = 0.5         # Center point of zoom (normalized 0-1)
        self.zoom_center_y = 0.5         # Center point of zoom (normalized 0-1)
        self.pan_step = 0.02             # How much to pan per key press
    
    def increase_zoom(self):
        self.zoom_factor = min(self.zoom_factor + ZOOM_STEP, MAX_ZOOM)
        return self.zoom_factor
        
    def decrease_zoom(self):
        self.zoom_factor = max(self.zoom_factor - ZOOM_STEP, 1.0)
        return self.zoom_factor
    
    def reset_zoom(self):
        self.zoom_factor = INITIAL_ZOOM
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5
    
    def apply_zoom(self, frame):
        """Apply digital zoom to the frame"""
        if self.zoom_factor <= 1.0:
            return frame  # No zoom needed
            
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate the region of interest based on zoom factor and center point
        # The higher the zoom, the smaller the ROI
        roi_size = 1.0 / self.zoom_factor
        
        # Calculate the top-left corner of the ROI
        x1 = int(w * (self.zoom_center_x - roi_size/2))
        y1 = int(h * (self.zoom_center_y - roi_size/2))
        
        # Calculate the bottom-right corner of the ROI
        x2 = int(w * (self.zoom_center_x + roi_size/2))
        y2 = int(h * (self.zoom_center_y + roi_size/2))
        
        # Ensure ROI is within the frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resize the ROI to the original frame size
        if roi.size > 0:  # Check if ROI is not empty
            return cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            return frame  # Return original frame if ROI is invalid
