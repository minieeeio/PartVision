import cv2
import numpy as np
from typing import List, Dict, Any
from config import settings

class PostProcessor:
    """
    Transforms raw neural network outputs (segmentation masks/logits)
    into normalized bounding box coordinates and confidence scores.
    """
    
    # Mapping model class index predictions to readable car part labels
    CLASS_LABELS: Dict[int, str] = {
        0: "BACKGROUND",
        1: "FRONT_BUMPER",
        2: "GRILLE",
        3: "HEADLIGHT_L",
        4: "HEADLIGHT_R",
        5: "HOOD",
        6: "REAR_BUMPER",
        7: "REAR_TAILLIGHT_L",
        8: "REAR_TAILLIGHT_R"
    }

    @classmethod
    def process_masks(
        cls, 
        raw_output: np.ndarray, 
        original_shape: tuple, 
        confidence_threshold: float = settings.CONFIDENCE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Converts model probability maps to normalized bounding box dictionaries.
        
        Args:
            raw_output (np.ndarray): Model output mask array of shape (Num_Classes, H, W).
            original_shape (tuple): Original image dimensions (Height, Width).
            confidence_threshold (float): Minimum confidence filter score.
            
        Returns:
            List[Dict[str, Any]]: List of detected parts with normalized x_min, y_min, width, height.
        """
        orig_h, orig_w = original_shape[:2]
        detections = []
        
        # Iterate through detected part class channels (skipping class 0: background)
        for class_idx in range(1, raw_output.shape[0]):
            class_mask = raw_output[class_idx]
            
            # Find maximum probability/confidence for this class region
            max_confidence = float(np.max(class_mask))
            if max_confidence < confidence_threshold:
                continue
                
            # Binarize mask threshold (values above confidence threshold become 255)
            binary_mask = (class_mask >= confidence_threshold).astype(np.uint8) * 255
            
            # Extract outer boundaries of the segmented part
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Ignore tiny noise artifacts
                if cv2.contourArea(contour) < 100:
                    continue
                    
                # Get bounding box in model pixel dimensions (x, y, width, height)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Normalize coordinates to 0.0 - 1.0 fractions relative to input image dimensions
                x_min_norm = round(float(x) / raw_output.shape[2], 4)
                y_min_norm = round(float(y) / raw_output.shape[1], 4)
                width_norm = round(float(w) / raw_output.shape[2], 4)
                height_norm = round(float(h) / raw_output.shape[1], 4)
                
                detections.append({
                    "label": cls.CLASS_LABELS.get(class_idx, "UNKNOWN_PART"),
                    "confidence": round(max_confidence, 3),
                    "x_min": x_min_norm,
                    "y_min": y_min_norm,
                    "width": width_norm,
                    "height": height_norm
                })
                
        return detections