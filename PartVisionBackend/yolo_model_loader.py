import time
import numpy as np
import cv2
from typing import Any, Optional, List, Dict

class YoloModelWrapper:
    """
    Wraps YOLOv8-seg model for inference.
    Model I/O:
      - input: RGB image (H, W, 3) uint8
      - output: YOLO results with boxes, masks, classes, confidences
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model: Any = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            print(f"[YOLO] Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print(f"[YOLO] Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                source=image_bgr,
                device=self.device,
                verbose=False,
                conf=0.25,
                iou=0.45,
            )

            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                masks = result.masks

                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        label = result.names[cls]

                        h, w = image_bgr.shape[:2]
                        x_min = float(x1) / w
                        y_min = float(y1) / h
                        width = float(x2 - x1) / w
                        height = float(y2 - y1) / h

                        polygon = []
                        if masks is not None and i < len(masks):
                            mask = masks[i].data[0].cpu().numpy()
                            contours, _ = cv2.findContours(
                                (mask * 255).astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )
                            if contours and len(contours) > 0:
                                largest = max(contours, key=cv2.contourArea)
                                epsilon = 0.01 * cv2.arcLength(largest, True)
                                approx = cv2.approxPolyDP(largest, epsilon, True)
                                for pt in approx:
                                    px = float(pt[0][0]) / w
                                    py = float(pt[0][1]) / h
                                    polygon.append({"x": round(px, 4), "y": round(py, 4)})

                        detections.append({
                            "label": label,
                            "confidence": round(conf, 3),
                            "x_min": round(x_min, 4),
                            "y_min": round(y_min, 4),
                            "width": round(width, 4),
                            "height": round(height, 4),
                            "polygon": polygon,
                        })

            return detections
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return []

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
