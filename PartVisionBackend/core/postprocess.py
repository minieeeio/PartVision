import cv2
import numpy as np
from typing import List, Dict, Any
from config import settings

CLASS_NAMES = [
    "background",
    "back_bumper", "back_door", "back_glass", "back_left_door",
    "back_left_light", "back_light", "back_right_door", "back_right_light",
    "front_bumper", "front_door", "front_glass", "front_left_door",
    "front_left_light", "front_light", "front_right_door", "front_right_light",
    "hood", "left_mirror", "object", "right_mirror",
    "tailgate", "trunk", "wheel",
]


class PostProcessor:
    CLASS_LABELS: Dict[int, str] = {
        i: name.upper() for i, name in enumerate(CLASS_NAMES)
    }

    @staticmethod
    def _inverse_transform_coords(x, y, w, h, meta, orig_w, orig_h):
        scale = meta.get("scale", 1.0)
        top = meta.get("top", 0)
        left = meta.get("left", 0)

        x_min = max(0.0, (x - left) / scale)
        y_min = max(0.0, (y - top) / scale)
        x_max = min(float(orig_w), ((x + w) - left) / scale)
        y_max = min(float(orig_h), ((y + h) - top) / scale)

        w_out = max(0.0, x_max - x_min)
        h_out = max(0.0, y_max - y_min)

        return x_min, y_min, w_out, h_out

    @staticmethod
    def _inverse_transform_point(px, py, meta, orig_w, orig_h):
        scale = meta.get("scale", 1.0)
        top = meta.get("top", 0)
        left = meta.get("left", 0)
        x = max(0.0, min(float(orig_w), (px - left) / scale))
        y = max(0.0, min(float(orig_h), (py - top) / scale))
        return x, y

    @classmethod
    def process_masks(
        cls,
        raw_output: np.ndarray,
        original_shape: tuple,
        confidence_threshold: float = settings.CONFIDENCE_THRESHOLD,
        letterbox_meta: dict = None,
    ) -> List[Dict[str, Any]]:
        model_h, model_w = raw_output.shape[1], raw_output.shape[2]
        orig_h, orig_w = original_shape
        detections = []

        exp_output = np.exp(raw_output - np.max(raw_output, axis=0, keepdims=True))
        probs = exp_output / np.sum(exp_output, axis=0, keepdims=True)

        for class_idx in range(1, raw_output.shape[0]):
            class_mask = probs[class_idx]
            max_confidence = float(np.max(class_mask))

            binary_mask = (class_mask >= confidence_threshold).astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            best_contour = None
            best_confidence = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                mask = np.zeros(class_mask.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                contour_confidence = float(np.mean(class_mask[mask == 255]))
                if contour_confidence > best_confidence:
                    best_confidence = contour_confidence
                    best_contour = contour

            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)

                if letterbox_meta:
                    x_min, y_min, w_out, h_out = cls._inverse_transform_coords(
                        float(x), float(y), float(w), float(h),
                        letterbox_meta, float(orig_w), float(orig_h)
                    )
                    x_min = round(x_min / orig_w, 4)
                    y_min = round(y_min / orig_h, 4)
                    width = round(w_out / orig_w, 4)
                    height = round(h_out / orig_h, 4)
                else:
                    x_min = round(float(x) / model_w, 4)
                    y_min = round(float(y) / model_h, 4)
                    width = round(float(w) / model_w, 4)
                    height = round(float(h) / model_h, 4)

                epsilon = 0.01 * cv2.arcLength(best_contour, True)
                approx = cv2.approxPolyDP(best_contour, epsilon, True)
                polygon = []
                for point in approx:
                    px = float(point[0][0])
                    py = float(point[0][1])
                    if letterbox_meta:
                        px, py = cls._inverse_transform_point(px, py, letterbox_meta, float(orig_w), float(orig_h))
                        polygon.append({"x": round(px / orig_w, 4), "y": round(py / orig_h, 4)})
                    else:
                        polygon.append({"x": round(px / model_w, 4), "y": round(py / model_h, 4)})

                detections.append({
                    "label": cls.CLASS_LABELS.get(class_idx, "UNKNOWN_PART"),
                    "confidence": round(max_confidence, 3),
                    "x_min": x_min,
                    "y_min": y_min,
                    "width": width,
                    "height": height,
                    "polygon": polygon,
                })

        if len(detections) == 0:
            top_class_idx = int(np.argmax(np.max(probs[1:], axis=(1, 2)))) + 1
            top_class_idx = max(1, min(top_class_idx, raw_output.shape[0] - 1))
            class_mask = probs[top_class_idx]
            y_idx, x_idx = np.unravel_index(np.argmax(class_mask), class_mask.shape)
            box_size = 40
            x1 = max(0, int(x_idx) - box_size // 2)
            y1 = max(0, int(y_idx) - box_size // 2)
            x2 = min(model_w, x1 + box_size)
            y2 = min(model_h, y1 + box_size)

            if letterbox_meta:
                x_min, y_min, w_out, h_out = cls._inverse_transform_coords(
                    float(x1), float(y1), float(x2 - x1), float(y2 - y1),
                    letterbox_meta, float(orig_w), float(orig_h)
                )
                x_min = round(x_min / orig_w, 4)
                y_min = round(y_min / orig_h, 4)
                width = round(w_out / orig_w, 4)
                height = round(h_out / orig_h, 4)
                polygon = [
                    {"x": round(max(0.0, (float(x1) - letterbox_meta.get("left", 0)) / letterbox_meta.get("scale", 1.0)) / orig_w, 4),
                     "y": round(max(0.0, (float(y1) - letterbox_meta.get("top", 0)) / letterbox_meta.get("scale", 1.0)) / orig_h, 4)},
                    {"x": round(min(float(orig_w), (float(x2) - letterbox_meta.get("left", 0)) / letterbox_meta.get("scale", 1.0)) / orig_w, 4),
                     "y": round(max(0.0, (float(y1) - letterbox_meta.get("top", 0)) / letterbox_meta.get("scale", 1.0)) / orig_h, 4)},
                    {"x": round(min(float(orig_w), (float(x2) - letterbox_meta.get("left", 0)) / letterbox_meta.get("scale", 1.0)) / orig_w, 4),
                     "y": round(min(float(orig_h), (float(y2) - letterbox_meta.get("top", 0)) / letterbox_meta.get("scale", 1.0)) / orig_h, 4)},
                    {"x": round(max(0.0, (float(x1) - letterbox_meta.get("left", 0)) / letterbox_meta.get("scale", 1.0)) / orig_w, 4),
                     "y": round(min(float(orig_h), (float(y2) - letterbox_meta.get("top", 0)) / letterbox_meta.get("scale", 1.0)) / orig_h, 4)},
                ]
            else:
                x_min = round(float(x1) / model_w, 4)
                y_min = round(float(y1) / model_h, 4)
                width = round(float(x2 - x1) / model_w, 4)
                height = round(float(y2 - y1) / model_h, 4)
                polygon = [
                    {"x": round(float(x1) / model_w, 4), "y": round(float(y1) / model_h, 4)},
                    {"x": round(float(x2) / model_w, 4), "y": round(float(y1) / model_h, 4)},
                    {"x": round(float(x2) / model_w, 4), "y": round(float(y2) / model_h, 4)},
                    {"x": round(float(x1) / model_w, 4), "y": round(float(y2) / model_h, 4)},
                ]

            detections.append({
                "label": cls.CLASS_LABELS.get(top_class_idx, "UNKNOWN_PART"),
                "confidence": round(float(np.max(class_mask)), 3),
                "x_min": x_min,
                "y_min": y_min,
                "width": width,
                "height": height,
                "polygon": polygon,
            })

        return detections
