import pytest
import numpy as np
from core.postprocess import PostProcessor, CLASS_NAMES


class TestPostProcessor:
    def test_class_labels_exist(self):
        """Test that all 24 class labels are defined."""
        assert len(PostProcessor.CLASS_LABELS) == 24
        assert PostProcessor.CLASS_LABELS[0] == "BACKGROUND"
        assert PostProcessor.CLASS_LABELS[1] == "BACK_BUMPER"
        assert PostProcessor.CLASS_LABELS[9] == "FRONT_BUMPER"
        assert PostProcessor.CLASS_LABELS[22] == "TRUNK"
        assert PostProcessor.CLASS_LABELS[23] == "WHEEL"

    def test_class_names_match_train(self):
        """Test that class names match the train.py CARPARTS_CLASSES."""
        assert CLASS_NAMES[0] == "background"
        assert "front_bumper" in CLASS_NAMES
        assert "hood" in CLASS_NAMES
        assert "wheel" in CLASS_NAMES
        assert "left_mirror" in CLASS_NAMES

    def test_process_masks_returns_list(self):
        """Test that process_masks returns a list."""
        raw_output = np.zeros((24, 640, 640))
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert isinstance(result, list)

    def test_process_masks_max_one_per_class(self):
        """Test that process_masks returns at most 1 detection per class."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 5.0   # BACK_BUMPER
        raw_output[1, 300:400, 300:400] = 5.0   # BACK_BUMPER again - should be deduped
        raw_output[17, 200:300, 200:300] = 5.0  # HOOD
        result = PostProcessor.process_masks(raw_output, (480, 640))
        labels = [d["label"] for d in result]
        assert labels.count("BACK_BUMPER") <= 1
        assert labels.count("HOOD") <= 1

    def test_process_masks_returns_top1_even_when_all_zeros(self):
        """Test that process_masks returns exactly 1 detection for flat output."""
        raw_output = np.zeros((24, 640, 640))
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) == 1
        assert "label" in result[0]
        assert "confidence" in result[0]

    def test_process_masks_detects_high_confidence(self):
        """Test that a high-confidence mask region produces a detection."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 5.0  # class 1 = BACK_BUMPER
        result = PostProcessor.process_masks(raw_output, (480, 320))
        assert len(result) >= 1
        assert result[0]["label"] == "BACK_BUMPER"
        assert "x_min" in result[0]
        assert "y_min" in result[0]
        assert "width" in result[0]
        assert "height" in result[0]

    def test_process_masks_confidence_is_valid(self):
        """Test that confidence is a valid probability between 0 and 1."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 10.0
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) >= 1
        conf = result[0]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_process_masks_normalizes_coordinates(self):
        """Test that bounding box coordinates are normalized to 0.0-1.0."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[2, 50:100, 50:100] = 5.0
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) >= 1
        det = result[0]
        assert 0.0 <= det["x_min"] <= 1.0
        assert 0.0 <= det["y_min"] <= 1.0
        assert 0.0 <= det["width"] <= 1.0
        assert 0.0 <= det["height"] <= 1.0

    def test_process_masks_skips_background(self):
        """Test that class 0 (background) is never the top prediction when other classes exist."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[0, 100:200, 100:200] = 5.0
        raw_output[1, 10:50, 10:50] = 3.0
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) >= 1
        assert result[0]["label"] != "BACKGROUND"

    def test_process_masks_multiple_classes(self):
        """Test that multiple different classes can each produce 1 detection."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 10:50, 10:50] = 5.0     # BACK_BUMPER
        raw_output[17, 100:150, 100:150] = 5.0  # HOOD
        result = PostProcessor.process_masks(raw_output, (480, 640))
        labels = [d["label"] for d in result]
        assert "BACK_BUMPER" in labels
        assert "HOOD" in labels
        assert labels.count("BACK_BUMPER") == 1
        assert labels.count("HOOD") == 1

    def test_process_masks_ignores_tiny_contours(self):
        """Test that tiny contours fall back to a single bounding box around the max region."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[2, 100:102, 100:102] = 5.0
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) >= 1

    def test_process_masks_applies_softmax(self):
        """Test that softmax is applied to convert logits to probabilities."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 10.0
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) >= 1
        assert 0.0 <= result[0]["confidence"] <= 1.0

    def test_process_masks_inverse_transform_bbox(self):
        """Test that letterbox metadata maps bbox back to original image coordinates."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 5.0
        meta = {"scale": 0.5, "top": 80, "left": 0}
        result = PostProcessor.process_masks(raw_output, (480, 640), letterbox_meta=meta)
        assert len(result) >= 1
        det = result[0]
        assert 0.0 <= det["x_min"] <= 1.0
        assert 0.0 <= det["y_min"] <= 1.0
        assert 0.0 <= det["width"] <= 1.0
        assert 0.0 <= det["height"] <= 1.0
        assert det["y_min"] > 0.0

    def test_process_masks_inverse_transform_polygon(self):
        """Test that polygon points are correctly inverse-transformed."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 5.0
        meta = {"scale": 0.5, "top": 80, "left": 0}
        result = PostProcessor.process_masks(raw_output, (480, 640), letterbox_meta=meta)
        assert len(result) >= 1
        assert len(result[0]["polygon"]) >= 3
        for pt in result[0]["polygon"]:
            assert 0.0 <= pt["x"] <= 1.0
            assert 0.0 <= pt["y"] <= 1.0

    def test_process_masks_no_letterbox_meta_falls_back(self):
        """Test that missing letterbox_meta still returns normalized model-space coords."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 5.0
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) >= 1
        det = result[0]
        assert 0.0 <= det["x_min"] <= 1.0
        assert 0.0 <= det["y_min"] <= 1.0
