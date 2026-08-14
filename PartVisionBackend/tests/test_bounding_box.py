import pytest
import numpy as np
from core.postprocess import PostProcessor


class TestBoundingBoxNormalization:
    """Test that normalized bounding box coordinates are correctly calculated."""

    def test_single_detection_coordinates(self):
        """Test coordinate normalization for a single detection in a 640x640 mask."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:300, 200:500] = 50.0
        result = PostProcessor.process_masks(raw_output, (480, 320))

        assert len(result) >= 1
        det = result[0]
        assert det["x_min"] == round(200 / 640, 4)
        assert det["y_min"] == round(100 / 640, 4)
        assert det["width"] == round(300 / 640, 4)
        assert det["height"] == round(200 / 640, 4)

    def test_coordinates_with_different_original_size(self):
        """Test that coordinate normalization uses mask dimensions, not original image dimensions."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[2, 0:640, 0:640] = 50.0
        result = PostProcessor.process_masks(raw_output, (1080, 1920))

        assert len(result) >= 1
        det = result[0]
        assert det["x_min"] == 0.0
        assert det["y_min"] == 0.0
        assert det["width"] == 1.0
        assert det["height"] == 1.0

    def test_confidence_is_rounded_to_3_decimals(self):
        """Test that confidence is rounded to 3 decimal places."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:200, 100:200] = 50.0
        result = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(result) >= 1
        assert result[0]["confidence"] == round(result[0]["confidence"], 3)

    def test_uses_correct_class_labels(self):
        """Test that the top-1 class label is returned."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[18, 50:100, 50:100] = 50.0  # left_mirror - highest global max
        raw_output[22, 200:250, 200:250] = 25.0  # trunk - lower
        raw_output[23, 300:350, 300:350] = 10.0  # wheel - lower
        result = PostProcessor.process_masks(raw_output, (480, 640))
        labels = [d["label"] for d in result]
        assert "LEFT_MIRROR" in labels
