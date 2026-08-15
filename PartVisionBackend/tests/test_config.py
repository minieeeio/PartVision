import pytest
from config import settings


class TestConfig:
    def test_host_defaults_to_all_interfaces(self):
        assert settings.HOST == "0.0.0.0"

    def test_port_is_5555(self):
        assert settings.PORT == 5555

    def test_input_size_is_640x640(self):
        assert settings.INPUT_SIZE == (640, 640)

    def test_confidence_threshold_is_float(self):
        assert isinstance(settings.CONFIDENCE_THRESHOLD, float)
        assert 0.0 < settings.CONFIDENCE_THRESHOLD <= 1.0

    def test_model_path_has_supported_extension(self):
        assert settings.MODEL_PATH.endswith(('.onnx', '.pt', '.pth'))

    def test_num_classes_matches_train(self):
        """NUM_CLASSES should match the train.py CARPARTS_CLASSES (23 parts + 1 bg)."""
        class_names = [
            "background", "back_bumper", "back_door", "back_glass", "back_left_door",
            "back_left_light", "back_light", "back_right_door", "back_right_light",
            "front_bumper", "front_door", "front_glass", "front_left_door",
            "front_left_light", "front_light", "front_right_door", "front_right_light",
            "hood", "left_mirror", "object", "right_mirror",
            "tailgate", "trunk", "wheel",
        ]
        assert len(class_names) == 24
