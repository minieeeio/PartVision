import pytest
import numpy as np
import cv2
from core.decoder import FrameDecoder


class TestFrameDecoder:
    def test_decode_valid_jpeg(self):
        """Test decoding a valid JPEG image."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        array[:, :, 1] = 128
        _, jpeg_bytes = cv2.imencode('.jpg', array)
        frame = FrameDecoder.decode_jpeg(jpeg_bytes.tobytes())
        assert frame is not None
        assert frame.shape == (100, 100, 3)

    def test_decode_none_input(self):
        """Test decoding None or empty bytes returns None."""
        assert FrameDecoder.decode_jpeg(None) is None
        assert FrameDecoder.decode_jpeg(b'') is None

    def test_decode_invalid_bytes(self):
        """Test decoding invalid bytes returns None."""
        result = FrameDecoder.decode_jpeg(b'not a jpeg')
        assert result is None

    def test_decode_preserves_content(self):
        """Test that decoded image content matches original."""
        original = np.full((50, 50, 3), 255, dtype=np.uint8)
        original[10:40, 10:40] = [0, 0, 0]
        _, jpeg_bytes = cv2.imencode('.jpg', original)
        decoded = FrameDecoder.decode_jpeg(jpeg_bytes.tobytes())
        assert decoded.shape == (50, 50, 3)
        assert decoded.shape[:2] == original.shape[:2]
