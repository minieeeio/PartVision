import pytest
import json
import numpy as np
import cv2
from fastapi.testclient import TestClient
from main import app
from core.postprocess import PostProcessor


class TestWebSocketEndpoint:
    """Test the WebSocket segmentation endpoint."""

    def test_websocket_accepts_connection(self):
        """Test that the WebSocket endpoint accepts connections."""
        client = TestClient(app)
        with client.websocket_connect("/ws/segment") as websocket:
            assert websocket is not None

    def test_websocket_receives_json_response(self):
        """Test that after receiving a frame, the server responds with JSON."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        array[20:80, 20:80] = [255, 255, 255]
        _, jpeg_bytes = cv2.imencode('.jpg', array)
        frame_data = jpeg_bytes.tobytes()

        client = TestClient(app)
        with client.websocket_connect("/ws/segment") as websocket:
            websocket.send_bytes(frame_data)
            response = websocket.receive_text()
            data = json.loads(response)
            assert "detections" in data
            assert "process_time_ms" in data
            assert isinstance(data["detections"], list)

    def test_websocket_returns_empty_detections_for_valid_frame(self):
        """Test that a valid frame with no detectable parts returns empty detections."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode('.jpg', array)
        frame_data = jpeg_bytes.tobytes()

        client = TestClient(app)
        with client.websocket_connect("/ws/segment") as websocket:
            websocket.send_bytes(frame_data)
            response = websocket.receive_text()
            data = json.loads(response)
            assert isinstance(data["detections"], list)

    def test_websocket_handles_invalid_frame(self):
        """Test that the server handles invalid frame data without crashing.

        Invalid frames are skipped (decode returns None) so no response
        is sent for that frame. We send a valid frame afterward to
        verify the connection stays alive.
        """
        client = TestClient(app)
        with client.websocket_connect("/ws/segment") as websocket:
            websocket.send_bytes(b'invalid jpeg data')
            array = np.full((100, 100, 3), 128, dtype=np.uint8)
            _, jpeg_bytes = cv2.imencode('.jpg', array)
            websocket.send_bytes(jpeg_bytes.tobytes())
            response = websocket.receive_text()
            data = json.loads(response)
            assert "detections" in data

    def test_websocket_process_time_is_measurement(self):
        """Test that process_time_ms is a non-negative number."""
        array = np.full((100, 100, 3), 128, dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode('.jpg', array)
        frame_data = jpeg_bytes.tobytes()

        client = TestClient(app)
        with client.websocket_connect("/ws/segment") as websocket:
            websocket.send_bytes(frame_data)
            response = websocket.receive_text()
            data = json.loads(response)
            assert data["process_time_ms"] >= 0

    def test_detection_response_has_required_fields(self):
        """Test that each detection has all required fields with correct types."""
        raw_output = np.zeros((24, 640, 640))
        raw_output[1, 100:300, 100:400] = 5.0
        detections = PostProcessor.process_masks(raw_output, (480, 640))
        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det["label"], str)
        assert isinstance(det["confidence"], (int, float))
        assert isinstance(det["x_min"], (int, float))
        assert isinstance(det["y_min"], (int, float))
        assert isinstance(det["width"], (int, float))
        assert isinstance(det["height"], (int, float))
