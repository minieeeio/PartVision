import pytest
import queue
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main import app
from core.reconstructor import get_manager, ReconstructionManager, SessionStatus


class TestReconstructionAPI:
    def setup_method(self):
        self.manager = get_manager()
        self.session = self.manager.create_session()
        self.manager.start_recording(self.session.session_id)

    def teardown_method(self):
        self.manager.mark_failed(self.session.session_id, "test cleanup")

    def test_start_session(self):
        client = TestClient(app)
        response = client.post("/start")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "status" in data
        assert "output_dir" in data

    def test_upload_frame(self):
        client = TestClient(app)
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        array[20:80, 20:80] = [255, 255, 255]
        _, jpeg_bytes = cv2.imencode('.jpg', array)
        frame_data = jpeg_bytes.tobytes()

        response = client.post(
            f"/{self.session.session_id}/frame",
            files={"file": ("frame.jpg", frame_data, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == self.session.session_id
        assert data["frame_count"] == 1

    def test_upload_blurry_frame_is_rejected(self):
        client = TestClient(app)
        blurry = np.zeros((100, 100, 3), dtype=np.uint8)
        blurry[:] = [128, 128, 128]
        _, jpeg_bytes = cv2.imencode('.jpg', blurry)
        frame_data = jpeg_bytes.tobytes()

        response = client.post(
            f"/{self.session.session_id}/frame",
            files={"file": ("frame.jpg", frame_data, "image/jpeg")},
        )
        assert response.status_code == 400

    def test_stop_session(self):
        client = TestClient(app)
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode('.jpg', array)
        client.post(
            f"/{self.session.session_id}/frame",
            files={"file": ("frame.jpg", jpeg_bytes.tobytes(), "image/jpeg")},
        )

        response = client.post(f"/{self.session.session_id}/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"

    def test_session_status(self):
        client = TestClient(app)
        response = client.get(f"/{self.session.session_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == self.session.session_id
        assert data["status"] == "recording"

    def test_update_location(self):
        client = TestClient(app)
        response = client.post(
            f"/{self.session.session_id}/location",
            json={"lat": 37.7749, "lon": -122.4194, "alt": 10.0},
        )
        assert response.status_code == 200
        assert self.session.gps is not None

    def test_websocket_reconstruction_progress(self):
        client = TestClient(app)
        with client.websocket_connect(f"/ws/reconstruction/{self.session.session_id}") as ws:
            msg = ws.receive_json()
            assert msg["stage"] == "heartbeat"
            assert msg["status"] == "recording"


class TestReconstructionManager:
    def test_blur_filtering(self):
        manager = ReconstructionManager(base_dir="/tmp/test_recon")
        session = manager.create_session()
        manager.start_recording(session.session_id)

        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[20:80, 20:80] = 255
        _, sharp_bytes = cv2.imencode('.jpg', sharp)
        result = manager.add_frame(session.session_id, sharp_bytes.tobytes())
        assert result is not None
        assert session.frame_count == 1

        blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
        _, blurry_bytes = cv2.imencode('.jpg', blurry)
        result = manager.add_frame(session.session_id, blurry_bytes.tobytes())
        assert result is None
        assert session.rejected_frames == 1

    def test_progress_queue(self):
        manager = ReconstructionManager(base_dir="/tmp/test_recon2")
        session = manager.create_session()
        q: queue.Queue = queue.Queue(maxsize=10)
        manager.register_progress_queue(session.session_id, q)

        manager._progress(session.session_id, {"stage": "test", "progress": 50})
        msg = q.get(timeout=1)
        assert msg["stage"] == "test"

        manager.unregister_progress_queue(session.session_id)
