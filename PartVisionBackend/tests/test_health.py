import pytest
from fastapi.testclient import TestClient
from main import app


class TestHealthCheck:
    def test_health_endpoint_returns_200(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_contains_status(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "online"

    def test_health_response_contains_device(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "device" in data

    def test_health_response_contains_threshold(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "confidence_threshold" in data
        assert isinstance(data["confidence_threshold"], (int, float))

    def test_health_response_contains_model_status(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_response_contains_num_classes(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "num_classes" in data
        assert data["num_classes"] == 24

    def test_health_response_contains_class_labels(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "class_labels" in data
        # JSON converts dict keys to strings
        assert data["class_labels"]["0"] == "BACKGROUND"
        assert data["class_labels"]["1"] == "BACK_BUMPER"
        assert data["class_labels"]["9"] == "FRONT_BUMPER"
        assert data["class_labels"]["22"] == "TRUNK"
        assert data["class_labels"]["23"] == "WHEEL"

    def test_health_response_contains_system_stats(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "system" in data
        assert "cpu" in data["system"]
        assert "memory" in data["system"]

    def test_health_response_contains_inference_stats(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "inference" in data
        assert "total_inferences" in data["inference"]


class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_200(self):
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_response_contains_flat_metrics(self):
        client = TestClient(app)
        response = client.get("/metrics")
        data = response.json()
        assert "metrics" in data
        assert "partvision_model_loaded" in data["metrics"]
        assert "partvision_total_inferences" in data["metrics"]

    def test_metrics_response_contains_detailed(self):
        client = TestClient(app)
        response = client.get("/metrics")
        data = response.json()
        assert "detailed" in data
        assert "inference" in data["detailed"]
        assert "system" in data["detailed"]
