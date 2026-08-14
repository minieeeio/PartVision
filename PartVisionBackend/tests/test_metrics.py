import pytest
import time
import numpy as np
from core.metrics import (
    ResourceMonitor,
    InferenceMetrics,
    get_resource_monitor,
    get_inference_metrics,
    start_monitoring,
)


class TestResourceMonitor:
    def test_get_stats_returns_dict(self):
        """Test that get_stats returns a valid dictionary."""
        monitor = ResourceMonitor(sample_interval=0.1, window_size=5)
        monitor.start()
        time.sleep(0.3)
        stats = monitor.get_stats()
        monitor.stop()
        assert isinstance(stats, dict)
        assert "timestamp" in stats
        assert "cpu" in stats
        assert "memory" in stats

    def test_get_stats_contains_cpu_fields(self):
        """Test that CPU stats include current, avg, and max."""
        monitor = ResourceMonitor(sample_interval=0.1, window_size=5)
        monitor.start()
        time.sleep(0.2)
        stats = monitor.get_stats()
        monitor.stop()
        assert "cpu_percent" not in stats  # raw, not in output
        assert "current_percent" in stats["cpu"]
        assert "avg_percent" in stats["cpu"]
        assert "max_percent" in stats["cpu"]

    def test_get_stats_contains_memory_fields(self):
        """Test that memory stats include RSS, VMS, and system percent."""
        monitor = ResourceMonitor(sample_interval=0.1, window_size=5)
        monitor.start()
        time.sleep(0.2)
        stats = monitor.get_stats()
        monitor.stop()
        assert "rss_mb" in stats["memory"]
        assert "vms_mb" in stats["memory"]
        assert "system_percent" in stats["memory"]

    def test_get_stats_no_samples(self):
        """Test get_stats when no samples have been collected."""
        monitor = ResourceMonitor(sample_interval=1.0, window_size=10)
        stats = monitor.get_stats()
        assert stats["status"] == "no_samples"


class TestInferenceMetrics:
    def test_record_model_load(self):
        """Test that model load time is recorded."""
        metrics = InferenceMetrics()
        metrics.record_model_load(0.123)
        m = metrics.get_metrics()
        assert m["model_load_time_seconds"] == 0.123

    def test_record_inference_latency(self):
        """Test that inference latency is tracked."""
        metrics = InferenceMetrics()
        metrics.start_inference()
        time.sleep(0.01)
        latency = metrics.record_inference(batch_size=1, frame_shape=(480, 640))
        assert latency is not None
        assert latency > 0

    def test_total_inferences_counter(self):
        """Test that total inference count increments."""
        metrics = InferenceMetrics()
        for _ in range(5):
            metrics.start_inference()
            time.sleep(0.001)
            metrics.record_inference()
        m = metrics.get_metrics()
        assert m["total_inferences"] == 5

    def test_error_recording(self):
        """Test that errors are tracked."""
        metrics = InferenceMetrics()
        for _ in range(3):
            metrics.record_error()
        m = metrics.get_metrics()
        assert m["total_errors"] == 3
        assert m["error_rate"] > 0

    def test_error_rate_calculation(self):
        """Test error rate = errors / total_inferences * 100."""
        metrics = InferenceMetrics()
        metrics.record_error()
        metrics.record_error()
        for _ in range(8):
            metrics.start_inference()
            metrics.record_inference()
        m = metrics.get_metrics()
        assert m["error_rate"] == 25.0  # 2 errors / 8 inferences

    def get_latency_stats(self):
        """Test that latency statistics are computed from samples."""
        metrics = InferenceMetrics()
        latencies = []
        for ms in [5, 8, 12, 15, 20]:
            metrics._latencies.append(float(ms))
        m = metrics.get_metrics()
        assert "latency" in m
        assert m["latency"]["mean_ms"] == 12.0
        assert m["latency"]["min_ms"] == 5.0
        assert m["latency"]["max_ms"] == 20.0

    def test_throughput_calculation(self):
        """Test that throughput (inferences/sec) is tracked."""
        metrics = InferenceMetrics()
        for _ in range(3):
            metrics.start_inference()
            time.sleep(0.01)
            metrics.record_inference(frame_shape=(480, 640))
        time.sleep(0.1)
        m = metrics.get_metrics()
        assert "throughput" in m
        assert "inferences_per_second" in m["throughput"]

    def test_empty_metrics(self):
        """Test metrics with no recordings."""
        metrics = InferenceMetrics()
        m = metrics.get_metrics()
        assert m["total_inferences"] == 0
        assert m["total_errors"] == 0
        assert m["error_rate"] == 0.0
