import time
import threading
import numpy as np
import psutil
from collections import deque
from typing import Dict, Any, Optional, List
from datetime import datetime


class ResourceMonitor:
    """Thread-safe system resource monitor.

    Samples CPU, memory, and disk I/O at regular intervals on a background
    thread. Returns rolling-window statistics (default: last 60 seconds).
    """

    def __init__(self, sample_interval: float = 1.0, window_size: int = 60):
        self.sample_interval = sample_interval
        self.window_size = window_size
        self._samples: Dict[str, deque] = {
            "cpu_percent": deque(maxlen=window_size),
            "memory_percent": deque(maxlen=window_size),
            "memory_rss_mb": deque(maxlen=window_size),
            "memory_vms_mb": deque(maxlen=window_size),
            "disk_read_mb": deque(maxlen=window_size),
            "disk_write_mb": deque(maxlen=window_size),
            "timestamp": deque(maxlen=window_size),
        }
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
        self._prev_disk_io = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _sample_loop(self):
        while self._running:
            self._sample()
            time.sleep(self.sample_interval)

    def _sample(self):
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem_info = self._process.memory_info()
            mem_percent = psutil.virtual_memory().percent
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)

            disk_read_mb = 0.0
            disk_write_mb = 0.0
            try:
                disk_io = psutil.disk_io_counters()
                if self._prev_disk_io is not None:
                    read_bytes = disk_io.read_bytes - self._prev_disk_io.read_bytes
                    write_bytes = disk_io.write_bytes - self._prev_disk_io.write_bytes
                    disk_read_mb = read_bytes / (1024 * 1024)
                    disk_write_mb = write_bytes / (1024 * 1024)
                self._prev_disk_io = disk_io
            except Exception:
                pass

            ts = time.time()
            with self._lock:
                for key, val in [
                    ("cpu_percent", cpu),
                    ("memory_percent", mem_percent),
                    ("memory_rss_mb", rss_mb),
                    ("memory_vms_mb", vms_mb),
                    ("disk_read_mb", disk_read_mb),
                    ("disk_write_mb", disk_write_mb),
                    ("timestamp", ts),
                ]:
                    self._samples[key].append(val)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Return current and rolling-average resource statistics."""
        with self._lock:
            if not self._samples["timestamp"]:
                return {
                    "status": "no_samples",
                    "timestamp": datetime.now().isoformat(),
                }

            stats = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "current_percent": round(self._samples["cpu_percent"][-1], 2),
                    "avg_percent": round(
                        float(np.mean(self._samples["cpu_percent"])), 2
                    ),
                    "max_percent": round(
                        float(np.max(self._samples["cpu_percent"])), 2
                    ),
                },
                "memory": {
                    "rss_mb": round(self._samples["memory_rss_mb"][-1], 2),
                    "vms_mb": round(self._samples["memory_vms_mb"][-1], 2),
                    "avg_rss_mb": round(
                        float(np.mean(self._samples["memory_rss_mb"])), 2
                    ),
                    "avg_vms_mb": round(
                        float(np.mean(self._samples["memory_vms_mb"])), 2
                    ),
                    "system_percent": round(
                        self._samples["memory_percent"][-1], 2
                    ),
                },
                "disk": {
                    "read_mb_last_interval": round(
                        self._samples["disk_read_mb"][-1], 2
                    ),
                    "write_mb_last_interval": round(
                        self._samples["disk_write_mb"][-1], 2
                    ),
                    "avg_read_mb": round(
                        float(np.mean(self._samples["disk_read_mb"])), 2
                    ),
                    "avg_write_mb": round(
                        float(np.mean(self._samples["disk_write_mb"])), 2
                    ),
                },
                "sample_count": len(self._samples["timestamp"]),
                "window_seconds": self.window_size * self.sample_interval,
            }
        return stats


class InferenceMetrics:
    """Tracks per-inference latency, throughput, and batch statistics.

    Thread-safe; stores a rolling window of latency samples and maintains
    aggregate counters.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._total_inferences = 0
        self._total_errors = 0
        self._model_load_time: Optional[float] = None
        self._inference_start: Optional[float] = None
        self._last_inference_time: Optional[float] = None
        self._batch_sizes: deque = deque(maxlen=window_size)
        self._frame_sizes: deque = deque(maxlen=window_size)

    def record_model_load(self, load_time_seconds: float):
        self._model_load_time = round(load_time_seconds, 4)

    def start_inference(self):
        self._inference_start = time.perf_counter()

    def record_inference(self, batch_size: int = 1, frame_shape: Optional[tuple] = None):
        if self._inference_start is not None:
            latency_ms = round(
                (time.perf_counter() - self._inference_start) * 1000, 2
            )
            with self._lock:
                self._latencies.append(latency_ms)
                self._total_inferences += 1
                self._batch_sizes.append(batch_size)
                if frame_shape is not None:
                    self._frame_sizes.append(frame_shape[0] * frame_shape[1])
                self._last_inference_time = time.time()
            self._inference_start = None
            return latency_ms
        return None

    def record_error(self):
        with self._lock:
            self._total_errors += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies)
            batch_sizes = list(self._batch_sizes)
            frame_sizes = list(self._frame_sizes)

            result = {
                "total_inferences": self._total_inferences,
                "total_errors": self._total_errors,
                "error_rate": round(
                    self._total_errors / max(self._total_inferences, 1) * 100, 2
                ),
                "model_load_time_seconds": self._model_load_time,
                "current_latency_ms": latencies[-1] if latencies else None,
                "last_inference": datetime.fromtimestamp(
                    self._last_inference_time
                ).isoformat() if self._last_inference_time else None,
            }

            if latencies:
                latencies_arr = np.array(latencies)
                result.update({
                    "latency": {
                        "mean_ms": round(float(latencies_arr.mean()), 2),
                        "median_ms": round(float(np.median(latencies_arr)), 2),
                        "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
                        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
                        "min_ms": round(float(latencies_arr.min()), 2),
                        "max_ms": round(float(latencies_arr.max()), 2),
                        "std_ms": round(float(latencies_arr.std()), 2),
                    },
                })

            if batch_sizes:
                batch_arr = np.array(batch_sizes)
                result["throughput"] = {
                    "avg_batch_size": round(float(batch_arr.mean()), 2),
                }

            if frame_sizes:
                result["input_resolution"] = {
                    "avg_pixels": round(float(np.mean(frame_sizes)), 0),
                    "last_pixels": frame_sizes[-1],
                }

            if latencies and self._last_inference_time:
                elapsed = time.time() - self._last_inference_time
                result["throughput"]["inferences_per_second"] = round(
                    len(latencies) / elapsed, 2
                ) if elapsed > 0 else 0.0

            return result


_resource_monitor = ResourceMonitor()
_inference_metrics = InferenceMetrics()

# Start the background monitoring thread at module load so it's always running
# (including in tests via TestClient, which doesn't trigger FastAPI lifecycle events).
_resource_monitor.start()


def get_resource_monitor() -> ResourceMonitor:
    return _resource_monitor


def get_inference_metrics() -> InferenceMetrics:
    return _inference_metrics


def start_monitoring():
    _resource_monitor.start()
