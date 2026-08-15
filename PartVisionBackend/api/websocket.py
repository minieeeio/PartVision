import time
import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.decoder import FrameDecoder
from core.postprocess import PostProcessor
from core.metrics import get_inference_metrics, get_resource_monitor
from models.model_loader import PartLiteUNetWrapper
from config import settings

router = APIRouter()

_model_load_start = time.perf_counter()
model_wrapper = PartLiteUNetWrapper()
if model_wrapper.is_loaded:
    get_inference_metrics().record_model_load(time.perf_counter() - _model_load_start)

latest_location_store: dict = {}


@router.get("/health")
def health_check():
    """REST endpoint returning server status, model info, and system resources."""
    metrics = get_inference_metrics()
    monitor = get_resource_monitor()

    health_info = {
        "status": "online",
        "device": settings.DEVICE,
        "model_path": settings.MODEL_PATH,
        "model_loaded": model_wrapper.is_loaded,
        "model_load_time_seconds": metrics.get_metrics().get("model_load_time_seconds"),
        "input_size": list(settings.INPUT_SIZE),
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "num_classes": len(PostProcessor.CLASS_LABELS),
        "class_labels": PostProcessor.CLASS_LABELS,
        "system": monitor.get_stats(),
        "inference": metrics.get_metrics(),
    }
    return health_info


@router.get("/metrics")
def server_metrics():
    """Detailed Prometheus-style metrics endpoint for monitoring dashboards."""
    metrics = get_inference_metrics()
    monitor = get_resource_monitor()

    m = metrics.get_metrics()
    s = monitor.get_stats()

    flat = {}

    flat["partvision_model_loaded"] = 1 if model_wrapper.is_loaded else 0
    flat["partvision_model_load_time_seconds"] = m.get("model_load_time_seconds") or 0.0
    flat["partvision_total_inferences"] = m["total_inferences"]
    flat["partvision_total_errors"] = m["total_errors"]
    flat["partvision_error_rate_percent"] = m["error_rate"]

    if "latency" in m:
        lat = m["latency"]
        flat["partvision_latency_mean_ms"] = lat["mean_ms"]
        flat["partvision_latency_p95_ms"] = lat["p95_ms"]
        flat["partvision_latency_p99_ms"] = lat["p99_ms"]

    if "throughput" in m:
        flat["partvision_inferences_per_second"] = m["throughput"].get(
            "inferences_per_second", 0.0
        )

    flat["partvision_cpu_percent"] = s.get("cpu", {}).get("current_percent", 0.0)
    flat["partvision_cpu_avg_percent"] = s.get("cpu", {}).get("avg_percent", 0.0)
    flat["partvision_memory_rss_mb"] = s.get("memory", {}).get("rss_mb", 0.0)
    flat["partvision_memory_vms_mb"] = s.get("memory", {}).get("vms_mb", 0.0)
    flat["partvision_memory_system_percent"] = s.get("memory", {}).get(
        "system_percent", 0.0
    )

    return {"metrics": flat, "detailed": {"inference": m, "system": s}}


@router.post("/location")
async def update_location(location: dict):
    """Receive GPS location updates from the client."""
    latest_location_store.clear()
    latest_location_store.update(location)
    return {"status": "ok"}


@router.websocket("/ws/segment")
async def websocket_segmentation_endpoint(websocket: WebSocket):
    """
    Persistent full-duplex WebSocket endpoint for receiving live camera video
    frames as binary JPEG data and streaming real-time car-part segmentation
    bounding boxes back to the client.

    Client sends:  binary JPEG bytes (one frame per message).
    Server responds: JSON with ``detections``, ``process_time_ms``, and optional ``location``.
    """
    await websocket.accept()
    print("[WebSocket] Client connected successfully.")

    metrics = get_inference_metrics()

    try:
        while True:
            print("[WebSocket] Waiting for frame bytes...")
            frame_bytes = await websocket.receive_bytes()
            print(f"[WebSocket] Received {len(frame_bytes)} bytes")
            metrics.start_inference()

            frame = FrameDecoder.decode_jpeg(frame_bytes)
            if frame is None:
                metrics.record_error()
                try:
                    await websocket.send_json({
                        "detections": [],
                        "process_time_ms": 0,
                        "error": "frame_decode_failed",
                    })
                except RuntimeError:
                    break
                continue

            orig_h, orig_w = frame.shape[:2]
            print(f"[WebSocket] Decoded frame: {orig_w}x{orig_h}")
            raw_output, meta = model_wrapper.predict(frame)

            if raw_output is None:
                metrics.record_error()
                raw_output = np.zeros(
                    (len(PostProcessor.CLASS_LABELS),
                     settings.INPUT_SIZE[1],
                     settings.INPUT_SIZE[0]),
                    dtype=np.float32,
                )
                meta = {}
            else:
                metrics.record_inference(
                    batch_size=1,
                    frame_shape=(orig_h, orig_w),
                )
                print(f"[WebSocket] Raw output stats: shape={raw_output.shape}, min={raw_output.min():.4f}, max={raw_output.max():.4f}, mean={raw_output.mean():.4f}")

            detections = PostProcessor.process_masks(
                raw_output=raw_output,
                original_shape=(orig_h, orig_w),
                letterbox_meta=meta,
            )

            print(f"[WebSocket] Sent {len(detections)} detections")
            response_payload = {
                "detections": detections,
                "process_time_ms": metrics.get_metrics().get("current_latency_ms", 0) or 0,
            }
            if latest_location_store:
                response_payload["location"] = dict(latest_location_store)

            try:
                await websocket.send_json(response_payload)
            except RuntimeError:
                print("[WebSocket] Send failed: connection already closed")
                break

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected cleanly.")
    except Exception as e:
        print(f"[WebSocket Error] Connection exception: {e}")
    finally:
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass
