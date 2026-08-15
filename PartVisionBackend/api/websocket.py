import time
import json
import threading
import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import traceback
from core.decoder import FrameDecoder
from core.postprocess import PostProcessor
from core.metrics import get_inference_metrics, get_resource_monitor
from config import settings
from model_manager import model_manager

router = APIRouter()

latest_location_store: dict = {}
location_lock = threading.Lock()


@router.get("/health")
def health_check():
    """REST endpoint returning server status, model info, and system resources."""
    metrics = get_inference_metrics()
    monitor = get_resource_monitor()

    current_model = model_manager.get_current_model_type()
    available = model_manager.get_available_models()

    health_info = {
        "status": "online",
        "device": settings.DEVICE,
        "model_path": settings.MODEL_PATH,
        "current_model": current_model,
        "available_models": available,
        "model_loaded": any(available.values()),
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
    flat["partvision_model_loaded"] = 1 if any(model_manager.get_available_models().values()) else 0
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
    with location_lock:
        latest_location_store.clear()
        latest_location_store.update(location)
    return {"status": "ok"}


@router.websocket("/ws/segment")
async def websocket_segmentation_endpoint(websocket: WebSocket):
    """
    Persistent full-duplex WebSocket endpoint for receiving live camera video
    frames as binary JPEG data and streaming real-time car-part segmentation
    bounding boxes back to the client.

    Client sends:
      - binary JPEG bytes (one frame per message)
      - JSON control messages: {"type": "switch_model", "model_type": "yolo"}

    Server responds: JSON with ``detections``, ``process_time_ms``, and optional ``location``.
    """
    await websocket.accept()
    print(f"[WebSocket] Client connected successfully. Current model: {model_manager.get_current_model_type()}")

    metrics = get_inference_metrics()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                frame_bytes = message["bytes"]
                print(f"[WebSocket] Received {len(frame_bytes)} bytes")
                metrics.start_inference()

                try:
                    frame = FrameDecoder.decode_jpeg(frame_bytes)
                except Exception:
                    print("[WebSocket] FrameDecoder.decode_jpeg raised exception:")
                    traceback.print_exc()
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

                start_time = time.perf_counter()
                current_model = model_manager.get_current_model_type()

                try:
                    if current_model == "yolo":
                        detections = _run_yolo_inference(frame)
                    else:
                        raw_output, meta = model_manager.current_wrapper.predict(frame)
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
                except Exception:
                    print(f"[WebSocket] Inference raised exception for model={current_model}:")
                    traceback.print_exc()
                    metrics.record_error()
                    detections = []

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"[WebSocket] Sent {len(detections)} detections in {elapsed_ms:.1f}ms (model={current_model})")

                response_payload = {
                    "detections": detections,
                    "process_time_ms": round(elapsed_ms, 1),
                }
                with location_lock:
                    if latest_location_store:
                        response_payload["location"] = dict(latest_location_store)

                try:
                    await websocket.send_json(response_payload)
                except RuntimeError:
                    print("[WebSocket] Send failed: connection already closed")
                    break

            elif "text" in message and message["text"] is not None:
                text = message["text"]
                print(f"[WebSocket] Received JSON control: {text}")
                try:
                    control = json.loads(text)
                    msg_type = control.get("type")
                    if msg_type == "switch_model":
                        model_type = control.get("model_type")
                        print(f"[WebSocket] Switch request: model_type={model_type}")
                        if model_type in ("partlitunet", "yolo"):
                            result = model_manager.switch_model(model_type)
                            print(f"[WebSocket] Switched model via WS: {result}")
                            try:
                                await websocket.send_json({
                                    "type": "model_switched",
                                    "current_model": model_manager.get_current_model_type(),
                                    "available_models": model_manager.get_available_models(),
                                    "switch_result": result,
                                })
                            except RuntimeError:
                                break
                        else:
                            print(f"[WebSocket] Rejected unknown model_type: {model_type}")
                            try:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Unknown model_type: {model_type}",
                                })
                            except RuntimeError:
                                break
                    else:
                        print(f"[WebSocket] Unknown control message type: {msg_type}")
                except json.JSONDecodeError:
                    print("[WebSocket] Failed to parse control message as JSON")
                    traceback.print_exc()

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected cleanly.")
    except Exception as e:
        print(f"[WebSocket Error] Connection exception: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


def _run_yolo_inference(frame: np.ndarray) -> list:
    yolo_wrapper = model_manager.current_wrapper
    if yolo_wrapper is None or not yolo_wrapper.is_loaded:
        print("[YOLO] Model not loaded, returning empty detections")
        return []

    try:
        import cv2
        h, w = frame.shape[:2]
        results = yolo_wrapper.model.predict(
            source=frame,
            device=yolo_wrapper.device,
            verbose=False,
            conf=0.25,
            iou=0.45,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            masks = result.masks

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    label = result.names[cls]

                    x_min = float(x1) / w
                    y_min = float(y1) / h
                    width = float(x2 - x1) / w
                    height = float(y2 - y1) / h

                    polygon = []
                    if masks is not None and i < len(masks):
                        mask = masks[i].data[0].cpu().numpy()
                        contours, _ = cv2.findContours(
                            (mask * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if contours and len(contours) > 0:
                            largest = max(contours, key=cv2.contourArea)
                            epsilon = 0.01 * cv2.arcLength(largest, True)
                            approx = cv2.approxPolyDP(largest, epsilon, True)
                            for pt in approx:
                                px = float(pt[0][0]) / w
                                py = float(pt[0][1]) / h
                                polygon.append({"x": round(px, 4), "y": round(py, 4)})

                    detections.append({
                        "label": label,
                        "confidence": round(conf, 3),
                        "x_min": round(x_min, 4),
                        "y_min": round(y_min, 4),
                        "width": round(width, 4),
                        "height": round(height, 4),
                        "polygon": polygon,
                    })

        return detections
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        import traceback
        traceback.print_exc()
        return []
