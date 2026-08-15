import time
import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from yolo_model_loader import YoloModelWrapper
from yolo_config import yolo_settings

router = APIRouter()

model_wrapper = YoloModelWrapper(
    model_path=yolo_settings.MODEL_PATH,
    device=yolo_settings.DEVICE,
)


@router.get("/health")
def health_check():
    return {
        "status": "online",
        "model_type": "yolo",
        "model_path": yolo_settings.MODEL_PATH,
        "model_loaded": model_wrapper.is_loaded,
        "device": yolo_settings.DEVICE,
        "confidence_threshold": yolo_settings.CONFIDENCE_THRESHOLD,
        "input_size": yolo_settings.INPUT_SIZE,
    }


@router.websocket("/ws/segment")
async def websocket_yolo_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[YOLO WS] Client connected successfully.")

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            print(f"[YOLO WS] Received {len(frame_bytes)} bytes")

            start_time = time.perf_counter()
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({
                    "detections": [],
                    "process_time_ms": 0,
                    "error": "frame_decode_failed",
                })
                continue

            detections = model_wrapper.predict(frame)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            print(f"[YOLO WS] Sent {len(detections)} detections in {elapsed_ms:.1f}ms")

            await websocket.send_json({
                "detections": detections,
                "process_time_ms": round(elapsed_ms, 1),
            })

    except WebSocketDisconnect:
        print("[YOLO WS] Client disconnected cleanly.")
    except Exception as e:
        print(f"[YOLO WS Error] Connection exception: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
