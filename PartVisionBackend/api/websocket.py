import time
import torch
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.decoder import FrameDecoder
from app.core.postprocess import PostProcessor
from app.core.connection_manager import connection_manager
from app.models.model_loader import PartLiteUNetWrapper
from config import settings

import logging
logger = logging.getLogger("PartVision")

router = APIRouter()

model_wrapper = PartLiteUNetWrapper()


@router.websocket("/ws/segment")
async def websocket_segmentation_endpoint(websocket: WebSocket):
    """
    Persistent full-duplex WebSocket endpoint for receiving live camera video frames
    and streaming real-time car part segmentation bounding boxes back to client.
    """
    client_id = await connection_manager.connect(websocket)

    try:
        while True:
            frame_bytes = await connection_manager.receive_bytes_with_timeout(
                websocket, timeout=10.0,
            )

            if frame_bytes is None:
                continue

            if frame_bytes == b"ping" or frame_bytes == b"pong":
                await websocket.ping()
                continue

            start_time = time.perf_counter()

            frame = FrameDecoder.decode_jpeg(frame_bytes)
            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]

            resized_frame = cv2.resize(frame, settings.INPUT_SIZE)
            input_tensor = (
                torch.from_numpy(resized_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )

            raw_output = model_wrapper.predict(input_tensor)

            if isinstance(raw_output, torch.Tensor):
                raw_output = raw_output.squeeze(0).cpu().numpy()
            else:
                raw_output = np.zeros(
                    (9, settings.INPUT_SIZE[1], settings.INPUT_SIZE[0])
                )

            detections = PostProcessor.process_masks(
                raw_output=raw_output,
                original_shape=(orig_h, orig_w),
            )

            process_time_ms = round(
                (time.perf_counter() - start_time) * 1000, 2
            )

            response_payload = {
                "detections": detections,
                "process_time_ms": process_time_ms,
            }

            sent = await connection_manager.send_json(websocket, response_payload)
            if not sent:
                break

    except WebSocketDisconnect:
        logger.info("Client %s disconnected cleanly.", client_id)
    except Exception as e:
        logger.error("WebSocket error [%s]: %s", client_id, e)
    finally:
        connection_manager.disconnect(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
