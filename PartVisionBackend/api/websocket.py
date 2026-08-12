import time
import asyncio
import logging
import torch
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.decoder import FrameDecoder
from app.core.postprocess import PostProcessor
from app.core.connection_manager import connection_manager
from app.models.model_loader import PartLiteUNetWrapper
from config import settings

logger = logging.getLogger("PartVision")

router = APIRouter()

model_wrapper = PartLiteUNetWrapper()

# Serialize GPU inference across concurrent WebSocket connections
inference_semaphore = asyncio.Semaphore(1)


@router.websocket("/ws/segment")
async def websocket_segmentation_endpoint(websocket: WebSocket):
    """
    Persistent full-duplex WebSocket endpoint.

    Receives live camera JPEG frames, runs car-part segmentation, and streams
    normalized bounding-box JSON back to the client. Implements frame dropping
    (skips a frame if the previous one is still being processed) so that the
    server always works on the most recent image rather than a stale queue.
    """
    client_id = await connection_manager.connect(websocket)
    processing: bool = False

    try:
        while True:
            # 1. Receive incoming binary JPEG with a timeout (prevents hanging)
            frame_bytes = await connection_manager.receive_bytes_with_timeout(
                websocket, timeout=10.0,
            )

            if frame_bytes is None:
                continue

            # 2. Heartbeat ping/pong handling
            if frame_bytes in (b"ping", b"pong"):
                try:
                    await websocket.ping()
                except Exception:
                    pass
                continue

            # 3. Frame dropping: skip if still processing the previous frame
            if processing:
                continue

            processing = True
            try:
                start_time = time.perf_counter()

                # 4. Decode JPEG → OpenCV BGR matrix
                frame = FrameDecoder.decode_jpeg(frame_bytes)
                if frame is None:
                    continue

                orig_h, orig_w = frame.shape[:2]

                # 5. Preprocess: resize + normalize to [0,1] tensor
                resized_frame = cv2.resize(frame, settings.INPUT_SIZE)
                input_tensor = (
                    torch.from_numpy(resized_frame)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )

                # 6. Execute model forward pass (offloaded to thread pool so
                #    the event loop stays responsive for WebSocket I/O)
                async with inference_semaphore:
                    raw_output = await asyncio.to_thread(
                        model_wrapper.predict, input_tensor,
                    )

                # Convert tensor → NumPy for postprocessing
                if isinstance(raw_output, torch.Tensor):
                    raw_output = raw_output.squeeze(0).cpu().numpy()
                else:
                    # Mock fallback shape before real weights are integrated
                    raw_output = np.zeros(
                        (9, settings.INPUT_SIZE[1], settings.INPUT_SIZE[0])
                    )

                # 7. Extract bounding boxes (normalized 0.0–1.0)
                detections = PostProcessor.process_masks(
                    raw_output=raw_output,
                    original_shape=(orig_h, orig_w),
                )

                process_time_ms = round(
                    (time.perf_counter() - start_time) * 1000, 2
                )

                # 8. Stream structured prediction JSON back to client
                response_payload = {
                    "detections": detections,
                    "process_time_ms": process_time_ms,
                }

                sent = await connection_manager.send_json(websocket, response_payload)
                if not sent:
                    break

            except asyncio.TimeoutError:
                logger.warning("Inference timeout for client %s", client_id)
            except Exception as e:
                logger.error("Frame processing error [%s]: %s", client_id, e)
            finally:
                processing = False

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
