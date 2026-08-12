import time
import torch
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.decoder import FrameDecoder
from app.core.postprocess import PostProcessor
from app.models.model_loader import PartLiteUNetWrapper
from config import settings

router = APIRouter()

# Instantiate model wrapper on GPU VRAM once during API router setup
model_wrapper = PartLiteUNetWrapper()

@router.websocket("/ws/segment")
async def websocket_segmentation_endpoint(websocket: WebSocket):
    """
    Persistent full-duplex WebSocket endpoint for receiving live camera video frames
    and streaming real-time car part segmentation bounding boxes back to iOS client.
    """
    await websocket.accept()
    print("[WebSocket] iOS Client connected successfully.")
    
    try:
        while True:
            # 1. Receive incoming binary JPEG image payload from iOS client
            frame_bytes = await websocket.receive_bytes()
            start_time = time.perf_counter()
            
            # 2. Decode JPEG binary bytes into an OpenCV BGR matrix
            frame = FrameDecoder.decode_jpeg(frame_bytes)
            if frame is None:
                continue
                
            orig_h, orig_w = frame.shape[:2]
            
            # 3. Preprocess frame matrix into a PyTorch tensor (Resize & Normalize)
            resized_frame = cv2.resize(frame, settings.INPUT_SIZE)
            input_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # 4. Execute AI model forward pass on GPU VRAM
            raw_output = model_wrapper.predict(input_tensor)
            
            # Convert tensor output to NumPy array for postprocessing
            if isinstance(raw_output, torch.Tensor):
                raw_output = raw_output.squeeze(0).cpu().numpy()
            else:
                # Mock fallback shape for initial testing before weight integration: (Classes, H, W)
                raw_output = np.zeros((9, settings.INPUT_SIZE[1], settings.INPUT_SIZE[0]))
            
            # 5. Extract bounding boxes and normalize coordinates
            detections = PostProcessor.process_masks(
                raw_output=raw_output,
                original_shape=(orig_h, orig_w)
            )
            
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            # 6. Transmit structured prediction JSON payload back to Swift
            response_payload = {
                "detections": detections,
                "process_time_ms": process_time_ms
            }
            
            await websocket.send_json(response_payload)
            
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected cleanly.")
    except Exception as e:
        print(f"[WebSocket Error] Connection exception: {e}")
        await websocket.close()