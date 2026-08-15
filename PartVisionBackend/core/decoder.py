import cv2
import numpy as np
from typing import Optional

class FrameDecoder:
    """
    Decodes compressed binary image payloads (JPEG/H.264 bytes)
    into NumPy arrays for neural network inference.
    """
    
    @staticmethod
    def decode_jpeg(binary_data: bytes) -> Optional[np.ndarray]:
        """
        Converts raw JPEG binary bytes into an OpenCV BGR NumPy matrix.
        
        Args:
            binary_data (bytes): The raw JPEG data sent over WebSocket.
            
        Returns:
            Optional[np.ndarray]: OpenCV image matrix of shape (H, W, C), or None if decoding fails.
        """
        if not binary_data:
            return None
            
        try:
            # Step 1: Convert raw Python bytes to 1D unsigned 8-bit integer array
            np_arr = np.frombuffer(binary_data, np.uint8)
            
            # Step 2: Decode memory buffer into a 3D OpenCV BGR image matrix
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            return frame
        except Exception as e:
            print(f"[Decoder Error] Failed to decode image frame: {e}")
            return None