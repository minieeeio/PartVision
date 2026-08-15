import os
from pydantic_settings import BaseSettings

class YoloSettings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 5556

    MODEL_PATH: str = ""
    DEVICE: str = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"

    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    INPUT_SIZE: int = 640

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.MODEL_PATH:
            weights_dir = "weights"
            candidates = [
                os.path.join(weights_dir, "best_yolo.pt"),
                os.path.join(weights_dir, "best_yolo.onnx"),
                os.path.join(weights_dir, "best.pt"),
                os.path.join(weights_dir, "best.onnx"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    self.MODEL_PATH = candidate
                    break
            else:
                self.MODEL_PATH = os.path.join(weights_dir, "best_yolo.pt")

yolo_settings = YoloSettings()
