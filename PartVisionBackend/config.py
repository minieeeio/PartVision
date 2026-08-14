import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server Host & Port Configurations
    HOST: str = "0.0.0.0"
    PORT: int = 5555
    
    # Model Weights & Execution Device
    MODEL_PATH: str = ""
    DEVICE: str = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
    
    # Inference Filtering Thresholds
    CONFIDENCE_THRESHOLD: float = 0.25
    
    # Target Tensor Dimensions expected by PartLiteUNet
    INPUT_SIZE: tuple = (640, 640)
    
    # Inference preprocessing
    USE_REMBG: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.MODEL_PATH:
            weights_dir = "weights"
            for ext in [".pt", ".pth", ".onnx"]:
                candidate = os.path.join(weights_dir, f"best{ext}")
                if os.path.exists(candidate):
                    self.MODEL_PATH = candidate
                    break
            if not self.MODEL_PATH:
                self.MODEL_PATH = os.path.join(weights_dir, "best.onnx")

settings = Settings()