import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server Host & Port Configurations
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model Weights & Execution Device
    MODEL_PATH: str = os.getenv("MODEL_PATH", "weights/partliteunet.pth")
    DEVICE: str = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
    
    # Inference Filtering Thresholds
    CONFIDENCE_THRESHOLD: float = 0.85
    
    # Target Tensor Dimensions expected by PartLiteUNet
    INPUT_SIZE: tuple = (640, 640)

settings = Settings()