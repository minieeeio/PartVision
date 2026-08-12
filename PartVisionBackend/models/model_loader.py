import torch
import torch.nn as nn
from typing import Any
from config import settings

# 1. Import your custom model architecture
from your_model_module import PartLiteUNet  # <--- Change this import to your actual file/class

class PartLiteUNetWrapper:
    def __init__(self, model_path: str = settings.MODEL_PATH, device: str = settings.DEVICE):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        print(f"[Model Loader] Loading trained PartLiteUNet on device: {self.device}")
        try:
            # 2. Instantiate your architecture and load the weights
            model = PartLiteUNet() 
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            print("[Model Loader] Model loaded successfully.")
            return model
        except Exception as e:
            print(f"[Model Loader Error] Failed to load model: {e}")
            raise e

    def predict(self, input_tensor: torch.Tensor) -> Any:
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output