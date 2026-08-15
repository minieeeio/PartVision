import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._current_model_type = "partlitunet"
        self._partlitunet_wrapper = None
        self._yolo_wrapper = None
        self._postprocessor = None
        self._decoder = None
        self._metrics = None
        self._settings = None
        self._init_backend()

    def _init_backend(self):
        from config import settings
        from core.decoder import FrameDecoder
        from core.postprocess import PostProcessor
        from core.metrics import get_inference_metrics
        from models.model_loader import PartLiteUNetWrapper
        self._settings = settings
        self._decoder = FrameDecoder()
        self._postprocessor = PostProcessor
        self._metrics = get_inference_metrics()
        self._partlitunet_wrapper = PartLiteUNetWrapper()
        if self._partlitunet_wrapper.is_loaded:
            print("[ModelManager] PartLiteUNet loaded successfully")
        else:
            print("[ModelManager] PartLiteUNet failed to load")
        self._preload_yolo()

    def load_yolo(self, model_path: str, device: str = "cuda"):
        from yolo_model_loader import YoloModelWrapper
        with self._lock:
            if self._yolo_wrapper is None or self._yolo_wrapper.model_path != model_path:
                print(f"[ModelManager] Loading YOLO model from: {model_path}")
                self._yolo_wrapper = YoloModelWrapper(model_path=model_path, device=device)
                if self._yolo_wrapper.is_loaded:
                    print("[ModelManager] YOLO loaded successfully")
                else:
                    print("[ModelManager] YOLO failed to load")

    def _preload_yolo(self):
        import os
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        candidates = ["best_yolo.pt", "best_yolo.onnx", "best.pt", "best.onnx"]
        for candidate in candidates:
            path = os.path.join(weights_dir, candidate)
            if os.path.exists(path):
                print(f"[ModelManager] Preloading YOLO from: {path}")
                self.load_yolo(path, device=self._settings.DEVICE)
                break

    def switch_model(self, model_type: str) -> Dict[str, Any]:
        with self._lock:
            if model_type not in ("partlitunet", "yolo"):
                return {"status": "error", "message": f"Unknown model type: {model_type}"}

            if model_type == "yolo" and self._yolo_wrapper is None:
                return {"status": "error", "message": "YOLO model not loaded"}

            self._current_model_type = model_type
            print(f"[ModelManager] Switched to {model_type}")
            return {
                "status": "ok",
                "current_model": model_type,
                "available_models": self.get_available_models(),
            }

    def get_current_model_type(self) -> str:
        with self._lock:
            return self._current_model_type

    def get_available_models(self) -> Dict[str, bool]:
        with self._lock:
            return {
                "partlitunet": self._partlitunet_wrapper is not None and self._partlitunet_wrapper.is_loaded,
                "yolo": self._yolo_wrapper is not None and self._yolo_wrapper.is_loaded,
            }

    @property
    def current_wrapper(self):
        with self._lock:
            if self._current_model_type == "yolo":
                return self._yolo_wrapper
            return self._partlitunet_wrapper

    @property
    def current_model_type(self) -> str:
        with self._lock:
            return self._current_model_type


model_manager = ModelManager()
