import time
import threading
import traceback
from typing import Optional, Dict, Any
from dataclasses import dataclass

class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._current_model_type = "partlitunet"
        self._partlitunet_wrapper = None
        self._yolo_wrapper = None
        self._yolo_load_error: Optional[str] = None
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
        try:
            self._partlitunet_wrapper = PartLiteUNetWrapper()
            if self._partlitunet_wrapper.is_loaded:
                print("[ModelManager] PartLiteUNet loaded successfully")
            else:
                print("[ModelManager] PartLiteUNet failed to load")
        except Exception:
            print("[ModelManager] PartLiteUNet raised exception during init:")
            traceback.print_exc()
        threading.Thread(target=self._preload_yolo, daemon=True).start()

    def load_yolo(self, model_path: str, device: str = "cuda"):
        from yolo_model_loader import YoloModelWrapper
        with self._lock:
            if self._yolo_wrapper is not None and self._yolo_wrapper.model_path == model_path:
                return
            print(f"[ModelManager] Loading YOLO model from: {model_path}")
            try:
                self._yolo_wrapper = YoloModelWrapper(model_path=model_path, device=device)
                if self._yolo_wrapper.is_loaded:
                    print("[ModelManager] YOLO loaded successfully")
                    self._yolo_load_error = None
                else:
                    print("[ModelManager] YOLO failed to load: wrapper reports not loaded")
                    self._yolo_load_error = "YOLO wrapper reports model not loaded"
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                print(f"[ModelManager] YOLO load exception: {error_msg}")
                traceback.print_exc()
                self._yolo_wrapper = None
                self._yolo_load_error = error_msg

    def _preload_yolo(self):
        import os
        weights_dir = "./weights"
        print(f"[ModelManager] Preloading YOLO from weights dir: {weights_dir}")
        candidates = ["best_yolo.pt"]
        found = False
        for candidate in candidates:
            path = os.path.join(weights_dir, candidate)
            print(path)
            if os.path.exists(path):
                print(f"[ModelManager] Found YOLO weight file: {path}")
                found = True
                try:
                    self.load_yolo(path, device=self._settings.DEVICE)
                except Exception:
                    print("[ModelManager] YOLO preload raised exception:")
                    traceback.print_exc()
                break
        if not found:
            print("[ModelManager] No YOLO weight files found in weights/")
            self._yolo_load_error = "No YOLO weight files found in weights/"
        print(f"[ModelManager] YOLO preload complete. _yolo_load_error={self._yolo_load_error}")

    def switch_model(self, model_type: str) -> Dict[str, Any]:
        with self._lock:
            try:
                if model_type not in ("partlitunet", "yolo"):
                    return {"status": "error", "message": f"Unknown model type: {model_type}"}

                if model_type == "yolo" and not self.get_available_models().get("yolo", False):
                    error_detail = self._yolo_load_error
                    print(f"[ModelManager] Switch to yolo rejected: {error_detail}")
                    return {
                        "status": "error",
                        "message": "YOLO model not loaded",
                        "detail": error_detail,
                    }

                self._current_model_type = model_type
                print(f"[ModelManager] Switched to {model_type}")
                return {
                    "status": "ok",
                    "current_model": model_type,
                    "available_models": self.get_available_models(),
                }
            except Exception:
                print("[ModelManager] switch_model raised exception:")
                traceback.print_exc()
                return {"status": "error", "message": "Internal server error during model switch"}

    def get_current_model_type(self) -> str:
        with self._lock:
            return self._current_model_type

    def get_available_models(self) -> Dict[str, bool]:
        with self._lock:
            try:
                return {
                    "partlitunet": self._partlitunet_wrapper is not None and self._partlitunet_wrapper.is_loaded,
                    "yolo": self._yolo_wrapper is not None and self._yolo_wrapper.is_loaded,
                }
            except Exception:
                print("[ModelManager] get_available_models raised exception:")
                traceback.print_exc()
                return {"partlitunet": False, "yolo": False}

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
