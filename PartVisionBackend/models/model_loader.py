import cv2
import numpy as np
import torch
import onnxruntime as ort
from typing import Any, Optional
from config import settings

from models.part_lite_unet import PartLiteUNet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PartLiteUNetWrapper:
    """
    Wraps the PartLiteUNet model for inference via ONNX Runtime or PyTorch.

    Model I/O:
      - input: [1, 3, 640, 640] float32 (NCHW, RGB, ImageNet-normalized)
      - output: [1, 24, 640, 640] float32 (24 = 1 background + 23 parts)

    The predict() method accepts a pre-resized BGR uint8 frame (already at
    640x640), handles BGR→RGB conversion, normalization, and returns the
    raw logits as a numpy array of shape [24, 640, 640].

    If the model file is missing or unloadable, predict() returns None and
    the caller should fall back to a zero-filled array.
    """

    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.device = torch.device(settings.DEVICE if settings.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        if not self.model_path or not self.model_path.endswith(('.onnx', '.pt', '.pth')):
            print(f"[Model Loader Warning] Unsupported model format: {self.model_path}")
            return

        if self.model_path.endswith('.onnx'):
            self._load_onnx()
        else:
            self._load_torch()

    def _load_onnx(self):
        print(f"[Model Loader] Loading ONNX model from: {self.model_path}")
        providers = []
        if settings.DEVICE == "cuda":
            available = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in available
                         else ["CPUExecutionProvider"])
        else:
            providers = ["CPUExecutionProvider"]

        print(f"[Model Loader] Providers: {providers}")
        try:
            sess = ort.InferenceSession(self.model_path, providers=providers)
            input_meta = sess.get_inputs()[0]
            print(f"[Model Loader] Input: name={input_meta.name}, shape={input_meta.shape}")
            for out in sess.get_outputs():
                print(f"[Model Loader] Output: name={out.name}, shape={out.shape}")
            self.model = sess
            self.model_type = "onnx"
            print("[Model Loader] ONNX model loaded successfully.")
        except Exception as e:
            print(f"[Model Loader Warning] Failed to load ONNX model: {e}")
            print("[Model Loader] Running in stub mode - predictions will return None.")
            self.model = None

    def _load_torch(self):
        print(f"[Model Loader] Loading PyTorch model from: {self.model_path}")
        print(f"[Model Loader] Device: {self.device}")
        try:
            loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)
            print(f"[Model Loader] Loaded object type: {type(loaded)}")

            if isinstance(loaded, dict):
                print("[Model Loader] Checkpoint is a state_dict, instantiating PartLiteUNet...")
                model = PartLiteUNet(
                    num_classes=23,
                    base_channels=loaded.get("base_channels", 32),
                    include_background=True,
                    backbone="scratch",
                    pretrained=False,
                    dropout=0.0,
                    use_coords=True,
                ).to(self.device)

                if "model_state" in loaded:
                    model.load_state_dict(loaded["model_state"])
                    print(f"[Model Loader] Loaded weights from checkpoint (epoch {loaded.get('epoch', '?')}, best mIoU {loaded.get('best_miou', '?'):.4f})")
                else:
                    model.load_state_dict(loaded)
                    print("[Model Loader] Loaded raw state_dict")

                model.eval()
                self.model = model
                self.model_type = "torch"
                print("[Model Loader] PyTorch model loaded successfully.")
                return

            if hasattr(loaded, 'eval'):
                loaded.eval()
            self.model = loaded
            self.model_type = "torch"
            print("[Model Loader] PyTorch model loaded successfully.")
        except Exception as e:
            print(f"[Model Loader Warning] Failed to load PyTorch model: {e}")
            import traceback
            traceback.print_exc()
            print("[Model Loader] Running in stub mode - predictions will return None.")
            self.model = None

    def predict(self, input_bgr: np.ndarray) -> tuple[Optional[np.ndarray], dict]:
        """
        Args:
            input_bgr: numpy array of shape [H, W, 3], BGR uint8 (from cv2).

        Returns:
            (logits, meta): logits is [24, H, W] float32 or None if model not loaded.
            meta contains letterbox transform info: scale, top, left, orig_shape.
        """
        if self.model is None:
            return None, {}

        orig_h, orig_w = input_bgr.shape[:2]
        img_rgb = input_bgr[:, :, ::-1]

        if settings.USE_REMBG:
            img_rgb = self._remove_background(img_rgb)

        img_letterboxed, scale, top, left = self._letterbox(img_rgb, settings.INPUT_SIZE[0])
        img_normalized = (img_letterboxed.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_batched = np.expand_dims(img_chw, axis=0).astype(np.float32)

        meta = {
            "scale": scale,
            "top": top,
            "left": left,
            "orig_shape": (orig_h, orig_w),
        }

        if self.model_type == "onnx":
            input_name = self.model.get_inputs()[0].name
            raw_output = self.model.run(None, {input_name: img_batched})
            logits = raw_output[0]
        else:
            tensor_input = torch.from_numpy(img_batched).to(self.device)
            with torch.no_grad():
                output = self.model(tensor_input)
                if isinstance(output, dict):
                    logits = output["seg"]
                else:
                    logits = output
            logits = logits.squeeze(0).cpu().numpy()

        return logits, meta

    @staticmethod
    def _remove_background(img_rgb: np.ndarray) -> np.ndarray:
        try:
            from rembg import remove
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            _, png_bytes = cv2.imencode('.png', img_bgr)
            result_bytes = remove(png_bytes.tobytes())
            result_bgr = cv2.imdecode(np.frombuffer(result_bytes, np.uint8), cv2.IMREAD_COLOR)
            if result_bgr is not None:
                return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[Model Loader Warning] Background removal failed: {e}")
        return img_rgb

    @staticmethod
    def _letterbox(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        top = (target_size - nh) // 2
        left = (target_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, top, left

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
