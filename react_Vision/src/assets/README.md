# PartVision On-Device Model

Place your ONNX model file here, named exactly:

```
partlite_unet.onnx
```

## How to convert from the trained .pt checkpoint

```python
import torch
from models.partlite_unet import PartLiteUNet

model = PartLiteUNet(num_classes=22, base_channels=32, include_background=True,
                     backbone="scratch", use_coords=True)
ckpt = torch.load("backend/models/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

# Wrap to output only seg (drop boundary/aux heads)
class SegOnly(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return {"seg": self.m(x)["seg"]}

dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    SegOnly(model), dummy, "partlite_unet.onnx",
    input_names=["input"], output_names=["seg"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "seg": {0: "batch"}},
)
```

Then copy `partlite_unet.onnx` to `src/assets/`.
