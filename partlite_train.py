#!/usr/bin/env python3
"""
Single-file PartLiteUNet training & evaluation on Carparts-Seg.

Changes vs original:
  - FIXED: config-dict aliasing bug (vars(args) aliasing caused --epochs to get
    silently clobbered by --optuna-epochs during Optuna search).
  - REMOVED: Grad-CAM generation.
  - ADDED: optional class-weighted loss (CE + Dice) to combat class imbalance.
  - ADDED: YOLO-style evaluation report (confusion matrix, PR/F1/Precision/Recall
    -vs-confidence curves, class distribution, training curves).
  - ADDED: optional instance-level (connected-component) mask mAP@0.5, the
    segmentation-native analogue of YOLO's box mAP.

Usage:
    python train.py                    # Train both variants (base32, base48)
    python train.py --variant 32       # Only base32
    python train.py --variant 32 --demo  # 1 epoch for demo
    python train.py --variant 32 --test_it  # Evaluate best checkpoint on test set
    python train.py --no-rembg         # Skip rembg preprocessing
    python train.py --variant 32 --optuna --optuna-trials 20 --optuna-epochs 10

    # Recommended full run with the new features:
    python train.py --variant 32 --optuna --optuna-trials 20 --optuna-epochs 15 \
        --epochs 150 --test_it --class-weighting inv_sqrt_freq \
        --ignore-classes object --compute-map
"""

import os
import sys
import json
import time
import glob
import random
import shutil
import argparse
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    _OPTUNA_AVAILABLE = False

CARPARTS_CLASSES = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light",
    "back_light", "back_right_door", "back_right_light", "front_bumper", "front_door",
    "front_glass", "front_left_door", "front_left_light", "front_light",
    "front_right_door", "front_right_light", "hood", "left_mirror", "object",
    "right_mirror", "tailgate", "trunk", "wheel",
]
NUM_CLASSES = len(CARPARTS_CLASSES)
NUM_CLASSES_WITH_BG = NUM_CLASSES + 1
CLASS_NAMES = ["background"] + CARPARTS_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_FLIP_LABEL_MAP = np.arange(256, dtype=np.uint8)
for src, dst in [(3, 6), (4, 7), (11, 14), (12, 15), (17, 19)]:
    _FLIP_LABEL_MAP[src + 1] = dst + 1
    _FLIP_LABEL_MAP[dst + 1] = src + 1

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.count = 0
    def update(self, val, n=1): self.sum += val * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)

@torch.no_grad()
def compute_confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    idx = num_classes * target[mask].long() + pred[mask].long()
    conf = torch.bincount(idx, minlength=num_classes**2)
    return conf.reshape(num_classes, num_classes)

def iou_from_confusion(conf, eps=1e-7):
    inter = torch.diag(conf).float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    return (inter + eps) / (union + eps)

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)

class PowerMonitor:
    def __init__(self, device="cuda"):
        self.device = device
        self.power_samples = []
        self._nvidia_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvidia_available = True
        except:
            pass
        try:
            import psutil
            self._psutil = psutil
        except:
            self._psutil = None

    def sample(self):
        if self._nvidia_available:
            try:
                power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self.power_samples.append(power)
            except:
                pass

    def get_stats(self):
        if not self.power_samples:
            return {"avg_power_W": None, "max_power_W": None, "min_power_W": None}
        return {
            "avg_power_W": float(np.mean(self.power_samples)),
            "max_power_W": float(np.max(self.power_samples)),
            "min_power_W": float(np.min(self.power_samples)),
            "num_samples": len(self.power_samples)
        }

def preprocess_with_rembg(src_dir, dst_dir):
    if os.path.exists(dst_dir) and len(os.listdir(dst_dir)) > 0:
        return
    os.makedirs(dst_dir, exist_ok=True)
    try:
        from rembg import remove
    except ImportError:
        for f in os.listdir(src_dir):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        return
    for f in os.listdir(src_dir):
        if not f.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        in_path = os.path.join(src_dir, f)
        out_path = os.path.join(dst_dir, f)
        with open(in_path, 'rb') as i:
            input_data = i.read()
        output_data = remove(input_data)
        with open(out_path, 'wb') as o:
            o.write(output_data)

def polygons_to_mask(label_path, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        poly = coords.astype(np.int32)
        cv2.fillPoly(mask, [poly], color=cls_id + 1)
    return mask

def mask_to_boundary(mask, dilation_ksize=3):
    kernel = np.ones((dilation_ksize, dilation_ksize), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (dilated != eroded).astype(np.uint8)

class CarPartsSegDataset(Dataset):
    def __init__(self, root, split="train", img_size=512, augment=False,
                 normalize="imagenet", use_rembg=False):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.normalize = normalize
        img_dir = os.path.join(root, "images", split)
        if use_rembg:
            rembg_dir = os.path.join(root, "images_rembg", split)
            if os.path.isdir(rembg_dir) and len(os.listdir(rembg_dir)) > 0:
                img_dir = rembg_dir
        self.image_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {img_dir}")
        self.label_dir = os.path.join(root, "labels", split)

    def __len__(self): return len(self.image_paths)

    def _label_path(self, image_path):
        base = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.label_dir, base + ".txt")

    def _letterbox(self, img, mask):
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        canvas_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        top = (self.img_size - nh) // 2
        left = (self.img_size - nw) // 2
        canvas_img[top:top+nh, left:left+nw] = img_r
        canvas_mask[top:top+nh, left:left+nw] = mask_r
        return canvas_img, canvas_mask

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = cv2.LUT(mask, _FLIP_LABEL_MAP)
        if np.random.rand() < 0.6:
            factor = 0.7 + np.random.rand() * 0.6
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-5, 5)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT_101,
                                 flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M, (w,h), borderMode=cv2.BORDER_CONSTANT,
                                  flags=cv2.INTER_NEAREST)
        return img, mask

    def _apply_normalization(self, img_np):
        img_f = img_np.astype(np.float32) / 255.0
        if self.normalize == "imagenet":
            img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalize == "minmax":
            img_f = (img_f - 0.5) / 0.5
        return img_f

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self._label_path(image_path)
        img_bgr = cv2.imread(image_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = polygons_to_mask(label_path, h, w)
        img, mask = self._letterbox(img, mask)
        if self.augment:
            img, mask = self._augment(img, mask)
        boundary = mask_to_boundary(mask)
        img_f = self._apply_normalization(img)
        img_t = torch.from_numpy(img_f).permute(2,0,1).float()
        mask_t = torch.from_numpy(mask).long()
        boundary_t = torch.from_numpy(boundary).float()
        return {"image": img_t, "mask": mask_t, "boundary": boundary_t, "path": image_path}

def get_stratified_indices(dataset, samples_per_class):
    class_to_indices = defaultdict(list)
    for idx, path in enumerate(dataset.image_paths):
        label_path = dataset._label_path(path)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                classes = set()
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0]) + 1
                        classes.add(cls)
                for c in classes:
                    class_to_indices[c].append(idx)
    selected = set()
    for cls, indices in class_to_indices.items():
        if len(indices) > samples_per_class:
            selected.update(np.random.choice(indices, samples_per_class, replace=False))
        else:
            selected.update(indices)
    return sorted(selected)

class StratifiedSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices
        self.image_paths = [base_dataset.image_paths[i] for i in indices]
        self.label_dir = base_dataset.label_dir
        self.img_size = base_dataset.img_size
        self.augment = False
        self.normalize = base_dataset.normalize
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.base[self.indices[idx]]

# ---------------------------------------------------------------------------
# NEW: class weight computation (combats class imbalance, e.g. tiny mirrors
# getting drowned out by huge background/door/hood regions).
# ---------------------------------------------------------------------------
def compute_class_weights(dataset, num_classes, method="inv_sqrt_freq",
                          ignore_class_ids=None, cache_path=None):
    """
    Computes per-class pixel-frequency-based loss weights from a dataset's
    training labels (polygon area, not full mask rendering, for speed).

    method:
      "none"          -> all weights 1.0 (no-op, matches original behavior)
      "inv_freq"      -> weight_c = 1 / freq_c
      "inv_sqrt_freq" -> weight_c = 1 / sqrt(freq_c)   (gentler, recommended)

    Weights are normalized to mean 1.0 so overall loss magnitude / other
    lambda hyperparameters stay comparable to an unweighted run.
    ignore_class_ids: iterable of int class ids (0 = background, 1..N = parts)
    whose weight is forced to 0 (excludes them from the loss entirely).
    """
    ignore_class_ids = set(ignore_class_ids or [])

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        if cached.get("method") == method and cached.get("num_classes") == num_classes \
           and sorted(cached.get("ignore_class_ids", [])) == sorted(ignore_class_ids):
            return torch.tensor(cached["weights"], dtype=torch.float32)

    if method == "none":
        weights = np.ones(num_classes, dtype=np.float64)
        for c in ignore_class_ids:
            if 0 <= c < num_classes:
                weights[c] = 0.0
    else:
        pixel_counts = np.zeros(num_classes, dtype=np.float64)
        image_paths = getattr(dataset, "image_paths", None)
        label_dir = dataset.label_dir
        for path in image_paths:
            base = os.path.splitext(os.path.basename(path))[0]
            label_path = os.path.join(label_dir, base + ".txt")
            if not os.path.exists(label_path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0]) + 1  # +1 for background offset
                    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    area = cv2.contourArea(coords.astype(np.float32))
                    if 0 <= cls_id < num_classes:
                        pixel_counts[cls_id] += max(area, 1.0)
            # crude background estimate: remaining area not covered by any polygon
        # background gets whatever isn't accounted for; approximate via a floor
        total_seen = pixel_counts.sum()
        if pixel_counts[0] <= 0:
            pixel_counts[0] = max(total_seen, 1.0)  # ensure background isn't zero/inf-weighted

        pixel_counts = np.clip(pixel_counts, 1.0, None)
        freq = pixel_counts / pixel_counts.sum()

        if method == "inv_freq":
            weights = 1.0 / freq
        elif method == "inv_sqrt_freq":
            weights = 1.0 / np.sqrt(freq)
        else:
            raise ValueError(f"Unknown class-weighting method: {method}")

        for c in ignore_class_ids:
            if 0 <= c < num_classes:
                weights[c] = 0.0

        # normalize so mean of *active* (non-ignored) weights is 1.0
        active = weights[weights > 0]
        if active.size > 0:
            weights = weights / active.mean()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "method": method,
                "num_classes": num_classes,
                "ignore_class_ids": sorted(ignore_class_ids),
                "weights": weights.tolist()
            }, f, indent=2)

    return torch.tensor(weights, dtype=torch.float32)

class GhostModule(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_kernel=3,
                 stride=1, act=True, dropout=0.0):
        super().__init__()
        primary = max(1, out_c // ratio)
        cheap = out_c - primary
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, primary, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(primary),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(primary, cheap, dw_kernel, 1, dw_kernel//2,
                      groups=primary, bias=False),
            nn.BatchNorm2d(cheap),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y1 = self.primary(x)
        y2 = self.cheap(y1)
        out = torch.cat([y1, y2], dim=1)
        return self.drop(out[:, :, :, :])

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch//reduction), bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, ch//reduction), ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.pool(x).view(b,c)
        w = self.fc(w).view(b,c,1,1)
        return x * w

class GhostBottleneck(nn.Module):
    def __init__(self, in_c, out_c, expand=2, stride=1, dropout=0.0):
        super().__init__()
        hidden = in_c * expand
        self.residual = (stride == 1 and in_c == out_c)
        layers = [GhostModule(in_c, hidden, kernel_size=1, dropout=dropout)]
        if stride > 1:
            layers += [
                nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True)
            ]
        layers += [SEBlock(hidden),
                   GhostModule(hidden, out_c, kernel_size=1, act=False, dropout=dropout)]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out += x
        return out

class LargeKernelStem(nn.Module):
    def __init__(self, in_c=3, out_c=32, k=7, dropout=0.0):
        super().__init__()
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.dw = nn.Conv2d(out_c, out_c, k, 2, k//2, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        x = self.pw(x)
        x = self.dw(x)
        return self.drop(self.act(self.bn(x)))

class SpatialSkipGate(nn.Module):
    def __init__(self, skip_ch, dec_ch, inter=None, dropout=0.0):
        super().__init__()
        inter = max(1, skip_ch // 2) if inter is None else inter
        self.se = SEBlock(skip_ch)
        self.theta = nn.Conv2d(dec_ch, inter, 1)
        self.phi = nn.Conv2d(skip_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, dec, skip):
        skip_se = self.se(skip)
        g = self.theta(dec)
        x = self.phi(skip_se)
        attn = torch.sigmoid(self.psi(F.silu(g + x)))
        return skip_se * attn

class GhostUpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, dropout=0.0):
        super().__init__()
        self.gate = SpatialSkipGate(skip_c, in_c, dropout=dropout)
        self.fuse = GhostModule(in_c + skip_c, out_c, kernel_size=3, dropout=dropout)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class CoordConv(nn.Module):
    @staticmethod
    def add_coords(x, normalize=True):
        b,c,h,w = x.shape
        device, dtype = x.device, x.dtype
        xs = torch.linspace(-1,1,w,device=device,dtype=dtype).view(1,1,1,w).expand(b,1,h,w)
        ys = torch.linspace(-1,1,h,device=device,dtype=dtype).view(1,1,h,1).expand(b,1,h,w)
        return torch.cat([x, xs, ys], dim=1)
    def forward(self, x): return self.add_coords(x)

class PartLiteUNet(nn.Module):
    def __init__(self, num_classes=23, base_channels=32, include_background=True,
                 backbone="scratch", pretrained=True, dropout=0.0, use_coords=True):
        super().__init__()
        self.use_coords = use_coords
        self.backbone_name = backbone
        self.dropout = dropout
        self.out_classes = num_classes + (1 if include_background else 0)

        if backbone == "mobilenetv3":
            from torchvision import models
            self.encoder = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
            skip_channels = [48, 24, 16, 16, 576]
            dec_channels   = [128, 64, 32, 16, 8]
            self.coord_conv = nn.Identity() if not use_coords else CoordConv()
            self.bottleneck = GhostModule(skip_channels[-1] + (2 if use_coords else 0),
                                          dec_channels[0], kernel_size=3, dropout=dropout)
        else:
            c1, c2, c3, c4, c5 = (base_channels, base_channels*2, base_channels*4,
                                  base_channels*8, base_channels*16)
            stem_in = 5 if use_coords else 3
            self.stem = LargeKernelStem(stem_in, c1, dropout=dropout)
            self.enc1 = GhostBottleneck(c1, c1, expand=2, stride=1, dropout=dropout)
            self.down1 = GhostBottleneck(c1, c2, expand=2, stride=2, dropout=dropout)
            self.enc2 = GhostBottleneck(c2, c2, expand=2, stride=1, dropout=dropout)
            self.down2 = GhostBottleneck(c2, c3, expand=3, stride=2, dropout=dropout)
            self.enc3 = GhostBottleneck(c3, c3, expand=3, stride=1, dropout=dropout)
            self.down3 = GhostBottleneck(c3, c4, expand=3, stride=2, dropout=dropout)
            self.enc4 = GhostBottleneck(c4, c4, expand=3, stride=1, dropout=dropout)
            self.down4 = GhostBottleneck(c4, c5, expand=4, stride=2, dropout=dropout)
            self.bottleneck = GhostBottleneck(c5, c5, expand=4, stride=1, dropout=dropout)
            skip_channels = [c4, c3, c2, c1, c1]
            dec_channels   = [c5, c4, c3, c2, c1]

        self.up4 = GhostUpBlock(dec_channels[0], skip_channels[0], dec_channels[1], dropout=dropout)
        self.up3 = GhostUpBlock(dec_channels[1], skip_channels[1], dec_channels[2], dropout=dropout)
        self.up2 = GhostUpBlock(dec_channels[2], skip_channels[2], dec_channels[3], dropout=dropout)
        self.up1 = GhostUpBlock(dec_channels[3], skip_channels[3], dec_channels[4], dropout=dropout)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            GhostModule(dec_channels[4], dec_channels[4], kernel_size=3, dropout=dropout),
        )

        self.seg_head = nn.Conv2d(dec_channels[4], self.out_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(dec_channels[4], 1, kernel_size=1)
        self.aux_head = nn.Conv2d(dec_channels[3], self.out_classes, kernel_size=1)

    def _extract_scratch(self, x):
        if self.use_coords:
            x = CoordConv.add_coords(x)
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.down1(x1); x2 = self.enc2(x2)
        x3 = self.down2(x2); x3 = self.enc3(x3)
        x4 = self.down3(x3); x4 = self.enc4(x4)
        x5 = self.down4(x4); x5 = self.bottleneck(x5)
        return [x4, x3, x2, x1, x5]

    def _extract_mobilenetv3(self, x):
        feats = self.encoder(x)
        x1 = feats[0]; x2 = feats[1]; x3 = feats[3]; x4 = feats[7]; x5 = feats[12]
        if self.use_coords:
            x5 = self.coord_conv(x5)
        x5 = self.bottleneck(x5)
        return [x4, x3, x2, x1, x5]

    def forward(self, x):
        input_size = x.shape[-2:]
        if self.backbone_name == "scratch":
            skips = self._extract_scratch(x)
        else:
            skips = self._extract_mobilenetv3(x)

        x4, x3, x2, x1, x5 = skips
        d4 = self.up4(x5, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        aux_logits = self.aux_head(d2)
        d1 = self.up1(d2, x1)
        d0 = self.final_up(d1)

        seg = self.seg_head(d0)
        boundary = self.boundary_head(d0)
        aux = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)

        return {"seg": seg, "boundary": boundary, "aux_seg": aux}

    def unfreeze_backbone(self):
        if self.backbone_name == "mobilenetv3":
            for p in self.encoder.parameters(): p.requires_grad = True

    def count_flops(self, img_size=512):
        try:
            from fvcore.nn import FlopCountAnalysis
            h,w = (img_size, img_size) if isinstance(img_size, int) else img_size
            dummy = torch.randn(1,3,h,w)
            flops = FlopCountAnalysis(self, dummy)
            return flops.total() / 1e6
        except:
            return None

def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights  # optional tensor [num_classes]

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = (probs * target_onehot).sum(dims)
        union = (probs + target_onehot).sum(dims)
        dice = (2*inter + self.smooth) / (union + self.smooth)
        if self.class_weights is not None:
            w = self.class_weights.to(dice.device)
            active = w > 0
            if active.any():
                return 1 - (dice[active] * w[active]).sum() / w[active].sum()
        return 1 - dice.mean()

class BoundaryBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
    def forward(self, logits, target):
        logits = logits.squeeze(1)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight.to(logits.device)
        )

class PartLiteUNetLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, lambda_aux=0.4,
                 lambda_boundary=0.5, pos_weight=5.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes, class_weights=class_weights)
        self.boundary = BoundaryBCELoss(pos_weight)
        self.lambda_aux = lambda_aux
        self.lambda_boundary = lambda_boundary

    def forward(self, outputs, mask, boundary):
        seg = outputs["seg"]
        aux = outputs["aux_seg"]
        bnd = outputs["boundary"]

        main_ce = self.ce(seg, mask)
        main_dice = self.dice(seg, mask)
        main_loss = main_ce + main_dice

        aux_ce = self.ce(aux, mask)
        aux_dice = self.dice(aux, mask)
        aux_loss = aux_ce + aux_dice

        bnd_loss = self.boundary(bnd, boundary)

        total = main_loss + self.lambda_aux * aux_loss + self.lambda_boundary * bnd_loss
        return {
            "total": total,
            "main_ce": main_ce.detach(),
            "main_dice": main_dice.detach(),
            "aux_loss": aux_loss.detach(),
            "boundary_loss": bnd_loss.detach(),
        }

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=20):
    model.train()
    loss_meter = AverageMeter()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        boundaries = batch["boundary"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss_dict = criterion(outputs, masks, boundaries)
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        if step % log_interval == 0:
            print(f"  Epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f} "
                  f"(avg {loss_meter.avg:.4f}) ce={loss_dict['main_ce']:.4f} "
                  f"dice={loss_dict['main_dice']:.4f} bnd={loss_dict['boundary_loss']:.4f}")
    return loss_meter.avg

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, ignore_class_ids=None):
    model.eval()
    loss_meter = AverageMeter()
    conf_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        boundaries = batch["boundary"].to(device)
        outputs = model(images)
        loss_dict = criterion(outputs, masks, boundaries)
        loss_meter.update(loss_dict["total"].item(), images.size(0))
        preds = outputs["seg"].argmax(dim=1)
        conf_total += compute_confusion_matrix(preds.flatten().cpu(), masks.flatten().cpu(), num_classes)
    per_class_iou = iou_from_confusion(conf_total)
    if ignore_class_ids:
        keep = torch.ones(num_classes, dtype=torch.bool)
        for c in ignore_class_ids:
            if 0 <= c < num_classes:
                keep[c] = False
        mean_iou = per_class_iou[keep].mean().item() if keep.any() else per_class_iou.mean().item()
    else:
        mean_iou = per_class_iou.mean().item()
    return loss_meter.avg, mean_iou, per_class_iou, conf_total

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes, ignore_class_ids=None):
    model.eval()
    conf_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    inference_times = []
    # collected for eval plots: per-pixel max-class probability + correctness,
    # and per-class probability maps for PR/F1 curves (subsampled to keep memory sane)
    all_probs = []   # list of [N_pix, num_classes] float16 chunks (subsampled)
    all_targets = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        inference_times.append(end - start)
        logits = outputs["seg"]
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        conf_total += compute_confusion_matrix(preds.flatten().cpu(), masks.flatten().cpu(), num_classes)

        # subsample pixels per-image to bound memory for curve computation
        b, c, h, w = probs.shape
        probs_flat = probs.permute(0,2,3,1).reshape(-1, c)
        masks_flat = masks.reshape(-1)
        n_pix = probs_flat.shape[0]
        n_sample = min(n_pix, 20000)
        idx = torch.randperm(n_pix)[:n_sample]
        all_probs.append(probs_flat[idx].half().cpu())
        all_targets.append(masks_flat[idx].cpu())

    per_class_iou = iou_from_confusion(conf_total)
    if ignore_class_ids:
        keep = torch.ones(num_classes, dtype=torch.bool)
        for c in ignore_class_ids:
            if 0 <= c < num_classes:
                keep[c] = False
        mean_iou = per_class_iou[keep].mean().item() if keep.any() else per_class_iou.mean().item()
    else:
        mean_iou = per_class_iou.mean().item()
    avg_inf_time = np.mean(inference_times) * 1000

    probs_cat = torch.cat(all_probs, dim=0) if all_probs else None
    targets_cat = torch.cat(all_targets, dim=0) if all_targets else None

    return mean_iou, per_class_iou, conf_total, avg_inf_time, probs_cat, targets_cat

# ---------------------------------------------------------------------------
# NEW: YOLO-style evaluation report (replaces Grad-CAM).
# All plots are pixel-probability-based sweeps (this is a segmentation model,
# not a detector, so "confidence" = per-pixel softmax probability).
# ---------------------------------------------------------------------------
def generate_eval_plots(probs, targets, per_class_iou, conf_total, class_names,
                        training_metrics, results_dir, ignore_class_ids=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping eval plots (pip install matplotlib)")
        return

    plots_dir = os.path.join(results_dir, "eval", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    ignore_class_ids = set(ignore_class_ids or [])
    num_classes = len(class_names)
    thresholds = np.linspace(0.01, 0.99, 50)

    probs_np = probs.numpy().astype(np.float32)      # [N, C]
    targets_np = targets.numpy()                      # [N]

    # --- per-class precision/recall/F1 vs threshold ---
    curves = {}  # cls -> dict(thr, precision, recall, f1)
    for c in range(num_classes):
        if c in ignore_class_ids:
            continue
        is_pos = (targets_np == c)
        if is_pos.sum() == 0:
            continue
        p_c = probs_np[:, c]
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            pred_pos = p_c >= t
            tp = np.logical_and(pred_pos, is_pos).sum()
            fp = np.logical_and(pred_pos, ~is_pos).sum()
            fn = np.logical_and(~pred_pos, is_pos).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
        curves[c] = dict(thr=thresholds, precision=np.array(precisions),
                         recall=np.array(recalls), f1=np.array(f1s))

    def _plot_curve(xkey, ykey, xlabel, ylabel, title, fname, sort_by_x=False):
        fig, ax = plt.subplots(figsize=(6,5))
        all_y_interp = []
        common_x = np.linspace(0, 1, 100)
        for c, d in curves.items():
            x, y = d[xkey], d[ykey]
            if sort_by_x:
                order = np.argsort(x)
                x, y = x[order], y[order]
            ax.plot(x, y, color="gray", linewidth=0.8, alpha=0.6)
            interp_y = np.interp(common_x, x, y) if sort_by_x else np.interp(common_x, d["thr"], y)
            all_y_interp.append(interp_y)
        if all_y_interp:
            mean_y = np.mean(all_y_interp, axis=0)
            ax.plot(common_x, mean_y, color="blue", linewidth=2.5, label="all classes")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.set_xlim(0,1)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close(fig)

    _plot_curve("recall", "precision", "Recall", "Precision", "Precision-Recall Curve",
               "pr_curve.png", sort_by_x=True)
    _plot_curve("thr", "f1", "Confidence", "F1", "F1-Confidence Curve",
               "f1_confidence_curve.png")
    _plot_curve("thr", "precision", "Confidence", "Precision", "Precision-Confidence Curve",
               "precision_confidence_curve.png")
    _plot_curve("thr", "recall", "Confidence", "Recall", "Recall-Confidence Curve",
               "recall_confidence_curve.png")

    # --- confusion matrices (raw + normalized) ---
    cm = conf_total.numpy().astype(np.float64)
    for normalize, fname, title in [(False, "confusion_matrix.png", "Confusion Matrix"),
                                     (True, "confusion_matrix_normalized.png", "Confusion Matrix (Normalized)")]:
        mat = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else cm
        fig, ax = plt.subplots(figsize=(10,9))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close(fig)

    # --- class distribution (pixel counts in ground truth, from confusion matrix row sums) ---
    counts = cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(class_names, counts, color="steelblue")
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_ylabel("Pixel count (sampled)")
    ax.set_title("Class Distribution (Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # --- per-class IoU bar chart ---
    fig, ax = plt.subplots(figsize=(12,5))
    ious = per_class_iou.numpy()
    colors = ["crimson" if i in ignore_class_ids else "seagreen" for i in range(num_classes)]
    ax.bar(class_names, ious, color=colors)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_ylabel("IoU"); ax.set_ylim(0,1)
    ax.set_title("Per-Class IoU (red = ignored in mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "per_class_iou.png"), dpi=150)
    plt.close(fig)

    # --- training curves ---
    if training_metrics:
        epochs = [m["epoch"] for m in training_metrics]
        fig, axes = plt.subplots(1, 3, figsize=(15,4))
        axes[0].plot(epochs, [m["train_loss"] for m in training_metrics], label="train_loss")
        axes[0].plot(epochs, [m["val_loss"] for m in training_metrics], label="val_loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_title("Loss"); axes[0].legend()
        axes[1].plot(epochs, [m["val_miou"] for m in training_metrics], color="green")
        axes[1].set_xlabel("Epoch"); axes[1].set_title("Val mIoU")
        axes[2].plot(epochs, [m["lr"] for m in training_metrics], color="orange")
        axes[2].set_xlabel("Epoch"); axes[2].set_title("Learning Rate")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=150)
        plt.close(fig)

    print(f"Eval plots saved to {plots_dir}")

# ---------------------------------------------------------------------------
# NEW: instance-level (connected-component) mask mAP@0.5 — the segmentation
# analogue of YOLO's box mAP. Each connected component in a per-class
# prediction mask is treated as a "detection" with confidence = mean softmax
# probability inside the component; matched against ground-truth components
# by mask IoU >= threshold.
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_mask_map(model, loader, device, num_classes, class_names,
                     iou_threshold=0.5, ignore_class_ids=None):
    ignore_class_ids = set(ignore_class_ids or [])
    # per class: list of (confidence, is_tp) across whole test set, plus total GT count
    detections = {c: [] for c in range(num_classes)}
    gt_counts = {c: 0 for c in range(num_classes)}

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].cpu().numpy()  # [B,H,W]
        outputs = model(images)
        probs = F.softmax(outputs["seg"], dim=1).cpu().numpy()  # [B,C,H,W]
        preds = probs.argmax(axis=1)  # [B,H,W]

        for b in range(masks.shape[0]):
            gt_mask = masks[b]
            pred_mask = preds[b]
            prob_map = probs[b]
            for c in range(num_classes):
                if c in ignore_class_ids:
                    continue
                gt_bin = (gt_mask == c).astype(np.uint8)
                pred_bin = (pred_mask == c).astype(np.uint8)

                n_gt, gt_labels = cv2.connectedComponents(gt_bin)
                n_pred, pred_labels = cv2.connectedComponents(pred_bin)
                gt_components = [(gt_labels == i) for i in range(1, n_gt)]
                pred_components = [(pred_labels == i) for i in range(1, n_pred)]
                gt_counts[c] += len(gt_components)

                matched_gt = set()
                # confidence = mean predicted prob for class c inside the component
                pred_with_conf = []
                for comp in pred_components:
                    conf = float(prob_map[c][comp].mean()) if comp.any() else 0.0
                    pred_with_conf.append((conf, comp))
                pred_with_conf.sort(key=lambda x: -x[0])

                for conf, comp in pred_with_conf:
                    best_iou, best_j = 0.0, -1
                    for j, gt_comp in enumerate(gt_components):
                        if j in matched_gt:
                            continue
                        inter = np.logical_and(comp, gt_comp).sum()
                        union = np.logical_or(comp, gt_comp).sum()
                        iou = inter / max(union, 1)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    is_tp = best_iou >= iou_threshold
                    if is_tp:
                        matched_gt.add(best_j)
                    detections[c].append((conf, is_tp))

    ap_per_class = {}
    for c in range(num_classes):
        if c in ignore_class_ids or gt_counts[c] == 0:
            continue
        dets = sorted(detections[c], key=lambda x: -x[0])
        if not dets:
            ap_per_class[class_names[c]] = 0.0
            continue
        tp = np.array([d[1] for d in dets], dtype=np.float64)
        fp = 1 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(gt_counts[c], 1)
        precisions = tp_cum / np.clip(tp_cum + fp_cum, 1, None)
        # 11-point interpolation (simple, robust)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            p = precisions[mask].max() if mask.any() else 0.0
            ap += p / 11.0
        ap_per_class[class_names[c]] = ap

    mean_ap = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return mean_ap, ap_per_class

# ---------------------------------------------------------------------------
# NEW: --track-internals helpers. These log AGGREGATE / REPRESENTATIVE weight
# and activation info, not full per-neuron dumps every epoch — full state_dicts
# every epoch for 150 epochs would be gigabytes and isn't visualizable anyway.
# Designed to feed a Manim explainer: layer norm curves, weight histograms at
# sparse checkpoints, and activation-map snapshots on a fixed image.
# ---------------------------------------------------------------------------
def log_layer_norms(model):
    """One float per parameter tensor (weight norm + grad norm). Tiny, cheap,
    safe to call every epoch. This is the data for a 'watch every layer's
    weight/gradient magnitude evolve over training' stacked line chart."""
    norms = {}
    for name, param in model.named_parameters():
        norms[name] = {
            "weight_norm": float(param.data.norm().item()),
            "grad_norm": float(param.grad.norm().item()) if param.grad is not None else None,
        }
    return norms

def weight_histogram_snapshot(model, bins=50):
    """Per-layer weight histogram (counts + bin edges), not raw values.
    Call this every N epochs (e.g. every 10), not every epoch."""
    hist_data = {}
    for name, param in model.named_parameters():
        vals = param.data.detach().cpu().numpy().flatten()
        h, edges = np.histogram(vals, bins=bins)
        hist_data[name] = {"counts": h.tolist(), "bin_edges": edges.tolist()}
    return hist_data

@torch.no_grad()
def save_activation_snapshot(model, fixed_image, device, epoch, out_dir):
    """Runs one fixed validation image through the model and saves intermediate
    activation maps at a few representative layers. This is the 'what is the
    network looking at as it learns' data — far more interpretable for video
    than raw weights, and much smaller than a full checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    layers_to_hook = []
    for name in ["stem", "enc1", "enc2", "enc3", "bottleneck"]:
        if hasattr(model, name):
            layers_to_hook.append((name, getattr(model, name)))

    acts = {}
    hooks = []
    def _make_hook(layer_name):
        def _hook(module, inp, outp):
            acts[layer_name] = outp.detach().cpu()
        return _hook
    for name, module in layers_to_hook:
        hooks.append(module.register_forward_hook(_make_hook(name)))

    was_training = model.training
    model.eval()
    model(fixed_image.unsqueeze(0).to(device))
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_activations.npz")
    np.savez_compressed(save_path, **{k: v.numpy() for k, v in acts.items()})
    return save_path

def run_training(config):
    config = dict(config)  # local copy; don't mutate caller's dict

    device = torch.device(config["device"])
    config["device"] = device

    base_channels = config["base_channels"]
    run_name = f"base{base_channels}"
    results_dir = os.path.join(config["results_root"], run_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "eval"), exist_ok=True)

    track_internals = config.get("track_internals", False)
    internals_every = config.get("track_internals_every", 10)
    if track_internals:
        os.makedirs(os.path.join(results_dir, "logs", "weight_histograms"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "logs", "activations"), exist_ok=True)

    print(f"\n{'='*60}\nTraining variant: {run_name}\n{'='*60}")

    if config["use_rembg"]:
        for split in ["train", "val", "test"]:
            src = os.path.join(config["data_root"], "images", split)
            dst = os.path.join(config["data_root"], "images_rembg", split)
            preprocess_with_rembg(src, dst)

    train_ds = CarPartsSegDataset(config["data_root"], split="train",
                                  img_size=config["img_size"], augment=True,
                                  normalize=config["normalize"], use_rembg=config["use_rembg"])
    val_ds_full = CarPartsSegDataset(config["data_root"], split="val",
                                    img_size=config["img_size"], augment=False,
                                    normalize=config["normalize"], use_rembg=config["use_rembg"])

    val_samples = config.get("stratified_samples", 10)
    val_indices = get_stratified_indices(val_ds_full, val_samples)
    val_ds = StratifiedSubset(val_ds_full, val_indices)
    print(f"Stratified val set: {len(val_ds)} images (sampled {val_samples} per class)")

    # Fixed image for activation-snapshot tracking across epochs (same image
    # every time so the sequence is directly comparable / animatable).
    fixed_viz_image = val_ds[0]["image"] if len(val_ds) > 0 else None

    test_ds_full = CarPartsSegDataset(config["data_root"], split="test",
                                      img_size=config["img_size"], augment=False,
                                      normalize=config["normalize"], use_rembg=config["use_rembg"])
    test_indices = get_stratified_indices(test_ds_full, val_samples)
    test_ds = StratifiedSubset(test_ds_full, test_indices)
    print(f"Stratified test set: {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"],
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"],
                             pin_memory=True)

    # --- class weighting / ignore-classes setup ---
    ignore_names = config.get("ignore_classes", []) or []
    ignore_class_ids = {CLASS_NAMES.index(n) for n in ignore_names if n in CLASS_NAMES}
    for n in ignore_names:
        if n not in CLASS_NAMES:
            print(f"WARNING: --ignore-classes '{n}' not a known class name, skipping")

    class_weight_method = config.get("class_weighting", "none")
    cache_path = os.path.join(config["results_root"], f"class_weights_{class_weight_method}.json")
    class_weights = compute_class_weights(train_ds, NUM_CLASSES_WITH_BG,
                                          method=class_weight_method,
                                          ignore_class_ids=ignore_class_ids,
                                          cache_path=cache_path)
    print(f"Class weighting: {class_weight_method}, ignored: {sorted(ignore_names)}")
    print(f"Class weights: {dict(zip(CLASS_NAMES, [round(w,3) for w in class_weights.tolist()]))}")

    model = PartLiteUNet(num_classes=NUM_CLASSES,
                         base_channels=base_channels,
                         include_background=True,
                         backbone="scratch",
                         dropout=config["dropout"],
                         use_coords=True).to(device)
    params = count_parameters(model)
    flops = model.count_flops(config["img_size"])
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {flops:.2f} MFLOPs" if flops else "FLOPs: n/a")

    criterion = PartLiteUNetLoss(num_classes=NUM_CLASSES_WITH_BG,
                                 class_weights=class_weights.to(device),
                                 lambda_aux=config["lambda_aux"],
                                 lambda_boundary=config["lambda_boundary"])
    optimizer = AdamW(model.parameters(), lr=config["lr"],
                      weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    power_mon = PowerMonitor(device=str(device))
    best_miou = 0.0
    start_epoch = 0
    if config["resume"]:
        start_epoch, best_miou = load_checkpoint(config["resume"], model, optimizer, str(device))
        print(f"Resumed from {config['resume']}, epoch {start_epoch}, best mIoU {best_miou:.4f}")

    training_metrics = []
    layer_norm_log = []
    for epoch in range(start_epoch, config["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        power_mon.sample()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, epoch+1, config["log_interval"])
        val_loss, val_miou, per_class_iou, conf = validate(model, val_loader, criterion,
                                                           device, NUM_CLASSES_WITH_BG,
                                                           ignore_class_ids=ignore_class_ids)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_mIoU={val_miou:.4f} lr={lr:.2e}")

        training_metrics.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "lr": lr,
            "per_class_iou": per_class_iou.tolist()
        })

        if track_internals:
            # Cheap, every epoch: one weight-norm + grad-norm float per layer.
            layer_norm_log.append({
                "epoch": epoch+1,
                "norms": log_layer_norms(model),
            })
            # Sparse (every `internals_every` epochs), heavier: weight histogram
            # per layer + activation-map snapshot on the fixed val image.
            if (epoch + 1) % internals_every == 0 or (epoch + 1) == config["epochs"]:
                hist = weight_histogram_snapshot(model)
                with open(os.path.join(results_dir, "logs", "weight_histograms",
                                       f"epoch_{epoch+1:03d}.json"), "w") as f:
                    json.dump(hist, f)
                if fixed_viz_image is not None:
                    save_activation_snapshot(model, fixed_viz_image, device, epoch+1,
                                            os.path.join(results_dir, "logs", "activations"))
                print(f"  -> Saved weight histogram + activation snapshot (epoch {epoch+1})")

        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou": best_miou,
                "base_channels": base_channels
            }, os.path.join(results_dir, "weights", "best.pt"))
            print(f"  -> New best model saved")

    power_stats = power_mon.get_stats()
    with open(os.path.join(results_dir, "logs", "power_stats.json"), "w") as f:
        json.dump(power_stats, f, indent=2)
    with open(os.path.join(results_dir, "logs", "training_metrics.json"), "w") as f:
        json.dump(training_metrics, f, indent=2)
    if track_internals:
        with open(os.path.join(results_dir, "logs", "layer_norms.json"), "w") as f:
            json.dump(layer_norm_log, f)
        print(f"Layer norms logged for {len(layer_norm_log)} epochs -> logs/layer_norms.json")

    save_checkpoint({
        "epoch": config["epochs"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_miou": best_miou,
        "base_channels": base_channels
    }, os.path.join(results_dir, "weights", "last.pt"))

    if config["test_it"]:
        print("\n--- Evaluating on test set ---")
        checkpoint_path = os.path.join(results_dir, "weights", "best.pt")
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, model, map_location=str(device))
        test_miou, test_iou, test_conf, avg_inf_time, probs_sample, targets_sample = evaluate_model(
            model, test_loader, device, NUM_CLASSES_WITH_BG, ignore_class_ids=ignore_class_ids
        )
        print(f"Test mIoU: {test_miou:.4f}, Avg inference time: {avg_inf_time:.2f} ms")
        np.save(os.path.join(results_dir, "eval", "confusion_matrix.npy"), test_conf.numpy())
        with open(os.path.join(results_dir, "eval", "per_class_iou.csv"), "w") as f:
            f.write("class,iou\n")
            for name, iou in zip(CLASS_NAMES, test_iou.tolist()):
                f.write(f"{name},{iou:.4f}\n")

        summary = {
            "test_miou": test_miou,
            "mean_iou": test_miou,
            "params": params,
            "flops_MFLOPs": flops,
            "inference_time_ms": avg_inf_time,
            "power": power_stats,
            "per_class_iou": dict(zip(CLASS_NAMES, test_iou.tolist())),
            "ignored_classes": sorted(ignore_names),
            "class_weighting": class_weight_method,
        }

        if config.get("compute_map"):
            print("\n--- Computing instance-level mask mAP@0.5 ---")
            mean_ap, ap_per_class = compute_mask_map(
                model, test_loader, device, NUM_CLASSES_WITH_BG, CLASS_NAMES,
                iou_threshold=0.5, ignore_class_ids=ignore_class_ids
            )
            print(f"mAP@0.5: {mean_ap:.4f}")
            summary["mAP@0.5"] = mean_ap
            summary["AP_per_class@0.5"] = ap_per_class
            with open(os.path.join(results_dir, "eval", "map_summary.json"), "w") as f:
                json.dump({"mAP@0.5": mean_ap, "AP_per_class@0.5": ap_per_class}, f, indent=2)

        with open(os.path.join(results_dir, "eval", "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n--- Generating evaluation plots ---")
        if probs_sample is not None:
            generate_eval_plots(probs_sample, targets_sample, test_iou, test_conf,
                                CLASS_NAMES, training_metrics, results_dir,
                                ignore_class_ids=ignore_class_ids)

    return {
        "variant": run_name,
        "best_miou": best_miou,
        "params": params,
        "flops": flops,
        "power": power_stats,
    }

def objective(trial, base_config):
    config = dict(base_config)  # per-trial copy, never mutate the shared config

    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    config["lambda_aux"] = trial.suggest_float("lambda_aux", 0.1, 0.8)
    config["lambda_boundary"] = trial.suggest_float("lambda_boundary", 0.1, 0.8)
    config["epochs"] = config["optuna_epochs"]
    config["stratified_samples"] = config.get("optuna_samples", 10)

    base_channels = config["base_channels"]
    results_dir = os.path.join(config["results_root"], "optuna", f"trial_{trial.number}")
    os.makedirs(results_dir, exist_ok=True)

    def _write_result(status, best_val_loss):
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump({
                "trial": trial.number,
                "status": status,
                "best_val_loss": best_val_loss,
                "hyperparams": {k: v for k, v in config.items()
                                if k in ["lr","weight_decay","dropout","lambda_aux","lambda_boundary"]}
            }, f, indent=2)

    try:
        train_ds = CarPartsSegDataset(config["data_root"], split="train",
                                      img_size=config["img_size"], augment=True,
                                      normalize=config["normalize"], use_rembg=config["use_rembg"])
        val_ds_full = CarPartsSegDataset(config["data_root"], split="val",
                                        img_size=config["img_size"], augment=False,
                                        normalize=config["normalize"], use_rembg=config["use_rembg"])
        val_indices = get_stratified_indices(val_ds_full, config["stratified_samples"])
        val_ds = StratifiedSubset(val_ds_full, val_indices)

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                  shuffle=True, num_workers=config["num_workers"],
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                                shuffle=False, num_workers=config["num_workers"],
                                pin_memory=True)

        device = torch.device(config["device"])

        ignore_names = config.get("ignore_classes", []) or []
        ignore_class_ids = {CLASS_NAMES.index(n) for n in ignore_names if n in CLASS_NAMES}
        class_weight_method = config.get("class_weighting", "none")
        cache_path = os.path.join(config["results_root"], f"class_weights_{class_weight_method}.json")
        class_weights = compute_class_weights(train_ds, NUM_CLASSES_WITH_BG,
                                              method=class_weight_method,
                                              ignore_class_ids=ignore_class_ids,
                                              cache_path=cache_path)

        model = PartLiteUNet(num_classes=NUM_CLASSES,
                             base_channels=base_channels,
                             include_background=True,
                             backbone="scratch",
                             dropout=config["dropout"],
                             use_coords=True).to(device)
        criterion = PartLiteUNetLoss(num_classes=NUM_CLASSES_WITH_BG,
                                     class_weights=class_weights.to(device),
                                     lambda_aux=config["lambda_aux"],
                                     lambda_boundary=config["lambda_boundary"])
        optimizer = AdamW(model.parameters(), lr=config["lr"],
                          weight_decay=config["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

        best_val_loss = float('inf')
        for epoch in range(config["epochs"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                         device, epoch+1, config["log_interval"])
            val_loss, val_miou, _, _ = validate(model, val_loader, criterion,
                                                device, NUM_CLASSES_WITH_BG,
                                                ignore_class_ids=ignore_class_ids)
            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # --- MedianPruner hook ---
            # Report the intermediate value so the pruner can compare this
            # trial's trajectory against the median of prior trials at the
            # same epoch. If it's clearly worse, stop this trial early and
            # free the GPU/time for a more promising one.
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                _write_result("pruned", best_val_loss)
                # free GPU memory before Optuna unwinds this trial
                del model, optimizer, scheduler, criterion
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

            if val_loss < 0.30:
                break

        _write_result("completed", best_val_loss)
        del model, optimizer, scheduler, criterion
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return best_val_loss

    except optuna.exceptions.TrialPruned:
        raise  # must propagate untouched so Optuna records it as pruned, not failed

    except Exception as e:
        # Any other failure (OOM, bad batch, transient I/O error, etc.) fails
        # THIS trial only. Optuna records it as a failed trial and moves on to
        # the next one instead of crashing the whole `study.optimize` call.
        print(f"[trial {trial.number}] FAILED with error: {e!r} -- skipping this trial")
        _write_result("failed", float('inf'))
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return float('inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./carparts-seg")
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--variant", default="both", choices=["32", "48", "both"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--test_it", action="store_true")
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--stratified_samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-aux", type=float, default=0.4)
    parser.add_argument("--lambda-boundary", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-epochs", type=int, default=10)
    parser.add_argument("--optuna-samples", type=int, default=8)
    # NEW flags
    parser.add_argument("--class-weighting", default="none",
                        choices=["none", "inv_freq", "inv_sqrt_freq"],
                        help="Loss class weighting scheme to combat class imbalance")
    parser.add_argument("--ignore-classes", nargs="*", default=[],
                        help="Class names to zero-weight in loss and exclude from mean IoU, e.g. object")
    parser.add_argument("--compute-map", action="store_true",
                        help="Also compute instance-level mask mAP@0.5 (slower; connected-component matching)")
    parser.add_argument("--track-internals", action="store_true",
                        help="Log per-layer weight/grad norms every epoch, plus weight "
                             "histograms and activation snapshots every --track-internals-every "
                             "epochs. Only applies to the final training run, never Optuna trials.")
    parser.add_argument("--track-internals-every", type=int, default=10,
                        help="Epoch interval for weight histogram + activation snapshots")
    args = parser.parse_args()

    if args.demo:
        args.epochs = 1

    seed_everything(42)

    if args.optuna:
        if not _OPTUNA_AVAILABLE:
            print("Please install optuna: pip install optuna")
            sys.exit(1)
        if args.variant == "both":
            print("Optuna only supports a single variant. Choose --variant 32 or 48.")
            sys.exit(1)
        base_channels = int(args.variant)

        config = dict(vars(args))          # independent copy, not an alias of args.__dict__
        final_epochs = args.epochs         # captured before anything can mutate it

        config["base_channels"] = base_channels
        config["use_rembg"] = not args.no_rembg
        config["optuna_epochs"] = args.optuna_epochs
        config["optuna_samples"] = args.optuna_samples
        config["normalize"] = "imagenet"

        # MedianPruner: after `n_warmup_steps` epochs, a trial is pruned if its
        # reported val_loss at that step is worse than the median of all other
        # trials at the same step. n_startup_trials lets the first few trials
        # run to completion unpruned so the pruner has something to compare
        # against before it starts culling.
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study = optuna.create_study(direction="minimize", pruner=pruner)

        try:
            study.optimize(
                lambda trial: objective(trial, config),
                n_trials=args.optuna_trials,
                # A failed trial (caught inside objective()) already returns
                # inf and does not raise, so this is a belt-and-suspenders
                # guard: even if something unexpected escapes objective()
                # itself, Optuna will record it as FAIL and keep going
                # instead of stopping the whole study.
                catch=(Exception,),
            )
        except KeyboardInterrupt:
            print("\nOptuna search interrupted by user -- using best trial found so far.")

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print("\nNo Optuna trials completed successfully. Falling back to CLI-provided "
                  "hyperparameters for the final training run.")
            config["epochs"] = final_epochs
            config["test_it"] = args.test_it
            run_training(config)
            return

        print("\n=== Optuna Best Trial ===")
        print(f"Best validation loss: {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_params}")
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        print(f"Trials: {len(study.trials)} total, {len(completed)} completed, "
              f"{n_pruned} pruned, {n_failed} failed")

        os.makedirs(os.path.join(args.results_root, "optuna"), exist_ok=True)
        with open(os.path.join(args.results_root, "optuna", "best_params.json"), "w") as f:
            json.dump({
                "best_val_loss": study.best_value,
                "best_params": study.best_params,
                "n_trials_total": len(study.trials),
                "n_completed": len(completed),
                "n_pruned": n_pruned,
                "n_failed": n_failed,
            }, f, indent=2)

        print("\n--- Training final model with best hyperparameters ---")
        best_hp = study.best_params
        config["lr"] = best_hp["lr"]
        config["weight_decay"] = best_hp["weight_decay"]
        config["dropout"] = best_hp["dropout"]
        config["lambda_aux"] = best_hp["lambda_aux"]
        config["lambda_boundary"] = best_hp["lambda_boundary"]
        config["epochs"] = final_epochs    # restores the real requested epoch count
        config["test_it"] = args.test_it
        run_training(config)
        return

    variants = [32] if args.variant != "both" else [32, 48]
    all_reports = {}
    for bc in variants:
        config = dict(vars(args))
        config["base_channels"] = bc
        config["use_rembg"] = not args.no_rembg
        config["normalize"] = "imagenet"
        report = run_training(config)
        all_reports[f"base{bc}"] = report

    with open(os.path.join(args.results_root, "report.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "variants": all_reports
        }, f, indent=2)
    print("\nAll results saved.")

if __name__ == "__main__":
    main()