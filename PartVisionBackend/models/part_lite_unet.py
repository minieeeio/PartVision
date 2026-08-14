import torch
import torch.nn as nn
import torch.nn.functional as F


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
