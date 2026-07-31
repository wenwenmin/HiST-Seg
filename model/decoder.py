"""H&E-detail decoder for the improved model.

Only high-resolution H&E skip features are used. ST is restricted to the semantic
bottleneck and never modulates the 1/8 or 1/4 boundary features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def cba(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1, d: int = 1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False),
        nn.GroupNorm(_groups(cout), cout),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(cba(cin, cout), cba(cout, cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASPP(nn.Module):
    def __init__(self, cin: int, cout: int, rates=(1, 6, 12), mid: int | None = None):
        super().__init__()
        mid = mid or max(32, cin // 4)
        self.branches = nn.ModuleList([
            cba(cin, mid, 1, 1, 0) if r == 1 else cba(cin, mid, 3, 1, r, r)
            for r in rates
        ])
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1), cba(cin, mid, 1, 1, 0))
        self.proj = cba(mid * (len(rates) + 1), cout, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        pooled = F.interpolate(self.gp(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat(feats + [pooled], dim=1))


class UpFuse(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.fuse = DoubleConv(cout + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sem = int(cfg.dims.sem_dim)
        c8, c4, c2, c1 = map(int, cfg.dims.decoder_channels)
        s2c, s3c, s4c = map(int, cfg.dims.stem_channels)
        self.deep = bool(cfg.model.deep_supervision)

        self.bottleneck = ASPP(sem, sem) if bool(cfg.model.aspp) else DoubleConv(sem, sem)
        self.up8 = UpFuse(sem, s4c, c8)
        self.up4 = UpFuse(c8, s3c, c4)
        self.up2 = UpFuse(c4, s2c, c2)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(c2, c1, 2, 2), DoubleConv(c1, c1))
        self.head = nn.Conv2d(c1, 1, 1)
        if self.deep:
            self.aux8 = nn.Conv2d(c8, 1, 1)
            self.aux4 = nn.Conv2d(c4, 1, 1)

    def forward(self, bottleneck: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor,
                s4: torch.Tensor, out_size: tuple[int, int]):
        b = self.bottleneck(bottleneck)
        d8 = self.up8(b, s4)
        d4 = self.up4(d8, s3)
        d2 = self.up2(d4, s2)
        d1 = self.up1(d2)
        main = self.head(d1)
        if main.shape[-2:] != out_size:
            main = F.interpolate(main, size=out_size, mode="bilinear", align_corners=False)
        outputs = [main]
        if self.deep:
            aux4 = F.interpolate(self.aux4(d4), size=out_size, mode="bilinear", align_corners=False)
            aux8 = F.interpolate(self.aux8(d8), size=out_size, mode="bilinear", align_corners=False)
            outputs.extend([aux4, aux8])
        return outputs
