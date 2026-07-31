import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class IBN(nn.Module):
    """Instance-Group Normalization (IBN) module.

    Splits channels into two portions:
    - Half undergoes InstanceNorm to strip sample-specific style/stain statistics.
    - Half undergoes GroupNorm to preserve structural content.
    """

    def __init__(self, channels: int, ratio: float = 0.5, gn_preferred: int = 8):
        super().__init__()
        half = int(round(channels * float(ratio)))
        half = max(1, min(channels - 1, half))
        self.half = half
        self.other = channels - half
        self.instance = nn.InstanceNorm2d(self.half, affine=True)
        self.group = nn.GroupNorm(_groups(self.other, gn_preferred), self.other)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xin, xgn = torch.split(x, [self.half, self.other], dim=1)
        out_in = self.instance(xin.contiguous())
        out_gn = self.group(xgn.contiguous())
        return torch.cat([out_in, out_gn], dim=1)


class MixStyle(nn.Module):
    """Domain Generalization with MixStyle (Zhou et al., ICLR 2021).

    Mixes feature statistics (mean and std) between random samples within a batch
    during training to simulate domain shifts.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        if float(torch.rand(1).item()) > self.p:
            return x
        b = x.size(0)
        if b < 2:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_norm = (x - mu) / sig
        lmda = self._beta.sample((b, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(b, device=x.device)
        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)
        return x_norm * sig_mix + mu_mix


def conv_gn_act(cin, cout, k=3, s=1, p=1):
    """Convolution + GroupNorm + ReLU block."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.GroupNorm(_groups(cout), cout),
        nn.ReLU(inplace=True),
    )


def _conv_norm_act(cin, cout, norm_factory, k=3, s=1, p=1):
    """Convolution + Pluggable Normalization + ReLU block."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        norm_factory(cout),
        nn.ReLU(inplace=True),
    )


class ConvStem(nn.Module):
    """Convolutional Stem extracting multi-scale local details (1/2, 1/4, 1/8).

    Applies IBN and MixStyle on shallow features for domain generalization,
    while retaining pure GroupNorm at the deepest stem level (1/8).
    """

    def __init__(
        self,
        chs=(64, 128, 256),
        *,
        use_ibn: bool = True,
        ibn_ratio: float = 0.5,
        use_mixstyle: bool = True,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
    ):
        super().__init__()
        c0, c1, c2 = chs

        def gn_factory(c):
            return nn.GroupNorm(_groups(c), c)

        def ibn_factory(c):
            return IBN(c, ratio=ibn_ratio)

        shallow_norm = ibn_factory if use_ibn else gn_factory

        self.b1 = nn.Sequential(
            _conv_norm_act(3, c0, shallow_norm, 3, 2, 1),
            _conv_norm_act(c0, c0, shallow_norm),
        )
        self.b2 = nn.Sequential(
            _conv_norm_act(c0, c1, shallow_norm, 3, 2, 1),
            _conv_norm_act(c1, c1, shallow_norm),
        )
        self.b3 = nn.Sequential(
            _conv_norm_act(c1, c2, gn_factory, 3, 2, 1),
            _conv_norm_act(c2, c2, gn_factory),
        )

        if use_mixstyle:
            self.ms1 = MixStyle(mixstyle_p, mixstyle_alpha)
            self.ms2 = MixStyle(mixstyle_p, mixstyle_alpha)
        else:
            self.ms1 = nn.Identity()
            self.ms2 = nn.Identity()

    def forward(self, x):
        s2 = self.ms1(self.b1(x))
        s3 = self.ms2(self.b2(s2))
        s4 = self.b3(s3)
        return s2, s3, s4


class DetailStream(nn.Module):
    """Detail Stream combining ConvStem and frozen ViT semantic backbone."""

    def __init__(self, cfg, build_uni=True):
        super().__init__()
        d = cfg.dims
        self.uni_layers = list(d.uni_layers)
        self.embed_dim = d.uni_embed_dim
        self.sem_dim = d.sem_dim
        self.patch = int(cfg.geometry.uni_patch_size)

        m = getattr(cfg, "model", None)
        stem_kwargs = dict(
            use_ibn=bool(getattr(m, "stem_ibn", True)),
            ibn_ratio=float(getattr(m, "stem_ibn_ratio", 0.5)),
            use_mixstyle=bool(getattr(m, "stem_mixstyle", True)),
            mixstyle_p=float(getattr(m, "stem_mixstyle_p", 0.5)),
            mixstyle_alpha=float(getattr(m, "stem_mixstyle_alpha", 0.1)),
        )
        self.stem = ConvStem(tuple(d.stem_channels), **stem_kwargs)

        self.vit = None
        if build_uni:
            self.vit = self._build_uni(cfg)

        per = self.sem_dim // len(self.uni_layers)
        self.layer_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, per, 1, bias=False),
                nn.GroupNorm(_groups(per), per),
                nn.ReLU(inplace=True),
            )
            for _ in self.uni_layers
        ])
        self.sem_fuse = nn.Sequential(
            nn.Conv2d(per * len(self.uni_layers), self.sem_dim, 1, bias=False),
            nn.GroupNorm(_groups(self.sem_dim), self.sem_dim),
            nn.ReLU(inplace=True),
        )

    def _build_uni(self, cfg):
        import timm

        model_name = str(getattr(cfg.model, "uni_model_name", "hf-hub:MahmoodLab/uni"))
        vit = timm.create_model(
            model_name, pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        for p in vit.parameters():
            p.requires_grad = False
        vit.eval()
        return vit

    @torch.no_grad()
    def extract_uni(self, x):
        vit = self.vit
        if vit is None:
            raise RuntimeError("UNI model is not built. Build the model with build_uni=True.")
        vit.eval()
        B, _, H, W = x.shape
        ph, pw = H // self.patch, W // self.patch

        x_patch = vit.patch_embed(x)
        try:
            seq = vit._pos_embed(x_patch)
        except Exception:
            if x_patch.dim() == 4:
                seq = x_patch.flatten(2).transpose(1, 2)
            else:
                seq = x_patch
            cls = vit.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, seq], 1)
            pos = vit.pos_embed
            cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
            side = int(patch_pos.shape[1] ** 0.5)
            p2d = patch_pos.transpose(1, 2).view(1, self.embed_dim, side, side)
            p2d = F.interpolate(p2d, size=(ph, pw), mode='bicubic', align_corners=False)
            patch_pos = p2d.flatten(2).transpose(1, 2)
            seq = seq + torch.cat([cls_pos, patch_pos], 1)

        out = {}
        for i, blk in enumerate(vit.blocks):
            seq = blk(seq)
            if i in self.uni_layers:
                out[i] = seq[:, 1:, :].transpose(1, 2).reshape(B, self.embed_dim, ph, pw)
        return [out[i] for i in self.uni_layers]

    def forward(self, x):
        s2, s3, s4 = self.stem(x)
        feats = self.extract_uni(x)
        proj = [p(f) for p, f in zip(self.layer_proj, feats)]
        F_sem = self.sem_fuse(torch.cat(proj, 1))
        return s2, s3, s4, F_sem


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from utils.config import load_config

    cfg_path = os.path.join(root_dir, "engine", "cross_dataset.yaml")
    cfg = load_config(cfg_path)
    detail = DetailStream(cfg, build_uni=False)
    grid = int(cfg.geometry.sem_grid)
    detail.extract_uni = lambda x: [
        torch.randn(x.shape[0], cfg.dims.uni_embed_dim, grid, grid, device=x.device)
        for _ in cfg.dims.uni_layers
    ]
    patch = int(cfg.geometry.patch_size)
    _, _, _, semantic = detail(torch.randn(1, 3, patch, patch))
    assert semantic.shape == (1, cfg.dims.sem_dim, grid, grid)
    print("DetailStream shape check passed:", tuple(semantic.shape))