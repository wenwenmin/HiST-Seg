"""
models/detail_stream.py
细节流（高分辨率，原生倍率）：
- ConvStem：可学习卷积，产出 [1/2, 1/4, 1/8] 的“真”高分辨率特征（UNI 给不了，边界靠它）。
- UNI ViT-L/16（冻结）：产出 1/16 强语义；4 个抽取层都是同一分辨率，
  不再假装多尺度，而是在 1/16 处 concat→1×1 conv 融合成 F_sem。
timm 惰性导入：未安装也能 import 本文件；测试时可传 build_uni=False 并重写 extract_uni。

★【本次新增：跨数据集域泛化的 stem 归一化】
  ConvStem 现在支持两种“风格无关化”手段，专门用于减小不同扫描仪/数据集之间的外观差异：

  1) IBN（Instance-Batch/Group Normalization，IBN-Net 思想）
     - 在浅层（1/2、1/4 两级）把通道拆两半：一半走 InstanceNorm（去掉每张图自身的
       颜色/风格统计），一半走 GroupNorm（保留内容、且对 batch=4 稳定）。
     - 最深的 1/8 级仍然用纯 GroupNorm，避免过度抹掉判别性内容。
     - InstanceNorm 使用 track_running_stats=False，训练/推理行为一致：测试时同样做实例
       归一化，这正是我们想要的跨域不变性。IN 的 affine 参数会随 checkpoint 保存/加载。

  2) MixStyle（Domain Generalization with MixStyle, ICLR'21）
     - 仅训练期生效（推理时为恒等），在浅层特征上随机混合“不同样本之间的风格统计
       （均值/方差）”，人为制造未见过的域，逼模型学习域不变特征。
     - 无可学习参数，不影响 checkpoint 结构；batch<2 时自动跳过。

  两者都由 YAML 的 model.stem_ibn / model.stem_mixstyle 等字段控制，默认开启。
  所有当前 YAML 都显式声明 IBN/MixStyle 开关，避免隐藏默认行为。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# 归一化 / 风格增强构件
# ---------------------------------------------------------------------------
class IBN(nn.Module):
    """半 InstanceNorm + 半 GroupNorm。

    InstanceNorm 去掉每个样本自身的外观/风格统计（对“不同染色/扫描仪”这类域偏移特别有效），
    GroupNorm 保留内容并且对小 batch 稳定。二者按通道拆分后拼回。

    track_running_stats 默认为 False：训练与推理都用实例统计，行为一致。
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
        # split 出来的是视图；显式 contiguous 以避免 channels_last 下的意外步幅问题。
        out_in = self.instance(xin.contiguous())
        out_gn = self.group(xgn.contiguous())
        return torch.cat([out_in, out_gn], dim=1)


class MixStyle(nn.Module):
    """在浅层特征上随机混合样本间的风格统计（均值/方差），仅训练期生效。

    参考: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
    - p:     以多大概率对一个 batch 施加 MixStyle。
    - alpha: Beta(alpha, alpha) 的形状参数，控制混合系数分布。
    统计量按论文做 detach（只混合风格，不通过统计量回传）。
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
    """保留旧接口（纯 GroupNorm 卷积块），供外部/自检引用。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.GroupNorm(_groups(cout), cout),
        nn.ReLU(inplace=True),
    )


def _conv_norm_act(cin, cout, norm_factory, k=3, s=1, p=1):
    """卷积 + 可插拔归一化(norm_factory(cout)) + ReLU。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        norm_factory(cout),
        nn.ReLU(inplace=True),
    )


class ConvStem(nn.Module):
    """输入 [B,3,H,W] -> s2(1/2,c0), s3(1/4,c1), s4(1/8,c2)。

    浅层（b1=1/2, b2=1/4）使用 IBN 做风格无关化，最深层（b3=1/8）保留纯 GroupNorm。
    MixStyle 施加在 b1、b2 的输出上（也就是 decoder skip s2、s3 之上）——训练期给 skip
    注入风格扰动、推理期为恒等，不影响测试时的边界特征。
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

        # 1/2 与 1/4：风格无关化（IBN 或回退 GN）
        self.b1 = nn.Sequential(
            _conv_norm_act(3, c0, shallow_norm, 3, 2, 1),
            _conv_norm_act(c0, c0, shallow_norm),
        )
        self.b2 = nn.Sequential(
            _conv_norm_act(c0, c1, shallow_norm, 3, 2, 1),
            _conv_norm_act(c1, c1, shallow_norm),
        )
        # 1/8：保留内容，纯 GroupNorm
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
    def __init__(self, cfg, build_uni=True):
        super().__init__()
        d = cfg.dims
        self.uni_layers = list(d.uni_layers)
        self.embed_dim = d.uni_embed_dim
        self.sem_dim = d.sem_dim
        self.patch = int(cfg.geometry.uni_patch_size)

        # Read domain-generalization settings explicitly from the experiment config.
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

        # 4 层各 1×1 降维到 sem_dim//4，concat 后再融合到 sem_dim
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
        import timm  # 惰性导入
        model_name = str(getattr(cfg.model, "uni_model_name", "hf-hub:MahmoodLab/uni"))
        vit = timm.create_model(model_name, pretrained=True,
                                init_values=1e-5, dynamic_img_size=True)
        for p in vit.parameters():
            p.requires_grad = False
        vit.eval()
        return vit

    @torch.no_grad()
    def extract_uni(self, x):
        """返回 list[[B, embed_dim, H/16, W/16]]，对应 self.uni_layers。冻结，无梯度。"""
        vit = self.vit
        if vit is None:
            raise RuntimeError("UNI model is not built. Build the model with build_uni=True.")
        # model.train() would otherwise switch the frozen UNI backbone to train
        # mode; force eval so online UNI features match cached/inference semantics.
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
        F_sem = self.sem_fuse(torch.cat(proj, 1))        # [B, sem_dim, 32, 32]
        return s2, s3, s4, F_sem


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.config import load_config

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "engine", "cross_dataset.yaml"))
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
