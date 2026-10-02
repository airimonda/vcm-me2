"""Six small from-scratch architectures sharing one front end and one Heads module.

Every backbone takes a log-mel map (B,1,64,T) or (B,64,T) and returns a
sequence (B, d, T') that goes through attention pooling (learned query) into
the shared Heads: command logits (B,20) and slot logits (B,6,3).

    A1 ds_cnn      depthwise-separable 2D CNN
    A2 bc_resnet   broadcast-residual CNN (optional SubSpectral norm)
    A3 tc_resnet   temporal conv ResNet (mel bins are channels, 1D convs)
    A4 matchbox    MatchboxNet: 1D time-channel separable conv blocks
    A5 crnn        small conv stack + unidirectional GRU
    A6 conformer   tiny conformer-style transformer (conv-subsampled tokens)

`FullModel` = LogMel + backbone + heads, raw waveform (B,80000) in [-1,1].
Tier S ~ 100k params, tier M ~ 300k params (trainable params, front end has none).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import LogMel
from .labels import N_CMD, N_SLOT_HEADS, N_SLOT_VALUES

TIER_TARGETS = {"S": 100_000, "M": 300_000}
TIER_TOLERANCE = 0.25
WINDOW = 80_000


# ---------------------------------------------------------------- shared parts
class AttnPool(nn.Module):
    """Attention pooling over time with a learned query: (B,d,T) -> (B,d)."""

    def __init__(self, d: int):
        super().__init__()
        self.key = nn.Linear(d, d, bias=False)
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))

    def forward(self, x):                                   # x: (B,d,T)
        xt = x.transpose(1, 2)                              # (B,T,d)
        score = torch.matmul(torch.tanh(self.key(xt)), self.query)      # (B,T)
        w = torch.softmax(score, dim=1).unsqueeze(-1)       # (B,T,1)
        return (xt * w).sum(dim=1)                          # (B,d)


class Heads(nn.Module):
    """Command head (20) + 6 slot heads (3 each) on the pooled embedding."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.cmd = nn.Linear(d, N_CMD)
        self.slots = nn.Linear(d, N_SLOT_HEADS * N_SLOT_VALUES)

    def forward(self, z):
        z = self.drop(z)
        return self.cmd(z), self.slots(z).view(-1, N_SLOT_HEADS, N_SLOT_VALUES)


def _bn_act(c, act=True):
    layers = [nn.BatchNorm2d(c)]
    if act:
        layers.append(nn.ReLU())
    return layers


def _as4d(x):
    return x.unsqueeze(1) if x.dim() == 3 else x


def _as3d(x):
    return x.squeeze(1) if x.dim() == 4 else x


# ---------------------------------------------------------------- A1 DS-CNN
class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class DSCNN(nn.Module):
    def __init__(self, width=64, depth=4, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 5, (2, 2), 2, bias=False), *_bn_act(width))
        blocks = []
        for i in range(depth):
            blocks.append(DSBlock(width, width, stride=(2, 2) if i == 1 else (1, 1)))
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = width

    def forward(self, x):
        x = self.blocks(self.stem(_as4d(x)))
        return x.mean(dim=2)                                # mean over freq -> (B,C,T')


# ---------------------------------------------------------------- A2 BC-ResNet
class SubSpectralNorm(nn.Module):
    def __init__(self, c, subbands=4):
        super().__init__()
        self.s = subbands
        self.bn = nn.BatchNorm2d(c * subbands)

    def forward(self, x):
        b, c, f, t = x.shape
        x = x.reshape(b, c * self.s, f // self.s, t)
        return self.bn(x).reshape(b, c, f, t)


class BCBlock(nn.Module):
    """Broadcast-residual block. f2: frequency-depthwise conv (carries the stride);
    f1: frequency-averaged temporal path, broadcast-added back over frequency."""

    def __init__(self, cin, cout, stride=(1, 1), dilation=1, subspectral=False):
        super().__init__()
        self.same = cin == cout and stride == (1, 1)
        self.pre = None
        if cin != cout:
            self.pre = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
        self.f2 = nn.Conv2d(cout, cout, (3, 3 if stride[1] > 1 else 1), stride, (1, 1 if stride[1] > 1 else 0),
                            groups=cout, bias=False)
        self.ssn = SubSpectralNorm(cout) if subspectral else nn.BatchNorm2d(cout)
        self.t_dw = nn.Conv2d(cout, cout, (1, 3), 1, (0, dilation), dilation=(1, dilation), groups=cout, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.pw = nn.Conv2d(cout, cout, 1, bias=False)
        self.drop = nn.Dropout2d(0.05)

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        f2 = self.ssn(self.f2(x))                           # (B,C,F',T')
        f1 = f2.mean(dim=2, keepdim=True)                   # (B,C,1,T')
        f1 = self.drop(self.pw(F.silu(self.bn(self.t_dw(f1)))))
        out = f2 + f1                                       # broadcast over frequency
        if self.same:
            out = out + x
        return F.relu(out)


class BCResNet(nn.Module):
    def __init__(self, scale=6, subspectral=False, **_):
        super().__init__()
        c = [int(round(k * scale)) for k in (2, 3, 4, 5)]
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 5, (2, 2), 2, bias=False), *_bn_act(c[0]))
        self.stages = nn.Sequential(
            BCBlock(c[0], c[0], subspectral=subspectral), BCBlock(c[0], c[0], dilation=1, subspectral=subspectral),
            BCBlock(c[0], c[1], stride=(2, 2), subspectral=subspectral), BCBlock(c[1], c[1], dilation=2, subspectral=subspectral),
            BCBlock(c[1], c[2], stride=(2, 2), subspectral=subspectral), BCBlock(c[2], c[2], dilation=3, subspectral=subspectral),
            BCBlock(c[2], c[2], dilation=3, subspectral=subspectral),
            BCBlock(c[2], c[3], dilation=4, subspectral=subspectral), BCBlock(c[3], c[3], dilation=4, subspectral=subspectral),
        )
        self.head = nn.Sequential(nn.Conv2d(c[3], c[3], (3, 1), 1, (1, 0), groups=c[3], bias=False),
                                  nn.Conv2d(c[3], c[3] * 2, 1, bias=False), *_bn_act(c[3] * 2))
        self.out_dim = c[3] * 2

    def forward(self, x):
        x = self.head(self.stages(self.stem(_as4d(x))))
        return x.mean(dim=2)


# ---------------------------------------------------------------- A3 TC-ResNet
class TCBlock(nn.Module):
    def __init__(self, cin, cout, stride, k=9):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = None
        if cin != cout or stride != 1:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class TCResNet(nn.Module):
    def __init__(self, width=1.5, n_mels=64, **_):
        super().__init__()
        ch = [int(round(k * width)) for k in (16, 24, 32, 48)]
        self.stem = nn.Sequential(nn.Conv1d(n_mels, ch[0], 3, 2, 1, bias=False), nn.BatchNorm1d(ch[0]), nn.ReLU())
        self.blocks = nn.Sequential(TCBlock(ch[0], ch[0], 1), TCBlock(ch[0], ch[1], 2),
                                    TCBlock(ch[1], ch[2], 2), TCBlock(ch[2], ch[3], 2))
        self.out_dim = ch[3]

    def forward(self, x):
        return self.blocks(self.stem(_as3d(x)))


# ---------------------------------------------------------------- A4 MatchboxNet
class MBSub(nn.Module):
    def __init__(self, cin, cout, k, relu=True):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, 1, k // 2, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.relu = relu

    def forward(self, x):
        y = self.bn(self.pw(self.dw(x)))
        return F.relu(y) if self.relu else y


class MBBlock(nn.Module):
    def __init__(self, cin, cout, k, repeat):
        super().__init__()
        subs = []
        for i in range(repeat):
            subs.append(MBSub(cin if i == 0 else cout, cout, k, relu=i < repeat - 1))
        self.subs = nn.Sequential(*subs)
        self.res = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        return F.relu(self.subs(x) + self.res(x))


class MatchboxNet(nn.Module):
    def __init__(self, width=64, blocks=3, repeat=2, n_mels=64, **_):
        super().__init__()
        self.pro = nn.Sequential(nn.Conv1d(n_mels, width, 11, 2, 5, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        ks = [13, 15, 17, 19, 21]
        cin, layers = width, []
        for i in range(blocks):
            layers.append(MBBlock(cin, width, ks[i % len(ks)], repeat))
            cin = width
        self.blocks = nn.Sequential(*layers)
        self.epi = nn.Sequential(
            nn.Conv1d(width, width, 29, 1, 14 * 2, dilation=2, groups=width, bias=False),
            nn.Conv1d(width, width * 2, 1, bias=False), nn.BatchNorm1d(width * 2), nn.ReLU())
        self.out_dim = width * 2

    def forward(self, x):
        return self.epi(self.blocks(self.pro(_as3d(x))))


# ---------------------------------------------------------------- A5 CRNN
class CRNN(nn.Module):
    def __init__(self, ch=16, hidden=96, layers=1, n_mels=64, **_):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1, bias=False), *_bn_act(ch), nn.MaxPool2d((2, 2)),
            nn.Conv2d(ch, ch * 2, 3, 1, 1, bias=False), *_bn_act(ch * 2), nn.MaxPool2d((2, 2)),
            nn.Conv2d(ch * 2, ch * 2, 3, 1, 1, bias=False), *_bn_act(ch * 2), nn.MaxPool2d((2, 1)),
        )
        f_out = n_mels // 8
        self.gru = nn.GRU(ch * 2 * f_out, hidden, num_layers=layers, batch_first=True)  # unidirectional
        self.out_dim = hidden

    def forward(self, x):
        x = self.conv(_as4d(x))                             # (B,C,F',T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)      # (B,T',C*F')
        y, _ = self.gru(x)
        return y.transpose(1, 2)                            # (B,H,T')


# ---------------------------------------------------------------- A6 tiny conformer
class MHSA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x):                                   # (B,T,d)
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk), dim=-1)
        return self.o(torch.matmul(att, v).transpose(1, 2).reshape(b, t, d))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, ff_mult=2, k=9):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * ff_mult), nn.SiLU(), nn.Linear(d * ff_mult, d))
        self.ln_att = nn.LayerNorm(d)
        self.att = MHSA(d, heads)
        self.ln_conv = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, 1, k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * ff_mult), nn.SiLU(), nn.Linear(d * ff_mult, d))
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x):                                   # (B,T,d)
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(self.ln_att(x))
        c = self.ln_conv(x).transpose(1, 2)
        c = F.glu(self.pw1(c), dim=1)
        c = self.pw2(F.silu(self.bn(self.dw(c))))
        x = x + c.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class TinyConformer(nn.Module):
    def __init__(self, d=64, layers=2, heads=4, ff_mult=2, n_mels=64, stride=4, **_):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv1d(n_mels, d, 2 * stride, stride, stride // 2 - 0, bias=False),
                                   nn.BatchNorm1d(d), nn.ReLU())
        self.blocks = nn.Sequential(*[ConformerBlock(d, heads, ff_mult) for _ in range(layers)])
        self.out_dim = d

    def forward(self, x):
        x = self.embed(_as3d(x)).transpose(1, 2)            # (B,T',d)
        return self.blocks(x).transpose(1, 2)               # (B,d,T')


# ---------------------------------------------------------------- registry
ARCHS = {
    "ds_cnn": DSCNN, "bc_resnet": BCResNet, "tc_resnet": TCResNet,
    "matchbox": MatchboxNet, "crnn": CRNN, "conformer": TinyConformer,
}
ARCH_NAMES = {"A1": "ds_cnn", "A2": "bc_resnet", "A3": "tc_resnet", "A4": "matchbox", "A5": "crnn", "A6": "conformer"}

# width/depth per tier (tuned with scripts/tune_tiers.py to hit ~100k / ~300k)
TIER_CONFIGS = {
    "ds_cnn": {"S": dict(width=128, depth=4), "M": dict(width=216, depth=5)},
    "bc_resnet": {"S": dict(scale=16), "M": dict(scale=28)},
    "tc_resnet": {"S": dict(width=1.2), "M": dict(width=2.0)},
    "matchbox": {"S": dict(width=56, blocks=3, repeat=2), "M": dict(width=104, blocks=4, repeat=2)},
    "crnn": {"S": dict(ch=12, hidden=96, layers=1), "M": dict(ch=16, hidden=128, layers=2)},
    "conformer": {"S": dict(d=56, layers=2, heads=4, ff_mult=1), "M": dict(d=64, layers=4, heads=4, ff_mult=2)},
}


class VCM(nn.Module):
    """Backbone + attention pooling + heads on log-mel input."""

    def __init__(self, arch: str, tier: str = "S", **overrides):
        super().__init__()
        cfg = dict(TIER_CONFIGS[arch][tier])
        cfg.update(overrides)
        self.arch, self.tier, self.cfg = arch, tier, cfg
        self.backbone = ARCHS[arch](**cfg)
        d = self.backbone.out_dim
        self.pool = AttnPool(d)
        self.heads = Heads(d)

    def forward(self, feats):
        return self.heads(self.pool(self.backbone(feats)))


class FullModel(nn.Module):
    """waveform (B, 80000) -> (command logits (B,20), slot logits (B,6,3))."""

    def __init__(self, arch: str, tier: str = "S", **overrides):
        super().__init__()
        self.frontend = LogMel()
        self.vcm = VCM(arch, tier, **overrides)
        self.arch, self.tier = arch, tier

    def features(self, wav):
        return self.frontend(wav)

    def forward_features(self, feats):
        return self.vcm(feats)

    def forward(self, wav):
        return self.vcm(self.frontend(wav))


class EnsembleModel(nn.Module):
    """Average of several FullModels' probabilities with one shared front end.

    Outputs log(mean softmax) for the command and slot heads. These are valid logits:
    softmax(log p) = p, so eval, tau and export treat the ensemble like a single model."""

    def __init__(self, members: list[FullModel]):
        super().__init__()
        self.frontend = members[0].frontend
        self.members = nn.ModuleList([m.vcm for m in members])
        self.arch = "ensemble(" + ",".join(m.arch for m in members) + ")"
        self.tier = members[0].tier

    def forward(self, wav):
        f = self.frontend(wav)
        cs, ss = zip(*(m(f) for m in self.members))
        c = torch.stack([F.softmax(x, -1) for x in cs]).mean(0)
        s = torch.stack([F.softmax(x, -1) for x in ss]).mean(0)
        return torch.log(c.clamp_min(1e-12)), torch.log(s.clamp_min(1e-12))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_tier(model: nn.Module, tier: str, tol: float = TIER_TOLERANCE) -> bool:
    t = TIER_TARGETS[tier]
    return abs(count_params(model) - t) <= tol * t


def build_model(arch: str, tier: str = "S", **overrides) -> FullModel:
    if arch in ARCH_NAMES:
        arch = ARCH_NAMES[arch]
    return FullModel(arch, tier, **overrides)


if __name__ == "__main__":
    for a in ARCHS:
        for t in ("S", "M"):
            m = build_model(a, t)
            n = count_params(m)
            print(f"{a:10s} {t}  {n:8d}  ({n / TIER_TARGETS[t]:.2f}x)  ok={check_tier(m, t)}")
