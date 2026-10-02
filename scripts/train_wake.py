"""Train the "Watson" wake-word model from scratch, tuned for high recall.

Model: log-mel front end (vcm.features.LogMel, in the graph) -> a vcm.models backbone -> attention
pooling -> 2 logits ["WATSON", "OTHER"]. Input "wav" (1, 24000) = 1.5 s at 16 kHz, same interface as
the runtime's wake model (runtime/wake.py scores the last 1.5 s every 0.25 s).

Training windows (built per batch from data/wake/train.npz + the command packs as extra negatives):
  positives  the whole "Watson" inside the window at a random place; clips that run on into a command
             have "Watson" start 0.05-0.6 s into the window
  negatives  random crops of Piper look-alikes, real speech (CommonVoice / SLURP / SNIPS / ...), your
             recorded talk and confusables, room noise, the command dataset (train pack) and its
             synthetic negatives, near-silence
Augmentation, on the GPU, for every class alike (so the channel never tells the classes apart):
  playback chain  a small loudspeaker (high-pass 120-500 Hz, low-pass 3.5-9 kHz, 1-3 resonances, soft
                  clipping): the benchmark plays your "Watson" from a laptop speaker into the Pi mic
  vcm.augment     speed, pitch, reverb, EQ, telephone band, babble, DEMAND / MS-SNSD noise at -3..30 dB SNR,
                  quantisation, clipping, gain, polarity
  music bed       synthetic chords + drums at 0-20 dB SNR (music playing while you say the wake word)
  SpecAugment     on the log-mel map
Selection (data/wake/tune.npz: Piper held-out voices, your earlier "Watson" takes, the benchmark takes,
real held-out speech): every epoch, the clip-level recall (any 0.25 s-hop window >= threshold, as the
runtime fires) at the lowest threshold whose false wakes per hour on the tune negative stream are
<= --fa-budget. Score = mean of recall on clean clips and on a fixed (seeded) playback-chain + noise
version. Early stop: no better score for --patience epochs (min --min-epochs).
At the end: tune table over thresholds, the chosen threshold (lowest within the budget = highest recall),
ONNX export with an onnxruntime parity check, models/wake/<name>.json sidecar.

  python scripts/train_wake.py --arch ds_cnn --width 96 --out exp/wake/ds_cnn_96
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vcm.augment import Augmenter, NoiseBank, _apply_resp, _mix_at_snr, _u, active_power, biquad_resp  # noqa: E402
from vcm.features import LogMel  # noqa: E402
from vcm.models import ARCHS, AttnPool  # noqa: E402

SR = 16000
WIN = 24000            # model window, 1.5 s
PAD = 2400             # extra context each side for speed perturbation
BUF = WIN + 2 * PAD
HOP = 4000             # runtime scoring hop, 0.25 s
LABELS = ["WATSON", "OTHER"]

AUG = dict(speed=0.5, shift=0.0, pitch=0.25, gain=0.8, polarity=0.5, clip=0.1, eq=0.5, tel=0.1,
           quant=0.1, noise=0.7, babble=0.2, reverb=0.5, specaug=0.4, timewarp=0.0, snr=(-3, 30))


# ------------------------------------------------------------------ model
class WakeNet(nn.Module):
    def __init__(self, arch: str, **cfg):
        super().__init__()
        self.frontend = LogMel()
        self.backbone = ARCHS[arch](**cfg)
        d = self.backbone.out_dim
        self.pool = AttnPool(d)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(d, 2)

    def forward_features(self, f):
        return self.head(self.drop(self.pool(self.backbone(f))))

    def forward(self, wav):
        return self.forward_features(self.frontend(wav))


# ------------------------------------------------------------------ data
class Clips:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.audio, self.off = d["audio"], d["offsets"]
        self.label, self.onset, self.end = d["label"], d["onset"], d["end"]
        self.lead, self.real, self.src = d["lead"], d["real"], d["src"]

    def __len__(self):
        return len(self.label)

    def clip(self, i):
        return self.audio[self.off[i]: self.off[i + 1]]


def place(clip, a, b, lead, rng, positive):
    """BUF samples from the clip; for positives, the speech span [a, b) lands inside the centre WIN."""
    out = np.zeros(BUF, np.int16)
    n = len(clip)
    if positive:
        if lead:
            pos = PAD + int(rng.uniform(0.05, 0.6) * SR)      # where the speech onset goes in the buffer
        else:
            dur = b - a
            lo, hi = PAD + 800, PAD + WIN - 800 - dur
            pos = int(rng.uniform(lo, hi)) if hi > lo else PAD + 400
        s = a - pos                                            # buffer index 0 = clip index s
    else:
        s = int(rng.uniform(-BUF // 3, max(1, n - BUF + BUF // 3)))
    lo, hi = max(0, s), min(n, s + BUF)
    if hi > lo:
        out[lo - s: hi - s] = clip[lo:hi]
    return out


class Sampler:
    def __init__(self, tr: Clips, packs: list[np.ndarray], pos_real_frac: float, seed: int):
        self.tr, self.packs, self.rng = tr, packs, np.random.default_rng(seed)
        lab, real, src = tr.label, tr.real, tr.src
        self.pos_piper = np.nonzero((lab == 1) & (real == 0))[0]
        self.pos_real = np.nonzero((lab == 1) & (real == 1))[0]
        self.pos_real_frac = pos_real_frac if len(self.pos_real) else 0.0
        rec = np.char.startswith(src.astype(str), "rec:")
        self.neg_groups = [
            (0.25, np.nonzero((lab == 0) & (real == 0))[0]),            # Piper look-alikes + bare commands
            (0.22, np.nonzero((lab == 0) & (real == 1) & ~rec)[0]),     # real speech / noise datasets
            (0.08, np.nonzero((lab == 0) & rec)[0]),                    # your talk + confusables
            (0.05, np.nonzero(lab == 2)[0]),                            # your room tone
        ]
        self.neg_groups = [(w, ix) for w, ix in self.neg_groups if len(ix)]
        self.pack_w = 0.35 if packs else 0.0
        self.silence_w = 0.05

    def batch(self, n, pos_frac):
        rng = self.rng
        xs, ys = [], []
        n_pos = int(round(n * pos_frac))
        for _ in range(n_pos):
            pool = self.pos_real if rng.random() < self.pos_real_frac else self.pos_piper
            i = pool[rng.integers(len(pool))]
            xs.append(place(self.tr.clip(i), self.tr.onset[i], self.tr.end[i], self.tr.lead[i], rng, True))
            ys.append(0)
        ws = [w for w, _ in self.neg_groups] + [self.pack_w, self.silence_w]
        ws = np.array(ws) / sum(ws)
        for _ in range(n - n_pos):
            g = rng.choice(len(ws), p=ws)
            if g < len(self.neg_groups):
                ix = self.neg_groups[g][1]
                i = ix[rng.integers(len(ix))]
                xs.append(place(self.tr.clip(i), 0, 0, 0, rng, False))
            elif g == len(self.neg_groups):
                p = self.packs[rng.integers(len(self.packs))]
                row = p[rng.integers(len(p))]
                s = rng.integers(0, len(row) - BUF)
                xs.append(np.asarray(row[s: s + BUF]))
            else:
                xs.append((rng.standard_normal(BUF) * 10 ** rng.uniform(1, 2.5)).astype(np.int16))
            ys.append(1)
        x = torch.from_numpy(np.stack(xs).astype(np.float32) / 32768.0)
        return x, torch.tensor(ys)


# ------------------------------------------------------------------ extra augmentation
def playback_chain(x: torch.Tensor) -> torch.Tensor:
    """Small loudspeaker: band-limit + resonances + soft clipping, level kept."""
    n, N = x.shape
    dev = x.device
    w = torch.linspace(0, math.pi, N // 2 + 1, device=dev)
    one, zero = torch.ones(n, device=dev), torch.zeros(n, device=dev)
    resp = biquad_resp("hp", _u(n, 120, 500, dev), 0.707 * one, zero, w) ** 2
    resp = resp * biquad_resp("lp", _u(n, 3500, 9000, dev), 0.707 * one, zero, w)
    for k in range(3):
        pk = biquad_resp("peak", torch.exp(_u(n, math.log(300), math.log(5000), dev)), _u(n, 1.0, 4.0, dev),
                         _u(n, -6, 9, dev), w)
        on = (torch.rand(n, device=dev) < (1.0 if k == 0 else 0.5))[:, None]
        resp = torch.where(on, resp * pk, resp)
    y = _apply_resp(x, resp)
    y = y / y.abs().amax(1, keepdim=True).clamp(min=1e-6)
    drive = _u(n, 1.0, 4.0, dev)[:, None]
    y = torch.tanh(y * drive) / torch.tanh(drive)
    return y * torch.sqrt(active_power(x).clamp(min=1e-10) / active_power(y).clamp(min=1e-10))[:, None]


def music_bed(n, N, dev):
    """Synthetic music: a few sustained chord tones with harmonics + a kick/hat pattern."""
    t = torch.arange(N, device=dev, dtype=torch.float32) / SR
    out = torch.zeros(n, N, device=dev)
    root = 110 * 2 ** (torch.randint(0, 24, (n,), device=dev).float() / 12)
    for iv in (0, 4, 7, 12):
        f = root * 2 ** (iv / 12 + torch.randint(-1, 2, (n,), device=dev).float() / 12 * 0)
        for h in range(1, 5):
            out += torch.sin(2 * math.pi * f[:, None] * h * t[None] + _u(n, 0, 6.28, dev)[:, None]) / h
    bpm = _u(n, 70, 140, dev)
    beat = (t[None] * bpm[:, None] / 60) % 1.0
    kick = torch.exp(-beat * 30) * torch.sin(2 * math.pi * 55 * beat / 1.0 * 0.5)
    hat = torch.exp(-((beat * 2) % 1.0) * 60) * torch.randn(n, N, device=dev) * 0.3
    return out * 0.3 + kick * 2 + hat


class Aug:
    def __init__(self, device, noise_bank, playback=0.5, music=0.15):
        self.base = Augmenter(dict(AUG), noise_bank=noise_bank, device=device, out_len=WIN)
        self.device, self.playback, self.music = torch.device(device), playback, music

    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device)
        n = x.shape[0]
        m = (torch.rand(n, device=self.device) < self.playback).nonzero(as_tuple=True)[0]
        if len(m):
            x[m] = playback_chain(x[m])
        x = self.base.wave(x)
        m = (torch.rand(n, device=self.device) < self.music).nonzero(as_tuple=True)[0]
        if len(m):
            x[m] = _mix_at_snr(x[m], music_bed(len(m), x.shape[1], self.device), _u(len(m), 0, 20, self.device))
        return x.clamp(-1, 1)


# ------------------------------------------------------------------ evaluation
def windows(x):
    """Pad like a live stream (0.5 s before, 1 s after) and cut 1.5 s windows every 0.25 s."""
    x = np.concatenate([np.zeros(SR // 2, np.float32), x, np.zeros(SR, np.float32)])
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    st = np.arange(0, len(x) - WIN + 1, HOP)
    return np.stack([x[s: s + WIN] for s in st])


@torch.no_grad()
def probs(model, wins, device, bs=1024):
    out = []
    for i in range(0, len(wins), bs):
        z = model(torch.from_numpy(np.ascontiguousarray(wins[i: i + bs])).to(device))
        out.append(F.softmax(z, -1)[:, 0].float().cpu())
    return torch.cat(out).numpy()


class TuneSet:
    """Positive clips (clean + seeded channel version) and one long negative stream."""

    def __init__(self, tu: Clips, device, noise_bank, seed=1234):
        self.device = device
        pos = np.nonzero(tu.label == 1)[0]
        self.pos_src = tu.src[pos].astype(str)
        self.pos_real = tu.real[pos].astype(bool)
        clean = [tu.clip(i).astype(np.float32) / 32768 for i in pos]
        torch.manual_seed(seed)
        ch = Augmenter(dict(reverb=1.0, eq=0.5, noise=1.0, snr=(5, 20)), noise_bank=noise_bank, device=device,
                       out_len=0)
        chan = []
        for x in clean:
            t = torch.from_numpy(x)[None].to(device)
            y = playback_chain(t)
            y = ch._reverb(y)
            y = ch._noise(y)
            chan.append(y[0].clamp(-1, 1).cpu().numpy())
        self.pos = {"clean": [windows(x) for x in clean], "channel": [windows(x) for x in chan]}
        neg = np.nonzero(tu.label != 1)[0]
        stream = np.concatenate([tu.clip(i).astype(np.float32) / 32768 for i in neg])
        self.neg_hours = len(stream) / SR / 3600
        self.neg_wins = np.lib.stride_tricks.sliding_window_view(stream, WIN)[::HOP]   # a view, no copy
        print(f"tune: {len(pos)} positive clips ({self.pos_real.sum()} real), negative stream "
              f"{self.neg_hours:.2f} h ({len(self.neg_wins)} windows)", flush=True)

    def score(self, model, thresholds, fa_budget, refractory_s=3.0):
        model.eval()
        maxp = {}
        for k, ws in self.pos.items():
            lens = [len(w) for w in ws]
            p = probs(model, np.concatenate(ws), self.device)
            maxp[k] = np.array([s.max() for s in np.split(p, np.cumsum(lens)[:-1])])
        pn = probs(model, self.neg_wins, self.device)
        ref = int(refractory_s * SR / HOP)
        rows = []
        for th in thresholds:
            fires = 0
            above = np.nonzero(pn >= th)[0]
            last = -10 ** 9
            for j in above:
                if j - last > ref:
                    fires += 1
                    last = j
            r = {"threshold": round(float(th), 3), "fa_per_hour": fires / self.neg_hours}
            for k, m in maxp.items():
                r[f"recall_{k}"] = float((m >= th).mean())
                r[f"recall_{k}_real"] = float((m[self.pos_real] >= th).mean()) if self.pos_real.any() else None
            rows.append(r)
        ok = [r for r in rows if r["fa_per_hour"] <= fa_budget]
        pick = min(ok, key=lambda r: r["threshold"]) if ok else max(rows, key=lambda r: r["threshold"])
        score = 0.5 * (pick["recall_clean"] + pick["recall_channel"])
        if pick["recall_clean_real"] is not None:
            score = 0.5 * score + 0.25 * (pick["recall_clean_real"] + pick["recall_channel_real"])
        model.train()
        return score, pick, rows, maxp


# ------------------------------------------------------------------ main
def export_onnx(model, path, device):
    import onnxruntime as ort
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, WIN)
    torch.onnx.export(model, (dummy,), str(path), opset_version=17, input_names=["wav"], output_names=["logits"],
                      dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = (torch.randn(1, WIN) * 0.1)
    d = np.abs(sess.run(None, {"wav": x.numpy()})[0] - model(x).detach().numpy()).max()
    model.to(device)
    return float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ds_cnn")
    ap.add_argument("--width", type=int, default=None, help="ds_cnn width / bc_resnet scale")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--data", default=str(ROOT / "data/wake"))
    ap.add_argument("--packs", nargs="*", default=[str(ROOT / "data/packs/train_audio.npy"),
                                                    str(ROOT / "data/packs/neg_train_audio.npy")])
    ap.add_argument("--noise", default=str(ROOT / "data/packs/noise.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300, help="steps per epoch")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pos-frac", type=float, default=0.4)
    ap.add_argument("--pos-real-frac", type=float, default=0.3, help="share of positives from your recordings")
    ap.add_argument("--pos-weight", type=float, default=2.0, help="loss weight on WATSON (recall)")
    ap.add_argument("--fa-budget", type=float, default=1.0, help="false wakes per hour allowed on the tune stream")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device

    cfg = {}
    if a.width is not None:
        cfg["width" if a.arch == "ds_cnn" else "scale"] = a.width
    if a.depth is not None:
        cfg["depth"] = a.depth
    model = WakeNet(a.arch, **cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{a.arch} {cfg} params={n_params}", flush=True)

    tr, tu = Clips(Path(a.data) / "train.npz"), Clips(Path(a.data) / "tune.npz")
    packs = [np.load(p, mmap_mode="r") for p in a.packs if Path(p).exists()]
    packs = [np.ascontiguousarray(p) for p in packs]
    print(f"train clips {len(tr)} (+{sum(len(p) for p in packs)} pack rows); packs: {[p.shape for p in packs]}",
          flush=True)
    nb = NoiseBank(a.noise, device=dev) if Path(a.noise).exists() else None
    sampler = Sampler(tr, packs, a.pos_real_frac, a.seed)
    aug = Aug(dev, nb)
    tune = TuneSet(tu, dev, nb)
    thresholds = np.round(np.arange(0.05, 0.96, 0.025), 3)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    total = a.max_epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    cw = torch.tensor([a.pos_weight, 1.0], device=dev)
    best, best_ep, log = -1.0, 0, open(out / "log.jsonl", "w")
    for ep in range(1, a.max_epochs + 1):
        t0, tot = time.time(), 0.0
        model.train()
        for _ in range(a.steps):
            x, y = sampler.batch(a.batch, a.pos_frac)
            x, y = aug(x), y.to(dev)
            f = aug.base.spec(model.frontend(x))
            loss = F.cross_entropy(model.forward_features(f), y, weight=cw, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
        score, pick, _, _ = tune.score(model, thresholds, a.fa_budget)
        rec = {"epoch": ep, "loss": tot / a.steps, "score": score, **{f"at_{k}": v for k, v in pick.items()},
               "sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if score > best:
            best, best_ep = score, ep
            torch.save({"model": model.state_dict(), "arch": a.arch, "cfg": cfg, "epoch": ep}, out / "best.pt")
        elif ep >= a.min_epochs and ep - best_ep >= a.patience:
            print(f"early stop at {ep} (best {best:.4f} @ {best_ep})", flush=True)
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    score, pick, rows, maxp = tune.score(model, thresholds, a.fa_budget)
    srcs = sorted(set(tune.pos_src))
    per_src = {s: {k: float((m[tune.pos_src == s] >= pick["threshold"]).mean()) for k, m in maxp.items()}
               for s in srcs}
    diff = export_onnx(model, out / "wake.onnx", dev)
    rep = {"arch": a.arch, "cfg": cfg, "params": n_params, "best_epoch": best_ep, "score": score,
           "fa_budget_per_hour": a.fa_budget, "threshold": pick["threshold"], "at_threshold": pick,
           "recall_by_source": per_src, "tune_neg_hours": tune.neg_hours, "onnx_max_abs_diff": diff,
           "table": rows, "args": vars(a)}
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("table", "args")}, indent=1))


if __name__ == "__main__":
    main()
