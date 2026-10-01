"""On-the-fly augmentation, batched on the training device (GPU/MPS/CPU).

Waveform stage (input: (B, L) float in [-1,1], L = 96000 train buffer or 80000):
  1. speed perturb 0.85-1.15 (linear-interp resample), random time shift
     (+-0.5 s) and the crop to the 5 s window are ONE gather, so no extra pass.
  2. pitch shift +-3 semitones: phase-vocoder time stretch + resample (applied
     before the crop; rare op, groups of 0.5-semitone steps, runs on CPU for MPS).
  3. synthetic reverb (exp-decay noise RIR, RT60 0.1-0.8 s, random DRR).
  4. random EQ: high-pass / low-pass / 1-2 peaking bands, from analytic biquad
     magnitude responses applied zero-phase in the frequency domain (no IIR loop,
     so it is fully batched). Telephone band-limit 300-3400 Hz the same way.
  5. babble: sum of 2-4 other clips of the batch at 10-25 dB SNR (babble is below speech).
  6. additive noise: noise files in data/noise (DEMAND / MS-SNSD) or synthetic
     white / pink / brown, SNR 0-30 dB (SNR measured on active-speech power).
  7. codec-like quantisation (bit-depth reduction or 8-bit mu-law), hard clipping,
     gain +-10 dB, polarity flip; final clamp to [-1,1].
Spectrogram stage (on normalised log-mel (B,64,T)): SpecAugment (2 time masks <= 40
frames, 2 freq masks <= 10 bins) and optional time warp. No mixup / CutMix.

Every op has its own per-sample probability; presets: none / light / heavy.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

SR = 16000
OUT_LEN = 80000

PRESETS = {
    "none": {},
    "light": dict(
        speed=0.2, shift=0.5, gain=0.4, polarity=0.3, eq=0.2, noise=0.3, reverb=0.1,
        specaug=0.5, snr=(5, 30),
    ),
    "heavy": dict(
        speed=0.5, shift=0.8, pitch=0.25, gain=0.8, polarity=0.5, clip=0.1, eq=0.5, tel=0.1,
        quant=0.1, noise=0.7, babble=0.2, reverb=0.3, specaug=0.8, timewarp=0.3, snr=(0, 30),
    ),
}


# ------------------------------------------------------------------ noise bank
class NoiseBank:
    """All noise wavs concatenated (int16) on the device; random segments by modular indexing."""

    def __init__(self, path: str | Path, device="cpu"):
        d = np.load(path, allow_pickle=False)
        self.data = torch.from_numpy(d["audio"]).to(device)
        self.off = torch.from_numpy(d["offsets"][:-1]).to(device)
        self.len = torch.from_numpy(np.diff(d["offsets"])).to(device)
        self.device = device

    def sample(self, n: int, length: int) -> torch.Tensor:
        fi = torch.multinomial(self.len.float(), n, replacement=True)
        flen, off = self.len[fi], self.off[fi]
        start = (torch.rand(n, device=self.device) * flen.float()).long()
        pos = (start[:, None] + torch.arange(length, device=self.device)[None]) % flen[:, None] + off[:, None]
        return self.data[pos].float() / 32768.0


# ------------------------------------------------------------------ helpers
def _u(n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev)


def active_power(x: torch.Tensor, frame: int = 400) -> torch.Tensor:
    """Mean power over frames within 10 dB of the loudest frame. (n,N)->(n,)"""
    n, N = x.shape
    nf = max(N // frame, 1)
    e = (x[:, : nf * frame].reshape(n, nf, -1) ** 2).mean(-1)
    m = (e >= 0.1 * e.max(1, keepdim=True).values).float()
    return (e * m).sum(1) / m.sum(1).clamp(min=1.0)


def _mix_at_snr(sig, noise, snr_db):
    ps = active_power(sig).clamp(min=1e-10)
    pn = (noise ** 2).mean(1).clamp(min=1e-12)
    scale = torch.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return sig + noise * scale[:, None]


def biquad_resp(kind: str, f0, q, gain_db, w):
    """|H| of an RBJ-cookbook biquad at normalised angular freq w (n_bins,). f0,q,gain_db: (n,)."""
    w0 = 2 * math.pi * f0 / SR
    alpha = torch.sin(w0) / (2 * q)
    cw = torch.cos(w0)
    A = 10 ** (gain_db / 40)
    if kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        raise ValueError(kind)
    e1 = torch.polar(torch.ones_like(w), -w)[None]       # z^-1  (1,bins)
    e2 = e1 * e1
    num = b[0][:, None] + b[1][:, None] * e1 + b[2][:, None] * e2
    den = a[0][:, None] + a[1][:, None] * e1 + a[2][:, None] * e2
    return (num / den).abs()


def _apply_resp(x, resp):
    N = x.shape[1]
    return torch.fft.irfft(torch.fft.rfft(x, dim=1) * resp, n=N, dim=1)


def _lerp_read(x, pos):
    """Linear interpolation read: x (n,L), pos (n,M) float source positions; OOB -> 0."""
    L = x.shape[1]
    i0 = pos.floor()
    fr = pos - i0
    i0l = i0.long()
    valid = (i0l >= 0) & (i0l < L - 1)
    i0c = i0l.clamp(0, L - 1)
    i1c = (i0l + 1).clamp(0, L - 1)
    y = torch.gather(x, 1, i0c) * (1 - fr) + torch.gather(x, 1, i1c) * fr
    return y * valid


def phase_vocoder(spec: torch.Tensor, rate: float, phase_advance: torch.Tensor) -> torch.Tensor:
    """Time-stretch a complex STFT (..., freq, time) by `rate` without changing pitch.
    Same algorithm as torchaudio.functional.phase_vocoder (kept local to avoid the torchaudio
    dependency, whose CUDA build must match torch's exactly)."""
    if rate == 1.0:
        return spec
    shape = spec.shape
    spec = spec.reshape(-1, shape[-2], shape[-1])
    steps = torch.arange(0, spec.size(-1), rate, device=spec.device, dtype=phase_advance.dtype)
    alphas = steps % 1.0
    phase_0 = spec[..., :1].angle()
    spec = torch.nn.functional.pad(spec, [0, 2])
    s0 = spec.index_select(-1, steps.long())
    s1 = spec.index_select(-1, (steps + 1).long())
    ang0, ang1 = s0.angle(), s1.angle()
    norm0, norm1 = s0.abs(), s1.abs()
    phase = ang1 - ang0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)
    mag = alphas * norm1 + (1 - alphas) * norm0
    out = torch.polar(mag, phase_acc)
    return out.reshape(shape[:-2] + out.shape[-2:])


def pitch_shift(x: torch.Tensor, semitones: torch.Tensor, n_fft=512, hop=128) -> torch.Tensor:
    """Pitch shift by phase-vocoder time stretch + resample. x (n,N), semitones (n,). Same length."""
    out = torch.empty_like(x)
    q = (semitones * 2).round() / 2                      # 0.5-semitone groups
    win = torch.hann_window(n_fft, device=x.device)
    for s in torch.unique(q):
        idx = (q == s).nonzero(as_tuple=True)[0]
        r = float(2 ** (float(s) / 12))
        xs = x[idx]
        N = xs.shape[1]
        spec = torch.stft(xs, n_fft, hop, window=win, return_complex=True)
        adv = torch.linspace(0, math.pi * hop, spec.shape[-2], device=x.device)[..., None]
        spec = phase_vocoder(spec, 1.0 / r, adv)
        y = torch.istft(spec, n_fft, hop, window=win)
        pos = torch.arange(N, device=x.device, dtype=torch.float32)[None].expand(len(idx), N) * r
        out[idx] = _lerp_read(y, pos)
    return out


# ------------------------------------------------------------------ augmenter
class Augmenter:
    def __init__(self, preset: str | dict = "heavy", noise_bank: NoiseBank | None = None,
                 device="cpu", out_len: int = OUT_LEN):
        self.cfg = dict(PRESETS[preset]) if isinstance(preset, str) else dict(preset)
        self.preset = preset if isinstance(preset, str) else "custom"
        self.noise_bank = noise_bank
        self.device = torch.device(device)
        self.out_len = out_len
        self.snr = self.cfg.get("snr", (0, 30))

    # ---- helpers
    def _p(self, name, n):
        p = self.cfg.get(name, 0.0)
        return torch.rand(n, device=self.device) < p

    def _sub(self, name, x):
        """Indices of samples for which op `name` fires."""
        m = self._p(name, x.shape[0])
        return m.nonzero(as_tuple=True)[0]

    def center_crop(self, x):
        L = x.shape[1]
        s = (L - self.out_len) // 2
        return x[:, s: s + self.out_len]

    # ---- waveform stage
    @torch.no_grad()
    def wave(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float().clone()
        B, L = x.shape
        if not self.cfg:
            return self.center_crop(x)
        dev = self.device

        # pitch (before crop, on the full buffer)
        idx = self._sub("pitch", x)
        if len(idx):
            xs = x[idx]
            sem = _u(len(idx), -3, 3, dev)
            if dev.type == "mps":
                x[idx] = pitch_shift(xs.cpu(), sem.cpu()).to(dev)
            else:
                x[idx] = pitch_shift(xs, sem)

        # speed + shift + crop in one gather
        speed = torch.ones(B, device=dev)
        m = self._p("speed", B)
        speed[m] = _u(int(m.sum()), 0.85, 1.15, dev)
        shift = torch.zeros(B, device=dev)
        m = self._p("shift", B)
        shift[m] = _u(int(m.sum()), -0.5 * SR, 0.5 * SR, dev)
        centre = L / 2
        j = torch.arange(self.out_len, device=dev, dtype=torch.float32) - self.out_len / 2
        pos = centre + shift[:, None] + j[None] * speed[:, None]
        x = _lerp_read(x, pos)
        base = x.clone()                                  # clean time-cropped clips (babble source)

        # reverb
        idx = self._sub("reverb", x)
        if len(idx):
            x[idx] = self._reverb(x[idx])

        # EQ and telephone band-limit
        idx = self._sub("eq", x)
        if len(idx):
            x[idx] = self._eq(x[idx])
        idx = self._sub("tel", x)
        if len(idx):
            x[idx] = self._telephone(x[idx])

        # babble
        idx = self._sub("babble", x)
        if len(idx) and B > 1:
            x[idx] = self._babble(x[idx], base, idx)

        # additive noise
        idx = self._sub("noise", x)
        if len(idx):
            x[idx] = self._noise(x[idx])

        # quantisation, clipping
        idx = self._sub("quant", x)
        if len(idx):
            x[idx] = self._quant(x[idx])
        idx = self._sub("clip", x)
        if len(idx):
            xs = x[idx]
            thr = xs.abs().max(1).values.clamp(min=1e-4) * _u(len(idx), 0.2, 0.7, dev)
            x[idx] = torch.maximum(torch.minimum(xs, thr[:, None]), -thr[:, None])

        # gain and polarity
        m = self._p("gain", B)
        if m.any():
            g = 10 ** (_u(int(m.sum()), -10, 10, dev) / 20)
            x[m] = x[m] * g[:, None]
        m = self._p("polarity", B)
        x[m] = -x[m]
        return x.clamp(-1.0, 1.0)

    def _reverb(self, x):
        n, N = x.shape
        dev = x.device
        max_len = int(0.8 * SR)
        rt60 = _u(n, 0.1, 0.8, dev)
        t = torch.arange(max_len, device=dev, dtype=torch.float32) / SR
        decay = torch.exp(-6.9078 * t[None] / rt60[:, None])
        tail = torch.randn(n, max_len, device=dev) * decay
        tail[:, : int(0.002 * SR)] = 0                    # direct path occupies the first 2 ms
        tail = tail / tail.pow(2).sum(1, keepdim=True).sqrt().clamp(min=1e-8)
        drr = _u(n, -2, 14, dev)                          # direct-to-reverberant ratio, dB
        h = tail * (10 ** (-drr / 20))[:, None]
        h[:, 0] = 1.0
        nfft = 1 << (N + max_len - 1).bit_length()
        y = torch.fft.irfft(torch.fft.rfft(x, nfft, dim=1) * torch.fft.rfft(h, nfft, dim=1), n=nfft, dim=1)[:, :N]
        scale = torch.sqrt(active_power(x).clamp(min=1e-10) / active_power(y).clamp(min=1e-10))
        return y * scale[:, None]

    def _eq(self, x):
        n, N = x.shape
        dev = x.device
        w = torch.linspace(0, math.pi, N // 2 + 1, device=dev)
        resp = torch.ones(n, N // 2 + 1, device=dev)
        one = torch.ones(n, device=dev)
        zero = torch.zeros(n, device=dev)
        pick = torch.rand(n, device=dev)
        # high-pass
        hp = biquad_resp("hp", _u(n, 50, 400, dev), 0.707 * one, zero, w)
        resp = torch.where((pick < 0.3)[:, None], resp * hp, resp)
        # low-pass
        lp = biquad_resp("lp", _u(n, 3000, 7500, dev), 0.707 * one, zero, w)
        resp = torch.where(((pick >= 0.3) & (pick < 0.55))[:, None], resp * lp, resp)
        # 1-2 peaking bands (always, so every EQ call changes the spectrum)
        for k in range(2):
            f0 = torch.exp(_u(n, math.log(200), math.log(4000), dev))
            pk = biquad_resp("peak", f0, _u(n, 0.5, 2.0, dev), _u(n, -8, 8, dev), w)
            on = (torch.rand(n, device=dev) < (1.0 if k == 0 else 0.5))[:, None]
            resp = torch.where(on, resp * pk, resp)
        y = _apply_resp(x, resp)
        return y * torch.sqrt(active_power(x).clamp(min=1e-10) / active_power(y).clamp(min=1e-10))[:, None]

    def _telephone(self, x):
        n, N = x.shape
        dev = x.device
        w = torch.linspace(0, math.pi, N // 2 + 1, device=dev)
        one = torch.ones(n, device=dev)
        zero = torch.zeros(n, device=dev)
        hp = biquad_resp("hp", _u(n, 250, 350, dev), 0.707 * one, zero, w)
        lp = biquad_resp("lp", _u(n, 3200, 3800, dev), 0.707 * one, zero, w)
        resp = (hp * lp) ** 2                              # 4th order each
        y = _apply_resp(x, resp)
        return y * torch.sqrt(active_power(x).clamp(min=1e-10) / active_power(y).clamp(min=1e-10))[:, None]

    def _babble(self, x, base, idx):
        n, N = x.shape
        B = base.shape[0]
        dev = x.device
        k = torch.randint(2, 5, (n,), device=dev)         # 2..4 talkers
        bab = torch.zeros_like(x)
        for j in range(1, 5):
            partner = base[(idx + j) % B]
            partner = torch.roll(partner, int(torch.randint(0, N, (1,))), dims=1)
            partner = partner / active_power(partner).clamp(min=1e-10).sqrt()[:, None]
            bab = bab + partner * (k >= j).float()[:, None]
        return _mix_at_snr(x, bab, _u(n, 10, 25, dev))

    def _noise(self, x):
        n, N = x.shape
        dev = x.device
        kind = torch.randint(1, 4, (n,), device=dev)      # 1 white, 2 pink, 3 brown (0 = noise file)
        if self.noise_bank is not None:
            kind = torch.where(torch.rand(n, device=dev) < 0.5, torch.zeros_like(kind), kind)
        noise = torch.randn(n, N, device=dev)
        f = torch.arange(N // 2 + 1, device=dev, dtype=torch.float32).clamp(min=1.0)
        alpha = torch.zeros(n, device=dev)
        alpha[kind == 2] = 1.0
        alpha[kind == 3] = 2.0
        shaped = torch.fft.irfft(torch.fft.rfft(noise, dim=1) * f[None] ** (-alpha[:, None] / 2), n=N, dim=1)
        noise = torch.where((kind >= 1)[:, None], shaped, noise)
        nb = (kind == 0).nonzero(as_tuple=True)[0]
        if len(nb):
            noise[nb] = self.noise_bank.sample(len(nb), N).to(dev)
        lo, hi = self.snr
        return _mix_at_snr(x, noise, _u(n, lo, hi, dev))

    def _quant(self, x):
        n = x.shape[0]
        dev = x.device
        bits = torch.randint(6, 13, (n,), device=dev).float()
        lv = (2 ** (bits - 1))[:, None]
        q = torch.round(x * lv) / lv
        mu = 255.0
        comp = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
        comp = torch.round(comp * 127) / 127
        mulaw = torch.sign(comp) * ((1 + mu) ** comp.abs() - 1) / mu
        use_mu = (torch.rand(n, device=dev) < 0.5)[:, None]
        return torch.where(use_mu, mulaw, q)

    # ---- spectrogram stage
    @torch.no_grad()
    def spec(self, f: torch.Tensor) -> torch.Tensor:
        """f: normalised log-mel (B, M, T). Masks fill with 0 (the per-utterance mean)."""
        if self.cfg.get("specaug", 0) <= 0:
            return f
        B, M, T = f.shape
        dev = f.device
        f = f.clone()
        if self.cfg.get("timewarp", 0) > 0:
            idx = self._sub("timewarp", f)
            if len(idx):
                f[idx] = self._timewarp(f[idx])
        on = self._p("specaug", B)
        tar = torch.arange(T, device=dev)[None]
        far = torch.arange(M, device=dev)[None]
        for _ in range(2):
            w = (torch.rand(B, device=dev) * 41).long()
            t0 = (torch.rand(B, device=dev) * (T - w).clamp(min=1)).long()
            m = (tar >= t0[:, None]) & (tar < (t0 + w)[:, None]) & on[:, None]
            f = f.masked_fill(m[:, None, :], 0.0)
            w = (torch.rand(B, device=dev) * 11).long()
            f0 = (torch.rand(B, device=dev) * (M - w).clamp(min=1)).long()
            m = (far >= f0[:, None]) & (far < (f0 + w)[:, None]) & on[:, None]
            f = f.masked_fill(m[:, :, None], 0.0)
        return f

    def _timewarp(self, f, W: int = 20):
        n, M, T = f.shape
        dev = f.device
        c = _u(n, W, T - W, dev)                           # warp anchor in the source
        d = _u(n, -W, W, dev)                              # displacement
        j = torch.arange(T, device=dev, dtype=torch.float32)[None]
        tgt = c + d                                        # anchor lands here in the output
        left = j * (c / tgt)[:, None]
        right = c[:, None] + (j - tgt[:, None]) * ((T - 1 - c) / (T - 1 - tgt))[:, None]
        src = torch.where(j <= tgt[:, None], left, right).clamp(0, T - 1)
        i0 = src.floor().long().clamp(0, T - 1)
        i1 = (i0 + 1).clamp(0, T - 1)
        fr = (src - i0.float())[:, None, :]
        g0 = torch.gather(f, 2, i0[:, None, :].expand(n, M, T))
        g1 = torch.gather(f, 2, i1[:, None, :].expand(n, M, T))
        return g0 * (1 - fr) + g1 * fr
