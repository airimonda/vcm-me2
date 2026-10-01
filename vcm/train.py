"""Train one VCM run.

    python -m vcm.train --arch bc_resnet --tier S --seed 0 --aug heavy --out exp/bc_resnet_S_s0
    python -m vcm.train --out exp/bc_resnet_S_s0 --resume

Each epoch: train, score on the test split, log one JSON line, save last.pt (full resume
state) and best.pt (best test variation balanced accuracy). Early stopping: see early_stop.py.
The final model is best.pt, not the last epoch.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from .augment import Augmenter, NoiseBank
from .data import PackedSplit, center_window, to_float
from .early_stop import EarlyStopper
from .eval import compute_metrics, pick_device
from .losses import multitask_loss
from .models import build_model, count_params

ROOT = Path(__file__).resolve().parent.parent


class EpochSampler(Sampler):
    """Weighted sampling with replacement; seeded per epoch so resume is reproducible."""

    def __init__(self, weights, n, seed):
        self.w, self.n, self.seed, self.epoch = weights, n, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.multinomial(self.w, self.n, replacement=True, generator=g).tolist())


def load_cfg(args) -> dict:
    cfg = yaml.safe_load(open(ROOT / "configs" / "vcm.yaml"))
    if args.config:
        cfg.update(yaml.safe_load(open(args.config)) or {})
    for k in ("arch", "tier", "seed", "aug", "epochs", "out", "batch_size", "lr", "num_workers", "device",
              "pack_dir", "max_train_clips", "max_test_clips"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    cfg["early_stop"]["max_epochs"] = min(cfg["early_stop"]["max_epochs"], cfg["epochs"])
    cfg["early_stop"]["min_epochs"] = min(cfg["early_stop"]["min_epochs"], cfg["early_stop"]["max_epochs"])
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state(device):
    st = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st, device):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])
    elif device.type == "mps" and "mps" in st:
        torch.mps.set_rng_state(st["mps"])


def make_scheduler(opt, steps_per_epoch, cfg):
    total = max(cfg["epochs"] * steps_per_epoch, 1)
    warm = max(int(cfg["warmup_epochs"] * steps_per_epoch), 1)

    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = min((step - warm) / max(total - warm, 1), 1.0)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def score_test(model, loader, device, tau, meta):
    model.eval()
    cp, sp = [], []
    for wav, _ in loader:
        wav = to_float(center_window(wav) if wav.shape[1] != 80000 else wav).to(device)
        c, s = model(wav)
        cp.append(torch.softmax(c.float(), -1).cpu().numpy())
        sp.append(torch.softmax(s.float(), -1).cpu().numpy())
    return compute_metrics(meta, np.concatenate(cp), np.concatenate(sp), tau, detail=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arch")
    ap.add_argument("--tier")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aug")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out")
    ap.add_argument("--batch-size", dest="batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--num-workers", dest="num_workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--pack-dir", dest="pack_dir")
    ap.add_argument("--max-train-clips", dest="max_train_clips", type=int)
    ap.add_argument("--max-test-clips", dest="max_test_clips", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg(args)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / "last.pt", out / "best.pt"

    resume = None
    if args.resume and last_path.exists():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        cfg = resume["cfg"]                    # keep the run's own config
        cfg["device"] = args.device or cfg["device"]
    device = pick_device(cfg["device"])
    seed_all(cfg["seed"])
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))

    train_ds = PackedSplit(cfg["pack_dir"], "train", cfg["max_train_clips"], seed=cfg["seed"])
    test_ds = PackedSplit(cfg["pack_dir"], "test", cfg["max_test_clips"], seed=0)
    sampler = EpochSampler(train_ds.sample_weights(cfg["real_weight"], cfg.get("oos_weight", 1.0)), len(train_ds), cfg["seed"])
    nw = cfg["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=nw,
                          drop_last=True, persistent_workers=nw > 0)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    bank = None
    noise_path = Path(cfg["noise"])
    if cfg["aug"] != "none" and noise_path.exists():
        bank = NoiseBank(noise_path, device)
    aug = Augmenter(cfg["aug"], bank, device)

    model = build_model(cfg["arch"], cfg["tier"]).to(device)
    n_params = count_params(model)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
    sched = make_scheduler(opt, len(train_dl), cfg)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopper(**cfg["early_stop"])
    start_epoch, best_metrics, times = 0, None, []

    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        scaler.load_state_dict(resume["scaler"])
        stopper.load_state_dict(resume["stopper"])
        start_epoch, best_metrics, times = resume["epoch"], resume["best_metrics"], resume["times"]
        set_rng_state(resume["rng"], device)
        print(f"resumed from epoch {start_epoch}", flush=True)

    print(f"{cfg['arch']} tier {cfg['tier']}: {n_params:,} params | device {device} | "
          f"train {len(train_ds)} test {len(test_ds)} | aug {cfg['aug']}", flush=True)
    log_f = open(out / "log.jsonl", "a")
    stopped = stopper.reason is not None

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        if stopped:
            break
        t0 = time.time()
        sampler.set_epoch(epoch)
        model.train()
        tot = tot_c = tot_s = 0.0
        nb = 0
        for wav, lab in train_dl:
            wav = to_float(wav).to(device, non_blocking=True)
            lab = lab.to(device)
            wav = aug.wave(wav)
            with torch.no_grad():
                feats = model.features(wav)            # fp32 front end
                feats = aug.spec(feats)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                cmd_l, slot_l = model.forward_features(feats)
            loss, lc, ls = multitask_loss(cmd_l.float(), slot_l.float(), lab[:, 0], lab[:, 1], lab[:, 2],
                                          cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            tot_c += lc.item()
            tot_s += ls.item()
            nb += 1
        m = score_test(model, test_dl, device, cfg["eval_tau"], test_ds.meta)
        dt = time.time() - t0
        times.append(dt)
        score = m["variation_bal_acc"]
        row = {"epoch": epoch, "train_loss": tot / nb, "train_loss_cmd": tot_c / nb, "train_loss_slot": tot_s / nb,
               "test_variation_bal_acc": score, "test_command_acc": m["command_acc"],
               "test_slot_acc": m["slot_acc"], "test_oos_false_accept": m["oos_false_accept"],
               "test_in_scope_false_reject": m["in_scope_false_reject"], "lr": opt.param_groups[0]["lr"],
               "time_s": dt}
        stopped = stopper.update(row["train_loss"], score)
        if stopper.best_epoch == epoch:
            best_metrics = {k: m[k] for k in ("variation_bal_acc", "command_acc", "slot_acc", "oos_false_accept",
                                              "in_scope_false_reject", "command_bal_acc")}
            best_metrics["epoch"] = epoch
            torch.save({"arch": cfg["arch"], "tier": cfg["tier"], "model": model.state_dict(), "epoch": epoch,
                        "score": score, "cfg": cfg, "params": n_params}, best_path)
        row["is_best"] = stopper.best_epoch == epoch
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(f"ep {epoch:3d} loss {row['train_loss']:.4f} test var-bal-acc {score:.4f} cmd {m['command_acc']:.4f} "
              f"slot {m['slot_acc']:.4f} oosFA {m['oos_false_accept']:.3f} {dt:.1f}s"
              f"{' *' if row['is_best'] else ''}", flush=True)
        torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "stopper": stopper.state_dict(), "epoch": epoch,
                    "best_metrics": best_metrics, "times": times, "rng": rng_state(device)}, last_path)

    reason = stopper.reason or f"max_epochs ({cfg['epochs']})"
    summary = {"arch": cfg["arch"], "tier": cfg["tier"], "seed": cfg["seed"], "aug": cfg["aug"], "params": n_params,
               "best_epoch": stopper.best_epoch, "stop_epoch": stopper.epoch, "stop_reason": reason,
               "best": best_metrics, "mean_epoch_time_s": float(np.mean(times)) if times else None,
               "device": str(device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("stop:", reason, flush=True)
    return summary


if __name__ == "__main__":
    main()
