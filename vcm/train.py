"""Train one VCM run.

    python -m vcm.train --arch bc_resnet --tier S --seed 0 --aug heavy --out exp/bc_resnet_S_s0
    python -m vcm.train --out exp/bc_resnet_S_s0 --resume

Each epoch: train, score on the selection split, log one JSON line, save last.pt (full resume
state) and best.pt (best selection score: (1-w) * variation balanced accuracy
+ w * the same on real voices only, w = select_real_weight). Early stopping: see early_stop.py.
The final model is best.pt, not the last epoch.

select_split (config / --select-split) picks the selection split:
  test  old behaviour (scores the gold test pack; kept so earlier runs reproduce exactly)
  tune  speaker-disjoint subset of TRAIN listed in tune_split.json (scripts/make_tune_split.py). Training uses only
        train_idx (+ neg_train_idx); scoring, best.pt, early stopping and the negatives misfire log use the tune
        subset (centre 5 s of the 96,000 buffer). The gold test / holdout packs are never opened in this mode.
Log keys are prefixed by the split name (test_select_score / tune_select_score, ...).
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
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from .augment import Augmenter, NoiseBank
from .data import PackedSplit, center_window, load_tune_split, to_float
from .early_stop import EarlyStopper
from .eval import compute_metrics, decide, pick_device
from .labels import OOS_IDX
from .losses import multitask_loss
from .models import build_model, count_params

ROOT = Path(__file__).resolve().parent.parent
SELECT_SPLITS = ("test", "tune")


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
              "pack_dir", "max_train_clips", "max_test_clips", "misfire_weight", "use_negatives",
              "negatives_weight", "select_split", "tune_split"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if cfg.get("select_split", "test") not in SELECT_SPLITS:
        raise ValueError(f"select_split must be one of {SELECT_SPLITS}, got {cfg['select_split']!r}")
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


def pack_exists(pack_dir, split) -> bool:
    p = Path(pack_dir)
    return (p / f"{split}_audio.npy").exists() and (p / f"{split}_meta.parquet").exists()


def tune_split_path(cfg) -> Path:
    return Path(cfg.get("tune_split") or Path(cfg["pack_dir"]) / "tune_split.json")


def build_train_set(cfg, tune=None):
    """Gold train set, plus the synthetic neg_train pack when cfg['use_negatives'] and it exists.

    Returns (dataset, sampling weights aligned with the dataset, n_negatives). Gold rows keep the
    real_weight / oos_weight logic unchanged; negative rows get negatives_weight times the normal
    synthetic weight (1.0; oos_weight is NOT applied to them).
    tune (dict from load_tune_split): train only on train_idx / neg_train_idx (the tune speakers are held out)."""
    gold = PackedSplit(cfg["pack_dir"], "train", cfg["max_train_clips"], seed=cfg["seed"],
                       indices=None if tune is None else tune["train_idx"])
    w = gold.sample_weights(cfg["real_weight"], cfg.get("oos_weight", 1.0))
    if not cfg.get("use_negatives", False):
        return gold, w, 0
    if not pack_exists(cfg["pack_dir"], "neg_train"):
        print(f"WARNING: use_negatives set but {cfg['pack_dir']}/neg_train pack not found; training without", flush=True)
        return gold, w, 0
    neg = PackedSplit(cfg["pack_dir"], "neg_train", cfg["max_train_clips"], seed=cfg["seed"],
                      indices=None if tune is None else tune["neg_train_idx"])
    wn = neg.sample_weights(cfg["real_weight"], 1.0) * float(cfg.get("negatives_weight", 1.0))
    return ConcatDataset([gold, neg]), torch.cat([w, wn]), len(neg)


@torch.no_grad()
def score_neg(model, ds, device, tau, bs=128):
    """Misfire rate on the synthetic-negative test pack: fraction of clips whose argmax command is not
    OUT_OF_SCOPE (tau 0), and the same after the reject threshold tau.
    Batches by hand instead of a DataLoader: creating a DataLoader iterator draws from the global torch RNG,
    which would shift the training run (gold metrics must not depend on the diagnostic being present)."""
    model.eval()
    cp = []
    for i in range(0, len(ds), bs):
        wav = torch.stack([ds[k][0] for k in range(i, min(i + bs, len(ds)))])
        wav = to_float(center_window(wav) if wav.shape[1] != 80000 else wav).to(device)
        c, _s = model(wav)
        cp.append(torch.softmax(c.float(), -1).cpu().numpy())
    cp = np.concatenate(cp)
    dummy = np.zeros((len(cp), 6, 3))
    argmax_pred, _ = decide(cp, dummy, 0.0)
    tau_pred, _ = decide(cp, dummy, tau)
    return {"neg_misfire": float((argmax_pred != OOS_IDX).mean()), "neg_misfire_tau": float((tau_pred != OOS_IDX).mean())}


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
    ap.add_argument("--misfire-weight", dest="misfire_weight", type=float,
                    help="weight of the OOS misfire loss term (0 = off)")
    ap.add_argument("--use-negatives", dest="use_negatives", action=argparse.BooleanOptionalAction, default=None,
                    help="also train on the neg_train pack (synthetic negatives)")
    ap.add_argument("--negatives-weight", dest="negatives_weight", type=float,
                    help="sampling weight of neg_train clips (times the normal synthetic weight of 1)")
    ap.add_argument("--select-split", dest="select_split", choices=SELECT_SPLITS,
                    help="split used for per-epoch scoring / best.pt / early stopping (tune = held-out train speakers)")
    ap.add_argument("--tune-split", dest="tune_split", help="tune_split.json (default <pack-dir>/tune_split.json)")
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

    sel = cfg.get("select_split", "test")          # runs resumed from older checkpoints have no key: test
    tune = load_tune_split(tune_split_path(cfg)) if sel == "tune" else None
    train_ds, train_w, n_neg = build_train_set(cfg, tune)
    if sel == "tune":
        # the selection set is a speaker-disjoint slice of the train pack; the gold test pack is never opened
        test_ds = PackedSplit(cfg["pack_dir"], "train", cfg["max_test_clips"], seed=0, indices=tune["tune_train_idx"])
        neg_test_ds = (PackedSplit(cfg["pack_dir"], "neg_train", cfg["max_test_clips"], seed=0, indices=tune["neg_tune_idx"])
                       if pack_exists(cfg["pack_dir"], "neg_train") and len(tune["neg_tune_idx"]) else None)
        assert test_ds.split == "train" and (neg_test_ds is None or neg_test_ds.split == "neg_train")
        assert not set(tune["train_idx"].tolist()) & set(tune["tune_train_idx"].tolist()), "tune clips leaked into train_idx"
    else:
        test_ds = PackedSplit(cfg["pack_dir"], "test", cfg["max_test_clips"], seed=0)
        neg_test_ds = (PackedSplit(cfg["pack_dir"], "neg_test", cfg["max_test_clips"], seed=0)
                       if pack_exists(cfg["pack_dir"], "neg_test") else None)
    sampler = EpochSampler(train_w, len(train_ds), cfg["seed"])
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
          f"train {len(train_ds)} ({n_neg} neg) {sel} {len(test_ds)}"
          f"{f' neg_{sel} {len(neg_test_ds)}' if neg_test_ds is not None else ''} | aug {cfg['aug']} | "
          f"misfire_weight {cfg.get('misfire_weight', 0.0)}", flush=True)
    log_f = open(out / "log.jsonl", "a")
    stopped = stopper.reason is not None

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        if stopped:
            break
        t0 = time.time()
        sampler.set_epoch(epoch)
        model.train()
        tot = tot_c = tot_s = tot_m = 0.0
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
            loss, lc, ls, lm = multitask_loss(cmd_l.float(), slot_l.float(), lab[:, 0], lab[:, 1], lab[:, 2],
                                              cfg["label_smoothing"], cfg.get("misfire_weight", 0.0))
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
            tot_m += lm.item()
            nb += 1
        m = score_test(model, test_dl, device, cfg["eval_tau"], test_ds.meta)
        mn = score_neg(model, neg_test_ds, device, cfg["eval_tau"]) if neg_test_ds is not None else None
        dt = time.time() - t0
        times.append(dt)
        # selection score: mix of all-clips and real-voice variation balanced accuracy
        w = cfg.get("select_real_weight", 0.0)
        score = (1 - w) * m["variation_bal_acc"] + w * m["real_variation_bal_acc"]
        row = {"epoch": epoch, "train_loss": tot / nb, "train_loss_cmd": tot_c / nb, "train_loss_slot": tot_s / nb,
               "train_loss_misfire": tot_m / nb,
               f"{sel}_select_score": score, f"{sel}_variation_bal_acc": m["variation_bal_acc"],
               f"{sel}_real_variation_bal_acc": m["real_variation_bal_acc"], f"{sel}_command_acc": m["command_acc"],
               f"{sel}_slot_acc": m["slot_acc"], f"{sel}_oos_false_accept": m["oos_false_accept"],
               f"{sel}_in_scope_false_reject": m["in_scope_false_reject"], "lr": opt.param_groups[0]["lr"],
               "time_s": dt}
        if mn is not None:                 # diagnostic only: never feeds the selection score / early stopping
            row[f"{sel}_neg_misfire"] = mn["neg_misfire"]
            row[f"{sel}_neg_misfire_tau"] = mn["neg_misfire_tau"]
        stopped = stopper.update(row["train_loss"], score)
        if stopper.best_epoch == epoch:
            best_metrics = {k: m[k] for k in ("variation_bal_acc", "real_variation_bal_acc", "command_acc", "slot_acc", "oos_false_accept",
                                              "in_scope_false_reject", "command_bal_acc")}
            if mn is not None:
                best_metrics["neg_misfire"] = mn["neg_misfire"]
                best_metrics["neg_misfire_tau"] = mn["neg_misfire_tau"]
            best_metrics["epoch"] = epoch
            best_metrics["select_score"] = score
            torch.save({"arch": cfg["arch"], "tier": cfg["tier"], "model": model.state_dict(), "epoch": epoch,
                        "score": score, "cfg": cfg, "params": n_params}, best_path)
        row["is_best"] = stopper.best_epoch == epoch
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        extra = f"mfloss {row['train_loss_misfire']:.4f} " if cfg.get("misfire_weight", 0.0) else ""
        if mn is not None:
            extra += f"negMF {mn['neg_misfire']:.3f}" + (f"/{mn['neg_misfire_tau']:.3f}@tau" if cfg["eval_tau"] else "") + " "
        print(f"ep {epoch:3d} loss {row['train_loss']:.4f} {sel} select {score:.4f} var-bal-acc {m['variation_bal_acc']:.4f} "
              f"real {m['real_variation_bal_acc']:.4f} cmd {m['command_acc']:.4f} "
              f"slot {m['slot_acc']:.4f} oosFA {m['oos_false_accept']:.3f} "
              f"{extra}{dt:.1f}s"
              f"{' *' if row['is_best'] else ''}", flush=True)
        torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "stopper": stopper.state_dict(), "epoch": epoch,
                    "best_metrics": best_metrics, "times": times, "rng": rng_state(device)}, last_path)

    reason = stopper.reason or f"max_epochs ({cfg['epochs']})"
    summary = {"arch": cfg["arch"], "tier": cfg["tier"], "seed": cfg["seed"], "aug": cfg["aug"], "params": n_params,
               "best_epoch": stopper.best_epoch, "stop_epoch": stopper.epoch, "stop_reason": reason,
               "best": best_metrics, "select_split": sel, "mean_epoch_time_s": float(np.mean(times)) if times else None,
               "device": str(device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("stop:", reason, flush=True)
    return summary


if __name__ == "__main__":
    main()
