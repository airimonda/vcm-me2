"""Make the figures used in docs/paper.md. Reads only files under results/.

    .venv/bin/python scripts/make_figures.py            # writes docs/figures/*.png

Palette: the first three slots of the dataviz reference palette (blue, orange, aqua), validated for
all-pairs colour-vision separation on a light surface. Aqua is below 3:1 contrast on white, so every
bar and line is also labelled directly or by value; series are additionally told apart by marker or hatch.
Neutral grey is used for "not the chosen setting".
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#9a9993", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})

ARCH_LABEL = {"ds_cnn": "DS-CNN", "bc_resnet": "BC-ResNet", "tc_resnet": "TC-ResNet", "matchbox": "MatchboxNet",
              "crnn": "CRNN", "conformer": "Conformer"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def vlabels(ax, bars, fmt="{:.3f}", dy=0.004, size=7.5):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()), ha="center",
                va="bottom", fontsize=size, color=INK2)


# ---------------------------------------------------------------- 1 bake-off
def fig_bakeoff():
    r1 = pd.read_csv(RES / "bakeoff_round1.csv")
    r2 = pd.read_csv(RES / "bakeoff_round2.csv")
    r1 = r1[r1.tier == "S"].sort_values("test_select_score", ascending=False)
    r2 = r2[r2.tier == "M"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [6, 3.4]})
    w = 0.38
    x = np.arange(len(r1))
    b1 = a1.bar(x - w / 2, r1.test_select_score, w, color=BLUE, label="selection score")
    b2 = a1.bar(x + w / 2, r1.test_real_variation_bal_acc, w, color=ORANGE, label="real-voice variation bal. acc.")
    vlabels(a1, b1)
    vlabels(a1, b2)
    a1.set_xticks(x, [f"{ARCH_LABEL[a]}\n{p/1000:.0f}k" for a, p in zip(r1.arch, r1.params)])
    a1.set_ylim(0.4, 0.92)
    a1.set_ylabel("score (higher is better)")
    a1.set_title("Round 1: six architectures, tier S (~100k), seed 0")
    a1.legend(loc="upper right", ncol=1)
    a1.grid(axis="x", visible=False)

    order = ["conformer", "matchbox", "crnn"]
    x = np.arange(len(order))
    for off, col, colname, lab, c in ((-w / 2, "test_select_score", None, "selection score", BLUE),
                                      (w / 2, "test_real_variation_bal_acc", None, "real-voice variation bal. acc.", ORANGE)):
        means = [r2[r2.arch == a][col].mean() for a in order]
        sds = [r2[r2.arch == a][col].std(ddof=1) for a in order]
        bars = a2.bar(x + off, means, w, yerr=sds, color=c, capsize=3, error_kw={"elinewidth": 1, "ecolor": INK2})
        for xi, a in zip(x + off, order):
            v = r2[r2.arch == a][col].to_numpy()
            a2.scatter(np.full(len(v), xi), v, s=12, color="white", edgecolor=INK, linewidth=0.8, zorder=4)
        for b, m, sd in zip(bars, means, sds):
            a2.text(b.get_x() + b.get_width() / 2, 0.405, f"{m:.3f}", ha="center", va="bottom", fontsize=7,
                    color="white", fontweight="bold", rotation=90)
    a2.set_xticks(x, [f"{ARCH_LABEL[a]}\n{r2[r2.arch == a].params.iloc[0]/1000:.0f}k" for a in order])
    a2.set_ylim(0.4, 0.92)
    a2.set_title("Round 2: tier M (~300k), 3 seeds")
    a2.grid(axis="x", visible=False)
    a2.set_ylim(0.4, 0.97)
    fig.tight_layout()
    save(fig, "fig1_bakeoff.png")


# ---------------------------------------------------------------- 2 training curves
def fig_curves():
    seeds = [(0, BLUE, "o"), (1, ORANGE, "s"), (2, AQUA, "^")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for s, c, mk in seeds:
        d = RES / "runs" / f"conformer_M_s{s}_tune_neg_mf1"
        log = pd.DataFrame([json.loads(l) for l in open(d / "log.jsonl")])
        summ = json.load(open(d / "summary.json"))
        be, se = summ["best_epoch"], summ["stop_epoch"]
        for ax, col in ((a1, "tune_select_score"), (a2, "train_loss")):
            ax.plot(log.epoch, log[col], color=c, lw=1.6, label=f"seed {s}")
            bi = log[log.epoch == be].iloc[0]
            ax.scatter([be], [bi[col]], marker="*", s=120, color=c, edgecolor=INK, linewidth=0.8, zorder=5)
            ei = log[log.epoch == se].iloc[0]
            ax.scatter([se], [ei[col]], marker="X", s=48, color=c, edgecolor=INK, linewidth=0.8, zorder=5)
    a1.set_title("Selection score on tune")
    a1.set_ylabel("selection score")
    a2.set_title("Training loss (CE + slot CE + misfire term)")
    a2.set_ylabel("train loss (epoch mean)")
    for ax in (a1, a2):
        ax.set_xlabel("epoch")
    a1.set_ylim(0.4, 0.85)
    a2.set_ylim(1.4, 5.0)
    h, l = a1.get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="*", ls="", color="white", markeredgecolor=INK, markerfacecolor=GREY, markersize=11),
          plt.Line2D([], [], marker="X", ls="", color="white", markeredgecolor=INK, markerfacecolor=GREY, markersize=7)]
    l += ["best epoch (best.pt)", "early stop"]
    a1.legend(h, l, loc="lower right")
    fig.tight_layout()
    save(fig, "fig3_training_curves.png")


# ---------------------------------------------------------------- 3 misfire ablation
def best(run):
    return json.load(open(RES / "runs" / run / "summary.json"))["best"]


def fig_misfire():
    settings = [
        ("base", "conformer_M_s0_tune_base"),
        ("misfire term only (λ=1)", "conformer_M_s0_tune_mf1"),
        ("negatives only", "conformer_M_s0_tune_neg"),
        ("negatives + λ=0.5", "conformer_M_s0_tune_neg_mf05"),
        ("negatives + λ=1  (chosen)", "conformer_M_s0_tune_neg_mf1"),
        ("negatives + λ=2", "conformer_M_s0_tune_neg_mf2"),
    ]
    vals = [(n, best(r)) for n, r in settings]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    y = np.arange(len(vals))[::-1]
    for ax, key, title in ((axs[0], "oos_false_accept", "Out-of-scope false accept (28 clips; lower is better)"),
                           (axs[1], "neg_misfire", "Synthetic-negative misfire (116 clips; lower is better)")):
        v = [b[key] for _, b in vals]
        cols = [BLUE if "chosen" in n else GREY for n, _ in vals]
        bars = ax.barh(y, v, 0.62, color=cols)
        for b, x in zip(bars, v):
            ax.text(x + 0.008, b.get_y() + b.get_height() / 2, f"{x:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, max(v) * 1.18)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [n for n, _ in vals])
    fig.text(0.5, -0.02, "Single run per setting (seed 0), conformer M, argmax decision (no reject threshold), best epoch on tune. "
             "28 clips: one clip = 3.6 points.", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig2_misfire_ablation.png")


# ---------------------------------------------------------------- 4 tau trade-off
def fig_tau():
    s = pd.read_csv(RES / "tau_single.csv")
    e = pd.read_csv(RES / "tau_ens3.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    for df, c, mk, lab in ((s, ORANGE, "s", "single model (seed 0)"), (e, BLUE, "o", "3-seed ensemble")):
        d = df[df.tau <= 0.85]
        a1.plot(d.in_scope_reject, d.oos_accept, color=c, lw=1.6, marker=mk, ms=3.5, label=lab)
        for t in (0.30, 0.40, 0.50, 0.70):
            r = df[np.isclose(df.tau, t)].iloc[0]
            big = np.isclose(t, 0.40)
            a1.scatter([r.in_scope_reject], [r.oos_accept], s=70 if big else 28, color=c, edgecolor=INK,
                       linewidth=1.2 if big else 0.6, zorder=5)
            dx, dy = (0.004, -0.014) if c == BLUE else (0.004, 0.008)
            a1.annotate(f"τ={t:.2f}", (r.in_scope_reject, r.oos_accept), xytext=(r.in_scope_reject + dx, r.oos_accept + dy),
                        fontsize=7.5, color=INK2, fontweight="bold" if big else "normal")
    a1.set_xlabel("in-scope false reject (1,525 tune clips)")
    a1.set_ylabel("OOS accepted as command")
    a1.set_title("Reject threshold trade-off (lower-left is better)")
    a1.legend(loc="upper right")
    a1.set_xlim(0.02, 0.22)
    a1.set_ylim(-0.02, 0.22)
    for df, c, mk, lab in ((s, ORANGE, "s", "single"), (e, BLUE, "o", "ensemble")):
        a2.plot(df.tau, df.var_bal_acc, color=c, lw=1.6, marker=mk, ms=3.5, label=lab)
    a2.axvline(0.40, color=INK2, lw=0.9, ls="--")
    a2.text(0.41, 0.63, "τ = 0.40", fontsize=8, color=INK2)
    a2.set_xlabel("reject threshold τ")
    a2.set_ylabel("variation balanced accuracy (tune)")
    a2.set_title("Accuracy cost of the threshold")
    a2.set_xlim(0, 0.9)
    a2.set_ylim(0.6, 0.95)
    a2.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08))
    fig.tight_layout()
    save(fig, "fig4_tau_tradeoff.png")


# ---------------------------------------------------------------- 5 final test comparison
def fig_final():
    F = RES / "final"
    ld = lambda d: json.load(open(F / d / "metrics.json"))
    models = [
        ("Conformer M ensemble (fp32, τ=0.40)", "vcm_conformer_M_ens3_test", "vcm_conformer_M_ens3_neg_test", BLUE, None),
        ("Conformer M single (fp32, τ=0.40)", "models_vcm_conformer_M_test", "models_vcm_conformer_M_neg_test", ORANGE, None),
        ("DS-CNN M (τ=0.40)", "ds_cnn_M_test", "ds_cnn_M_neg_test", AQUA, None),
        ("DS-CNN M (τ=0.55)", "ds_cnn_M_test_tau055", "ds_cnn_M_neg_test", AQUA, "////"),
    ]
    groups_hi = [("variation_bal_acc", "variation\nbal. acc."), ("real_variation_bal_acc", "real-voice\nvariation bal. acc."),
                 ("command_acc", "command\naccuracy"), ("slot_acc", "slot\naccuracy")]
    groups_lo = [("oos_false_accept", "OOS false\naccept"), ("in_scope_false_reject", "in-scope\nfalse reject"),
                 ("neg_misfire", "negatives\nmisfire")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [4, 3]})
    for ax, groups, title, ylim in ((a1, groups_hi, "Higher is better", (0.4, 1.02)), (a2, groups_lo, "Lower is better", (0, 0.36))):
        n = len(models)
        w = 0.8 / n
        x = np.arange(len(groups))
        for i, (lab, td, nd, c, hatch) in enumerate(models):
            m = ld(td)
            vals = []
            for k, _ in groups:
                if k == "neg_misfire":
                    # neg_test was scored for DS-CNN at tau 0.40 only, so the tau 0.55 model has no bar here
                    vals.append(float("nan") if "tau055" in td else ld(nd)["misfire_rate"])
                else:
                    vals.append(m[k])
            bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=c, hatch=hatch, edgecolor="white" if hatch is None else INK,
                          linewidth=0 if hatch is None else 0.4, label=lab)
            for b, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, v + (0.006 if ylim[1] > 0.5 else 0.004), f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.3, rotation=90, color=INK2)
        ax.set_xticks(x, [g[1] for g in groups])
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    a2.text(0.98, 0.98, "negatives were scored for DS-CNN\nat τ=0.40 only (no bar at 0.55)", transform=a2.transAxes, ha="right",
            va="top", fontsize=7, color=INK2)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Final test split (4,443 clips; negatives: 250 clips). Same recipe, same data, scored once per model.",
                 fontsize=9, color=INK2, y=1.0)
    fig.tight_layout()
    save(fig, "fig5_final_test.png")


if __name__ == "__main__":
    fig_bakeoff()
    fig_curves()
    fig_misfire()
    fig_tau()
    fig_final()
