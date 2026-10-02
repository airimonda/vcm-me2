# vcm-me2 — tiny on-device Voice Command Model

A small voice-command recogniser trained from scratch (random init, no pretrained weights
or feature extractors anywhere) for AI231 ME2. It listens to one 5 second window of 16 kHz
audio and outputs:

* a **command** (19 commands + `OUT_OF_SCOPE`), and
* for six commands (TIMER, ALARM, TEMPERATURE, BRIGHTNESS, COLOR, CREATE_REMINDER) one of
  **3 slot values**.

The whole thing (log-mel front end included) exports to one ONNX file that runs with
`onnxruntime` + `numpy` only. Six architectures (DS-CNN, BC-ResNet, TC-ResNet,
MatchboxNet, CRNN, tiny Conformer) at two sizes (tier S ~100k and tier M ~300k parameters)
share the same front end, heads, data, augmentation and stopping rule so they can be compared.

## Headline result

Released model: **3-seed conformer ensemble** (882,738 parameters, 4.46 MB fp32 ONNX, 2.78 MB int8), reject threshold
tau = 0.40, scored with `--final-test` on the speaker-disjoint **test** split (4,443 clips, 76 out-of-scope,
250 synthetic negatives; note that the test split was scored more than once during the project, see `docs/paper.md` Section 6.3). Source: `results/final/vcm_conformer_M_ens3_test/metrics.json` and `..._neg_test`.

| Metric (ensemble, fp32, tau 0.40) | Test |
| --- | ---: |
| Variation balanced accuracy (93 variations + OOS) | 0.9431 |
| ... on real (non-synthetic) voices only | 0.7727 |
| Command accuracy | 0.9458 |
| Slot accuracy | 0.9937 |
| Out-of-scope false accept (7 of 76) | 0.0921 |
| In-scope false reject | 0.0398 |
| Synthetic-negative misfire (250 clips) | 0.0640 |

Single model (294,246 parameters): 0.9354 / 0.7738 real-voice / OOS false accept 0.1711. DS-CNN M baseline with the
same recipe: 0.8610. int8 ensemble: 0.9400. The real-voice gap, the test-split handling and other limits are in the
paper. On a Raspberry Pi 4 the fp32 ensemble takes 89 ms per 5-s window on one core (56 ms on two); int8 is not faster there (`results/bench_pi4.json`).

**Live on a Raspberry Pi 4** (class benchmark, holdout split, 196 + 10 trials, injected audio with air-conditioner
noise at 10 dB, wake model v2): intent accuracy 90.8% (real voices 82.3%, synthetic 99.0%), wake word detected in
99.0% of trials, 2 of 10 out-of-scope clips accepted, 1 false wake in 10, latency p95 1.72 s, inference 76 ms
(RTF 0.015). The first run with a loudspeaker and the old wake model reached 46.9%, limited by the wake word. See
`docs/paper.md` Section 8.x and `results/pi_holdout/`.

## Documentation

* [`docs/paper.md`](docs/paper.md) - the full write-up: task, data, model, training, evaluation protocol, experiments, deployment, limitations, reproduction.
* [`docs/model_card.md`](docs/model_card.md) - one-page model card.
* [`docs/runtime.md`](docs/runtime.md) - the on-device runtime: wake word, capture, Spotify / light / thermostat / timers, dashboard, Pi setup, metrics.
* [`docs/wake.md`](docs/wake.md) - the "Watson" wake word: why it was rebuilt after the first live run, data, training, tuning for high recall, validation on the Pi.
* Figures: `docs/figures/` (made by `scripts/make_figures.py` from `results/`; needs matplotlib).

Dataset: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) on the Hugging Face Hub (default config `train` / `test` / `holdout`; config `synthetic_negatives`).

## Released files (tag `v1.0`)

| File | What |
| --- | --- |
| `models/vcm_conformer_M_ens3.onnx` + `.json` | **ensemble of 3 conformers, fp32** (882,738 params, 4.46 MB) |
| `models/vcm_conformer_M_ens3_int8.onnx` + `.json` | ensemble, int8 (2.78 MB) |
| `models/vcm_conformer_M.onnx` + `.json` | single conformer (seed 0), fp32 (294,246 params, 1.94 MB) |
| `models/vcm_conformer_M_int8.onnx` + `.json` | single conformer, int8 (1.36 MB) |
| `models/ckpt/conformer_M_s{0,1,2}.pt` | PyTorch checkpoints of the three ensemble members |
| `models/ckpt/ds_cnn_M_s0_baseline.pt` | DS-CNN M baseline checkpoint (same recipe) |
| `results/` | bake-off tables, run summaries and logs, tau reports, per-clip predictions and metrics of every test scoring |

## Quick start (onnxruntime)

The ONNX graph contains the log-mel front end. Input `waveform`: float32 `(batch, 80000)`, 16 kHz mono, exactly 5 s,
values in [-1, 1]. Outputs: `cmd_logits` `(batch, 20)` and `slot_logits` `(batch, 6, 3)`. The `.json` next to each model
lists `classes` (the 20 command names; `OUT_OF_SCOPE` is the last, index 19), `slot_commands` (order of the 6 slot heads),
`slot_values` (the 3 values per head), `tau`, `window_samples`, `input`, `outputs`, `frontend`, `decision_rule`, `params`
and `opset`. The ensemble outputs `log(mean softmax)`, so applying softmax gives the averaged probabilities.

```python
import json
import numpy as np
import onnxruntime as ort

ONNX = "models/vcm_conformer_M_ens3.onnx"
meta = json.load(open(ONNX.replace(".onnx", ".json")))
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def predict(wav, tau=meta["tau"]):
    """wav: float32 mono, 16 kHz, in [-1, 1]; zero-padded or cut to exactly 80,000 samples."""
    n = meta["window_samples"]
    x = np.zeros(n, np.float32)
    x[: min(n, len(wav))] = wav[:n]
    cmd_logits, slot_logits = sess.run(None, {meta["input"]["name"]: x[None]})   # input name: "waveform"
    p = softmax(cmd_logits[0])
    k = int(p.argmax())
    if p[k] < tau:                                  # reject rule: low confidence -> OUT_OF_SCOPE
        return "OUT_OF_SCOPE", None, float(p[k])
    command = meta["classes"][k]                    # may still be "OUT_OF_SCOPE" (index 19)
    slot = None
    if command in meta["slot_commands"]:
        h = meta["slot_commands"].index(command)
        slot = meta["slot_values"][command][int(slot_logits[0, h].argmax())]
    return command, slot, float(p[k])
```

Note: the evaluation clips were speech-trimmed and centred in the 5-s window by `scripts/pack_data.py` (an offline step
that is not part of the model). Streaming or untrimmed microphone input has not been evaluated.

## Reproduce

Step-by-step commands (dataset download, packing, tune split, bake-off, final recipe, tau report, export, quantise,
final test, figures) are in [`docs/paper.md`, Section 10](docs/paper.md#10-reproducibility). The final recipe is

```bash
python -m vcm.train --arch conformer --tier M --seed 0 --select-split tune --use-negatives --misfire-weight 1.0 --out exp/conformer_M_s0_tune_neg_mf1
python -m vcm.export export --ckpt a.pt+b.pt+c.pt --out models/vcm_conformer_M_ens3.onnx --tau 0.40 --pack data/packs
```

Splits are called **train**, **tune**, **test** and **holdout**. **tune** is a speaker-disjoint ~15% slice of
train; *every* choice (checkpoint, early stopping, hyper-parameters, reject threshold tau, misfire weight,
architecture, ablations) is made on tune, and the model is fitted on the rest of train. **test** is scored once, at
the very end, for the final report (`--final-test`); **holdout** is only for live testing on the Pi. Neither ever
selects anything. The `numerals` split is not used.

```bash
python scripts/make_tune_split.py --pack-dir data/packs --frac 0.15 --seed 0 --out data/packs/tune_split.json
python -m vcm.train --select-split tune ...        # train on train_idx only, select on tune; test is never loaded
```

`make_tune_split.py` picks whole speakers, stratified (group recordings, Xela's speakers, group synthetic voices,
each open-source dataset: ~15% of each stratum's clips; a one-speaker stratum stays in train), tops up so every
variation and OUT_OF_SCOPE has >= 3 tune clips, and assigns synthetic negatives (`neg_tune` only if all their train
sources are tune speakers, mixed rows dropped). Default `select_split: test` in `configs/vcm.yaml` is the old
behaviour so earlier runs reproduce exactly.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # or the package list in the task
export PYTHONPATH=.            # or: uv pip install -e .
pytest                         # unit tests (a few need data/packs)
```

Noise files (DEMAND / MS-SNSD wavs, data only) go in `data/noise/`. `data/` is git-ignored.

## Pack the data

```bash
python scripts/pack_data.py --dataset ~/ai231-me2-collated/dataset \
       --splits train test holdout --noise data/noise --out data/packs
```

Each clip gets an energy-based speech trim, then:

* **test / holdout**: speech centred in exactly 5.0 s (80,000 samples); if longer, the loudest 5 s is kept.
* **train**: trimmed speech (up to 6 s) centred in a 6.0 s buffer (96,000 samples). Training takes
  a random 5 s crop from it (time shift), so shifting never runs out of audio.

Output: `<split>_audio.npy` (int16), `<split>_meta.parquet` (labels + metadata), `noise.npz`.

## Train

```bash
python -m vcm.train --arch bc_resnet --tier S --seed 0 --aug heavy --out exp/bc_resnet_S_s0
python -m vcm.train --out exp/bc_resnet_S_s0 --resume      # continue from last.pt
```

Defaults are in `configs/vcm.yaml` (`--config other.yaml` overlays it; CLI flags win).
AdamW, cosine schedule with warmup, label smoothing 0.1, real voices sampled 5x more often (`real_weight`), out-of-scope clips 3x (`oos_weight`).
Device: CUDA, else Apple MPS, else CPU. Every epoch the model is scored on the selection split (`tune`, or `test`
for old-style runs) and a JSON line with `tune_*` (or `test_*`) keys is appended to `log.jsonl`. `best.pt` = best
selection score; `last.pt` = full
resume state. Early stopping (see `vcm/early_stop.py`) stops when both the train loss has
stopped improving (< 2% over 3 epochs) and the selection score is below its best and below its value
3 epochs ago (min 8, max 60 epochs). The final model is `best.pt`. `summary.json` records the
stop reason.

Augmentation presets: `none`, `light`, `heavy` (`vcm/augment.py` documents every op).

## Evaluate

```bash
python -m vcm.eval --model exp/bc_resnet_S_s0/best.pt --split tune --sweep-tau --out results/bc_S_s0
python -m vcm.eval --model exp/bc_resnet_S_s0/model.onnx --split tune --tau 0.6 --out results/bc_S_s0_onnx
python -m vcm.eval --model exp/bc_resnet_S_s0/model.onnx --split test --tau 0.6 --final-test --out results/final   # final report only
python scripts/compare.py results/a/predictions.csv results/b/predictions.csv   # paired sign test
```

A clip is right if the command is right and, for slotted commands, the slot is right.
The headline number is **variation balanced accuracy**: mean recall over the 93 phrase
variations plus the out-of-scope group. Also reported: command accuracy, per-slot-head accuracy,
out-of-scope false-accept rate, in-scope false-reject rate, breakdown by accent group and real vs
synthetic voices, Wilson 95% intervals, confusion matrices and per-clip predictions.
Choosing the reject threshold tau on tune is what `--sweep-tau` does. `--split test|holdout|neg_test` is refused
unless `--final-test` is given.

## Misfires and synthetic negatives

A *misfire* is non-command audio (talk, noise, partial speech) predicted as a command. Two optional,
default-off features (old configs reproduce bit-for-bit):

* `misfire_weight` (`--misfire-weight`): adds `w * mean(1 - softmax(cmd)[OOS])` over the true-OOS rows of each
  batch. `train_loss` includes it; `train_loss_misfire` is logged separately.
* Synthetic negatives, kept in a **separate folder, not part of the gold splits**:

```bash
python scripts/make_negatives.py --dataset ~/ai231-me2-collated/dataset --noise data/noise \
       --out ~/ai231-me2-collated/dataset/synthetic_negatives --n-train 1000 --n-test 250 --seed 0
python scripts/pack_data.py --negatives ~/ai231-me2-collated/dataset/synthetic_negatives   # neg_train, neg_test
python -m vcm.train --use-negatives --negatives-weight 1.0 --misfire-weight 1.0 --out exp/bc_resnet_S_s0_neg
python -m vcm.eval --model exp/x/best.pt --split neg_tune          # misfire rate + per neg_kind (tune slice of neg_train)
python -m vcm.eval --model exp/x/best.pt --split tune --tau-table  # per tau: var bal acc, OOS FA, neg misfire, false reject
```

Every epoch also logs `<split>_neg_misfire` (argmax) and `<split>_neg_misfire_tau` (at `eval_tau`), on `neg_tune`
with `--select-split tune` (or on `neg_test` if packed, in old-style runs). It is a diagnostic only: selection score
and early stopping are unchanged.
`scripts/bakeoff.sh` takes `TAG=_neg` to suffix the run dir name.

## Export

```bash
python -m vcm.export export --ckpt exp/bc_resnet_S_s0/best.pt --out exp/bc_resnet_S_s0/model.onnx --tau 0.6 --pack data/packs
python -m vcm.export export --ckpt a.pt+b.pt+c.pt --out ens.onnx --tau 0.40 --pack data/packs   # probability-averaging ensemble
python -m vcm.export quantize --onnx exp/bc_resnet_S_s0/model.onnx --out exp/bc_resnet_S_s0/model_int8.onnx
```

`export` writes an ONNX opset 17 graph (input `waveform`, float `batch x 80000` in [-1, 1], dynamic batch;
outputs `cmd_logits` and `slot_logits`), checks torch vs onnxruntime (max abs logit difference
< 1e-3) and writes a `.json` sidecar with labels, slot values, tau and window size.
`quantize` makes a static int8 (QDQ, per-channel) model calibrated on 400 train clips, with the
log-mel front end left in fp32. Evaluate both fp32 and int8 on the same test set.

## Bake-off

```bash
scripts/bakeoff.sh                              # round 1: 6 archs, tier S, seed 0
scripts/bakeoff.sh runs.txt                     # lines of "arch tier seed"
EXTRA="--num-workers 4" scripts/bakeoff.sh      # extra train.py flags
```

Use `TAG=_tune EXTRA="--select-split tune" scripts/bakeoff.sh` for tune-selected runs. Round 2 of the bake-off was `scripts/bakeoff.sh results/round2_runs.txt`.
Runs go to `exp/<arch>_<tier>_s<seed>/`, finished runs are skipped, unfinished ones resume,
and `results/bakeoff.csv` is written at the end. `scripts/smoke.py` is the quick all-architecture
check (500 clips, 1 epoch, export, ONNX eval, int8; evaluates on train clips, never on test).

## Licence

MIT, author airimonda.
