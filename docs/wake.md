# Wake word "Watson" (v2)

The runtime only listens for a command after the wake word. This document covers why the wake model was
rebuilt, how the new one is trained and tuned, and how it is validated on the Pi. Code: `scripts/record_wake.py`,
`scripts/gen_piper.py --templates wake`, `scripts/build_wake_set.py`, `scripts/train_wake.py`,
`scripts/eval_wake.py`.

## Why: the first live holdout run

The first live run of the class benchmark (`vcm-benchmark`, run `20261002-182511`; laptop speaker about 1 m
from the Pi, 196 holdout commands with the wake word plus 10 holdout commands replayed without it, so 10 clips are heard twice) gave:

| Measure | Value |
| --- | ---: |
| Intent accuracy (19 intents, all 196 trials) | 46.9% |
| Trials where the wake word was detected | 48.0% (94 of 196) |
| Correct command, given the wake word fired | 86 of 94 (human 38 of 46, synthetic 48 of 48) |
| Slot exact (intent right) | 100% (n = 47) |
| False accept (out-of-scope fired) | 0 of 10 |
| False wake (command without wake word fired) | 0 of 10 |
| Inference on the Pi (mean / p95) | 69 / 101 ms, RTF 0.014 / 0.020 |

All 102 misses were trials in which the wake word was not detected; the command model itself was almost always
right once it ran. Detection depended on which recorded take of "Watson" was played (about 60% for two takes,
21% for the third) and not on real versus synthetic command voices (48% each), so the failure is acoustic: the
old wake model (trained in the earlier project, threshold 0.55) loses most of its margin after a laptop
loudspeaker, the room and the Pi microphone.

On the tune set described below, the old model (int8, threshold 0.55) detects 96.5% of clean positive clips
but only 53.2% after a simulated loudspeaker + room + noise chain, and 5 of the 9 benchmark takes (55.6%) in
that condition, close to the 48% seen live. The simulated chain is therefore used as a proxy for the live setup.

## Data

| Part | Train | Tune (selection only) |
| --- | --- | --- |
| Your recordings (`record_wake.py`, session `mac1`): "Watson" in 8 styles (normal, quick, slow, soft, loud, far, varied tones, "Watson + command") | 124 positives | - |
| Your recordings: sound-alikes and free talk (clips and the two whole blocks), room tone | 54 + 1 | - |
| Earlier own recordings (`ailene_wake`: 40 "Watson", 20 near-miss) | - | 40 + 20 |
| Benchmark "Watson" takes from earlier runs | - | 9 |
| Piper TTS, ~1,060 voices, "Watson" alone / accented / followed by a benchmark command | 10,763 | 1,237 (held-out voices) |
| Piper TTS sound-alikes ("what's on", "Wilson", "Dawson", "Wesson", ...) and bare commands | 8,000 | - (no synthetic negatives in tune) |
| Real speech and noise from the earlier project (CommonVoice, SLURP, SNIPS, TimersAndSuch, RIRS / Speech Commands noise) | 9,000 | 1,803 (incl. 241 own command recordings) |
| Command dataset, train rows only (`train_idx`, `neg_train_idx` of `data/packs/tune_split.json`) | 9,180 + 778 | - |

Phrases containing "Watson" (for example "Watsonville") are not used as negatives. The tune negatives are
concatenated into one 1.79 h stream for the false-wake measurement.

**Data discipline.** The command model's test, holdout and validation (tune, itself a slice of train) sets are never used. One exception, disclosed: the final wake model was resumed from a checkpoint whose first 3 epochs ran before this filtering, on data that included the command model's tune clips and up to 481 public-corpus clips whose transcript matched a test / holdout / tune clip, all as "not Watson" negatives. All later epochs used the filtered data. The rules:

* only the command packs' train rows are read (the trainer refuses any other pack);
* old real negatives whose dataset and normalised transcript, or CommonVoice speaker, appear in the command
  model's test, holdout, tune or tune_oos metadata are dropped (481 clips);
* the wake tune set contains no synthetic negatives.

## Model

`WakeNet`: the same log-mel front end as the command model (in the graph), a DS-CNN backbone (width 64,
depth 4), attention pooling and 2 logits `["WATSON", "OTHER"]`. 25,730 parameters. Input `wav` (1, 24000) =
1.5 s at 16 kHz, the interface `runtime/wake.py` already uses (scored every 0.25 s).

## Training

* Windows: positives place the whole "Watson" inside the 1.5 s window (clips that run on into a command put
  "Watson" 0.05-0.6 s from the start); negatives are random crops.
* Augmentation for every class alike: a small-loudspeaker chain (high-pass 120-500 Hz, low-pass 3.5-9 kHz,
  1-3 resonances, soft clipping), then speed, pitch, reverb, EQ, telephone band, babble, DEMAND / MS-SNSD noise
  at -3 to 30 dB SNR, quantisation, clipping, gain and polarity, a synthetic music bed (0-20 dB SNR) and
  SpecAugment.
* Loss: cross-entropy with weight 2 on WATSON (misses cost more than false triggers). AdamW, one-cycle
  learning rate (peak 3e-3), batch 256, 300 steps per epoch, bf16 autocast.
* Selection, per epoch on tune: clip-level recall, as the runtime fires (any 0.25 s-hop window at or above
  the threshold), at the lowest threshold whose false wakes per hour on the tune stream stay within a budget;
  averaged over budgets 0.5, 1, 2 and 4 per hour. Recall is the mean of clean clips and a fixed, seeded
  loudspeaker + room + noise version, half from Piper voices and half from real takes. Early stop: no better
  score for 10 epochs (minimum 15).
* The final threshold is the lowest one within 1 false wake per hour on the tune stream, i.e. the highest
  recall that budget allows.

The tune stream is dense speech with sound-alikes and spoken commands, much harder than a normal room, so its
false-wake rates are only for comparing models and thresholds.

## Validation on the Pi without a loudspeaker

`vcm-benchmark` can send each trial to the runtime instead of playing it (`--inject`), with room noise mixed
in (`--room-noise`, `--room-snr`). The runtime, started with `--inject`, feeds the posted audio to the live
pipeline in place of the microphone signal, at real-time speed. See `docs/runtime.md`.

```
# Pi
.venv/bin/python scripts/pi_runtime.py --inject
# laptop
python benchmark.py --inject http://<pi>:8080 \
  --room-noise ~/vcm-me2/data/noise/mssnsd_AirConditioner_*.wav --room-snr 10
```

## Result on the tune set

Training resumed from the epoch-3 checkpoint of an interrupted run (weights only; that run had seen the
earlier, unfiltered data for 3 epochs) and stopped early at epoch 33; the best epoch is 23
(`results/wake/v2_report.json`, `v2_log.jsonl`). Comparison on the same tune set (`results/wake/v2_vs_old.json`):

| Model, threshold | False wakes / h (tune stream) | Recall clean | Recall loudspeaker + noise | ... your real takes |
| --- | ---: | ---: | ---: | ---: |
| old (int8), 0.55 | 13.4 | 96.5% | 53.2% | 73.5% |
| v2 (fp32), 0.90 | 10.6 | 89.2% | 60.8% | 81.6% |
| v2 (fp32), 0.80 | 30.2 | 95.1% | 73.0% | 85.7% |
| **v2 (fp32), 0.70 (deployed)** | **59.4** | **97.5%** | **80.2%** | **93.9%** |
| v2 (int8), 0.70 | 58.3 | 97.5% | 79.9% | 89.8% |

The deployed setting trades more false wakes on this dense-speech stream for recall (the old model had 0 of
10 false wakes in the live run). fp32 is deployed because int8 loses recall on the real takes.

## Status

Deployed on the Pi (`config/runtime.yaml`: `models/wake/vcm_wake_v2.onnx`, threshold 0.70). In the injected
holdout run with air-conditioner noise at 10 dB SNR (`results/pi_holdout/20261002-225814_inject_aircon_wake_v2`)
the wake word was detected in 194 of 196 trials (99.0%; real 97.9%, synthetic 100%), against 48.0% in the
first, loudspeaker run with the old model; 1 of the 10 trials without the wake word fired. The two runs differ
in the audio path as well as the wake model (`docs/paper.md` Section 8.x).
