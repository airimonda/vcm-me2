# Model card: vcm-me2 voice command model (v1.0)

## Intended use

Closed-set voice commands for smart-device control on a small computer (target: Raspberry Pi 4/5). Input is one 5-second window of 16 kHz mono audio. Output is one of 19 commands or `OUT_OF_SCOPE`, plus a 3-way slot value for six commands (`TIMER`, `ALARM`, `TEMPERATURE`, `BRIGHTNESS`, `COLOR`, `CREATE_REMINDER`). Runs offline with `onnxruntime` and `numpy`; no speech recogniser, language model or cloud service. Intended for research, teaching and prototypes of home-device control.

Out of scope: open-vocabulary speech recognition, speaker identification or verification, wake-word detection (none is included), and any use where a wrong or missed command can cause harm.

## Model details

| | |
| --- | --- |
| Architecture | log-mel front end inside the graph (n_fft 400, hop 160, 64 mels, per-utterance mean/std norm); conformer backbone (d_model 64, 4 layers, 4 heads, FF ×2, conv kernel 9, stride-4 conv subsampling to 124 tokens); attention pooling; command head (20) and 6 × 3 slot heads |
| Parameters | 294,246 per network; **882,738** for the released ensemble (3 seeds, mean of softmax, one shared front end) |
| Files | `models/vcm_conformer_M_ens3.onnx` (4.46 MB), `..._ens3_int8.onnx` (2.78 MB), single `vcm_conformer_M.onnx` (1.94 MB), `..._int8.onnx` (1.36 MB) |
| Input / output | `waveform` float32 `(batch, 80000)` in [−1, 1]; `cmd_logits` `(batch, 20)`, `slot_logits` `(batch, 6, 3)` |
| Decision rule | softmax over `cmd_logits`; if max probability < τ = 0.40 then `OUT_OF_SCOPE`, else argmax; slot = argmax of that command's slot head |
| Training | from scratch (no pretrained weights); AdamW, lr 3e-3, cosine with 2 warm-up epochs, batch 64, label smoothing 0.1, misfire term λ = 1, heavy augmentation; seeds 0, 1, 2 |
| Precision | fp32 and static int8 (QDQ, per-channel; front end kept fp32) |

## Training data

Hugging Face dataset `airimonda/ai231-me2-voice-commands`, speaker-disjoint splits: train 10,733 clips (2,359 real, 8,374 synthetic; 270 out-of-scope), test 4,443 (824 real), holdout 202 (96 real). Training also uses 778 synthetic negatives (noise, babble, reversed speech, truncated commands, near-silence). Model fitting used 9,180 train clips; a speaker-disjoint 1,553-clip slice of train (*tune*; part of train, not of test or holdout) made every choice after the architecture bake-off. No test or holdout clip was used for training. Real recordings are few in speaker count: the group's own Filipino recordings in train come from 6 speaker IDs (680 clips), of which one (72 clips) is held out as the only group speaker in tune, so most speech seen in training is synthetic. See `docs/paper.md`, Section 3.

## Metrics

Test split (4,443 clips; 76 out-of-scope; 250 synthetic negatives), τ = 0.40, scored with `--final-test` (source: `results/final/*/metrics.json`). Variation balanced accuracy is the mean recall over 93 phrase variations plus the out-of-scope group.

| Model | Var. bal. acc. | Real-voice var. bal. acc. | Command acc. | Slot acc. | OOS false accept | In-scope false reject | Neg. misfire |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ensemble fp32 (882,738 params) | 0.9431 | 0.7727 | 0.9458 | 0.9937 | 0.0921 | 0.0398 | 0.0640 |
| Ensemble int8 | 0.9400 | 0.7640 | 0.9422 | 0.9945 | 0.1053 | 0.0426 | 0.0600 |
| Single fp32 (294,246) | 0.9354 | 0.7738 | 0.9397 | 0.9886 | 0.1711 | 0.0366 | 0.0760 |
| Single int8 | 0.9343 | 0.7742 | 0.9386 | 0.9882 | 0.1711 | 0.0373 | 0.0720 |
| DS-CNN M baseline (308,270), same recipe | 0.8610 | 0.5552 | 0.8728 | 0.9712 | 0.1447 | 0.0994 | 0.1160 |
| DS-CNN M 3-seed ensemble (924,810), size-matched baseline | 0.8727 | 0.5393 | 0.8812 | 0.9795 | 0.0658 | 0.1058 | 0.0840 |

Ensemble fp32 by voice type: clip accuracy 0.9895 on synthetic voices (3,619 clips) against 0.7379 on real voices (824 clips). Per-kind negative misfire (ensemble fp32): babble 0.20, truncated 0.10, reversed 0.02, noise-only 0.00, near-silence 0.00. Paired sign test, single against ensemble: 32 against 68 discordant clips, p = 0.0004. Raspberry Pi 4 (onnxruntime, 1 thread): 89 ms per 5-s window for the fp32 ensemble (RTF 0.018), 56 ms with 2 threads; int8 is not faster, so fp32 is deployed (`results/bench_pi4.json`).

## Ethical considerations

* The real recordings include voices of students who took part in the project. The release uses speaker identifiers, not names. Consent records are held by the project group and were not checked for this card; anyone redistributing the data or retraining on it should confirm consent and the licences of the open corpora that are included (Common Voice, SLURP, Fluent Speech Commands, SNIPS, Speech Commands, TimersAndSuch).
* The model is not a speaker-recognition system and does not store audio, but a device that listens continuously raises privacy questions that the deployer must handle (indicator, on-device processing, retention).
* Performance differs by speaker group (Section 7e of the paper): clip accuracy for the ensemble is 0.64 for the "Other / non-native" group and 0.78 for native English against 0.87 for the project's Filipino group speakers. Group sizes are small. Do not assume equal accuracy for accents or ages that are not represented.
* Not for safety-critical use: do not use it to control equipment where an unintended or missed command could hurt people or damage property (heaters, locks, medical devices).

## Caveats

* Real-voice variation balanced accuracy is 0.7727, against 0.9431 over all clips; the test set is about 81% synthetic speech.
* 9.2% of out-of-scope test clips (7 of 76) are accepted as commands; babble (20%) and truncated commands (10%) are the weakest negatives. The threshold was tuned on data with no Filipino-accented out-of-scope speech, while 14 of the 76 test out-of-scope clips (18%) are Filipino-accented Common Voice speech.
* The test split was used to pick the architecture (bake-off rounds 1 and 2) and was scored more than once in the final evaluation (single model first, then the ensemble, plus int8 and baseline variants). The holdout split was used only for the live test on the Raspberry Pi (`docs/paper.md` Section 8.x).
* int8 costs the ensemble about 0.3 points of variation balanced accuracy and 0.9 points of real-voice score (sign test p = 0.016).
* Input must be a 5-second window. Evaluation clips were speech-trimmed and centred by an offline step that is not part of the model; streaming behaviour is untested.
* Intended for English commands from the fixed phrase list (`configs/variations.csv`). Other phrasings and languages are not evaluated.
