# VCM benchmark - ailene - 20261002-225814

Wake word: **watson** - trials: 196 with the wake word + 10 without - shuffle seed: 14784 - connection: http - holdout: huggingface

Pi log file: ~/vcm-me2/logs/live.log

Audio: **injected** into the assistant in place of its microphone (no loudspeaker, no room acoustics); room noise mssnsd_AirConditioner_1.wav, mssnsd_AirConditioner_10.wav, mssnsd_AirConditioner_2.wav, mssnsd_AirConditioner_3.wav, mssnsd_AirConditioner_4.wav, mssnsd_AirConditioner_5.wav, mssnsd_AirConditioner_6.wav, mssnsd_AirConditioner_7.wav, mssnsd_AirConditioner_8.wav, mssnsd_AirConditioner_9.wav at 10 dB SNR

## At a glance

|                                   | overall      | real voice   | synthetic voice |
|-----------------------------------|--------------|--------------|-----------------|
| intent accuracy (19)              | 90.8%        | 82.3%        | 99.0%           |
| command accuracy (93)             | 90.3%        | 81.2%        | 99.0%           |
| false accept (out of scope fired) | 20.0% (2/10) | 20.0% (2/10) | -               |
| false reject (command ignored)    | 7.0%         | 14.0%        | 1.0%            |
| false wake (no wake word, fired)  | 10.0% (1/10) | 16.7% (1/6)  | 0.0% (0/4)      |
| slot exact                        | 99.0%        | 97.6%        | 100.0%          |
| latency p95                       | 1.72 s       | 1.66 s       | 1.74 s          |

**Pi:** real-time factor 0.015 (p95 0.021), inference 76 ms, CPU temp max 51.1 C

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 90.8%                | 90.3%                 |
| balanced accuracy                                 | 88.4%                | 90.7%                 |
| precision (macro)                                 | 94.1%                | 96.3%                 |
| recall (macro)                                    | 88.4%                | 90.7%                 |
| F1 (macro)                                        | 90.2%                | 91.9%                 |
| F2 (macro)                                        | 88.9%                | 90.9%                 |
| false accept rate (OOS fired)                     | 20.0%                | 20.0%                 |
| false reject rate (in-scope silent/rejected)      | 7.0%                 | 7.0%                  |
| misfire rate (wrong command fired)                | 1.6%                 | 2.2%                  |
| accuracy 95% CI                                   | [86-94%]             | [85-94%]              |
| false accept 95% CI                               | [6-51%] (2/10)       | [6-51%]               |
| false wake rate (command without wake word fired) | 10.0% [2-40%] (1/10) | 10.0% [2-40%] (1/10)  |

Responses: 99.0% of trials fired a command; no response: 2; extra fires: 0; wake detect rate: 99.0%

**False wakes (no wake word, Pi fired):** Message -> OUT_OF_SCOPE

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 196            | 96             | 100             |
| **19 intents** accuracy        | 90.8% [86-94%] | 82.3% [73-89%] | 99.0% [95-100%] |
| balanced accuracy              | 88.4%          | 78.9%          | 98.2%           |
| F1 (macro)                     | 90.2%          | 82.4%          | 98.9%           |
| F2 (macro)                     | 88.9%          | 79.9%          | 98.5%           |
| false accept rate              | 20.0% (2/10)   | 20.0% (2/10)   | -               |
| false reject rate              | 7.0%           | 14.0%          | 1.0%            |
| misfire rate                   | 1.6%           | 3.5%           | 0.0%            |
| **93 commands** accuracy       | 90.3%          | 81.2%          | 99.0%           |
| balanced accuracy              | 90.7%          | 81.4%          | 98.9%           |
| F1 (macro)                     | 91.9%          | 79.0%          | 98.9%           |
| F2 (macro)                     | 90.9%          | 80.1%          | 98.9%           |
| misfire rate                   | 2.2%           | 4.7%           | 0.0%            |
| slot exact (intent right)      | 99.0% (n=103)  | 97.6% (n=42)   | 100.0% (n=61)   |
| latency p50 / p95              | 1.45 / 1.72 s  | 1.43 / 1.66 s  | 1.47 / 1.74 s   |
| false wake rate (no wake word) | 10.0% (1/10)   | 16.7% (1/6)    | 0.0% (0/4)      |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n   | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|-----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 17  | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 17  | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 17  | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 18  | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 16  | 93.8%  | 0.2 deg        | 0.031          | 0.031         | 0.031     |
| TIMER           | 18  | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 103 | 99.0%  | -              | 0.007          | 0.005         | 0.005     |

## Raspberry Pi

- **Raspberry Pi 4 Model B Rev 1.1**, 4 cores  up to 1500.0 MHz, RAM 3795.7 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-v8, Python 3.13.5
- packages: numpy 2.2.4

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 1.399 / 1.720 / 2.736 s  |
| latency p50 / p99                           | 1.451 / 2.302 s          |
| inference time (Pi-reported)                | 75.9 / 107.1 / 131.5 ms  |
| real-time factor (infer / audio window)     | 0.015 / 0.021 / 0.026    |
| CPU temperature                             | 48.0 / 49.7 / 51.1 C     |
| CPU use, whole Pi                           | 12.0 / 29.8 / 67.1 %     |
| CPU use, your runtime process               | -                        |
| RAM (RSS), your runtime process             | -                        |
| RAM used, whole Pi                          | 559.7 / 592.0 / 630.1 MB |
| CPU clock                                   | 1281 / 1500 / 1500 MHz   |
| load average (1 min)                        | 0.55 / 0.95 / 1.33       |
| runtime CPU-seconds per second of speech    | -                        |
| runtime CPU share of wall time              | -                        |
| throttling flags seen                       | none                     |
| test wall time                              | 56.3 min                 |

## Most frequent confusions

**intent level:** PLAY_MUSIC -> REJECT (2); TEMPERATURE -> REJECT (2); LIGHT_OFF -> REJECT (2); BRIGHTNESS -> REJECT (1); PAUSE -> REJECT (1); CALL -> REJECT (1); VOLUME_UP -> REJECT (1); REJECT -> TEMPERATURE (1); LIGHT_ON -> REJECT (1); ALARM -> REJECT (1)

**command level:** Shut off the lights -> REJECT (2); Play music -> REJECT (1); Change the temperature to 26 degrees -> REJECT (1); Brightness 100 percent -> REJECT (1); Pause -> REJECT (1); Change the temperature to 18 degrees -> REJECT (1); Make a phone call -> REJECT (1); Increase the volume -> REJECT (1); REJECT -> Temperature 26 degrees (1); Temperature 18 degrees -> Temperature 22 degrees (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| BRIGHTNESS      | 18 | 94.4%     | 94.4%  | 94.4%  | 94.4%  |
| CALL            | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| COLOR           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| CREATE_REMINDER | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| LIGHT_OFF       | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| LIGHT_ON        | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| LIST_REMINDERS  | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| NEXT            | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| PAUSE           | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| PLAY_MUSIC      | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| REJECT          | 10 | 38.1%     | 80.0%  | 51.6%  | 65.6%  |
| STOP            | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| TEMPERATURE     | 18 | 94.1%     | 88.9%  | 91.4%  | 89.9%  |
| TIME            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| TIMER           | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_DOWN     | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| VOLUME_UP       | 6  | 71.4%     | 83.3%  | 76.9%  | 80.6%  |
| WEATHER         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
