# VCM benchmark - ailene - 20261002-182511

Wake word: **watson** - trials: 196 with the wake word + 10 without - shuffle seed: 32302 - connection: http - holdout: huggingface

Pi log file: ~/vcm-me2/logs/live.log

## At a glance

|                                   | overall     | real voice  | synthetic voice |
|-----------------------------------|-------------|-------------|-----------------|
| intent accuracy (19)              | 46.9%       | 45.8%       | 48.0%           |
| command accuracy (93)             | 46.9%       | 45.8%       | 48.0%           |
| false accept (out of scope fired) | 0.0% (0/10) | 0.0% (0/10) | -               |
| false reject (command ignored)    | 54.3%       | 57.0%       | 52.0%           |
| false wake (no wake word, fired)  | 0.0% (0/10) | 0.0% (0/3)  | 0.0% (0/7)      |
| slot exact                        | 100.0%      | 100.0%      | 100.0%          |
| latency p95                       | 1.66 s      | 1.72 s      | 1.61 s          |

**Pi:** real-time factor 0.014 (p95 0.020), inference 69 ms, CPU temp max 55.0 C

**Check:**

- the Pi answered only 48.0% of commands: check volume, distance, wake word and the log path

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 46.9%                | 46.9%                 |
| balanced accuracy                                 | 47.2%                | 44.7%                 |
| precision (macro)                                 | 91.1%                | 72.1%                 |
| recall (macro)                                    | 47.2%                | 44.7%                 |
| F1 (macro)                                        | 56.3%                | 53.2%                 |
| F2 (macro)                                        | 48.2%                | 47.1%                 |
| false accept rate (OOS fired)                     | 0.0%                 | 0.0%                  |
| false reject rate (in-scope silent/rejected)      | 54.3%                | 54.3%                 |
| misfire rate (wrong command fired)                | 1.6%                 | 1.6%                  |
| accuracy 95% CI                                   | [40-54%]             | [40-54%]              |
| false accept 95% CI                               | [0-28%] (0/10)       | [0-28%]               |
| false wake rate (command without wake word fired) | 0.0% [0-28%] (0/10)  | 0.0% [0-28%] (0/10)   |

Responses: 48.0% of trials fired a command; no response: 102; extra fires: 0; wake detect rate: 48.0%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 196            | 96             | 100             |
| **19 intents** accuracy        | 46.9% [40-54%] | 45.8% [36-56%] | 48.0% [38-58%]  |
| balanced accuracy              | 47.2%          | 39.2%          | 52.3%           |
| F1 (macro)                     | 56.3%          | 44.6%          | 64.1%           |
| F2 (macro)                     | 48.2%          | 39.6%          | 56.1%           |
| false accept rate              | 0.0% (0/10)    | 0.0% (0/10)    | -               |
| false reject rate              | 54.3%          | 57.0%          | 52.0%           |
| misfire rate                   | 1.6%           | 3.5%           | 0.0%            |
| **93 commands** accuracy       | 46.9%          | 45.8%          | 48.0%           |
| balanced accuracy              | 44.7%          | 40.2%          | 47.8%           |
| F1 (macro)                     | 53.2%          | 39.0%          | 48.4%           |
| F2 (macro)                     | 47.1%          | 39.5%          | 48.0%           |
| misfire rate                   | 1.6%           | 3.5%           | 0.0%            |
| slot exact (intent right)      | 100.0% (n=47)  | 100.0% (n=21)  | 100.0% (n=26)   |
| latency p50 / p95              | 1.16 / 1.66 s  | 1.11 / 1.72 s  | 1.19 / 1.61 s   |
| false wake rate (no wake word) | 0.0% (0/10)    | 0.0% (0/3)     | 0.0% (0/7)      |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 10 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 8  | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 2  | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 9  | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 8  | 100.0% | 0.0 deg        | 0.000          | 0.000         | 0.000     |
| TIMER           | 10 | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 47 | 100.0% | -              | 0.000          | 0.000         | 0.000     |

## Raspberry Pi

- **Raspberry Pi 4 Model B Rev 1.1**, 4 cores  up to 1500.0 MHz, RAM 3795.7 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-v8, Python 3.13.5
- packages: numpy 2.2.4

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 1.135 / 1.657 / 2.378 s  |
| latency p50 / p99                           | 1.159 / 2.268 s          |
| inference time (Pi-reported)                | 69.4 / 101.0 / 106.7 ms  |
| real-time factor (infer / audio window)     | 0.014 / 0.020 / 0.021    |
| CPU temperature                             | 51.2 / 53.1 / 55.0 C     |
| CPU use, whole Pi                           | 11.4 / 26.3 / 50.4 %     |
| CPU use, your runtime process               | -                        |
| RAM (RSS), your runtime process             | -                        |
| RAM used, whole Pi                          | 537.9 / 576.0 / 612.8 MB |
| CPU clock                                   | 1180 / 1500 / 1500 MHz   |
| load average (1 min)                        | 0.50 / 0.88 / 1.43       |
| runtime CPU-seconds per second of speech    | -                        |
| runtime CPU share of wall time              | -                        |
| throttling flags seen                       | none                     |
| test wall time                              | 56.3 min                 |

## Most frequent confusions

**intent level:** COLOR -> REJECT (16); TEMPERATURE -> REJECT (10); BRIGHTNESS -> REJECT (10); CREATE_REMINDER -> REJECT (9); ALARM -> REJECT (8); TIMER -> REJECT (8); VOLUME_DOWN -> REJECT (4); PAUSE -> REJECT (4); MESSAGE -> REJECT (4); CALL -> REJECT (4)

**command level:** Set color to Blue -> REJECT (2); Wake me up at 8:00 AM -> REJECT (2); Change color to Red -> REJECT (2); Set an alarm for 9:00 PM -> REJECT (2); Create a reminder to Study -> REJECT (2); Change the temperature to 26 degrees -> REJECT (2); Change color to Blue -> REJECT (2); Switch color to Green -> REJECT (2); Stop -> REJECT (2); Pause -> REJECT (2)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 55.6%  | 71.4%  | 61.0%  |
| BRIGHTNESS      | 18 | 100.0%    | 44.4%  | 61.5%  | 50.0%  |
| CALL            | 6  | 100.0%    | 16.7%  | 28.6%  | 20.0%  |
| COLOR           | 18 | 100.0%    | 11.1%  | 20.0%  | 13.5%  |
| CREATE_REMINDER | 18 | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| LIGHT_OFF       | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| LIGHT_ON        | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| LIST_REMINDERS  | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| NEXT            | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| PAUSE           | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| PLAY_MUSIC      | 6  | 66.7%     | 33.3%  | 44.4%  | 37.0%  |
| REJECT          | 10 | 9.0%      | 100.0% | 16.5%  | 33.1%  |
| STOP            | 6  | 66.7%     | 33.3%  | 44.4%  | 37.0%  |
| TEMPERATURE     | 18 | 100.0%    | 44.4%  | 61.5%  | 50.0%  |
| TIME            | 6  | 80.0%     | 66.7%  | 72.7%  | 69.0%  |
| TIMER           | 18 | 100.0%    | 55.6%  | 71.4%  | 61.0%  |
| VOLUME_DOWN     | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| VOLUME_UP       | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| WEATHER         | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
