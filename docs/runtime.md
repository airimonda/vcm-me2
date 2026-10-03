# On-device runtime

The runtime turns the released command model into a voice assistant on a Raspberry Pi 4: it listens for
**"Watson"**, captures the command, runs the ensemble, and acts on it (Spotify music, an emulated light and
thermostat, real timers / alarms / reminders, weather, time), answers with a short spoken reply and shows
everything on a dashboard. Code: `runtime/` (package) and `scripts/pi_runtime.py` (entry point). Config:
`config/runtime.yaml`.

```
mic (16 kHz) -> ring buffer -> wake model every 0.25 s on the last 1.5 s ->  P(WATSON) >= 0.55 ?
                                                                              |  yes
        earcon + duck music  <-------------------------------------------------+
        capture <= 5 s, energy endpointing (stop 0.7 s after speech, min 0.8 s)
        trim / best 5 s / centre in 80,000 samples   (runtime/prep.py = scripts/pack_data.py)
        command model (ONNX ensemble)  ->  softmax; max prob < tau (0.40) -> OUT_OF_SCOPE; slot = argmax(head)
        dispatcher -> actuator -> reply clip (Piper, espeak-ng fallback) -> restore music volume
                                      |
                       state / heard / turn / metrics -> bus -> aiohttp WebSocket -> dashboard (Chromium kiosk)
```

## Modules

| module | job |
|---|---|
| `runtime/app.py` | `Runtime` (builds everything, one turn path), the audio loop, CLI (`main()`) |
| `runtime/audio.py` | sources: `SoundDeviceSource`, `ArecordSource` (fallback), `WavSource` (`--input-wav`) |
| `runtime/wake.py` | `RingBuffer`, `WakeDetector` (sliding window, threshold, consecutive, cooldown) |
| `runtime/endpoint.py` | `Endpointer`: 20 ms energy frames, noise-floor relative threshold, hangover |
| `runtime/prep.py` | `trim_speech`, `best_window`, `centre_pad`, `prepare_window` (shared with `scripts/pack_data.py`) |
| `runtime/model.py` | `CommandModel` + `decide()` (the decision rule), `WakeModel`; sessions use `session.intra_op.allow_spinning=0` |
| `runtime/dispatcher.py` | command -> actuator mapping, OUT_OF_SCOPE handling, STOP dismisses a ringing timer |
| `runtime/devices.py` | builds the actuators from the config |
| `runtime/actuators/` | `spotify.py` (Web API + mock player), `light.py`, `thermostat.py`, `comms.py`, `clock.py` (timers/alarms/reminders), `info.py` (weather + time) |
| `runtime/feedback.py` | reply clips by key, espeak-ng / `say` fallback, ducking while speaking |
| `runtime/sounds.py` | earcons (synthesised WAVs, amplitude 0.15; Mac System sounds) |
| `runtime/convo.py` | conversation log; the "Understood as" sentence is rebuilt from command + slot |
| `runtime/metrics.py` | per-turn stage latency, CPU/RAM/temperature sampler, run folder |
| `runtime/state.py`, `bus.py`, `ui_server.py`, `ui/` | shared device state, pub/sub, aiohttp + WebSocket server, dashboard |

## Commands, actions and replies

The decision rule is the one in the model's sidecar (`models/vcm_conformer_M_ens3.json`): `softmax(cmd_logits)`;
below `tau` (0.40, `model.tau` overrides) the clip is `OUT_OF_SCOPE`; the slot is the argmax of the command's own
slot head. Everything else is the dispatcher (`runtime/dispatcher.py`).

| command (slot) | action | device |
|---|---|---|
| PLAY_MUSIC | transfer to the Pi's Spotify Connect device if needed, then start `music.context_uri` (default `liked` = the account's Liked Songs, shuffled), or resume it if it is already loaded and paused | Spotify (real) |
| PAUSE / STOP | Spotify pause (stop = pause; STOP also silences a ringing timer/alarm instead) | Spotify |
| NEXT | skip to the next track | Spotify |
| VOLUME_UP / VOLUME_DOWN | Spotify device volume +/- 15 % (`music.volume_step`), clamped 0-100 | Spotify |
| LIGHT_ON / LIGHT_OFF | power | light (emulated) |
| BRIGHTNESS (20 / 60 / 100 percent) | brightness, turns the light on | light |
| COLOR (Red / Blue / Green) | colour, turns the light on | light |
| TEMPERATURE (18 / 22 / 26 degrees) | thermostat setpoint (16-30), simulated room temperature drifts toward it | thermostat (emulated) |
| TIMER (10 seconds / 30 seconds / 1 minute) | real asyncio timer, rings until STOP / dashboard Stop (60 s max) | clock (real) |
| ALARM (6:00 AM / 8:00 AM / 9:00 PM) | next occurrence of that time, rings the same way | clock |
| CREATE_REMINDER (Drink water / Study / Exercise) | saved in the reminder list (persisted in `runtime_logs/state.json`) | clock |
| LIST_REMINDERS | speaks the count and the last five | clock |
| WEATHER | Open-Meteo, Quezon City, 3 s deadline, IPv4 forced | real |
| TIME | the device clock (`clock.timezone` or the system zone) | real |
| CALL / MESSAGE | dashboard "Calling Mom" / "Message to Mom: I'm on my way" card for 3 s | phone (emulated) |
| OUT_OF_SCOPE | "Sorry, I didn't catch that." (`dispatcher.oos_reply: silent` = reject chime only) | - |

Notes: the model has no contact slot, so CALL / MESSAGE go to `comms.contact`. Replies are keys in
`config/replies.json`. Errors are spoken: "I can't reach Spotify.", "I can't find the speaker on Spotify.",
"Spotify isn't set up yet.", "Spotify needs to be set up again.", "Spotify Premium is required.", "Nothing is
playing.", "I can't reach the weather service."

## Configuration (`config/runtime.yaml`)

Everything has a default in `runtime/config.py`; the YAML only needs the keys you change. The main ones:

| key | default | meaning |
|---|---|---|
| `model.path` | `models/vcm_conformer_M_ens3.onnx` | command model (fp32; int8 is not faster on the Pi 4) |
| `model.tau` | `null` | override the sidecar tau (0.40) |
| `wake.path` | `models/wake/vcm_wake_int8.onnx` | wake model (fp32: `vcm_wake.onnx`); labels `["WATSON","OTHER"]` |
| `wake.threshold` / `consecutive` / `hop_s` / `cooldown_s` | 0.55 / 1 / 0.25 / 1.0 | single-window threshold on P(WATSON), score every 0.25 s |
| `audio.backend` / `device` | auto / null | sounddevice, else `arecord`; default devices = the PipeWire echo-cancel source/sink on the Pi |
| `audio.earcon_volume` | 0.15 | chime amplitude; there is no mute/guard after the chime (it cut off the first word) |
| `capture.*` | max 5 s, min 0.8 s, 0.7 s silence | endpointing; `start_ignore_s` (0.25) only affects when speech *starts* being detected, all audio is kept |
| `dispatcher.oos_reply` | `sorry` | or `silent` |
| `dispatcher.ducking` / `duck_to` | true / 20 | lower Spotify to 20 % (never raises it) on wake, restore after the command |
| `mock.light` / `aircon` / `spotify` | true / true / false | emulated devices. Light and thermostat have no real driver (always emulated); `spotify: true` = in-memory player |
| `music.*` | device "Watson", `credentials: config/spotify.json` | see below |
| `ui.*` | enabled, 0.0.0.0:8080 | dashboard server |
| `metrics.dir` | `runtime_logs` | one folder per run |

### CLI

```
.venv/bin/python scripts/pi_runtime.py [--config F] [--mock all|none|light,aircon,spotify] [--no-ui] [--port N]
    [--input-wav WAV|DIR ...] [--use-wake] [--realtime] [--keep-open] [--no-mic] [--silent] [--fresh-state]
    [--model ONNX] [--tau X] [--wake-model ONNX] [--wake-threshold X] [--metrics-dir D] [--json] [--inject]
pkill -f "[s]cripts/pi_runtime.py"      # stop a running instance (the bracket keeps pkill from matching itself)
```

* `--input-wav` feeds files instead of the mic with a **mocked wake trigger** at the start of each file (so the
  endpointer, window prep, model, dispatcher and reply all run). `--use-wake` runs the real wake model on the
  stream instead (the file must contain "Watson ..."). `--realtime` paces the file; file replays are silent
  and have no dashboard unless `--realtime` or `--keep-open`.
* `--no-mic` = dashboard and its test buttons only.
* `--inject` accepts a WAV POSTed to `/inject` on the dashboard port and feeds it to the live pipeline in place of
  the microphone signal (`runtime/audio.py` `InjectSource`): one block per mic block, so it plays out in real
  time, and blocks dropped while the runtime was busy still advance it. Used by `vcm-benchmark --inject` to run
  the holdout benchmark without a loudspeaker (see `docs/wake.md`). Off by default (`audio.inject: false`):
  while it is on, anyone who can reach the port can feed audio to the assistant.

Example (Mac, no hardware):

```
$ .venv/bin/python scripts/pi_runtime.py --mock all --input-wav .../holdout_000134_real_voice_BRIGHTNESS_V1_2_t1.wav
  BRIGHTNESS [60 percent]  p=0.78  accepted=True  reply='Brightness 60 percent.'
```

## Spotify

Music goes through the Spotify Web API (plain `requests`, `runtime/actuators/spotify.py`) to a **Spotify Connect
device on the Pi** (raspotify / librespot). A Spotify **Premium** account is required for remote playback control.

1. **Create an app** at <https://developer.spotify.com/dashboard>. Add the redirect URI
   `http://127.0.0.1:8888/callback` (Edit settings -> Redirect URIs; it must match exactly). Note the client id
   and client secret.
2. **Install the Spotify Connect player on the Pi.** The setup used on the demo Pi needs no sudo: the
   `librespot` binary is taken out of the raspotify `.deb` and run as a systemd *user* service
   (`deploy/librespot.service`, device name "Watson", PipeWire/pulseaudio backend, so music goes to the
   echo-cancel sink like the replies):
   ```
   url=$(curl -s https://api.github.com/repos/dtcooper/raspotify/releases/latest | grep -o 'https://[^"]*/raspotify_[^"]*arm64\.deb' | head -1)
   curl -sL -o /tmp/raspotify.deb "$url" && mkdir -p /tmp/rs && dpkg-deb -x /tmp/raspotify.deb /tmp/rs
   mkdir -p ~/.local/bin ~/.cache/librespot && cp /tmp/rs/usr/bin/librespot ~/.local/bin/
   cp deploy/librespot.service ~/.config/systemd/user/ && systemctl --user daemon-reload
   systemctl --user enable --now librespot
   ```
   Sign the player in once. Picking "Watson" in the Spotify app needs zeroconf, which a router with client
   isolation blocks; the route used on the demo Pi is device pairing, which needs no browser on the Pi:
   ```
   systemctl --user stop librespot
   ~/.local/bin/librespot --name Watson --backend pulseaudio --cache ~/.cache/librespot --enable-device-auth
   # open the printed https://spotify.com/pair?code=XXXXXX, approve, wait for "Authenticated as ...", Ctrl+C
   systemctl --user start librespot
   ```
   librespot caches the login in `~/.cache/librespot/credentials.json`, after which "Watson" stays listed in the
   Web API's device list. The system-wide alternative is raspotify itself:
   ```
   curl -sL https://dtcooper.github.io/raspotify/install.sh | sh
   sudo nano /etc/raspotify/conf        # set LIBRESPOT_NAME="Watson"  (must equal music.device_name)
   sudo systemctl restart raspotify
   ```
   (`scripts/pi_setup.sh --raspotify` does this.) To make Spotify go through the same echo-cancelled output as the
   replies, point it at the default device: `LIBRESPOT_BACKEND=alsa`, `LIBRESPOT_DEVICE=default` (with PipeWire's
   ALSA plugin the default device is the default sink).
3. **Log in once** (Mac or Pi): `python scripts/spotify_auth.py --client-id ... --client-secret ...`. It prints
   the authorize URL (scopes `user-modify-playback-state user-read-playback-state`); open it, approve, and paste
   the address you are redirected to (nothing needs to listen on 127.0.0.1:8888, the browser just shows an error
   page). It writes `config/spotify.json` (mode 600, gitignored). `config/spotify.example.json` shows the fields;
   optional `device_name`, `context_uri` (playlist/album URI to start when nothing is queued) and `shuffle` live
   there or under `music:` in `runtime.yaml`.
4. If the credentials were made on the Mac, copy `config/spotify.json` to the Pi (`scripts/sync_to_pi.sh` does,
   unless `--no-secrets`).

How it behaves: the access token is refreshed from the refresh token (cached, renewed 60 s early; a rotated
refresh token is written back); the device is found by name in `/me/player/devices`; PLAY transfers playback to it
if it is not active, then resumes, or starts `context_uri` if nothing is queued (with no `context_uri` and nothing
queued it says so); everything runs on one worker thread so it never blocks listening. **Ducking:** when the wake
word fires the Spotify volume drops to `duck_to` (20 %) and is restored after the turn (nested with the reply
clips' own duck; a VOLUME command during a turn changes the level that is restored). Offline / no device / auth /
Premium errors become a spoken reply and a message on the Music tile. Without credentials the runtime still starts
and answers "Spotify isn't set up yet." to music commands.


**After a power cut.** Spotify can keep a stale session for the "Watson" device, and then answers every play / pause
with 500 / 502 even though the device is listed. `deploy/librespot.service` therefore signs in a throw-away player
(`WatsonReset`) for about 12 s before starting the real one (with a 5 s gap); a different device taking over the
account's session clears the stale one. `play()` starts playback first and sets shuffle afterwards (a just-started
device answers "Restriction violated" to shuffle), and retries play for ~6 s if Spotify answers "Restriction violated"
right after a hand-over.

**Wake word over music.** The echo canceller removes 30-40 dB of the music once adapted (measured on the Pi with Spotify at 100 %), but it also damps the user's voice while both play, so the wake word scores lower. `wake.threshold_music` (0.50) replaces `wake.threshold` (0.70) while Spotify is playing.

**Volume at boot.** Music starts at 100 % (librespot `--initial-volume 100`) and the assistant's voice plays at full volume:
`deploy/vcm-volume.service` runs `scripts/pi_boot_volume.sh 100%`, which sets the real outputs (Bluetooth speaker,
built-in jack) and the echo-cancel sink to 100 %. It watches for 3 minutes, so a Bluetooth speaker that connects late is
covered.

## Setting up the Pi

The Mac is the source of truth. Nothing here is run automatically against the Pi.

```
# on the Mac
scripts/sync_to_pi.sh --dry-run          # abnunez@100.75.251.43:~/vcm-me2  (override with PI_HOST / PI_DIR)
scripts/sync_to_pi.sh                    # code + models + config + replies; excludes data/, exp/, .venv, results/runs
# on the Pi
cd ~/vcm-me2 && bash scripts/pi_setup.sh --echo-cancel --raspotify --service --kiosk
.venv/bin/python scripts/spotify_auth.py    # unless config/spotify.json came with the sync
```

`pi_setup.sh` installs the apt packages (python3-venv, libportaudio2, libsndfile1, mpv, alsa-utils, espeak-ng,
pipewire + pipewire-pulse + wireplumber, pulseaudio-utils, bluez, chromium, ffmpeg, curl, unclutter), creates
`.venv` with numpy onnxruntime sounddevice aiohttp requests pyyaml soundfile psutil, and optionally:

* `--echo-cancel`: `deploy/pipewire-echo-cancel.conf` (WebRTC AEC). Then `wpctl set-default` the
  `echo-cancel-source` and `echo-cancel-sink` ids. **Audio in and out use the default devices**, so the mic
  hears the speaker minus its own output. Pair the Bluetooth speaker first and make it the sink's target.
* `--raspotify`: see above.
* `--service`: `deploy/vcm-me2.service` as a **systemd user unit** (`systemctl --user status vcm-me2`,
  `journalctl --user -u vcm-me2 -f`) plus `loginctl enable-linger` so it starts at boot.
* `--kiosk`: `deploy/kiosk.desktop` autostarts Chromium full screen on `http://localhost:8080/?theme=dark`.

Hand start for debugging: `systemctl --user stop vcm-me2; .venv/bin/python scripts/pi_runtime.py`. Stop it with
`pkill -f "[s]cripts/pi_runtime.py"`.

Reply clips are rendered on the Mac (`pip install piper-tts soundfile`, voice files in `data/piper_voices/`):
`python scripts/gen_reply_audio.py --limit 0` writes `replies/*.mp3` (1,772 clips, ~30 MB, gitignored, synced to
the Pi). Voice: Piper `en_US-amy-medium` with the `ship` effect (length 1.1, noise 0.4). A missing clip falls back
to espeak-ng on the Pi (`say` on the Mac). `--list` counts clips without rendering, `--samples` auditions.

## Dashboard

`http://<pi>:8080/` (kiosk: `?theme=dark`). Apple HIG plain style (system colours, Inter, no theme/mascot):
a **Now** view (current exchange, accessory tiles: Music / Light / Thermostat / Phone / Weather, timers &
reminders with a Stop button while ringing, a Performance widget with per-stage latency, the activity log), **Home**
(tiles in detail), **Timers**, **Activity** (table with the ✓ / ✗ marks) and **Diagnostics** (latency / CPU / RAM /
temperature charts, counters, model facts, system log). The log shows **"Understood as"**, a sentence rebuilt from
command + slot, never a transcript. Emulated devices are labelled "Emulated". URL options: `?theme=dark|light`,
`?test=1` (adds a test panel in Diagnostics that sends any command + slot), `?lowfx=1`, `?mascot=<state>` (pins
the status orb for screenshots). Developing without hardware:
`.venv/bin/python scripts/pi_runtime.py --mock all --no-mic --silent` and open `http://localhost:8080/?test=1#diagnostics`.

## Metrics

Every run writes `runtime_logs/<YYYYmmdd-HHMMSS>/` (flushed every 60 s and on exit; the dashboard gets the same
numbers every 2 s):

* `turns.csv`: one row per turn: command, slot, prob, accepted, reply key, reason, ✓/✗ mark, and the stage latencies
  in ms: `wake_infer`, `wake_to_chime`, `capture`, `prep`, `vcm_infer`, `decode_dispatch`, `actuator`,
  `reply_start`, **`end_to_end`** (capture closed -> reply starts), `speech_end_to_reply` (last voiced frame ->
  reply starts, includes the 0.7 s endpointing wait on a real mic), `wake_to_reply`, `weather_api`, `spotify_api`.
* `resources.csv` (every 2 s): process CPU, total CPU, RSS MB, SoC temperature, `vcgencmd get_throttled`, music on/off.
* `summary.json` / `report.md`: p50 / p90 / p99 / max per stage, rejection rate, false wakes (a wake not followed by an
  accepted command), live ✓/✗ accuracy, per-command counts, model and device info.
* `convo.jsonl` + `clips/<id>.wav`: the turns and the command windows the model saw (playable from the dashboard).

`runtime_logs/state.json` (outside the run folders) keeps reminders across restarts; `--fresh-state` forgets them.

## Design decisions and limits

* The 1.5 s wake window is scored every 0.25 s on one thread (~16 ms per window for the int8 model on a Pi 4). The
  command capture starts right after the wake fires; audio spoken before the wake fired is not included, and a
  short tail of "Watson" can enter the window (the start-detection guard and the model's training on trimmed
  clips cover it; the 5 s cap leaves ~3.4 s for the command after a 0.8 s pause).
* Endpointing needs the speech to rise ~10 dB above the noise floor (measured on the second before the wake word).
  In a very noisy room the capture runs to the 5 s cap, which is still correct, only slower.
* Light and thermostat are emulated for the demo (the bulb speaks an encrypted Tuya protocol; the aircon has no remote).
* The model has no contact slot and no timer/alarm/reminder values beyond the three each, so replies and actions use
  exactly those values.
* Test on the Mac with `--input-wav`; the numbers on a Pi come from `runtime_logs/*/summary.json`.

## Tests

`.venv/bin/pytest -q` (~20 s). Runtime tests: `tests/test_runtime_*.py` cover the decision rule with tau, the
dispatcher for all 19 commands + OUT_OF_SCOPE, endpointing and wake logic on synthetic audio, window-prep parity
with the packed holdout, the Spotify actuator against a scripted Web API (token refresh, device lookup, play /
pause / next / volume, ducking, offline and auth errors), the reply catalogue, metrics files, the dashboard
WebSocket, and end-to-end runs of real holdout recordings through the real ensemble ONNX (mocked wake) plus the
CLI `--input-wav --json`.
