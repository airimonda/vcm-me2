# Demo-day cheat sheet (Raspberry Pi "watsonpi")

Run from the Mac terminal. The Pi is `abnunez@100.75.251.43` (Tailscale; the Mac cannot use the Pi's LAN address).

## 0. Before you leave home
```
ssh abnunez@100.75.251.43 'systemctl --user is-active vcm-me2 librespot'      # both: active
/Applications/Tailscale.app/Contents/MacOS/Tailscale status | grep watsonpi    # Pi online?
```
Charge the littleFUN speaker. Bring the USB mic, Pi power supply, a screen + HDMI for the kiosk (optional).
If the venue Wi-Fi is new: the Pi must join it (or your phone hotspot) for Tailscale and Spotify to work.

## 1. Set-up at the venue (in order)
```
# 1. Pi reachable?
ssh abnunez@100.75.251.43 'uptime'

# 2. Speaker on, then connect it (speaker must NOT be connected to your phone/Mac)
ssh abnunez@100.75.251.43 'bluetoothctl connect 41:42:B2:79:0B:D1'
ssh abnunez@100.75.251.43 'bluetoothctl info 41:42:B2:79:0B:D1 | grep Connected'   # Connected: yes

# 3. Echo-cancel is the default in/out (both lines start with *)
ssh abnunez@100.75.251.43 'wpctl status | grep -E "echo-cancel-(source|sink)"'

# 4. Restart Spotify player + runtime (picks up the speaker)
ssh abnunez@100.75.251.43 'systemctl --user restart librespot vcm-me2; sleep 5; systemctl --user is-active librespot vcm-me2'
```
Dashboard: open **http://100.75.251.43:8080/?theme=dark** on the Mac (or the Pi screen in kiosk mode).

## 2. What to say
Say **"Watson"**, wait for the chime, then one command. Phrases the model was trained on (any of the 3 wordings works best):

| Command | Example phrases |
|---|---|
| Music | "Play music" / "Pause" / "Stop" / "Next song" / "Volume up" / "Lower the volume" |
| Weather / time | "What's the weather?" / "What time is it?" |
| Lights | "Turn on the lights" / "Kill the lights" / "Change color to Blue" / "Adjust brightness to 60 percent" |
| Thermostat | "Set the temperature to 22 degrees" (18 / 22 / 26) |
| Timer / alarm | "Start a timer for 10 seconds" (10 s / 30 s / 1 min), "Set an alarm for 6:00 AM" (6 AM / 8 AM / 9 PM) |
| Reminders | "Remind me to drink water" (drink water / study / exercise), "Show my reminders" |
| Calls | "Make a phone call" / "Send a message" (contact is always Mom) |

"Play music" starts your Spotify **Liked Songs** (shuffled). Anything else (other questions, chatter) should be ignored or answered "Sorry, I didn't catch that."
Only the values above exist (e.g. 22 degrees, not 23). Speak after the chime, not on top of it.

## 3. Quick fixes
| Problem | Do this |
|---|---|
| No reaction to "Watson" | `ssh abnunez@100.75.251.43 'systemctl --user restart vcm-me2'` ; check the USB mic is plugged in |
| Music says playing but silent | speaker disconnected: run step 1.2, then `systemctl --user restart librespot` |
| "Spotify not available" | `ssh abnunez@100.75.251.43 'systemctl --user restart librespot'`; Pi needs internet |
| Dashboard won't load | `systemctl --user restart vcm-me2`, reload the page after ~10 s |
| Pi unreachable over ssh | Pi off Wi-Fi/Tailscale: plug a screen + keyboard into the Pi, or check the hotspot |
| Everything stuck | `ssh abnunez@100.75.251.43 'sudo reboot'` (asks your password); services start by themselves |

## 4. Watch it live (debug view)
```
ssh abnunez@100.75.251.43
systemctl --user stop vcm-me2
cd ~/vcm-me2 && .venv/bin/python scripts/pi_runtime.py      # prints every wake/command; Ctrl+C to quit
systemctl --user start vcm-me2                             # back to normal afterwards
```
Never run the manual copy and the service at the same time (they fight over the mic and port 8080).

## 5. Results after the demo
```
ssh abnunez@100.75.251.43 'ls -t ~/vcm-me2/runtime_logs | head -3'
ssh abnunez@100.75.251.43 'cat ~/vcm-me2/runtime_logs/$(ls -t ~/vcm-me2/runtime_logs | grep -v state | head -1)/report.md'
scp -r abnunez@100.75.251.43:vcm-me2/runtime_logs ~/Desktop/demo_logs      # copy everything to the Mac
```
Each run folder has `report.md` (summary), `turns.csv` (every command: what it understood, confidence, latency),
`resources.csv` (CPU/RAM/temperature) and `clips/` (the recorded commands).

## 6. Numbers to quote
- Test split (4,443 clips): 94.3% variation balanced accuracy, 77.3% on real voices; 9.2% of out-of-scope accepted.
- Pi 4: 89 ms per 5-s window for the 3-model ensemble (1 core), wake word 16 ms per check.
- 882,738 parameters, 4.5 MB ONNX, no cloud models; Spotify and weather are the only network calls.
