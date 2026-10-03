"""The runtime: mic -> wake word -> capture + endpointing -> command model -> dispatcher -> actuators -> reply.

    Runtime(cfg)            builds everything (models, actuators, bus, metrics, ...)
    await rt.run_stream(s)  drive it from an audio Source (microphone or WAV)
    await rt.handle_command a decided Command through dispatch, reply, logging and metrics

scripts/pi_runtime.py is a thin CLI around `main()` here.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import platform
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from runtime import audio as audio_mod
from runtime import sounds
from runtime.bus import Bus
from runtime.command import Command
from runtime.config import REPO_ROOT, load_config, parse_mock, resolve
from runtime.convo import ConvoLog
from runtime.devices import build_actuators
from runtime.dispatcher import dispatch
from runtime.endpoint import Endpointer, noise_floor_db  # noqa: F401
from runtime.feedback import Feedback
from runtime.metrics import Metrics, Turn
from runtime.model import CommandModel, WakeModel
from runtime.prep import prepare_window
from runtime.state import State
from runtime.wake import WakeDetector

SR = 16000
log = logging.getLogger("runtime")


class Runtime:
    """holds every long-lived piece and the one turn-processing path all input sources use"""

    def __init__(self, cfg: dict, fresh_state: bool = False, load_wake: bool = True, run_dir=None, state_path=None):
        self.cfg = cfg
        self.model = CommandModel(resolve(cfg["model"]["path"]), cfg["model"]["tau"], cfg["model"]["intra_threads"])
        self.wake_model = None
        wcfg = cfg["wake"]
        if load_wake and wcfg["enabled"]:
            self.wake_model = WakeModel(resolve(wcfg["path"]), wcfg["intra_threads"])
        self.infer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self.model.warmup()

        self.bus = Bus()
        self.metrics = Metrics(model_info={**self.model.info(), "wake": str(wcfg["path"]),
                                           "wake_threshold": wcfg["threshold"]},
                               device_info={"platform": platform.platform(), "python": platform.python_version(),
                                            "machine": platform.machine()},
                               root=str(resolve(cfg["metrics"]["dir"])), run_dir=run_dir)
        self.run_dir = Path(self.metrics.dir)
        sp = state_path or (self.run_dir.parent / "state.json")
        if fresh_state and Path(sp).exists():
            Path(sp).unlink()
        self.state = State(path=sp)
        self.convo = ConvoLog(self.run_dir)

        sounds.set_enabled(cfg["audio"]["play"])
        self.feedback = Feedback(cfg["replies"]["dir"], cfg["replies"]["catalogue"], bus=self.bus,
                                 play=cfg["audio"]["play"], earcon_volume=cfg["audio"]["earcon_volume"])
        self.actuators = self.build_actuators()
        self.music = self.actuators["music"]
        self.feedback.media = self.music
        self.counters = {"turns": 0}
        self._tasks: list[asyncio.Task] = []

    # -- construction -----------------------------------------------------------------
    def build_actuators(self) -> dict:
        return build_actuators(self.cfg, self.notify, speak=self.speak, chime=self.feedback.chime)

    def notify(self):
        self.state.save()
        self.bus.publish({"type": "state", "state": self.state.snapshot()})

    async def speak(self, key, **values):
        return await self.feedback.say(key, **values)

    def set_mode(self, mode):
        self.state.set_mode(mode)
        self.notify()

    # -- ducking ----------------------------------------------------------------------
    def duck_music(self):
        if self.cfg["dispatcher"]["ducking"]:
            try:
                self.music.duck()
            except Exception:
                log.exception("duck failed")

    def restore_music(self):
        if self.cfg["dispatcher"]["ducking"]:
            try:
                self.music.unduck()
            except Exception:
                log.exception("unduck failed")

    # -- background tasks -----------------------------------------------------------------
    async def start_background(self, ui=True, port=None):
        self.music.start_polling(self.state)
        self._tasks.append(asyncio.ensure_future(self.metrics.sampler_loop(bus=self.bus, is_playing=self.music.is_playing)))
        self._tasks.append(asyncio.ensure_future(self.inbound_loop()))
        self.ui_runner = None
        self.inject_source = getattr(self, "inject_source", None)
        if ui and self.cfg["ui"]["enabled"]:
            try:
                from runtime.ui_server import start_ui_server
                u = self.cfg["ui"]
                inject = (lambda: self.inject_source) if self.cfg["audio"].get("inject") else None
                self.ui_runner = await start_ui_server(self.bus, self.state, self.convo, self.feedback.dir,
                                                       u["host"], port or u["port"], inject=inject)
                log.info("dashboard on http://%s:%s", u["host"], port or u["port"])
            except ImportError as e:
                log.warning("dashboard unavailable (%s); continuing without it", e)
            except OSError as e:
                log.warning("dashboard could not start: %s", e)
        self.notify()

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        if getattr(self, "ui_runner", None):
            await self.ui_runner.cleanup()
        for a in self.actuators.values():
            try:
                a.cancel()
            except Exception:
                log.exception("actuator shutdown failed")
        self.infer_pool.shutdown(wait=False)
        self.metrics.flush()

    async def inbound_loop(self):
        """dashboard -> runtime: test buttons, dismiss, ✓/✗ marks"""
        while True:
            msg = await self.bus.inbound.get()
            try:
                t = msg.get("type")
                if t == "command":
                    cmd = Command(msg["command"], msg.get("slot") or None, 1.0, source="client")
                    await self.handle_command(cmd, Turn())
                elif t == "dismiss":
                    if self.actuators["clock"].dismiss(self.state):
                        self.notify()
                elif t == "mark":
                    self.metrics.apply_mark(msg.get("turn_id"), msg.get("correct"), msg.get("intended_command"))
                    self.convo.mark(msg.get("turn_id"), msg.get("correct"), msg.get("intended_command"))
            except Exception:
                log.exception("inbound message failed: %r", msg)

    def live_log(self, rec: dict):
        """append one JSON line to the live log (cfg metrics.live_log, default logs/live.log); never raises"""
        path = self.cfg["metrics"].get("live_log")
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), **rec}) + "\n")
        except OSError:
            log.exception("live log write failed")

    # -- one turn -----------------------------------------------------------------------------
    async def classify(self, window: np.ndarray, turn: Turn):
        decision = await asyncio.get_running_loop().run_in_executor(self.infer_pool, self.model.classify, window)
        turn.mark("vcm_done")
        return Command.from_decision(decision)

    async def handle_command(self, cmd: Command, turn: Turn | None = None, audio=None):
        turn = turn or Turn()
        bus = self.bus
        bus.publish({"type": "heard", "command": cmd.command, "slot": cmd.slot, "prob": cmd.prob,
                     "accepted": None, "reason": ""})
        self.set_mode("thinking")
        turn.mark("actuator_start")
        resp = await dispatch(cmd, self.state, self.actuators, self.cfg["dispatcher"]["oos_reply"])
        turn.mark("actuator_done")
        data = resp.data
        for name in ("weather", "spotify"):
            if f"_{name}_start" in data:
                turn.mark(f"{name}_start", data.pop(f"_{name}_start"))
                turn.mark(f"{name}_done", data.pop(f"_{name}_done"))
        if resp.answer:
            bus.publish({"type": "answer", **resp.answer})
        self.notify()
        bus.publish({"type": "heard", "command": cmd.command, "slot": cmd.slot, "prob": cmd.prob,
                     "accepted": resp.accepted, "reason": resp.reason})
        reply_text, reply_urls = None, []
        if resp.say is not None:
            self.set_mode("speaking")
            reply_text, reply_urls = await self.feedback.say_response(resp, on_start=lambda: turn.mark("reply_start"))
        else:
            if resp.sound:
                self.feedback.chime(resp.sound)
            turn.mark("reply_start")
        self.set_mode("idle")
        turn_id = self.convo.log_turn(cmd, resp.accepted, reply_text=reply_text,
                                      reply_audio_url=reply_urls[0] if reply_urls else None, audio=audio,
                                      bus=bus, reason=resp.reason)
        self.metrics.record_turn(turn_id, turn, command=cmd.command, slot=cmd.slot, prob=cmd.prob,
                                 accepted=resp.accepted, reply_key=resp.say, reason=resp.reason, source=cmd.source)
        self.counters["turns"] += 1
        slot = f" [{cmd.slot}]" if cmd.slot else ""
        print(f"  {cmd.command}{slot}  p={cmd.prob:.2f}  accepted={resp.accepted}"
              + (f"  reply={reply_text!r}" if reply_text else "") + (f"  ({resp.reason})" if resp.reason else ""))
        return resp, turn_id

    async def capture_turn(self, it, turn: Turn, pre_audio=None, noise_db=None):
        """after a wake: capture the command with endpointing, prepare the window, classify, dispatch.
        Returns (Command|None, Response|None)."""
        c = self.cfg["capture"]
        if noise_db is None:
            noise_db = noise_floor_db(pre_audio) if pre_audio is not None and len(pre_audio) else -60.0
        ep = Endpointer(noise_db, c["max_s"], c["min_s"], c["end_silence_s"], c["no_speech_timeout_s"],
                        c["start_ignore_s"], c["margin_db"], c["min_db"], c["min_speech_s"])
        turn.mark("capture_start")
        try:
            while not ep.feed(await it.__anext__()):
                pass
        except StopAsyncIteration:
            ep.finish_eof()
        turn.mark("window_closed")
        if ep.last_voiced_wall is not None:
            turn.mark("speech_end", ep.last_voiced_wall)       # wall clock: last voiced frame processed
        log.info("capture: %.2fs, %s", ep.n_samples / SR, ep.reason)
        if not ep.has_speech:
            cmd = Command("NO_SPEECH", None, 0.0)
            turn.mark("actuator_start")
            turn.mark("actuator_done")
            turn.mark("reply_start")
            self.bus.publish({"type": "heard", "command": cmd.command, "slot": None, "prob": 0.0,
                              "accepted": False, "reason": "no_speech"})
            self.feedback.chime("reject")
            turn_id = self.convo.log_turn(cmd, False, reason="no_speech", bus=self.bus)
            self.metrics.record_turn(turn_id, turn, command=cmd.command, accepted=False, reason="no_speech")
            print("  (wake without speech)")
            return cmd, None
        window = prepare_window(ep.audio)
        turn.mark("prep_done")
        cmd = await self.classify(window, turn)
        # one line per recognised command for the class benchmark (vcm-benchmark): window prep + log-mel + model
        self.live_log({"intent": cmd.command, "slot": cmd.slot,
                       "infer_ms": round((turn.marks["vcm_done"] - turn.marks["window_closed"]) * 1000, 1),
                       "audio_ms": round(len(window) / SR * 1000), "prob": round(cmd.prob, 3)})
        resp, _ = await self.handle_command(cmd, turn, audio=window)
        return cmd, resp

    def wake_threshold(self) -> float:
        """wake.threshold, or wake.threshold_music while music is playing: the echo canceller also damps the user's
        voice when it talks over the music, so the wake word scores lower then"""
        w = self.cfg["wake"]
        tm = w.get("threshold_music")
        try:
            playing = tm is not None and self.music.is_playing()
        except Exception:
            playing = False
        return float(tm) if playing else float(w["threshold"])

    async def run_stream(self, source, use_wake=True, noise_db=None, once=False):
        """Drive the runtime from an audio Source until it ends (WAV) or forever (mic).

        use_wake=False is the mocked wake trigger: the stream is treated as already woken, one turn per
        source (--input-wav default). noise_db overrides the endpointer's noise floor in that mode."""
        it = source.__aiter__()
        wcfg = self.cfg["wake"]
        det = None
        if use_wake:
            if self.wake_model is None:
                raise RuntimeError("wake word model not loaded (wake.enabled/path in config/runtime.yaml)")
            det = WakeDetector(self.wake_model.timed_score, self.wake_model.window, wcfg["hop_s"], wcfg["threshold"],
                               wcfg["consecutive"], wcfg["cooldown_s"])
            print('ready. Say "Watson", then your command. Ctrl+C to stop.')
        try:
            while True:
                turn = Turn()
                pre = None
                if det is not None:
                    event = None
                    while event is None:
                        det.threshold = self.wake_threshold()
                        event = det.feed(await it.__anext__())
                    turn.mark("wake_done", event.t_done)
                    self.live_log({"event": "wake", "msg": "wake word detected"})
                    turn.mark("wake_start", event.t_done - event.infer_ms / 1000.0)
                    pre = det.ring.last(int(1.5 * SR))
                self.set_mode("listening")
                self.feedback.chime("wake")
                turn.mark("chime")
                self.duck_music()
                try:
                    cmd, resp = await self.capture_turn(it, turn, pre, noise_db)
                except Exception:
                    log.exception("turn failed")        # one bad turn must not end the listening loop
                    cmd = resp = None
                finally:
                    self.restore_music()
                    self.set_mode("idle")
                if det is not None:
                    self.metrics.note_wake(bool(resp and resp.accepted and cmd.command != "OUT_OF_SCOPE"))
                    det.release()
                    source.flush()      # audio queued while the reply played (its own sound) must not reach the wake word
                if det is None or once:
                    return
        except StopAsyncIteration:
            return


# ===================================================================== CLI
def collect_wavs(paths):
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.wav")))
        else:
            files.append(p)
    return files


async def run_wavs(rt: Runtime, files, use_wake: bool, realtime: bool):
    results = []
    block_ms = rt.cfg["audio"]["block_ms"]
    for f in files:
        wav = audio_mod.load_wav(f)
        src = audio_mod.WavSource(wav, block_ms, lead_s=0.5 if use_wake else 0.0, tail_s=1.5, realtime=realtime)
        before = len(rt.metrics.turns)
        print(f)
        await rt.run_stream(src, use_wake=use_wake)
        for row in rt.metrics.turns[before:]:
            results.append({"file": f, "command": row["command"], "slot": row["slot"] or None, "prob": row["prob"],
                            "accepted": row["accepted"], "reply_key": row["reply_key"]})
    return results


def build_arg_parser():
    ap = argparse.ArgumentParser(description="VCM-ME2 on-device runtime", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="""examples:
  scripts/pi_runtime.py                                  # mic + wake word + dashboard (the Pi)
  scripts/pi_runtime.py --mock all --input-wav clip.wav  # one clip through capture -> model -> dispatcher (mock wake)
  scripts/pi_runtime.py --mock all --no-mic              # dashboard + test buttons only
  pkill -f "[s]cripts/pi_runtime.py"                     # stop it
""")
    ap.add_argument("--config", default=None, help="YAML config (default config/runtime.yaml)")
    ap.add_argument("--mock", default=None, help='devices to emulate: "all", "none" or a comma list of light,aircon,spotify')
    ap.add_argument("--no-ui", action="store_true", help="do not start the dashboard server")
    ap.add_argument("--port", type=int, default=None, help="dashboard port (default ui.port)")
    ap.add_argument("--input-wav", nargs="+", metavar="WAV", help="feed these WAV files/directories instead of the mic")
    ap.add_argument("--use-wake", action="store_true",
                    help="with --input-wav: run the real wake model on the file (it must contain 'Watson ...'); "
                         "default is a mocked wake trigger at the start of each file")
    ap.add_argument("--keep-open", action="store_true", help="with --input-wav: keep the dashboard up after the files (Ctrl+C to stop)")
    ap.add_argument("--realtime", action="store_true", help="with --input-wav: pace the file at real time")
    ap.add_argument("--no-mic", action="store_true", help="no audio input; dashboard test buttons only")
    ap.add_argument("--silent", action="store_true", help="never play sounds or replies (audio.play=false)")
    ap.add_argument("--fresh-state", action="store_true", help="forget saved reminders (runtime_logs/state.json)")
    ap.add_argument("--metrics-dir", default=None, help="parent of the per-run folders (default metrics.dir = runtime_logs)")
    ap.add_argument("--model", default=None, help="command model .onnx (default model.path)")
    ap.add_argument("--tau", type=float, default=None, help="override the sidecar tau")
    ap.add_argument("--wake-model", default=None, help="wake word .onnx (default wake.path)")
    ap.add_argument("--wake-threshold", type=float, default=None)
    ap.add_argument("--inject", action="store_true",
                    help="accept WAVs POSTed to /inject on the dashboard port in place of the mic signal "
                         "(vcm-benchmark --inject; anyone on the network can then feed audio)")
    ap.add_argument("--json", action="store_true", help="with --input-wav: print one JSON result line per file at the end")
    return ap


def overrides_from_args(a) -> dict:
    o: dict = {"mock": parse_mock(a.mock)}
    if a.no_ui:
        o["ui"] = {"enabled": False}
    if a.silent or (a.input_wav and not a.realtime):          # a file replay is silent unless asked to be real time
        o["audio"] = {"play": False}
    if a.metrics_dir:
        o["metrics"] = {"dir": a.metrics_dir}
    if a.model:
        o.setdefault("model", {})["path"] = a.model
    if a.tau is not None:
        o.setdefault("model", {})["tau"] = a.tau
    if a.wake_model:
        o.setdefault("wake", {})["path"] = a.wake_model
    if a.wake_threshold is not None:
        o.setdefault("wake", {})["threshold"] = a.wake_threshold
    if a.inject:
        o.setdefault("audio", {})["inject"] = True
    return o


async def amain(a) -> int:
    cfg = load_config(a.config, overrides_from_args(a))
    need_wake = not a.no_mic and (not a.input_wav or a.use_wake)
    rt = Runtime(cfg, fresh_state=a.fresh_state, load_wake=need_wake)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, task.cancel)
        except (NotImplementedError, RuntimeError):
            pass
    ui = not a.no_ui and not (a.input_wav and not a.realtime and not a.keep_open)         # a one-shot file replay needs no dashboard
    await rt.start_background(ui=ui, port=a.port)
    results = []
    source = None
    try:
        if a.input_wav:
            files = collect_wavs(a.input_wav)
            if not files:
                print("no WAV files found", file=sys.stderr)
                return 2
            results = await run_wavs(rt, files, a.use_wake, a.realtime)
            if a.keep_open:
                print("replay done; dashboard stays up (Ctrl+C to stop)")
                await asyncio.Event().wait()
        elif a.no_mic:
            print("no-mic mode: waiting for dashboard messages (Ctrl+C to stop)")
            await asyncio.Event().wait()
        else:
            source = audio_mod.open_mic(cfg["audio"]["backend"], cfg["audio"]["device"], cfg["audio"]["block_ms"])
            if cfg["audio"].get("inject"):
                source = audio_mod.InjectSource(source)
                rt.inject_source = source
                print("audio injection ON: POST a WAV to /inject on the dashboard port (benchmark without a speaker)")
            await rt.run_stream(source, use_wake=True)
    except asyncio.CancelledError:
        pass
    finally:
        if source is not None:
            source.close()
        rt.music.unduck_all()
        await rt.stop()
        print(f"metrics written to {rt.metrics.dir}")
    if a.json:
        for r in results:
            print(json.dumps(r))
    return 0


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(amain(a)) or 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
