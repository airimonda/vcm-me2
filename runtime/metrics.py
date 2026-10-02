"""Per-turn stage timings + a resource sampler. Every run writes runtime_logs/<YYYYmmdd-HHMMSS>/
{turns.csv, resources.csv, summary.json, report.md} (flushed every 60 s and on exit) and publishes
{"type":"metrics",...} on the bus every 2 s for the dashboard.

    m = Metrics(model_info={...}, device_info={...}, root="runtime_logs")
    turn = Turn(); turn.mark("window_closed"); ...; turn.mark("reply_start")
    m.record_turn(turn_id, turn, command=..., slot=..., prob=..., accepted=...)
    m.apply_mark(turn_id, correct=True)          # dashboard check / cross
    asyncio.ensure_future(m.sampler_loop(bus=bus, is_playing=lambda: music.is_playing()))
    m.flush()
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
import subprocess
import time

try:
    import psutil
except ImportError:                       # pragma: no cover
    psutil = None

# (stage name, start mark, end mark). A stage is only computed when both marks exist (a mock-wake
# turn has no wake_infer, a turn without a reply has no reply stages, ...).
STAGE_PAIRS = [
    ("wake_infer", "wake_start", "wake_done"),
    ("wake_to_chime", "wake_done", "chime"),
    ("capture", "capture_start", "window_closed"),
    ("prep", "window_closed", "prep_done"),
    ("vcm_infer", "prep_done", "vcm_done"),
    ("decode_dispatch", "vcm_done", "actuator_start"),
    ("actuator", "actuator_start", "actuator_done"),
    ("reply_start", "actuator_done", "reply_start"),
    ("end_to_end", "window_closed", "reply_start"),
    ("speech_end_to_reply", "speech_end", "reply_start"),
    ("wake_to_reply", "wake_done", "reply_start"),
    ("weather_api", "weather_start", "weather_done"),
    ("spotify_api", "spotify_start", "spotify_done"),
]
STAGE_NAMES = [s[0] for s in STAGE_PAIRS]
DASH_STAGES = ["wake_infer", "prep", "vcm_infer", "decode_dispatch", "actuator", "reply_start", "end_to_end"]


class Turn:
    """one turn's named checkpoints (monotonic clock)"""

    def __init__(self):
        self.marks = {}

    def mark(self, name, t=None):
        self.marks[name] = t if t is not None else time.perf_counter()
        return self

    def stages(self):
        out = {}
        for name, a, b in STAGE_PAIRS:
            if a in self.marks and b in self.marks:
                out[name] = (self.marks[b] - self.marks[a]) * 1000.0
        return out


def _vcgencmd():
    """(soc_temp_c, throttled_flags) from vcgencmd, or (None, "") off-Pi"""
    if not shutil.which("vcgencmd"):
        return None, ""
    temp, throttled = None, ""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=2)
        temp = float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        pass
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=2)
        throttled = out.strip().split("=")[1]
    except Exception:
        pass
    return temp, throttled


def _percentiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None

    def pct(p):
        return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]
    return {"p50": pct(50), "p90": pct(90), "p99": pct(99), "max": values[-1], "n": len(values)}


class Metrics:
    def __init__(self, model_info=None, device_info=None, root="runtime_logs", run_dir=None):
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.dir = str(run_dir or os.path.join(root, self.run_id))
        os.makedirs(self.dir, exist_ok=True)
        self.turns = []
        self._rows_by_id = {}
        self.resources = []
        self.model_info = model_info or {}
        self.device_info = device_info or {}
        self.false_wakes = 0
        self.wake_count = 0
        self.started = time.time()
        self._last_flush = time.time()
        self._proc = psutil.Process() if psutil else None
        if self._proc is not None:
            self._proc.cpu_percent(interval=None)      # prime the counter (the first call always reads 0)
            psutil.cpu_percent(interval=None)
        self.sample_resources()

    # -- turns ----------------------------------------------------------------
    def record_turn(self, turn_id, turn, command=None, slot=None, prob=None, accepted=None, reply_key=None,
                    reason="", source="voice"):
        row = {"turn_id": turn_id, "t": time.time(), "command": command, "slot": slot or "", "prob": prob,
               "accepted": accepted, "source": source,
               "reply_key": reply_key if isinstance(reply_key, str) or reply_key is None else json.dumps(reply_key),
               "reason": reason, "mark_correct": "", "mark_intended_command": ""}
        row.update(turn.stages())
        self.turns.append(row)
        self._rows_by_id[turn_id] = row
        self._maybe_flush()
        return row

    def apply_mark(self, turn_id, correct, intended_command=None):
        row = self._rows_by_id.get(turn_id)
        if row is None:
            return False
        row["mark_correct"] = bool(correct)
        row["mark_intended_command"] = intended_command or ""
        return True

    def note_wake(self, accepted_command_followed):
        """a wake fired; no accepted command after it = a false wake"""
        self.wake_count += 1
        if not accepted_command_followed:
            self.false_wakes += 1

    # -- resources --------------------------------------------------------------
    def sample_resources(self, music_playing=False):
        row = {"t": time.time(), "cpu_process": None, "cpu_total": None, "rss_mb": None,
               "soc_temp_c": None, "throttled": "", "music_playing": bool(music_playing)}
        if self._proc is not None:
            try:
                row["cpu_process"] = self._proc.cpu_percent(interval=None)
                row["rss_mb"] = round(self._proc.memory_info().rss / 1e6, 1)
            except Exception:
                pass
        if psutil is not None:
            try:
                row["cpu_total"] = psutil.cpu_percent(interval=None)
            except Exception:
                pass
        row["soc_temp_c"], row["throttled"] = _vcgencmd()
        self.resources.append(row)
        self._maybe_flush()
        return row

    def snapshot(self, row=None):
        """the {"type":"metrics"} bus message (flat fields + nested {last,p50} per stage)"""
        s = self.summary()
        last = self.turns[-1] if self.turns else {}
        row = row or (self.resources[-1] if self.resources else {})
        stages = {name: {"last": last.get(name), "p50": (s["stages_ms"].get(name) or {}).get("p50"),
                         "p90": (s["stages_ms"].get(name) or {}).get("p90")} for name in STAGE_NAMES}
        thr = row.get("throttled") or ""
        try:
            throttled_now = bool(int(thr, 16) & 0xF) if thr else False
        except ValueError:
            throttled_now = False
        return {
            "type": "metrics", "stages": stages,
            "end_to_end": stages["end_to_end"], "vcm_infer": stages["vcm_infer"], "wake_infer": stages["wake_infer"],
            "cpu_pct": row.get("cpu_process"), "cpu_total": row.get("cpu_total"), "ram_mb": row.get("rss_mb"),
            "soc_temp_c": row.get("soc_temp_c"), "throttled": throttled_now,
            "turns": len(self.turns), "marked_accuracy": s["marked_accuracy"],
            "false_wakes": self.false_wakes, "wake_count": self.wake_count,
            "model": self.model_info,
        }

    async def sampler_loop(self, bus=None, is_playing=None, interval=2.0):
        while True:
            playing = False
            if is_playing is not None:
                try:
                    playing = bool(is_playing())
                except Exception:
                    playing = False
            row = self.sample_resources(music_playing=playing)
            if bus is not None:
                bus.publish(self.snapshot(row))
            await asyncio.sleep(interval)

    # -- flush / summary -----------------------------------------------------------
    def _maybe_flush(self):
        if time.time() - self._last_flush >= 60:
            self.flush()

    def flush(self):
        self._write_csv(os.path.join(self.dir, "turns.csv"), self.turns)
        self._write_csv(os.path.join(self.dir, "resources.csv"), self.resources)
        summary = self.summary()
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(os.path.join(self.dir, "report.md"), "w") as f:
            f.write(self._report_md(summary))
        self._last_flush = time.time()

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            open(path, "w").close()
            return
        fields = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)

    def summary(self):
        stages = {name: _percentiles([r.get(name) for r in self.turns]) for name in STAGE_NAMES}
        n_turns = len(self.turns)
        n_accepted = sum(1 for r in self.turns if r.get("accepted"))
        marked = [r for r in self.turns if r.get("mark_correct") != ""]
        n_correct = sum(1 for r in marked if r.get("mark_correct") is True)
        per_command = {}
        for r in self.turns:
            d = per_command.setdefault(r.get("command") or "?", {"count": 0, "accepted": 0})
            d["count"] += 1
            if r.get("accepted"):
                d["accepted"] += 1
        res = lambda k: [r.get(k) for r in self.resources if isinstance(r.get(k), (int, float))]  # noqa: E731
        return {
            "run_id": self.run_id, "model": self.model_info, "device": self.device_info,
            "duration_s": round(time.time() - self.started, 1),
            "n_turns": n_turns, "n_accepted": n_accepted,
            "rejection_rate": (1 - n_accepted / n_turns) if n_turns else None,
            "n_marked": len(marked), "n_marked_correct": n_correct,
            "marked_accuracy": (n_correct / len(marked)) if marked else None,
            "wake_count": self.wake_count, "false_wakes": self.false_wakes,
            "resources": {"cpu_process_pct": _percentiles(res("cpu_process")), "rss_mb": _percentiles(res("rss_mb")),
                          "soc_temp_c": _percentiles(res("soc_temp_c")),
                          "throttled_seen": sorted({r["throttled"] for r in self.resources if r.get("throttled")})},
            "per_command": per_command, "stages_ms": stages,
        }

    def _report_md(self, summary):
        lines = [f"# Run {summary['run_id']}", "",
                 f"- turns: {summary['n_turns']}  accepted: {summary['n_accepted']}  rejection rate: {summary['rejection_rate']}",
                 f"- marked: {summary['n_marked']}  marked accuracy: {summary['marked_accuracy']}",
                 f"- wakes: {summary['wake_count']}  false wakes: {summary['false_wakes']}",
                 f"- model: {summary['model']}", f"- device: {summary['device']}",
                 "", "## Stage latency (ms)", "", "| stage | p50 | p90 | p99 | max | n |", "|---|---|---|---|---|---|"]
        for name in STAGE_NAMES:
            p = summary["stages_ms"].get(name)
            if p:
                lines.append(f"| {name} | {p['p50']:.1f} | {p['p90']:.1f} | {p['p99']:.1f} | {p['max']:.1f} | {p['n']} |")
            else:
                lines.append(f"| {name} | - | - | - | - | 0 |")
        lines += ["", "## Per command", "", "| command | count | accepted |", "|---|---|---|"]
        for c, d in sorted(summary["per_command"].items()):
            lines.append(f"| {c} | {d['count']} | {d['accepted']} |")
        return "\n".join(lines) + "\n"
