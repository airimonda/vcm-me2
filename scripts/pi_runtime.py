#!/usr/bin/env python3
"""VCM-ME2 on-device runtime: mic -> "Watson" wake word -> command model -> Spotify / lights / thermostat /
timers / weather -> spoken reply, with a dashboard on :8080 and per-run metrics in runtime_logs/<ts>/.

All the logic lives in runtime/app.py; see docs/runtime.md. Stop a running instance with
    pkill -f "[s]cripts/pi_runtime.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
