"""Builds the actuator set from the config (kept apart from runtime.app so tests need no models or audio)."""
from __future__ import annotations

import logging

from runtime.actuators.clock import ClockActuator
from runtime.actuators.comms import CommsActuator
from runtime.actuators.info import InfoActuator
from runtime.actuators.light import LightActuator
from runtime.actuators.spotify import build_music
from runtime.actuators.thermostat import ThermostatActuator

log = logging.getLogger("devices")


def build_actuators(cfg: dict, notify=None, speak=None, chime=None) -> dict:
    """dict keyed like runtime.dispatcher.ACTUATOR_FOR's values: music, light, thermostat, comms, clock, info"""
    mock = cfg["mock"]
    for dev in ("light", "aircon"):
        if not mock[dev]:
            log.warning("mock.%s is false but there is no real %s driver; using the emulated one", dev, dev)
    tc, wc = cfg["thermostat"], cfg["weather"]
    return {
        "music": build_music(cfg, mock["spotify"], notify=notify),
        "light": LightActuator(notify),
        "thermostat": ThermostatActuator(notify, tc["min"], tc["max"], tc["drift_period_s"], tc["start_room_temp"]),
        "comms": CommsActuator(notify, cfg["comms"]["contact"], cfg["comms"]["message_text"]),
        "clock": ClockActuator(notify, speak=speak, chime=chime, timezone=cfg["clock"]["timezone"]),
        "info": InfoActuator(wc["city"], wc["latitude"], wc["longitude"], wc["timeout_s"], cfg["clock"]["timezone"]),
    }
