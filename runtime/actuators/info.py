"""WEATHER (Open-Meteo, Quezon City by default) and TIME (the device clock).

DNS on the Pi (and on phone tethers) returns a NAT64 IPv6 address first; Python tries it before IPv4 with no
happy-eyeballs race, so a call took 4-6 s instead of ~1.3 s. Weather is the only HTTP user besides Spotify, so
forcing IPv4 process-wide is fine (the Spotify API is reachable over IPv4 too).
"""
from __future__ import annotations

import asyncio
import datetime
import socket
import time

import requests
import urllib3.util.connection

from runtime.actuators.base import Actuator
from runtime.command import Response

urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

# WMO weather code -> one of ~11 spoken condition groups (Open-Meteo docs)
WMO_GROUPS = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy",
    45: "foggy", 48: "foggy",
    51: "drizzly", 53: "drizzly", 55: "drizzly", 56: "drizzly", 57: "drizzly",
    61: "rainy", 63: "rainy", 65: "rainy", 66: "rainy", 67: "rainy", 80: "rainy", 81: "rainy", 82: "rainy",
    71: "snowy", 73: "snowy", 75: "snowy", 77: "snowy", 85: "snowy", 86: "snowy",
    95: "stormy", 96: "stormy", 99: "stormy",
}
CONDITIONS = sorted(set(WMO_GROUPS.values()))


def now_local(tz: str | None = None) -> datetime.datetime:
    if tz:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo(tz))
    return datetime.datetime.now()


def time_parts(now: datetime.datetime):
    """(hour12 1-12, minute, 'AM'|'PM')"""
    return now.hour % 12 or 12, now.minute, "AM" if now.hour < 12 else "PM"


class InfoActuator(Actuator):
    name = "info"
    kind = "real"

    def __init__(self, city="Quezon City", latitude=14.676, longitude=121.0437, timeout_s=3.0, timezone=None,
                 fetch=None):
        self.city, self.lat, self.lon = city, latitude, longitude
        self.timeout_s = timeout_s
        self.timezone = timezone
        self._fetch = fetch or self._open_meteo        # injectable for tests

    async def handle(self, cmd, state):
        if cmd.command == "TIME":
            h, mm, ampm = time_parts(now_local(self.timezone))
            return Response(say="time_now", data={"h": h, "mm": mm, "ampm": ampm},
                            answer={"kind": "time", "text": f"It's {h}:{mm:02d} {ampm}."})
        if cmd.command == "WEATHER":
            t0 = time.perf_counter()
            try:       # requests' timeout is per phase (DNS/connect/read); enforce a total deadline too
                temp, condition = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(None, self._fetch), self.timeout_s)
            except Exception:
                resp = Response(say="weather_offline", ok=False)
            else:
                state.data["weather"] = {"temp": temp, "condition": condition, "city": self.city, "t": time.time()}
                resp = Response(say="weather", data={"temp": temp, "condition": condition},
                                answer={"kind": "weather", "text": f"{temp}°C and {condition} in {self.city}",
                                        "temp": temp, "condition": condition, "city": self.city})
            resp.data = dict(resp.data, _weather_start=t0, _weather_done=time.perf_counter())
            return resp
        return Response(say=None, ok=False)

    def _open_meteo(self):
        url = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&current=temperature_2m,weather_code").format(lat=self.lat, lon=self.lon)
        r = requests.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        cur = r.json()["current"]
        return round(cur["temperature_2m"]), WMO_GROUPS.get(int(cur["weather_code"]), "cloudy")
