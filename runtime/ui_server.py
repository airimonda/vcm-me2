"""aiohttp dashboard server: serves runtime/ui/ and bridges it to the runtime bus over a WebSocket.

    server -> client: {"type": "state", "state": {...}}
                      {"type": "turn", "id", "t", "heard": {...}, "reply": {...}|null}
                      plus "heard" / "reply" / "answer" / "metrics" / "log", forwarded verbatim from the bus
    client -> server: {"type": "command", "command", "slot"?}   (test buttons, ?test=1)
                      {"type": "dismiss"}
                      {"type": "mark", "turn_id", "correct": bool, "intended_command"?}

Routes: / (index), /ws, /static/*, /clips/* (command windows of this run), /replies/* (reply clips).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

LOG = logging.getLogger("ui_server")
UI_DIR = Path(__file__).resolve().parent / "ui"
MAX_TURNS = 50
LATEST_TYPES = ("heard", "answer", "metrics")      # replayed to a page opened later


async def _pump(ws, queue):
    while True:
        message = await queue.get()
        if ws.closed:
            return
        try:
            await ws.send_str(json.dumps(message, default=str))
        except (ConnectionResetError, RuntimeError):
            return


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    bus, state, convo = request.app["bus"], request.app["state"], request.app["convo"]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    try:
        snapshot = state.snapshot()
    except Exception:
        LOG.exception("state.snapshot() failed")
        snapshot = {}
    await ws.send_str(json.dumps({"type": "state", "state": snapshot}, default=str))
    for turn in (convo.last(MAX_TURNS) if convo is not None else []):
        await ws.send_str(json.dumps({"type": "turn", **turn}, default=str))
    for msg in list(request.app["latest"].values()):
        await ws.send_str(json.dumps(msg, default=str))

    queue = bus.subscribe()
    pump = asyncio.create_task(_pump(ws, queue))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    LOG.warning("dropping malformed ws message: %r", msg.data[:200])
                    continue
                if not isinstance(data, dict) or "type" not in data:
                    LOG.warning("dropping ws message without a type: %r", data)
                    continue
                await bus.inbound.put(data)
            elif msg.type == WSMsgType.ERROR:
                LOG.warning("ws closed with exception: %s", ws.exception())
    finally:
        pump.cancel()
        bus.unsubscribe(queue)
    return ws


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(UI_DIR / "index.html")


@web.middleware
async def no_cache(request: web.Request, handler):
    """The page and its JS/CSS must never come from the browser cache: an old cached app.js (e.g. the previous
    runtime's dashboard on the same port) talks to this server in another message format and shows blank turns."""
    resp = await handler(request)
    if request.path == "/" or request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


async def inject_handler(request: web.Request) -> web.Response:
    """POST a WAV (any rate / channels): it replaces the mic signal until it ends. Off unless the runtime
    was started with audio.inject: true (anyone on the network could otherwise talk to the assistant)."""
    get = request.app.get("inject")
    src = get() if callable(get) else get
    if src is None:
        return web.json_response({"ok": False, "error": "injection is off (audio.inject: false)"}, status=403)
    import io
    import time

    import soundfile as sf
    body = await request.read()
    try:
        x, rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    except Exception as e:                           # not a readable WAV
        return web.json_response({"ok": False, "error": f"bad wav: {e}"}, status=400)
    x = x.mean(axis=1)
    if rate != 16000:
        n = int(round(len(x) * 16000 / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    t = time.time()
    dur = src.inject(x)
    return web.json_response({"ok": True, "t": t, "duration_s": dur})


def build_app(bus, state, convo=None, replies_dir=None, inject=None) -> web.Application:
    """Assemble the Application (separate from start_ui_server so tests can use aiohttp's test client)."""
    app = web.Application(middlewares=[no_cache])
    app["bus"], app["state"], app["convo"] = bus, state, convo
    app["latest"] = {}
    app["inject"] = inject
    app.router.add_get("/", index_handler)
    app.router.add_post("/inject", inject_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/static/", UI_DIR, name="static", show_index=False)
    if convo is not None:
        Path(convo.clips_dir).mkdir(parents=True, exist_ok=True)
        app.router.add_static("/clips/", convo.clips_dir, name="clips", show_index=False)
    if replies_dir is not None:
        Path(replies_dir).mkdir(parents=True, exist_ok=True)
        app.router.add_static("/replies/", str(replies_dir), name="replies", show_index=False)
    return app


async def start_ui_server(bus, state, convo=None, replies_dir=None, host="0.0.0.0", port=8080,
                          inject=None) -> web.AppRunner:
    app = build_app(bus, state, convo, replies_dir, inject)
    cache_queue = bus.subscribe()

    async def _cache_latest():
        while True:
            msg = await cache_queue.get()
            if isinstance(msg, dict) and msg.get("type") in LATEST_TYPES:
                app["latest"][msg["type"]] = msg

    app["cache_task"] = asyncio.ensure_future(_cache_latest())
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    LOG.info("UI server listening on http://%s:%s", host, port)
    return runner
