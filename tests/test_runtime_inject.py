"""Audio injection: InjectSource replaces mic blocks in step with the mic clock; /inject is off by default."""
import asyncio
import io

import numpy as np
import soundfile as sf
from aiohttp.test_utils import TestClient, TestServer

from runtime.audio import InjectSource, Source
from runtime.bus import Bus
from runtime.state import State


class FakeMic(Source):
    def __init__(self, block=4):
        self.block = block
        self.queued = 0

    async def __anext__(self):
        return np.full(self.block, -1.0, np.float32)

    def flush(self):
        n, self.queued = self.queued, 0
        return n


def test_inject_replaces_blocks_then_returns_to_mic():
    async def go():
        src = InjectSource(FakeMic())
        assert (await src.__anext__() == -1).all()
        src.inject(np.arange(10, dtype=np.float32))
        assert (await src.__anext__()).tolist() == [0, 1, 2, 3]
        assert (await src.__anext__()).tolist() == [4, 5, 6, 7]
        assert (await src.__anext__()).tolist() == [8, 9, 0, 0]       # tail padded with silence
        assert not src.active
        assert (await src.__anext__() == -1).all()
    asyncio.run(go())


def test_flush_advances_the_clip_like_real_time():
    async def go():
        mic = FakeMic()
        src = InjectSource(mic)
        src.inject(np.arange(16, dtype=np.float32))
        await src.__anext__()                                        # 0..3
        mic.queued = 2                                               # two blocks dropped while busy
        assert src.flush() == 2
        assert (await src.__anext__()).tolist() == [12, 13, 14, 15]
    asyncio.run(go())


def _wav(x, sr=16000):
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def test_inject_endpoint_off_by_default_and_on_when_enabled():
    from runtime.ui_server import build_app

    async def go():
        app = build_app(Bus(), State())
        async with TestClient(TestServer(app)) as c:
            r = await c.post("/inject", data=_wav(np.zeros(1600, np.float32)))
            assert r.status == 403
        src = InjectSource(FakeMic())
        app = build_app(Bus(), State(), inject=lambda: src)
        async with TestClient(TestServer(app)) as c:
            r = await c.post("/inject", data=_wav(np.zeros(8000, np.float32), sr=8000))
            j = await r.json()
            assert r.status == 200 and j["ok"] and abs(j["duration_s"] - 1.0) < 1e-3
            assert src.active
            assert (await c.post("/inject", data=b"not a wav")).status == 400
    asyncio.run(go())
