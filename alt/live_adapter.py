"""gpt-live-1 over a raw WebSocket, client delegation mode.

Protocol facts this adapter relies on (verified against the docs and test sessions):
- `wss://api.openai.com/v1/live/sessions`; the first message is `session.start`; everything in it is immutable.
- Audio in and out are continuous real-time streams of 24 kHz PCM16 (output includes silence).
- Transcripts arrive as deltas with `start_ms`/`end_ms` on the session clock; there are no items or turn events.
- The client steers only with `session.commentary.append` (say this, paraphrase allowed),
  `session.instructions.append` (behaviour, or "say exactly") and `session.thinking.append` (silent facts).
- Billing is per second of session time; `session.usage.updated` is cumulative.
"""
from __future__ import annotations

import asyncio
import base64
import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np
import websockets

from backend.keystore import get_secret

URL = "wss://api.openai.com/v1/live/sessions"


@dataclass
class LiveEvent:
    type: str
    data: dict = field(default_factory=dict)
    audio: bytes | None = None


class GptLiveAdapter:
    provider = "openai"

    def __init__(self):
        self.ws = None
        self._queue: asyncio.Queue[LiveEvent | None] = asyncio.Queue()
        self._rx: asyncio.Task | None = None
        self.raw_log: list[dict] = []          # every non-audio event, for runs/<id>/events.jsonl
        self.seconds = 0.0
        self.session_id: str | None = None
        self.closed = asyncio.Event()
        self.started = asyncio.Event()

    async def connect(self, *, model: str, instructions: str, voice: str, seed: list[dict] | None = None) -> dict:
        self.ws = await websockets.connect(URL, max_size=None, additional_headers={
            "Authorization": f"Bearer {get_secret('OPENAI_API_KEY')}"})
        session = {"model": model, "instructions": instructions,
                   "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": voice}},
                   "delegation": {"type": "client"}, "store": False}
        if seed:
            session["input"] = seed
        await self._send({"type": "session.start", "event_id": "start", "session": session})
        self._rx = asyncio.create_task(self._receive())
        await asyncio.wait_for(self.started.wait(), 15)
        return {"model": model, "voice": voice, "delegation": "client", "session_id": self.session_id}

    async def _send(self, msg: dict):
        if self.ws is not None:
            await self.ws.send(json.dumps(msg))

    async def send_audio(self, pcm16: bytes) -> None:
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm16).decode()})

    async def append(self, kind: str, content: str, delegation_id: str | None = None) -> str:
        """kind: commentary | instructions | thinking. Returns the client event id (matched by the ack)."""
        eid = f"{kind[:3]}_{uuid.uuid4().hex[:10]}"
        await self._send({"type": f"session.{kind}.append", "event_id": eid, "delegation_id": delegation_id,
                          "content": content})
        return eid

    async def events(self) -> AsyncIterator[LiveEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def _receive(self):
        q = self._queue
        try:
            async for raw in self.ws:
                ev = json.loads(raw)
                typ = ev.get("type", "")
                if typ == "session.output_audio.delta":
                    pcm = base64.b64decode(ev["delta"])
                    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
                    q.put_nowait(LiveEvent("audio", {"rms": rms}, pcm))
                    continue
                if typ in ("session.input_transcript.delta", "session.output_transcript.delta"):
                    q.put_nowait(LiveEvent("in_delta" if typ.startswith("session.input") else "out_delta",
                                           {"text": ev.get("delta", ""), "start_ms": ev.get("start_ms"),
                                            "end_ms": ev.get("end_ms")}))
                    continue
                if typ == "session.usage.updated":
                    self.seconds = float((ev.get("usage") or {}).get("seconds") or self.seconds)
                    q.put_nowait(LiveEvent("usage", {"seconds": self.seconds}))
                    continue
                self.raw_log.append(ev)
                if typ == "session.started":
                    self.session_id = (ev.get("session") or {}).get("id")
                    self.started.set()
                    q.put_nowait(LiveEvent("started", {"session_id": self.session_id,
                                                       "expires_at": (ev.get("session") or {}).get("expires_at")}))
                elif typ == "session.delegation.created":
                    d = ev.get("delegation") or {}
                    if d.get("target", "client") == "client":
                        q.put_nowait(LiveEvent("delegation", {"id": d.get("id"), "offset_ms": ev.get("offset_ms")}))
                elif typ.endswith(".appended"):
                    q.put_nowait(LiveEvent("appended", {"kind": typ.split(".")[1], "id": ev.get("client_event_id"),
                                                        "start_ms": ev.get("start_ms")}))
                elif typ == "session.closed":
                    self.seconds = float((ev.get("usage") or {}).get("seconds") or self.seconds)
                    q.put_nowait(LiveEvent("closed", {"reason": ev.get("reason"), "seconds": self.seconds}))
                    self.closed.set()
                    return
                elif typ == "error":
                    q.put_nowait(LiveEvent("error", ev.get("error") or {}))
        except websockets.ConnectionClosed as e:
            q.put_nowait(LiveEvent("error", {"message": f"connection closed: {e}"}))
        finally:
            self.started.set()
            self.closed.set()
            q.put_nowait(None)

    async def close(self) -> float:
        """Close cleanly and return the billed session seconds."""
        try:
            if not self.closed.is_set():
                await self._send({"type": "session.close"})
                await asyncio.wait_for(self.closed.wait(), 8)
        except Exception:
            pass
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self._rx:
            self._rx.cancel()
        return self.seconds
