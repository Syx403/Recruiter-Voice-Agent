"""OpenAI Realtime (GA) over a raw WebSocket.

Facts this adapter relies on:
- `reasoning` defaults to null, so effort is always set explicitly.
- There is no automatic response after `function_call_output`; the orchestrator calls `respond()`.
- Audio is generated faster than real time, so on barge-in the client reports how much it actually
  played and we send `conversation.item.truncate` with that position.
- Turn taking: semantic VAD with `create_response` and `interrupt_response` off. The provider
  detects speech; the orchestrator decides whether it is a real turn and whether it interrupts.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator

import websockets

from backend.adapters.base import SessionConfig, VoiceEvent
from backend.keystore import get_secret

URL = "wss://api.openai.com/v1/realtime?model={model}"
BYTES_PER_MS = 48  # 24 kHz × 16-bit mono


class OpenAIRealtimeAdapter:
    provider = "openai"

    def __init__(self):
        self.ws = None
        self.model = ""
        self._queue: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        self._rx: asyncio.Task | None = None
        self.raw_log: list[dict] = []          # events without audio payloads, for runs/<id>/events.jsonl
        self._audio_ms: dict[str, float] = {}  # item_id → generated audio ms (capped by truncation)
        self._started: set[str] = set()
        self.response_active = False
        self._respond_pending: dict | None = None   # the response.create waiting for the active response to end

    async def connect(self, config: SessionConfig) -> None:
        self.model = config.model
        headers = {"Authorization": f"Bearer {get_secret('OPENAI_API_KEY')}"}
        self.ws = await websockets.connect(URL.format(model=config.model), additional_headers=headers,
                                           max_size=None)
        audio_in: dict = {"format": {"type": "audio/pcm", "rate": 24000},
                          "transcription": {"model": config.transcription_model}}
        if config.transcription_language:
            audio_in["transcription"]["language"] = config.transcription_language
        if config.noise_reduction:
            audio_in["noise_reduction"] = {"type": config.noise_reduction}
        if config.turn_detection:
            audio_in["turn_detection"] = {**config.turn_detection, "create_response": False,
                                          "interrupt_response": False}
        await self._send({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": config.instructions,
            "output_modalities": ["audio"],
            "reasoning": {"effort": config.reasoning_effort},
            "tools": [{"type": "function", **t} for t in config.tools],
            "tool_choice": "auto",
            "audio": {"input": audio_in,
                      "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": config.voice}},
        }})
        self._rx = asyncio.create_task(self._receive())

    async def _send(self, msg: dict):
        await self.ws.send(json.dumps(msg))

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16).decode()})

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        await self._send({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": call_id, "output": json.dumps(result, ensure_ascii=False)}})

    async def respond(self, instructions: str | None = None) -> None:
        """Ask for a response, optionally with instructions for this response only (the turn directive).
        If one is in progress, the request waits until it finishes; a newer request replaces an older one."""
        msg = {"type": "response.create"}
        if instructions:
            msg["response"] = {"instructions": instructions}
        if self.response_active:
            self._respond_pending = msg
            return
        self.response_active = True   # set now so a second call before `response.created` also waits
        await self._send(msg)

    async def cancel_response(self) -> None:
        if self.response_active:
            await self._send({"type": "response.cancel"})

    async def send_system_note(self, text: str) -> None:
        """Append a system message to the conversation (verified 2026-10-14: the next response follows it).
        Unlike per-response `instructions`, this keeps the instruction prefix byte-stable for caching."""
        await self._send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "system", "content": [{"type": "input_text", "text": text}]}})

    async def send_user_text(self, text: str) -> None:
        await self._send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})

    async def report_playback(self, item_id: str, played_ms: int) -> None:
        """Truncate the item at what the candidate actually heard. Clamped to what was generated, so the
        provider never receives a position past the end of the audio."""
        generated = self._audio_ms.get(item_id, 0)
        played = max(0, min(int(played_ms), int(generated)))
        if played < generated - 50:
            self._audio_ms[item_id] = played
            await self._send({"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0,
                              "audio_end_ms": played})

    async def events(self) -> AsyncIterator[VoiceEvent]:
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
                if typ == "response.output_audio.delta":
                    pcm = base64.b64decode(ev["delta"])
                    item = ev["item_id"]
                    if item not in self._started:
                        self._started.add(item)
                        q.put_nowait(VoiceEvent("agent_audio_start", {"item_id": item}))
                    self._audio_ms[item] = self._audio_ms.get(item, 0) + len(pcm) / BYTES_PER_MS
                    q.put_nowait(VoiceEvent("agent_audio_chunk", {"item_id": item}, pcm))
                    continue
                if typ == "response.output_audio_transcript.delta":
                    q.put_nowait(VoiceEvent("agent_transcript_delta", {"item_id": ev["item_id"],
                                                                       "delta": ev.get("delta", "")}))
                    continue
                if typ == "conversation.item.input_audio_transcription.delta":
                    q.put_nowait(VoiceEvent("user_transcript_delta", {"item_id": ev.get("item_id"),
                                                                      "delta": ev.get("delta", "")}))
                    continue
                if typ.endswith(".delta"):
                    continue
                self.raw_log.append(ev)
                if typ == "response.created":
                    self.response_active = True
                elif typ == "response.output_audio.done":
                    q.put_nowait(VoiceEvent("agent_audio_end", {"item_id": ev["item_id"],
                                                                "ms": self._audio_ms.get(ev["item_id"], 0)}))
                elif typ == "input_audio_buffer.speech_started":
                    q.put_nowait(VoiceEvent("user_speech_start", {"item_id": ev.get("item_id"),
                                                                  "audio_start_ms": ev.get("audio_start_ms")}))
                elif typ == "input_audio_buffer.speech_stopped":
                    q.put_nowait(VoiceEvent("user_speech_end", {"item_id": ev.get("item_id"),
                                                                "audio_end_ms": ev.get("audio_end_ms")}))
                elif typ == "conversation.item.input_audio_transcription.completed":
                    q.put_nowait(VoiceEvent("user_transcript", {"item_id": ev.get("item_id"),
                                                                "text": ev.get("transcript", "")}))
                    seconds = (ev.get("usage") or {}).get("seconds")
                    if seconds:
                        q.put_nowait(VoiceEvent("usage", {"kind": "transcription", "seconds": seconds}))
                elif typ == "conversation.item.input_audio_transcription.failed":
                    q.put_nowait(VoiceEvent("user_transcript", {"item_id": ev.get("item_id"), "text": "",
                                                                "failed": True}))
                elif typ == "response.output_audio_transcript.done":
                    q.put_nowait(VoiceEvent("agent_transcript", {"item_id": ev.get("item_id"),
                                                                 "text": ev.get("transcript", "")}))
                elif typ == "conversation.item.truncated":
                    q.put_nowait(VoiceEvent("agent_truncated", {"item_id": ev.get("item_id"),
                                                                "audio_end_ms": ev.get("audio_end_ms")}))
                elif typ == "response.done":
                    resp = ev.get("response", {})
                    calls = [{"call_id": o["call_id"], "name": o["name"], "arguments": o.get("arguments", "{}")}
                             for o in resp.get("output", [])
                             if o.get("type") == "function_call" and o.get("status", "completed") == "completed"]
                    if resp.get("usage"):
                        q.put_nowait(VoiceEvent("usage", {"kind": "response", "usage": resp["usage"]}))
                    if resp.get("status") == "cancelled":
                        q.put_nowait(VoiceEvent("interrupted", {"response_id": resp.get("id")}))
                    if calls:
                        q.put_nowait(VoiceEvent("tool_calls", {"calls": calls}))
                    q.put_nowait(VoiceEvent("response_done", {"status": resp.get("status"),
                                                              "has_calls": bool(calls)}))
                    self.response_active = False
                    if self._respond_pending:
                        msg, self._respond_pending = self._respond_pending, None
                        self.response_active = True
                        await self._send(msg)
                elif typ == "error":
                    err = ev.get("error", {})
                    # A truncate past the end of already-cut audio is harmless; keep it out of the error stream.
                    if err.get("code") == "invalid_value" and "shorter than" in err.get("message", ""):
                        continue
                    if err.get("code") == "conversation_already_has_active_response":
                        self._respond_pending = self._respond_pending or {"type": "response.create"}
                    q.put_nowait(VoiceEvent("error", err))
        except websockets.ConnectionClosed as e:
            q.put_nowait(VoiceEvent("error", {"message": f"connection closed: {e}"}))
        finally:
            q.put_nowait(None)

    async def close(self) -> dict:
        if self.ws is not None:
            await self.ws.close()
        if self._rx:
            try:
                await asyncio.wait_for(self._rx, 3)
            except (asyncio.TimeoutError, Exception):
                self._rx.cancel()
        return {}
