"""Provider-agnostic voice adapter interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


@dataclass
class SessionConfig:
    model: str
    instructions: str
    tools: list[dict]
    voice: str = "marin"
    reasoning_effort: str = "low"
    transcription_model: str = "gpt-realtime-whisper"
    transcription_language: str | None = None
    noise_reduction: str | None = None
    turn_detection: dict | None = None


@dataclass
class VoiceEvent:
    """Normalised event. `type` is one of: agent_audio_start, agent_audio_chunk, agent_audio_end,
    agent_transcript_delta, agent_transcript, agent_truncated, user_speech_start, user_speech_end,
    user_transcript_delta, user_transcript, tool_calls, interrupted, response_done, error, usage."""
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    audio: bytes | None = None


class VoiceAdapter(Protocol):
    async def connect(self, config: SessionConfig) -> None: ...
    async def send_audio(self, pcm16: bytes) -> None: ...
    def events(self) -> AsyncIterator[VoiceEvent]: ...
    async def send_tool_result(self, call_id: str, result: dict) -> None: ...
    async def respond(self, instructions: str | None = None) -> None: ...   # one response, with this turn's directive
    async def send_user_text(self, text: str) -> None: ...     # session start ("candidate has joined")
    async def cancel_response(self) -> None: ...
    async def send_system_note(self, text: str) -> None: ...   # steering note for the next reply
    async def report_playback(self, item_id: str, played_ms: int) -> None: ...
    async def close(self) -> dict: ...
