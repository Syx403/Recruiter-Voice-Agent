"""Call recording: `recording.wav` is a mono mix for playback; `candidate.wav` and `interviewer.wav` keep
the two speakers apart for analysis (overlap, talk time, re-transcription).

The interviewer track follows a server-side playback clock (the same model the call page uses): each
chunk starts when the previous one ends, or now if the queue is empty, and a confirmed barge-in cuts
the track at that moment. It doubles as the orchestrator's "is the interviewer speaking" signal.
Recordings stay local (runs/, gitignored).
"""
from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

RATE = 24000
BYTES_PER_S = RATE * 2


class PlaybackClock:
    def __init__(self):
        self.queue_end = 0.0                      # monotonic time the queued audio finishes playing
        self.segments: list[list] = []           # [start_monotonic, pcm bytes, item_id]

    def add(self, item_id: str, pcm: bytes):
        now = time.monotonic()
        start = max(now, self.queue_end)
        self.segments.append([start, pcm, item_id])
        self.queue_end = start + len(pcm) / BYTES_PER_S

    def playing(self) -> bool:
        return time.monotonic() < self.queue_end

    def cut(self) -> dict[str, float]:
        """A confirmed barge-in: drop everything not yet played. Returns {item_id: ms heard} for the items
        that were cut, so code knows exactly which words the candidate heard."""
        now = time.monotonic()
        kept, cut_items = [], set()
        for start, pcm, item in self.segments:
            if start >= now:
                cut_items.add(item)
                continue
            end = start + len(pcm) / BYTES_PER_S
            if end > now:
                pcm = pcm[: int((now - start) * RATE) * 2]
                cut_items.add(item)
            kept.append([start, pcm, item])
        self.segments = kept
        self.queue_end = now
        return {i: sum(len(p) for _, p, it in kept if it == i) / BYTES_PER_S * 1000 for i in cut_items}


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self.mic_start: float | None = None      # monotonic time of the first mic chunk
        self.mic = bytearray()
        self.agent = PlaybackClock()

    def add_mic(self, pcm: bytes):
        if self.mic_start is None:
            self.mic_start = time.monotonic()
        self.mic.extend(pcm)

    def save(self) -> Path | None:
        if self.mic_start is None and not self.agent.segments:
            return None
        t0 = min([self.mic_start or float("inf")] + [s[0] for s in self.agent.segments])
        left = np.frombuffer(bytes(self.mic), dtype=np.int16)
        end = max([(self.mic_start or t0) + len(left) / RATE]
                  + [s[0] + len(s[1]) / BYTES_PER_S for s in self.agent.segments])
        n = int((end - t0) * RATE) + 1
        tracks = np.zeros((n, 2), dtype=np.int32)
        if len(left):
            off = int(((self.mic_start or t0) - t0) * RATE)
            tracks[off:off + len(left), 0] = left[: n - off]
        for start, pcm, _ in self.agent.segments:
            a = np.frombuffer(pcm, dtype=np.int16)
            off = int((start - t0) * RATE)
            tracks[off:off + len(a), 1] = a[: n - off]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mix = np.clip(tracks.sum(axis=1), -32768, 32767).astype(np.int16)
        _write(self.path, mix)
        _write(self.path.with_name("candidate.wav"), tracks[:, 0].astype(np.int16))
        _write(self.path.with_name("interviewer.wav"), tracks[:, 1].astype(np.int16))
        return self.path


def _write(path: Path, mono: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(mono.tobytes())
