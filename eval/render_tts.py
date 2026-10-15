"""Azure Speech TTS (F0, japaneast) → 24 kHz 16-bit mono PCM, cached under runs/tts_cache."""
from __future__ import annotations

import hashlib
from pathlib import Path
from xml.sax.saxutils import escape

import httpx

from backend.keystore import get_secret

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "tts_cache"
URL = "https://japaneast.tts.speech.microsoft.com/cognitiveservices/v1"


def render(text: str, voice: str) -> bytes:
    lang = "-".join(voice.split("-")[:2])
    key = hashlib.sha256(f"{voice}|{text}".encode()).hexdigest()[:16]
    path = CACHE / f"{key}.pcm"
    if path.exists():
        return path.read_bytes()
    ssml = f"<speak version='1.0' xml:lang='{lang}'><voice name='{voice}'>{escape(text)}</voice></speak>"
    r = httpx.post(URL, content=ssml.encode(), timeout=60, headers={
        "Ocp-Apim-Subscription-Key": get_secret("AZURE_SPEECH_KEY"), "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm", "User-Agent": "ae-voice-screener"})
    r.raise_for_status()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return r.content
