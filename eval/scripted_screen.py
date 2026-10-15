"""Scripted full screen helpers: a synthetic candidate answers from its answer bank.

It behaves like the browser call page: streams 24 kHz PCM in real time (silence between answers),
plays the agent's audio on a real-time clock, reports the played position on barge-in, and answers
after each agent turn with the next answer for the current card (read from the monitor stream).

    uv run python -m eval.scripted_screen arjun_inflated_attainment [--base http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time

import httpx
import numpy as np
import websockets
import yaml

from eval.render_tts import ROOT, render

CHUNK_MS = 100
CHUNK_BYTES = CHUNK_MS * 48
REPLY_GAP_S = 2.0       # silence after the agent stops before the candidate answers
MAX_WALL_S = 17 * 60
NO_QUESTION_WAIT_S = 8.0


class Player:
    """Real-time playback clock for the agent's audio."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.queue_end = 0.0
        self.last_audio = 0.0

    def add(self, item_id: str, nbytes: int):
        now = time.monotonic()
        start = max(now, self.queue_end)
        it = self.items.setdefault(item_id, {"start": start, "ms": 0.0})
        it["ms"] += nbytes / 48
        self.queue_end = start + nbytes / 48 / 1000
        self.last_audio = now

    def playing(self) -> bool:
        return time.monotonic() < self.queue_end

    def flush(self) -> tuple[str, int] | None:
        now = time.monotonic()
        current = None
        for item_id, it in self.items.items():
            if it["start"] <= now < it["start"] + it["ms"] / 1000:
                current = (item_id, int((now - it["start"]) * 1000))
        self.queue_end = now
        return current


def cough_pcm(ms: int = 500, seed: int = 0) -> bytes:
    """A short broadband burst with a fast attack and decay, roughly like a cough or a knock on the mic."""
    rng = np.random.default_rng(seed)
    n = int(24 * ms)
    env = np.exp(-np.linspace(0, 6, n)) * np.minimum(1, np.linspace(0, 20, n))
    x = rng.normal(0, 1, n) * env * 9000
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


async def cough(outbox: asyncio.Queue, counter: dict):
    """Inject one non-speech burst 1 s into an interviewer turn (only if the candidate is silent)."""
    await asyncio.sleep(1.0)
    if not outbox.empty():
        return
    counter["sent"] += 1
    pcm = cough_pcm(seed=counter["sent"])
    for k in range(0, len(pcm), CHUNK_BYTES):
        outbox.put_nowait(pcm[k:k + CHUNK_BYTES].ljust(CHUNK_BYTES, b"\0"))


async def barge(outbox: asyncio.Queue, text: str, voice: str):
    """Talk over the interviewer 1.5 s into its turn."""
    await asyncio.sleep(1.5)
    print(f"         (candidate interrupts: {text})")
    pcm = render(text, voice)
    for k in range(0, len(pcm), CHUNK_BYTES):
        outbox.put_nowait(pcm[k:k + CHUNK_BYTES].ljust(CHUNK_BYTES, b"\0"))


def pick_answer(card: dict, answers: dict, used: dict) -> tuple[str, str]:
    """Answer the card's question first; then answer the probed field; then any further turns."""
    cid = card["id"]
    bank = answers.get(cid, {})
    u = used.setdefault(cid, {"answer": False, "probes": 0, "more": 0})
    if card.get("probes_used", 0) > u["probes"] and u["answer"]:
        u["probes"] = card["probes_used"]
        field = card.get("last_probe")
        text = (bank.get("probes") or {}).get(field)
        if text:
            return text, f"{cid} probe:{field}"
    if not u["answer"]:
        u["answer"] = True
        return bank.get("answer", "Sorry, could you repeat the question?"), f"{cid} answer"
    more = bank.get("more") or []
    if u["more"] < len(more):
        u["more"] += 1
        return more[u["more"] - 1], f"{cid} more#{u['more']}"
    return "I think I've covered that.", f"{cid} fallback"


async def run(candidate_id: str, base: str, max_wall_s: float = MAX_WALL_S, noise: bool = False,
              interrupt: bool = False):
    cand = yaml.safe_load((ROOT / "data/sim_candidates" / f"{candidate_id}.yaml").read_text())
    voice = cand["accent_voice"]
    answers: dict[str, dict] = cand["answers"]
    used: dict[str, dict] = {}
    inv = httpx.post(f"{base}/api/invites", json={"candidate_id": candidate_id}).json()
    token = inv["token"]
    wsbase = base.replace("http", "ws")
    state: dict = {}
    player = Player()
    outbox: asyncio.Queue[bytes] = asyncio.Queue()
    agent_turns = {"n": 0, "answered": 0, "asked": False}
    ended = asyncio.Event()
    printed: set[str] = set()
    interrupts: dict[str, str] = cand.get("interrupts") or {}
    interrupted: set[str] = set()
    noise_bursts = {"sent": 0}

    async def monitor():
        sid = None
        while sid is None:          # the session exists once the candidate has joined
            await asyncio.sleep(0.2)
            sid = next((i["session_id"] for i in httpx.get(f"{base}/api/invites").json()
                        if i["token"] == token and i["session_id"]), None)
        print(f"session {sid}")
        async with websockets.connect(f"{wsbase}/ws/monitor/{sid}", max_size=None) as m:
            async for raw in m:
                msg = json.loads(raw)
                if msg["type"] == "state":
                    state.update(msg["state"])
                elif msg["type"] == "timeline":
                    e = msg["entry"]
                    if e.get("partial") or e["id"] in printed or e["kind"] == "stage":
                        continue
                    printed.add(e["id"])
                    if e["kind"] == "agent":
                        if "?" in e["text"]:
                            agent_turns["asked"] = True
                        print(f"[{e['t']:6.1f}] AGENT {e['text']}")
                    elif e["kind"] in ("candidate", "noise"):
                        print(f"[{e['t']:6.1f}] {'CAND ' if e['kind'] == 'candidate' else 'NOISE'} {e['text']}")
                    elif e["kind"] == "note":
                        print(f"         STEER {e.get('intent')}: {e['text'][38:240]}")
                    elif e["kind"] == "tool":
                        out = e.get("output") or {}
                        nxt = out.get("next") or {}
                        print(f"         tool {e.get('name')} {json.dumps(e.get('arguments'))[:160]} → "
                              f"{out.get('status')} next={nxt.get('type')}:{nxt.get('card', '')}")
                elif msg["type"] == "ended":
                    ended.set()
                    return

    async with websockets.connect(f"{wsbase}/ws/interview/{token}", max_size=None) as ws:
        async def mic():
            silence = bytes(CHUNK_BYTES)
            nxt = time.monotonic()
            while not ended.is_set():
                chunk = outbox.get_nowait() if not outbox.empty() else silence
                await ws.send(chunk)
                nxt += CHUNK_MS / 1000
                await asyncio.sleep(max(0, nxt - time.monotonic()))

        async def rx():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "audio":
                    player.add(msg["item_id"], len(base64.b64decode(msg["pcm"])))
                elif msg["type"] == "agent_speaking":
                    agent_turns["n"] += 1
                    card = (state.get("card") or {}).get("id")
                    if interrupt and card in interrupts and card not in interrupted:
                        interrupted.add(card)
                        asyncio.create_task(barge(outbox, interrupts[card], voice))
                    if noise and agent_turns["n"] % 3 == 0:
                        asyncio.create_task(cough(outbox, noise_bursts))
                elif msg["type"] == "flush":
                    cur = player.flush()
                    if cur:
                        await ws.send(json.dumps({"type": "playback", "item_id": cur[0], "played_ms": cur[1]}))
                elif msg["type"] == "ended":
                    ended.set()
                    return

        async def speaker():
            idle_reported = 0.0
            while not ended.is_set():
                await asyncio.sleep(0.2)
                quiet = (not player.playing() and outbox.empty()
                         and time.monotonic() - max(player.queue_end, player.last_audio) > REPLY_GAP_S)
                if quiet and player.items and idle_reported != player.queue_end:
                    idle_reported = player.queue_end
                    await ws.send(json.dumps({"type": "idle"}))
                if not quiet or agent_turns["n"] <= agent_turns["answered"]:
                    continue
                # Answer only after a turn that asked something; a bare "Okay." before a tool call is not a
                # question. If the agent stays silent for long, answer anyway so the call cannot deadlock.
                silent_for = time.monotonic() - max(player.queue_end, player.last_audio)
                if not agent_turns["asked"] and silent_for < NO_QUESTION_WAIT_S:
                    continue
                agent_turns["answered"] = agent_turns["n"]
                agent_turns["asked"] = False
                text, label = pick_answer(state.get("card") or {"id": "close_next"}, answers, used)
                print(f"         (candidate answers {label})")
                pcm = render(text, voice)
                for k in range(0, len(pcm), CHUNK_BYTES):
                    outbox.put_nowait(pcm[k:k + CHUNK_BYTES].ljust(CHUNK_BYTES, b"\0"))

        mon = asyncio.create_task(monitor())
        tasks = [asyncio.create_task(c()) for c in (mic, rx, speaker)]
        try:
            await asyncio.wait_for(ended.wait(), max_wall_s)
        except asyncio.TimeoutError:
            print("wall-clock limit reached")
            await ws.send(json.dumps({"type": "hangup"}))
            await asyncio.sleep(2)
        for t in tasks + [mon]:
            t.cancel()
    print(json.dumps({k: state.get(k) for k in ("stage", "clock", "cost_usd", "overrides", "refusals",
                                                "consistency", "flags", "end_reason")}, indent=1, default=str))
    print(f"ledger entries: {len(state.get('ledger', []))}; noise bursts sent: {noise_bursts['sent']}")
    return token


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_id")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--max-seconds", type=float, default=MAX_WALL_S)
    ap.add_argument("--noise", action="store_true", help="inject cough-like bursts while the interviewer speaks")
    ap.add_argument("--interrupt", action="store_true", help="talk over the interviewer on the cards in `interrupts`")
    a = ap.parse_args()
    asyncio.run(run(a.candidate_id, a.base, a.max_seconds, a.noise, a.interrupt))
