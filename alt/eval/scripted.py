"""Scripted full screen against the running app (:8100), adapted to gpt-live-1, whose audio is a continuous stream:
the candidate answers after an interviewer segment that asked a question has finished (from the monitor's
transcript), not when audio stops. Answers come from data/candidates/<id>/answers.yaml (keyed by card id), or the
older data/sim_candidates/<id>.yaml. The run goes through the pipeline: question plan → approval → link (the role's
screen configuration must already be approved by a recruiter).

    uv run --group alt python -m alt.eval.scripted arjun_inflated_attainment [--role R] [--voice en-IN-PrabhatNeural]
        [--interrupt] [--max-seconds 720]
Costs about USD 0.05 per minute of call (gpt-live-1) plus Haiku. Check `python -m alt.budget` first.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import websockets
import yaml

from alt import budget
from eval.render_tts import ROOT, render
from eval.scripted_screen import CHUNK_BYTES, CHUNK_MS, pick_answer

REPLY_GAP_S = 0.8          # after the interviewer's question segment is final
NO_QUESTION_WAIT_S = 12.0  # answer anyway if the interviewer stays silent this long after a candidate turn


def load_answers(candidate_id: str) -> tuple[dict, dict, str | None]:
    new = ROOT / "data" / "candidates" / candidate_id / "answers.yaml"
    if new.exists():
        d = yaml.safe_load(new.read_text())
        bank = d.get("answers", d)
        if "perf_clarify" in bank:                   # one bank for either clarification code may insert
            bank.setdefault("perf_cv", bank["perf_clarify"])
        return bank, {}, None
    cand = yaml.safe_load((ROOT / "data/sim_candidates" / f"{candidate_id}.yaml").read_text())
    return cand["answers"], cand.get("interrupts") or {}, cand.get("accent_voice")


async def run(candidate_id: str, base: str, max_wall_s: float, interrupt: bool, role_id: str | None = None,
              voice: str | None = None):
    budget.require(max_wall_s / 60 * 0.05 + 0.05, "scripted screen")
    answers, interrupts, default_voice = load_answers(candidate_id)
    voice = voice or default_voice or "en-IN-PrabhatNeural"
    used: dict = {}
    body = {"candidate_id": candidate_id, **({"role_id": role_id} if role_id else {})}
    r = httpx.post(f"{base}/api/invites", json=body, timeout=120)
    if r.status_code != 200:
        raise SystemExit(f"cannot prepare the plan: {r.text}")
    pipe = r.json()
    if "approve_plan" in (pipe.get("next") or []):
        print(f"plan: {len(pipe['plan']['cards'])} personalised cards, {len(pipe['plan']['extra_probes'])} probes;"
              " approved by the script for this test run")
        pipe = httpx.post(f"{base}/api/pipelines/{pipe['id']}/plan", json={}, timeout=60).json()
    token = pipe["token"]
    wsbase = base.replace("http", "ws")
    state: dict = {}
    outbox: asyncio.Queue[bytes] = asyncio.Queue()
    ended = asyncio.Event()
    shared = {"question_at": None, "last_audio_end": time.monotonic(), "speaking": False, "barged": set()}
    printed: set[str] = set()

    def speak(text: str):
        pcm = render(text, voice)
        for k in range(0, len(pcm), CHUNK_BYTES):
            outbox.put_nowait(pcm[k:k + CHUNK_BYTES].ljust(CHUNK_BYTES, b"\0"))

    async def barge(text: str):
        await asyncio.sleep(1.5)
        print(f"         (candidate interrupts: {text})")
        speak(text)

    async def monitor():
        sid = None
        while sid is None:
            await asyncio.sleep(0.2)
            sid = next((i["session_id"] for i in httpx.get(f"{base}/api/invites").json()
                        if i["token"] == token and i["session_id"]), None)
        print(f"session {sid}  →  {base}/#/sessions/{sid}")
        async with websockets.connect(f"{wsbase}/ws/monitor/{sid}", max_size=None) as m:
            async for raw in m:
                msg = json.loads(raw)
                if msg["type"] == "state":
                    state.update(msg["state"])
                elif msg["type"] == "timeline":
                    e = msg["entry"]
                    if e["kind"] == "agent" and e.get("partial") and interrupt:
                        card = (state.get("card") or {}).get("id")
                        if (card in interrupts and card not in shared["barged"] and e.get("role") == "directed"
                                and len(e.get("text", "").split()) >= 4):
                            shared["barged"].add(card)
                            asyncio.create_task(barge(interrupts[card]))
                    if e.get("partial") or e["kind"] == "stage":
                        continue
                    key = f"{e['id']}:{e.get('text', '')}:{e.get('first_sound_s')}"
                    if key in printed:
                        continue
                    printed.add(key)
                    if e["kind"] == "agent":
                        print(f"[{e['t']:6.1f}] AGENT{' (bridge)' if e.get('role') == 'bridge' else ''} {e['text']}")
                        if "?" in e["text"]:
                            shared["question_at"] = time.monotonic()
                    elif e["kind"] in ("candidate", "noise"):
                        print(f"[{e['t']:6.1f}] {'CAND ' if e['kind'] == 'candidate' else 'NOISE'} {e['text']}")
                    elif e["kind"] == "note":
                        nodes = " → ".join(f"{s['node']}" + (f"({s['ms']}ms)" if s.get('ms') and s['ms'] > 5 else "")
                                           for s in e.get("trace") or [])
                        print(f"         BRAIN {e.get('intent')} [{nodes}] first_sound={e.get('first_sound_s')} "
                              f"next_line={e.get('next_line_s')} :: {e['text'][:200]}")
                    elif e["kind"] == "tool":
                        out = e.get("output") or {}
                        print(f"         tool {e.get('name')} {json.dumps(e.get('arguments'))[:140]} → {out.get('status')}")
                elif msg["type"] == "ended":
                    ended.set()
                    return

    async with websockets.connect(f"{wsbase}/ws/interview/{token}", max_size=None) as ws:
        async def mic():
            silence = bytes(CHUNK_BYTES)
            nxt = time.monotonic()
            while not ended.is_set():
                if not outbox.empty():
                    chunk = outbox.get_nowait()
                    shared["last_audio_end"] = time.monotonic()
                else:
                    chunk = silence
                await ws.send(chunk)
                nxt += CHUNK_MS / 1000
                await asyncio.sleep(max(0, nxt - time.monotonic()))

        async def rx():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "speaking":
                    shared["speaking"] = bool(msg["on"])
                    shared["last_agent"] = time.monotonic()
                elif msg["type"] == "ended":
                    ended.set()
                    return
                elif msg["type"] == "error":
                    print("ERROR", msg.get("error"))

        async def answerer():
            while not ended.is_set():
                await asyncio.sleep(0.2)
                if not outbox.empty() or shared["speaking"]:
                    continue
                q = shared["question_at"]
                ready = q is not None and time.monotonic() - q > REPLY_GAP_S
                quiet_for = time.monotonic() - max(shared["last_audio_end"], shared.get("last_agent", 0))
                stuck = q is None and quiet_for > NO_QUESTION_WAIT_S
                if not (ready or stuck):
                    continue
                shared["question_at"] = None
                shared["last_audio_end"] = time.monotonic()
                text, label = pick_answer(state.get("card") or {"id": "close_next"}, answers, used)
                print(f"         (candidate answers {label}{' — no question heard' if stuck else ''})")
                speak(text)

        mon = asyncio.create_task(monitor())
        tasks = [asyncio.create_task(c()) for c in (mic, rx, answerer)]
        try:
            await asyncio.wait_for(ended.wait(), max_wall_s)
        except asyncio.TimeoutError:
            print("wall-clock limit reached")
            await ws.send(json.dumps({"type": "hangup"}))
            await asyncio.sleep(3)
        for t in tasks + [mon]:
            t.cancel()
    print(json.dumps({k: state.get(k) for k in ("stage", "clock", "cost_usd", "cost_split", "overrides",
                                                "consistency", "flags", "end_reason")}, indent=1, default=str))
    turns = state.get("turns") or []
    fs = sorted(t["first_sound_s"] for t in turns if t.get("first_sound_s") is not None)
    nl = sorted(t["next_line_s"] for t in turns if t.get("next_line_s") is not None)
    print(f"ledger entries: {len(state.get('ledger', []))}; turns: {len(turns)}; "
          f"first sound median {fs[len(fs)//2] if fs else None}s; next line median {nl[len(nl)//2] if nl else None}s "
          f"(max {nl[-1] if nl else None}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_id")
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--max-seconds", type=float, default=720)
    ap.add_argument("--interrupt", action="store_true")
    ap.add_argument("--role", default=None)
    ap.add_argument("--voice", default=None, help="Azure TTS voice, e.g. en-IN-PrabhatNeural, en-SG-WayneNeural")
    a = ap.parse_args()
    asyncio.run(run(a.candidate_id, a.base, a.max_seconds, a.interrupt, a.role, a.voice))
