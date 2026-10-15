"""Directive-driven session orchestrator (gpt-realtime-2.1 + turn controller). It is not served by the app;
the app imports its helpers (Subscriber, has_content, is_answerable, load_config). It bridges the call page, the
voice adapter, the tools and the live monitor, and owns the stage machine, turn taking, the time and cost caps,
the session timeline, the recording and all logging.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import traceback
from pathlib import Path

import yaml
from fastapi import WebSocket, WebSocketDisconnect

from backend import costs, store
from backend.adapters.base import SessionConfig, VoiceEvent
from backend.adapters.openai_realtime import OpenAIRealtimeAdapter
from backend.content_store import ContentStore, candidate
from backend.context_builder import build_instructions, turn_note
from backend.monitor import check_turn
from backend.recording import Recorder
from backend.controller import decide, opening, plan, sounds_unfinished
from backend.session import SessionState

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("uvicorn.error")
WORDS = re.compile(r"[A-Za-z]{2,}")
FILLERS = {"mhm", "mm", "mmm", "hmm", "hm", "uh", "um", "uhm", "erm", "ah", "oh", "eh", "huh", "yeah", "yep", "ok",
           "okay", "right", "sure", "yes", "no", "so", "and", "well"}


def has_content(text: str) -> bool:
    """True if the transcript has a word that is not a filler or backchannel ("mhm", "uh", "yeah").
    Only such a turn interrupts the interviewer; a lone "yes"/"no" still gets a reply."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text or "")]
    return any(w.strip("'") not in FILLERS for w in words)


def is_answerable(text: str) -> bool:
    """A turn worth replying to: content words, or a bare yes/no/okay (consent, eligibility)."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text or "")]
    return has_content(text) or any(w in ("yes", "no", "yeah", "yep", "okay", "ok", "sure") for w in words)
HOLD_S = 2.5             # wait this long for the rest of an unfinished sentence ("I sell air product to")
CLOSE_TIMEOUT_S = 60      # the close stage is 30 s by design; end the call if it drags on
JOIN_TEXT = "(The candidate has joined the call.)"
A_STAGES = "stages_a_frozen.yaml"


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config/config.yaml").read_text())


class Subscriber:
    """One outbound WebSocket with its own queue and sender task. A slow or hidden monitor tab must
    never block the call: when its queue is full, messages to it are dropped."""

    def __init__(self, ws: WebSocket, maxsize: int = 400):
        self.ws = ws
        self.q: asyncio.Queue[str | None] = asyncio.Queue(maxsize)
        self.dropped = 0
        self.alive = True
        self.task = asyncio.create_task(self._run())

    def put(self, data: str):
        if not self.alive:
            return
        try:
            self.q.put_nowait(data)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _run(self):
        try:
            while (data := await self.q.get()) is not None:
                await self.ws.send_text(data)
        except Exception:
            pass
        self.alive = False

    def close(self):
        self.put(None)


class Orchestrator:
    registry: dict[str, "Orchestrator"] = {}

    def __init__(self, role_id: str, persona_id: str, candidate_id: str, step: str = "dev",
                 model: str | None = None, invite: str | None = None):
        cfg = load_config()
        self.voice_cfg = {**cfg["voice"], **({"model": model} if model else {})}
        self.store = ContentStore(stages_file=A_STAGES)     
        h: dict = {}
        self.state = SessionState(role=self.store.role(role_id, h), persona=self.store.persona(persona_id, h),
                                  candidate=candidate(candidate_id), stages=self.store.stages(h),
                                  rubric=self.store.rubric(h), content_hashes=h)
        self.state.enter_stage("consent")
        self.instructions = build_instructions(self.state, self.store)
        self.ids = {"role": role_id, "persona": persona_id, "candidate": candidate_id}
        self.names = {"role": self.state.role["title"], "company": self.state.role.get("company"),
                      "persona": self.state.persona.get("title", persona_id),
                      "interviewer": self.state.persona.get("name"),
                      "candidate": self.state.candidate.get("name", candidate_id)}
        self.voice_info = voice_info(self.voice_cfg)
        self.step = step
        self.cost = 0.0
        self.cap = cfg["budget"]["session_cap_usd"].get(self.voice_cfg["model"], 1.0)
        self.monitors: dict[WebSocket, Subscriber] = {}
        self.client: WebSocket | None = None
        self.client_out: Subscriber | None = None
        self.adapter = OpenAIRealtimeAdapter()
        self.log = store.EventLog(self.state.id)
        self.recorder = Recorder(ROOT / "runs" / self.state.id / "recording.wav")
        self.done = asyncio.Event()
        self.playback_idle = asyncio.Event()
        self.recent_cards: list[str] = []
        self.pending_transcripts: set[str] = set()     # committed candidate turns not yet transcribed
        self.transcripts_ready = asyncio.Event()
        self.transcripts_ready.set()
        self.tool_lock = asyncio.Lock()                # one tool batch at a time
        self.turns: dict[str, dict] = {}               # candidate speech segments by item_id
        self.timeline: dict[str, dict] = {}
        self.mic_t0: float | None = None               # session time of the first mic chunk
        self.interrupt: dict | None = None             # the last barge-in: what the candidate heard
        self.turn_lock = asyncio.Lock()                # one controller decision at a time
        self.wait_task: asyncio.Task | None = None     # holding for the rest of an unfinished sentence
        self.closing = False                           # the closing line has been requested; end after it plays
        self.turn_notes = 0
        self.acts = 0
        self.status = "created"
        self.invite = invite
        store.add(store.ScreenSession(id=self.state.id, role_id=role_id, persona_id=persona_id,
                                      candidate_id=candidate_id, provider="openai",
                                      model=self.voice_cfg["model"], content_hashes=h,
                                      summary={"voice": self.voice_info, "names": self.names}))
        Orchestrator.registry[self.state.id] = self

    # ── broadcasting and timeline ────────────────────────────────────────
    def add_monitor(self, ws: WebSocket):
        self.monitors[ws] = Subscriber(ws)

    def remove_monitor(self, ws: WebSocket):
        sub = self.monitors.pop(ws, None)
        if sub:
            sub.close()

    async def broadcast(self, msg: dict, to_client: bool = False):
        data = json.dumps(msg, default=str)
        for ws, sub in list(self.monitors.items()):
            if not sub.alive:
                self.monitors.pop(ws, None)
            else:
                sub.put(data)
        if to_client and self.client_out is not None:
            self.client_out.put(data)

    def snapshot(self) -> dict:
        snap = self.state.snapshot()
        snap.update(cost_usd=round(self.cost, 4), cap_usd=self.cap, status=self.status, ids=self.ids,
                    names=self.names, voice=self.voice_info, model=self.voice_cfg["model"],
                    candidate_speaking=any(not t.get("ended") for t in self.turns.values()),
                    agent_speaking=self.recorder.agent.playing())
        return snap

    async def push_state(self):
        await self.broadcast({"type": "state", "state": self.snapshot()})

    async def tl(self, entry_id: str, persist: bool = True, **fields):
        """Create or update one timeline entry, stream it to monitors, and persist it."""
        e = self.timeline.setdefault(entry_id, {"id": entry_id, "stage": self.state.stage})
        e.update(fields)
        if persist:
            store.upsert_timeline(self.state.id, e)
        await self.broadcast({"type": "timeline", "entry": e})

    # ── call lifecycle ───────────────────────────────────────────────────
    async def run_call(self, ws: WebSocket):
        self.client = ws
        self.client_out = Subscriber(ws, maxsize=10000)   # the call page gets everything; it reads continuously
        tools: list[dict] = []        # the voice model only speaks; code owns every decision
        v = self.voice_cfg
        await self.adapter.connect(SessionConfig(
            model=v["model"], instructions=self.instructions, tools=tools, voice=v["voice"],
            reasoning_effort=v["reasoning_effort"], transcription_model=v["transcription_model"],
            transcription_language=v.get("transcription_language"), noise_reduction=v.get("noise_reduction"),
            turn_detection=v.get("turn_detection")))
        self.log.write("session_config", voice=self.voice_info, hashes=self.state.content_hashes,
                       instructions=self.instructions, tools=tools)
        self.status = "live"
        store.update_session(self.state.id, status="live", started_at=time.time())
        tasks = [asyncio.create_task(self._provider_loop()), asyncio.create_task(self._client_loop()),
                 asyncio.create_task(self._timer())]
        await self.adapter.send_user_text(JOIN_TEXT)
        first = opening(self.state)
        await self._speak(first["text"])
        self.state.start_clock()
        await self.tl("stage-0", kind="stage", t=0.0, stage="consent", text="consent")
        await self.push_state()
        await self.done.wait()
        for t in tasks:
            t.cancel()
        await self.finish()

    async def end(self, reason: str):
        if self.done.is_set():
            return
        if not self.state.ended:
            self.state.ended, self.state.end_reason = True, reason
        self.log.write("end", reason=reason, elapsed=self.state.elapsed)
        self.done.set()

    async def finish(self):
        await self.adapter.close()
        self.status = "ended"
        s = self.state
        rec = self.recorder.save()
        store.add(*[store.EvidenceRow(session_id=s.id, **ev.__dict__) for ev in s.ledger],
                  *[store.FlagRow(session_id=s.id, t=f["t"], type=f["type"], detail=f["detail"],
                                  source=f.get("source", "agent")) for f in s.flags])
        store.update_session(s.id, status="ended", ended_at=time.time(), end_reason=s.end_reason,
                             cost_usd=round(self.cost, 6),
                             summary={"elapsed_s": round(s.elapsed, 1), "final_stage": s.stage,
                                      "voice": self.voice_info, "names": self.names, "cap_usd": self.cap,
                                      "stage_log": s.stage_log, "overrides": s.overrides, "refusals": s.refusals,
                                      "rejections": s.rejections, "card_log": s.card_log,
                                      "consistency": s.consistency, "recording": bool(rec),
                                      "order": s.stages["order"],
                                      "stage_targets": {k: v["target_s"] for k, v in s.stages["stages"].items()}})
        for ev in self.adapter.raw_log:
            self.log.write("provider", payload=ev)
        self.log.close()
        await self.push_state()
        await self.broadcast({"type": "ended", "reason": s.end_reason}, to_client=True)
        if self.invite:
            store.update_invite(self.invite, status="completed")
        dropped = {"client": self.client_out.dropped if self.client_out else 0,
                   "monitors": sum(m.dropped for m in self.monitors.values())}
        if any(dropped.values()):
            log.warning("session %s dropped outbound messages: %s", s.id, dropped)
        if self.client_out:
            self.client_out.close()
            await asyncio.wait([self.client_out.task], timeout=2)

    # ── client (call page or harness) ────────────────────────────────────
    async def _client_loop(self):
        ws = self.client
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if msg.get("bytes"):
                    if self.mic_t0 is None:
                        self.mic_t0 = self.state.elapsed
                    self.recorder.add_mic(msg["bytes"])
                    await self.adapter.send_audio(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    if data["type"] == "playback":
                        self.log.write("playback", **data)
                        await self.adapter.report_playback(data["item_id"], int(data["played_ms"]))
                    elif data["type"] == "idle":
                        self.playback_idle.set()
                    elif data["type"] == "hangup":
                        await self.end("candidate_hangup")
        except (WebSocketDisconnect, RuntimeError):
            await self.end("client_disconnected")

    # ── provider events ──────────────────────────────────────────────────
    async def _provider_loop(self):
        async for ev in self.adapter.events():
            try:
                await self._handle(ev)
            except Exception:
                err = traceback.format_exc()
                log.error("event handler failed: %s", err)
                self.log.write("handler_error", event=ev.type, traceback=err)
        await self.end("provider_closed")

    def _speech_t(self, audio_ms) -> float:
        """Session time of a point in the input audio stream (provider audio_start_ms/audio_end_ms)."""
        if audio_ms is None or self.mic_t0 is None:
            return round(self.state.elapsed, 2)
        return round(self.mic_t0 + audio_ms / 1000, 2)

    async def _handle(self, ev: VoiceEvent):
        s = self.state
        t = round(s.elapsed, 2)
        d = ev.data
        if ev.type == "agent_audio_chunk":
            item = d["item_id"]
            lead = max(0.0, self.recorder.agent.queue_end - time.monotonic())
            self.recorder.agent.add(item, ev.audio)
            if item not in self.timeline:
                await self.tl(item, kind="agent", t=round(t + lead, 2), text="", partial=True)
            await self.broadcast({"type": "audio", "item_id": item,
                                  "pcm": base64.b64encode(ev.audio).decode()}, to_client=True)
            return
        if ev.type == "agent_transcript_delta":
            e = self.timeline.get(d["item_id"])
            text = (e or {}).get("text", "") + d["delta"]
            await self.tl(d["item_id"], persist=False, kind="agent", t=(e or {}).get("t", t), text=text,
                          partial=True)
            return
        if ev.type == "user_transcript_delta":
            turn = self.turns.get(d["item_id"])
            if turn is not None:
                turn["partial"] = turn.get("partial", "") + d["delta"]
                if WORDS.search(turn["partial"]):
                    await self.tl(d["item_id"], persist=False, kind="candidate", t=turn["t"],
                                  text=turn["partial"], partial=True)
                if has_content(turn["partial"]):
                    await self._maybe_barge(d["item_id"], reason="words")
            return
        self.log.write(ev.type, t=t, **d)
        if ev.type == "agent_audio_end":
            if d["item_id"] in self.timeline:
                await self.tl(d["item_id"], audio_ms=round(d.get("ms", 0)))
        elif ev.type == "agent_audio_start":
            self.playback_idle.clear()
            await self.broadcast({"type": "agent_speaking", "t": t}, to_client=True)
        elif ev.type == "user_speech_start":
            self._cancel_wait()
            item = d.get("item_id")
            self.turns[item] = {"t": self._speech_t(d.get("audio_start_ms")), "start_ms": d.get("audio_start_ms"),
                                "barge_pending": self.recorder.agent.playing()}
            await self.push_state()
        elif ev.type == "user_speech_end":
            item = d.get("item_id")
            turn = self.turns.setdefault(item, {"t": t})
            turn["ended"] = True
            if turn.get("start_ms") is not None and d.get("audio_end_ms") is not None:
                turn["ms"] = d["audio_end_ms"] - turn["start_ms"]
            if item:
                self.pending_transcripts.add(item)
                self.transcripts_ready.clear()
            await self.push_state()
        elif ev.type == "user_transcript":
            await self._candidate_turn(d.get("item_id"), d["text"])
        elif ev.type == "agent_transcript":
            await self._agent_turn(d["item_id"], d["text"])
        elif ev.type == "agent_truncated":
            e = self.timeline.get(d["item_id"])
            if e:
                heard = d.get("audio_end_ms") or 0
                await self.tl(d["item_id"], interrupted=True, heard_ms=heard)
        elif ev.type == "tool_calls":
            self.log.write("unexpected_tool_calls", calls=d["calls"])   # the voice model has no tools
        elif ev.type == "usage":
            self._usage(d)
            await self.push_state()
        elif ev.type == "error":
            await self.broadcast({"type": "error", "t": t, "error": d})

    # ── turn taking ────────────────────────────────────────────────
    async def _maybe_barge(self, item: str, reason: str):
        turn = self.turns.get(item)
        if not turn or not turn.get("barge_pending") or turn.get("barged"):
            return
        turn["barge_pending"] = False
        if not self.recorder.agent.playing():
            return
        turn["barged"] = True
        heard_ms = self.recorder.agent.cut()
        await self.broadcast({"type": "flush", "t": round(self.state.elapsed, 2)}, to_client=True)
        await self.adapter.cancel_response()
        self.interrupt = self._heard(heard_ms)
        if self.interrupt and self.interrupt["unheard_question"]:
            self.state.unhear_question()
        self.log.write("barge_in", item=item, reason=reason, interrupt=self.interrupt)

    def _heard(self, heard_ms: dict[str, float]) -> dict | None:
        """Which interviewer words the candidate heard before cutting in (proportional to audio played)."""
        if not heard_ms:
            return None
        item = max(heard_ms, key=lambda i: self.timeline.get(i, {}).get("t", 0))
        e = self.timeline.get(item, {})
        text = e.get("text", "")
        total = self.recorder_total_ms(item)
        cut = int(len(text) * min(1.0, heard_ms[item] / total)) if total else len(text)
        heard = text[:cut].rsplit(" ", 1)[0] if cut < len(text) else text
        unheard = text[len(heard):]
        return {"item": item, "heard": heard, "unheard_question": "?" in unheard and "?" not in heard}

    def recorder_total_ms(self, item: str) -> float:
        e = self.timeline.get(item, {})
        return e.get("audio_ms") or self.adapter_audio_ms(item)

    def adapter_audio_ms(self, item: str) -> float:
        return float(getattr(self.adapter, "_audio_ms", {}).get(item, 0) or 0)

    async def _candidate_turn(self, item: str | None, text: str):
        s = self.state
        self.pending_transcripts.discard(item)
        if not self.pending_transcripts:
            self.transcripts_ready.set()
        turn = self.turns.setdefault(item, {"t": round(s.elapsed, 2)})
        turn["transcribed"] = True
        if has_content(text):
            await self._maybe_barge(item, reason="words")      # the transcript usually completes after speech ends
        if not is_answerable(text):
            # Coughs, breaths and fillers: kept in the record for audit, never answered.
            await self.tl(item, kind="noise", t=turn["t"], text=text.strip(), partial=False)
            turn["responded"] = True
            if any(t.get("deferred") for t in self.turns.values()) and not self._candidate_speaking(exclude=item):
                await self._maybe_decide()
            await self.push_state()
            return
        s.candidate_turns.append((turn["t"], s.stage, text))
        card = s.current_card()
        await self.tl(item, kind="candidate", t=turn["t"], text=text, partial=False,
                      card=card["id"] if card else None)
        if not turn.get("responded"):
            turn["deferred"] = True
            if not self._candidate_speaking(exclude=item):
                await self._maybe_decide()
        await self.push_state()

    def _candidate_speaking(self, exclude: str | None = None) -> bool:
        return any(not t.get("ended") for k, t in self.turns.items() if k != exclude and "start_ms" in t)

    def _pending(self) -> list[str]:
        """Transcribed candidate segments not yet answered, oldest first."""
        items = [k for k, t in self.turns.items() if t.get("deferred") and not t.get("responded")]
        return sorted(items, key=lambda k: self.turns[k]["t"])

    def _cancel_wait(self):
        if self.wait_task and not self.wait_task.done():
            self.wait_task.cancel()
        self.wait_task = None

    async def _maybe_decide(self):
        """End of turn: if the sentence sounds unfinished, hold briefly for the rest (a new segment cancels the
        hold and is merged); otherwise decide now."""
        pending = self._pending()
        if not pending:
            return
        utterance = " ".join(self.timeline.get(k, {}).get("text", "") for k in pending)
        self._cancel_wait()
        if sounds_unfinished(utterance):
            self.wait_task = asyncio.create_task(self._decide_after(HOLD_S))
        else:
            await self._decide_turn()

    async def _decide_after(self, seconds: float):
        await asyncio.sleep(seconds)
        if not self._candidate_speaking():
            await self._decide_turn()

    async def _decide_turn(self):
        """One decision per complete candidate turn, then one directed interviewer turn."""
        async with self.turn_lock:
            s = self.state
            pending = self._pending()
            if not pending or s.ended:
                return
            for k in pending:
                self.turns[k]["responded"], self.turns[k]["deferred"] = True, False
            utterance = " ".join(self.timeline.get(k, {}).get("text", "") for k in pending).strip()
            interrupt, self.interrupt = self.interrupt, None
            decision = await decide(s, utterance, self._recent())
            await self.tl(pending[-1], intent=decision["kind"])
            n_stages = len(s.stage_log)
            stage_before = s.stage
            d = plan(s, decision, utterance, interrupt)
            t = round(s.elapsed, 2)
            for a in d["actions"]:
                self.acts += 1
                await self.tl(f"act-{self.acts}", kind="tool", t=t, stage=stage_before, name=a["name"],
                              arguments=a["arguments"], output=a["output"], text="")
                self.log.write("tool", t=t, name=a["name"], arguments=json.dumps(a["arguments"]), output=a["output"])
            for i, mark in enumerate(s.stage_log[n_stages:], start=n_stages):
                await self.tl(f"stage-{i}", kind="stage", t=t, stage=mark["stage"], text=mark["stage"])
            text = d["text"]
            if s.corrections and not d["end"]:
                text += " Also: " + " ".join(s.corrections)
                s.corrections.clear()
            self.acts += 1
            await self.tl(f"act-{self.acts}", kind="note", t=t, text=text, intent=decision["kind"])
            self.log.write("decision", utterance=utterance, decision=decision, directive=text, end=d["end"])
            if d["end"]:
                self.closing = True
            await self._speak(text)
            await self.push_state()

    async def _speak(self, directive: str):
        """A labelled system note plus a plain response.create. Replacing the instructions per response
        breaks the prompt cache from the first token."""
        self.turn_notes += 1
        await self.adapter.send_system_note(turn_note(directive, self.turn_notes))
        await self.adapter.respond()

    def _recent(self) -> list[tuple[str, str]]:
        rows = [e for e in sorted(self.timeline.values(), key=lambda e: e["t"]) if e["kind"] in ("agent", "candidate")]
        return [("Interviewer" if e["kind"] == "agent" else "Candidate", e.get("text", "")) for e in rows[-8:]]

    async def _agent_turn(self, item: str, text: str):
        s = self.state
        e = self.timeline.get(item, {})
        t = e.get("t", round(s.elapsed, 2))
        s.agent_turns.append((t, s.stage, text))
        card = s.current_card()
        await self.tl(item, kind="agent", t=t, text=text, partial=False, card=card["id"] if card else None)
        texts = list(self.recent_cards)
        if card:
            texts += [card["primary"], *card.get("probes", {}).values()]
        last = s.last_instruction or {}
        texts += [v for k, v in last.items() if k in ("question", "say") and isinstance(v, str)]
        texts += [c.get("suggested_clarification", "") for c in s.consistency]
        for issue in check_turn(text, s.stage, texts):
            s.flags.append({"t": t, "type": issue["type"], "detail": issue["detail"], "source": "monitor"})
            s.corrections.append(issue["correction"])
        if self.closing:
            self.closing = False
            asyncio.create_task(self._end_after_playback())

    async def _end_after_playback(self):
        """Let the closing line finish playing before hanging up."""
        remaining = max(0.0, self.recorder.agent.queue_end - time.monotonic())
        try:
            await asyncio.wait_for(self.playback_idle.wait(), timeout=max(1.0, remaining + 1.5))
        except asyncio.TimeoutError:
            pass
        await self.end(self.state.end_reason or "end_interview")

    def _usage(self, data: dict):
        model = self.voice_cfg["model"]
        if data["kind"] == "response":
            usd = costs.openai_response_usd(model, data["usage"])
            costs.log("openai", model, usd, data["usage"], self.state.id, self.step, "response")
        else:
            tm = self.voice_cfg["transcription_model"]
            usd = costs.transcription_usd(tm, data["seconds"])
            costs.log("openai", tm, usd, {"seconds": data["seconds"]}, self.state.id, self.step, "transcription")
        self.cost += usd
        if self.cost >= self.cap and not self.state.force_close:
            self.state.force_close = "cost_cap"
            self.log.write("cost_cap", cost=self.cost)

    async def _timer(self):
        hard = self.state.stages["time"]["hard_stop_s"]
        while not self.done.is_set():
            await asyncio.sleep(1)
            ce = self.state.close_entered_at
            if self.state.elapsed >= hard:
                await self.end("hard_stop")
            elif ce is not None and self.state.elapsed - ce > CLOSE_TIMEOUT_S and self.state.ended:
                await self.end(self.state.end_reason or "close_timeout")
            elif self.cost >= self.cap * 1.25:
                await self.end("cost_cap_hard")
            elif int(self.state.elapsed) % 5 == 0:
                await self.push_state()


def voice_info(v: dict) -> dict:
    """What the session page shows under "Models": the tunable interviewer and the fixed transcriber."""
    return {
        "interviewer": {"model": v["model"], "reasoning_effort": v["reasoning_effort"], "voice": v["voice"],
                        "turn_detection": v.get("turn_detection"), "noise_reduction": v.get("noise_reduction"),
                        "barge_in": "only a transcript with a non-filler word interrupts"},
        "transcriber": {"model": v["transcription_model"], "language": v.get("transcription_language")},
    }
