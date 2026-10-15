"""Session orchestrator for one screening call.

Bridges the candidate page, gpt-live-1 and the LangGraph brain. It owns: turn segmentation from the transcript
stream, end-of-turn detection, interruption accounting, the revision/commit rule for brain results, the
timeline, recording, caps and logging. It reuses the shared core's session state, store, recorder and monitor
(`backend/`).
"""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import re
import time
import traceback
import uuid
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from alt import budget
from alt.brain import llm
from alt.brain.graph import Brain, run_turn, speech, _say
from alt.live_adapter import GptLiveAdapter, LiveEvent
from backend import costs, store
from alt.pipeline import inputs as I
from alt.pipeline import screen as S
from backend.content_store import ContentStore, candidate, sha
from backend.context_builder import candidate_context, role_summary
from backend.controller import opening, sounds_unfinished
from backend.monitor import check_turn
from backend.orchestrator import Subscriber, has_content, is_answerable
from backend.recording import Recorder
from backend.session import SessionState, render
from backend.tools.evidence import _share, _stems

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("uvicorn.error")

EOT_S = 0.6            # transcript silence that starts the brain (speculative: if the candidate goes on, the
                       # result is dropped and the brain re-runs on the merged turn; transcripts trail audio ~0.4 s)
MERGE_MS = 2000        # a candidate segment starting this soon after an unanswered one continues it
HOLD_LONG_S = 6.0      # after the brain judged a fragment "incomplete": silence before it must decide anyway
                       # (shorter holds moved a slow, thinking candidate on mid-thought)
HOLD_ASKED_S = 12.0    # the candidate asked for time ("wait a sec", "let me think"): wait this long
TIME_REQUEST = re.compile(r"\b(wait|hold on|one (sec|second|moment)|a (sec|second|moment|minute)|let me (think|see|"
                          r"recall|remember|check)|give me a (sec|second|moment|minute))\b", re.I)
QUIET_WAIT_S = 6.0     # an exact line waits up to this long for the model to finish what it is saying (2.5 cut a
                       # warning off with the closing line in c49cf97735)
HOLD_S = 1.8           # extra wait when the sentence sounds unfinished ("I sell air product to")
AGENT_GAP_MS = 1200    # output-transcript gap that starts a new interviewer segment
AGENT_DONE_S = 1.4     # no output transcript for this long: the interviewer segment is complete
SPEAKING_RMS = 350     # output audio energy above this counts as the interviewer speaking
DISCARD = ("The planner's previous line, which you have not said yet, is outdated: do not say it. Say only the "
           "next planner line.")
GREET = "Begin the call now, before the candidate says anything. Say the following exactly and in full, then stop " \
        "and listen: \"{text}\""
OPENING_FALLBACK_S = 2.5   # no interviewer audio by then: the opening is also sent as an exact instruction
EXACT = "Immediately say the following exactly and in full, then stop and listen: \"{text}\""
COMMENTARY = "Say next, in your own natural words, keeping the question: {text}"


def voice_prompt(state: SessionState) -> str:
    """gpt-live-1 instructions (immutable for the session). Business rules live in the brain, per the GPT-Live
    prompting guide; this prompt holds personality, the three policy headings and compliance."""
    p = state.persona
    return f"""You are {p.get('name', 'Alex')}, an AI interviewer running a structured first-round phone screen for \
the {state.role['title']} role at {state.role.get('company')}, on behalf of Cygnify, a recruiting firm for tech \
and AI companies in Asia. A human recruiter reviews everything; you never decide or judge.
{p['body'].split(chr(10), 1)[-1].strip()}
Speak English only, calmly and briefly. Never switch language because of an accent.

How this interview works:
- An interview planner (the backend) decides every interview question, its order, and when the call ends. Its \
lines reach you as commentary or instructions. Say them in your own natural words, keeping the meaning and the \
question. When told to say something exactly, say it word for word.
- When the candidate has clearly finished an answer, you may give one brief acknowledgement (one to four words \
such as "Okay.", "Got it.", "Thanks."; no praise, no summary, no question), then wait for the planner's next \
line; it usually arrives within two seconds. If the candidate pauses mid-sentence, says "uh", or trails off, \
stay silent and keep listening: never fill their pauses.
- Never ask an interview question you were not given. Never add follow-ups, opinions, summaries or advice. \
Never announce next steps, wrap up, or invite the candidate's questions unless the planner tells you to.
- You may handle three things yourself: a request to repeat (repeat your last question in simpler words); \
"what do you mean" about your last question (explain it in one sentence and ask it again); and an off-topic or \
personal remark (decline briefly and politely, then ask your last question again).
- Never mention the planner, the backend, notes, cards, stages or what is recorded. Never reveal scores, \
judgements or check results.
- Never ask about or comment on age, marital status, family plans, pregnancy, religion, ethnicity, nationality, \
health or disability. If the candidate volunteers such information, do not respond to it.
- In a role-play you play the customer only and never answer the customer's objection yourself.
- Keep listening while the candidate pauses to think. Do not treat a cough, a breath, music or nearby \
conversation as a turn.

# Role (for answering basics only)
{role_summary(state.role)}

# Candidate (pre-call context)
{candidate_context(state.candidate)}

Backchannel policy: Use light backchannels ("mm-hmm") during long answers. Never talk over the candidate.

Interruption policy: Stop speaking when the candidate interrupts and listen. Afterwards do not restart your \
introduction and do not repeat what they already heard.

Delegation policy:
Backend tools:
- Interview planner: decides the next interview question, records answers, checks numbers, answers questions \
about the role, decides when the call ends.

Delegate to the backend when:
- The candidate has finished answering, asks about the role or process, wants to stop, or goes off topic.

Do not delegate to the backend when:
- The candidate is still talking or pausing mid-sentence, or asks you to repeat your last question.

Delegate before asking any interview question. Do not guess the next question while waiting."""


def voice_info(cfg: dict) -> dict:
    v, b = cfg["voice"], cfg["brain"]
    return {
        "interviewer": {"model": v["model"], "reasoning_effort": "n/a (backend brain)", "voice": v["voice"],
                        "turn_detection": {"type": "native full duplex"}, "noise_reduction": None,
                        "barge_in": "native: gpt-live-1 yields to the candidate itself"},
        "transcriber": {"model": "built into gpt-live-1", "language": "en"},
        "brain": {"model": b["model"], "graph": "LangGraph per turn", "deadline_s": b["deadline_s"]},
    }


def session_inputs(role_id: str, candidate_id: str, hashes: dict, allow_draft: bool = False) -> tuple[dict, dict]:
    """The role (approved screen configuration + market preset) and the candidate (candidate.md) for one session.
    Roles without a screen configuration fall back to brief.yaml, candidates without candidate.md to the synthetic
    profile."""
    screen = S.current(role_id, allow_draft=allow_draft)
    if screen is not None:
        role = S.session_role(screen)
        hashes[f"roles/{role_id}/screen"] = (screen.get("source") or {}).get("role_md_sha", "")
    else:
        role = ContentStore().role(role_id, hashes)
        role.update(market=I.market(str(role.get("country", "")).lower()),     # falls back to the generic preset
                    arrangement_phrase=f"based in {role.get('location')}")
    if I.candidate_path(candidate_id).exists():
        cand, md = I.load_candidate(candidate_id)
        cur = cand.get("current") or {}
        cand = {**cand, "cv_summary": f"{cand.get('headline', '')} Current: {cur.get('title', '')} at "
                                      f"{cur.get('company', '')}."}
        hashes[f"candidates/{candidate_id}/candidate.md"] = sha(md)
    else:
        cand = candidate(candidate_id)
    return role, cand


def build_stages(stages: dict, role: dict, plan: dict | None, seed: int, permute: bool) -> dict:
    return S.session_stages(stages, role, plan, seed=seed, permute=permute)


class AltOrchestrator:
    registry: dict[str, "AltOrchestrator"] = {}

    def __init__(self, role_id: str, persona_id: str, candidate_id: str, invite: str | None = None,
                 step: str = "alt-dev", plan: dict | None = None, seed: int | None = None,
                 allow_draft_screen: bool = False):
        self.cfg = budget.cfg()
        self.store = ContentStore()
        h: dict = {}
        role, cand = session_inputs(role_id, candidate_id, h, allow_draft_screen)
        self.seed = seed if seed is not None else uuid.uuid4().int % 100000
        b = self.cfg["brain"]
        stages = build_stages(self.store.stages(h), role, plan, self.seed, bool(b.get("permute_middle_stages")))
        self.state = SessionState(role=role, persona=self.store.persona(persona_id, h), candidate=cand,
                                  stages=stages, rubric=self.store.rubric(h), content_hashes=h,
                                  id="c" + uuid.uuid4().hex[:9])
        self.plan = plan or {}
        if b.get("shuffle_within_stage"):
            self.state.shuffle_seed = self.seed
        self.state.enter_stage("consent")
        self.instructions = voice_prompt(self.state)
        self.brain_extra = ("# Role brief (authoritative; the only source for answers to the candidate)\n"
                            + role_summary(self.state.role))
        self.ids = {"role": role_id, "persona": persona_id, "candidate": candidate_id}
        self.names = {"role": self.state.role["title"], "company": self.state.role.get("company"),
                      "persona": self.state.persona.get("title", persona_id),
                      "interviewer": self.state.persona.get("name"),
                      "candidate": self.state.candidate.get("name", candidate_id)}
        self.voice_info = voice_info(self.cfg)
        self.step = step
        self.cap = self.cfg["budget"]["session_cap_usd"]
        self.voice_cost = 0.0
        self.brain_cost = 0.0
        self.logged_seconds = 0.0
        self.monitors: dict[WebSocket, Subscriber] = {}
        self.client: WebSocket | None = None
        self.client_out: Subscriber | None = None
        self.adapter = GptLiveAdapter()
        self.log = store.EventLog(self.state.id)
        self.recorder = Recorder(ROOT / "runs" / self.state.id / "recording.wav")
        self.done = asyncio.Event()
        self.timeline: dict[str, dict] = {}
        self.status = "created"
        self.invite = invite
        self.live = False
        # transcript segmentation
        self.agent_seg: dict | None = None        # the interviewer segment being spoken
        self.agent_segs: list[dict] = []          # recent interviewer segments (for latency attribution)
        self.cand_segs: dict[str, dict] = {}
        self.agent_n = 0
        self.cand_seg: dict | None = None         # the candidate segment being spoken
        self.cand_n = 0
        self.pending: list[str] = []              # candidate segment ids not yet answered by the brain
        self.last_in = 0.0                        # monotonic time of the last input transcript delta
        self.last_in_ms = 0
        self.eot_task: asyncio.Task | None = None
        self.agent_done_task: asyncio.Task | None = None
        # brain
        self.rev = 0                              # bumped by every input delta; a result from an older rev is stale
        self.brain_lock = asyncio.Lock()
        self.rerun = False
        self.delegation: str | None = None        # the latest unanswered delegation id
        self.turn_n = 0
        self.turn_metrics: list[dict] = []
        self.awaiting: dict | None = None         # the last committed turn, waiting for its spoken line
        self.last_question: str = ""
        self.interrupt: dict | None = None
        self.closing: str | None = None
        self.opening_heard = False                # candidate speech before the opening question is not a turn
        self.allow_wait = True                    # the brain may hold a fragment ("incomplete") once
        self.force_task: asyncio.Task | None = None
        self.prev: dict | None = None             # the state before the last commit (provisional until heard)
        self.closing_line: str = ""
        self.speaking = False
        self.last_loud = 0.0
        store.add(store.ScreenSession(id=self.state.id, role_id=role_id, persona_id=persona_id,
                                      candidate_id=candidate_id, provider="openai", model=self.cfg["voice"]["model"],
                                      content_hashes=h, summary={"voice": self.voice_info, "names": self.names}))
        AltOrchestrator.registry[self.state.id] = self

    # ── broadcasting and timeline (as in the shared core) ──────────────────────
    def add_monitor(self, ws: WebSocket):
        self.monitors[ws] = Subscriber(ws)

    def remove_monitor(self, ws: WebSocket):
        sub = self.monitors.pop(ws, None)
        if sub:
            sub.close()

    def set_mirror(self, ws: WebSocket, on: bool):
        if (sub := self.monitors.get(ws)) is not None:
            sub.audio = on

    async def broadcast(self, msg: dict, to_client: bool = False):
        data = json.dumps(msg, default=str)
        audio = msg.get("type") == "audio"
        for ws, sub in list(self.monitors.items()):
            if not sub.alive:
                self.monitors.pop(ws, None)
            elif not audio or getattr(sub, "audio", False):
                # gpt-live-1 streams audio continuously; sent to every monitor it filled their queues, and state
                # updates could be dropped (the stage bar stopped following). Audio only on request.
                sub.put(data)
        if to_client and self.client_out is not None:
            self.client_out.put(data)

    @property
    def cost(self) -> float:
        return self.voice_cost + self.brain_cost

    def snapshot(self) -> dict:
        snap = self.state.snapshot()
        snap.update(cost_usd=round(self.cost, 4), cap_usd=self.cap, status=self.status, ids=self.ids,
                    names=self.names, voice=self.voice_info, model=self.cfg["voice"]["model"],
                    candidate_speaking=self.cand_seg is not None, agent_speaking=self.speaking,
                    turns=self.turn_metrics[-30:], cost_split={"voice": round(self.voice_cost, 4),
                                                               "brain": round(self.brain_cost, 4)},
                    layout=self.layout())
        return snap

    def layout(self) -> dict:
        """What this session asks and why: the seed, the stage order, each card's variant, which
        cards the candidate's plan personalised and which are extra probes or role questions."""
        st, ctx = self.state.stages, self.state.ctx
        role = self.state.role
        return {"seed": self.seed, "order": st["order"], "market": (role.get("market") or {}).get("name"),
                "role_id": role.get("role_id"), "plan_summary": self.plan.get("summary", ""),
                "cards": {stage: [{"id": c["id"], "variant": c.get("variant"),
                                   "personalised": bool(c.get("personalised")), "extra": bool(c.get("extra")),
                                   "role_question": c.get("role_question") is not None,
                                   "text": render(c.get("primary", ""), ctx),
                                   "standard": render(c["standard"], ctx) if c.get("standard") else None,
                                   "verifies": c.get("verifies") or [],
                                   "requirements": c.get("requirement_ids") or
                                   ([c["requirement_id"]] if c.get("requirement_id") else [])}
                                  for c in cfg["cards"] if not c.get("action")]
                          for stage, cfg in st["stages"].items()}}

    async def push_state(self):
        await self.broadcast({"type": "state", "state": self.snapshot()})

    async def tl(self, entry_id: str, persist: bool = True, **fields):
        e = self.timeline.setdefault(entry_id, {"id": entry_id, "stage": self.state.stage})
        e.update(fields)
        if persist:
            store.upsert_timeline(self.state.id, e)
        await self.broadcast({"type": "timeline", "entry": e})

    # ── call lifecycle ───────────────────────────────────────────────────────
    async def run_call(self, ws: WebSocket):
        self.client = ws
        self.client_out = Subscriber(ws, maxsize=10000)
        budget.require(self.cap, "gpt-live-1 session")
        v = self.cfg["voice"]
        self.opening_line = _say(self.state, speech()["cards"]["consent_ack"]["say"])
        greet = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": GREET.format(
            text=self.opening_line)}]}]
        info = await self.adapter.connect(model=v["model"], instructions=self.instructions, voice=v["voice"],
                                          seed=greet)
        self.log.write("session_config", voice=self.voice_info, live=info, hashes=self.state.content_hashes,
                       instructions=self.instructions, brain_system_extra=self.brain_extra)
        self.status = "live"
        self.live = True
        self.state.start_clock()
        store.update_session(self.state.id, status="live", started_at=time.time())
        tasks = [asyncio.create_task(self._provider_loop()), asyncio.create_task(self._client_loop()),
                 asyncio.create_task(self._timer())]
        asyncio.create_task(llm.warm(self.brain_extra, model=self.cfg["brain"]["model"], run_id=self.state.id,
                                     step=self.step))
        await self._open()
        await self.push_state()
        await self.done.wait()
        for t in tasks:
            t.cancel()
        await self.finish()

    async def _open(self):
        """The interviewer speaks first. The greeting goes in `session.input` at startup, the documented way to have
        gpt-live-1 open the conversation (live calls started speaking 2–8 s late, often only after the candidate made
        a sound, when the line was appended after `session.started`). If no interviewer audio follows within
        OPENING_FALLBACK_S, the line is also sent as an exact instruction, which reliably starts the opening."""
        first = opening(self.state)
        line = self.opening_line
        self.last_question = line
        await self.tl("stage-0", kind="stage", t=0.0, stage="consent", text="consent")
        end = time.monotonic() + OPENING_FALLBACK_S
        while time.monotonic() < end and not self.last_loud:
            await asyncio.sleep(0.05)
        fallback = not self.last_loud
        if fallback:
            await self._send_lines([{"channel": "exact", "text": line}], first_turn=True)
        self.log.write("opening", directive=first["text"], line=line, via="instruction" if fallback else "startup input",
                       first_sound_s=round(self.state.elapsed, 2) if not fallback else None)

    async def end(self, reason: str):
        if self.done.is_set():
            return
        if not self.state.ended:
            self.state.ended, self.state.end_reason = True, reason
        self.log.write("end", reason=reason, elapsed=self.state.elapsed)
        self.done.set()

    async def finish(self):
        seconds = await self.adapter.close()
        self._log_voice(seconds)
        self.status = "ended"
        s = self.state
        rec = self.recorder.save()
        store.add(*[store.EvidenceRow(session_id=s.id, **ev.__dict__) for ev in s.ledger],
                  *[store.FlagRow(session_id=s.id, t=f["t"], type=f["type"], detail=f["detail"],
                                  source=f.get("source", "agent")) for f in s.flags])
        lat = [m["first_sound_s"] for m in self.turn_metrics if m.get("first_sound_s") is not None]
        nxt = [m["next_line_s"] for m in self.turn_metrics if m.get("next_line_s") is not None]
        store.update_session(s.id, status="ended", ended_at=time.time(), end_reason=s.end_reason,
                             cost_usd=round(self.cost, 6),
                             summary={"elapsed_s": round(s.elapsed, 1), "final_stage": s.stage,
                                      "voice": self.voice_info, "names": self.names, "cap_usd": self.cap,
                                      "stage_log": s.stage_log, "overrides": s.overrides, "refusals": s.refusals,
                                      "rejections": s.rejections, "card_log": s.card_log,
                                      "consistency": s.consistency, "recording": bool(rec),
                                      "order": s.stages["order"], "layout": self.layout(),
                                      "stage_targets": {k: v["target_s"] for k, v in s.stages["stages"].items()},
                                      "turns": self.turn_metrics, "voice_seconds": seconds,
                                      "cost_split": {"voice": round(self.voice_cost, 4),
                                                     "brain": round(self.brain_cost, 4)},
                                      "latency": {"first_sound_median_s": _median(lat),
                                                  "next_line_median_s": _median(nxt)}})
        for ev in self.adapter.raw_log:
            self.log.write("provider", payload=ev)
        self.log.close()
        await self.push_state()
        await self.broadcast({"type": "ended", "reason": s.end_reason}, to_client=True)
        if self.invite:
            store.update_invite(self.invite, status="completed")
        from alt.pipeline.graph import Pipelines      # resume the candidate pipeline: it writes the brief
        asyncio.create_task(Pipelines.call_ended(self.invite, s.id))
        if self.client_out:
            self.client_out.close()
            await asyncio.wait([self.client_out.task], timeout=2)

    def _log_voice(self, seconds: float):
        """Voice spend goes to the shared ledger in increments, so a crash still leaves a record."""
        delta = max(0.0, seconds - self.logged_seconds)
        if delta <= 0:
            return
        usd = delta / 60 * self.cfg["voice"]["usd_per_minute"]
        costs.log("openai", self.cfg["voice"]["model"], usd, {"seconds": round(delta, 2)}, self.state.id, self.step,
                  "session")
        self.logged_seconds = seconds

    # ── client (call page or harness) ────────────────────────────────────────
    async def _client_loop(self):
        ws = self.client
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if msg.get("bytes"):
                    if self.live:
                        self.recorder.add_mic(msg["bytes"])
                        await self.adapter.send_audio(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    if data["type"] == "hangup":
                        await self.end("candidate_hangup")
        except (WebSocketDisconnect, RuntimeError):
            await self.end("client_disconnected")

    # ── provider events ──────────────────────────────────────────────────────
    async def _provider_loop(self):
        async for ev in self.adapter.events():
            try:
                await self._handle(ev)
            except Exception:
                err = traceback.format_exc()
                log.error("alt event handler failed: %s", err)
                self.log.write("handler_error", event=ev.type, traceback=err)
        await self.end("provider_closed")

    async def _handle(self, ev: LiveEvent):
        d = ev.data
        if ev.type == "audio":
            self.recorder.agent.add("live", ev.audio)
            await self.broadcast({"type": "audio", "item_id": "live", "pcm": base64.b64encode(ev.audio).decode()},
                                 to_client=True)
            now = time.monotonic()
            if d["rms"] > SPEAKING_RMS:
                self.last_loud = now
            speaking = now - self.last_loud < 0.35
            if speaking != self.speaking:
                self.speaking = speaking
                await self.broadcast({"type": "speaking", "on": speaking}, to_client=True)
                if speaking:
                    await self.broadcast({"type": "agent_speaking", "t": round(self.state.elapsed, 2)},
                                         to_client=True)
            return
        if ev.type == "out_delta":
            await self._out_delta(d)
            return
        if ev.type == "in_delta":
            await self._in_delta(d)
            return
        if ev.type == "usage":
            self.voice_cost = d["seconds"] / 60 * self.cfg["voice"]["usd_per_minute"]
            if d["seconds"] - self.logged_seconds >= 30:
                self._log_voice(d["seconds"])
            if self.cost >= self.cap and not self.state.force_close:
                self.state.force_close = "cost_cap"
                self.log.write("cost_cap", cost=self.cost)
            return
        self.log.write(ev.type, t=round(self.state.elapsed, 2), **d)
        if ev.type == "delegation":
            self.delegation = d["id"]
            if self.pending:
                self._schedule_eot(0.2)      # the model thinks the turn is over: end it sooner
        elif ev.type == "appended" and self.awaiting and d.get("id") == self.awaiting.get("say_id"):
            self.awaiting["appended_ms"] = d.get("start_ms")
        elif ev.type == "error":
            await self.broadcast({"type": "error", "t": round(self.state.elapsed, 2), "error": d})
        elif ev.type == "closed":
            await self.end(self.state.end_reason or f"session closed: {d.get('reason')}")

    # ── interviewer transcript ───────────────────────────────────────────────
    async def _out_delta(self, d: dict):
        start, end = d.get("start_ms") or 0, d.get("end_ms") or 0
        seg = self.agent_seg
        if seg is None or start - seg["end_ms"] > AGENT_GAP_MS:
            if seg is not None:
                await self._agent_final()
            self.agent_n += 1
            seg = self.agent_seg = {"id": f"agent-{self.agent_n}", "start_ms": start, "end_ms": end, "text": ""}
            self.agent_segs = (self.agent_segs + [seg])[-12:]
            a = self.awaiting
            if a and a.get("sent") and not a.get("reported"):
                # The planner's line is spoken once it has reached the model's context (ack start_ms); a segment
                # that starts before that is the model's own bridge.
                directed_from = a.get("appended_ms") or (a.get("sent_ms", 0) + 300)
                if start >= directed_from:
                    seg["role"] = "directed"
                    a["started"] = True
                    a["next_line_ms"] = start
                    a.setdefault("first_sound_ms", start)
                    await self._report_turn(a)
                elif "first_sound_ms" not in a:
                    seg["role"] = "bridge"
                    a["first_sound_ms"] = start
        seg["end_ms"] = end
        seg["text"] += d.get("text", "")
        await self.tl(seg["id"], persist=False, kind="agent", t=round(seg["start_ms"] / 1000, 2),
                      text=seg["text"].strip(), partial=True, role=seg.get("role"))
        if self.agent_done_task:
            self.agent_done_task.cancel()
        self.agent_done_task = asyncio.create_task(self._agent_done_after(AGENT_DONE_S))

    async def _agent_done_after(self, s: float):
        await asyncio.sleep(s)
        await self._agent_final()

    async def _agent_final(self):
        seg, self.agent_seg = self.agent_seg, None
        if seg is None:
            return
        st = self.state
        text = seg["text"].strip()
        t = round(seg["start_ms"] / 1000, 2)
        st.agent_turns.append((t, st.stage, text))
        self.opening_heard = True
        self.log.write("agent_seg", id=seg["id"], start_ms=seg["start_ms"], end_ms=seg["end_ms"], role=seg.get("role"),
                       text=text)
        card = st.current_card()
        await self.tl(seg["id"], kind="agent", t=t, text=text, partial=False, card=card["id"] if card else None,
                      audio_ms=seg["end_ms"] - seg["start_ms"], role=seg.get("role"))
        texts = [self.last_question]
        if card:
            texts += [card["primary"], *card.get("probes", {}).values()]
        texts += [c.get("suggested_clarification", "") for c in st.consistency]
        for issue in check_turn(text, st.stage, texts):
            st.flags.append({"t": t, "type": issue["type"], "detail": issue["detail"], "source": "monitor"})
        a = self.awaiting
        if a and a.get("sent") and not a.get("started") and _overlap(self.last_question, text) >= 0.5:
            a["started"] = True                      # the model said the planner's line inside its own segment
        if self.closing and ("bye" in text.lower() or _overlap(self.closing_line, text) >= 0.5):
            # Hang up only after the closing line itself was spoken (otherwise a short
            # "I hear you." can end the call and cut the closing line).
            asyncio.create_task(self._end_soon())
        await self.push_state()

    async def _end_soon(self):
        await asyncio.sleep(1.5)
        await self.end(self.closing or "completed")

    # ── candidate transcript ─────────────────────────────────────────────────
    async def _in_delta(self, d: dict):
        text = d.get("text", "")
        start, end = d.get("start_ms") or 0, d.get("end_ms") or 0
        self.last_in = time.monotonic()
        self.last_in_ms = end
        self.rev += 1
        self.allow_wait = True
        if self.force_task and not self.force_task.done():
            self.force_task.cancel()
        seg = self.cand_seg
        if seg is None:
            prev = self.cand_segs.get(self.pending[-1]) if self.pending else None
            if prev is not None and start - prev["end_ms"] < MERGE_MS:
                seg = self.cand_seg = prev          # a pause inside one answer: continue the same turn
                seg["text"] += " "
            else:
                self.cand_n += 1
                seg = self.cand_seg = {"id": f"cand-{self.cand_n}", "start_ms": start, "end_ms": end, "text": ""}
                self.cand_segs[seg["id"]] = seg
                self.pending.append(seg["id"])
        seg["end_ms"] = end
        seg["text"] += text
        if has_content(seg["text"]) and self.agent_seg is not None and not seg.get("barge_checked"):
            seg["barge_checked"] = True
            a = self.agent_seg
            if (start - a["end_ms"] < 600 and a["start_ms"] < seg["start_ms"]
                    and len(a["text"].split()) >= 3):      # the interviewer was talking (not a backchannel)
                heard = self.agent_seg["text"].strip()
                self.interrupt = {"heard": heard, "unheard_question": "?" not in heard and "?" in self.last_question,
                                  "agent": self.agent_seg["id"]}
                await self.tl(self.agent_seg["id"], persist=False, interrupted=True,
                              heard_ms=start - self.agent_seg["start_ms"])
                self.log.write("barge_in", seg=seg["id"], interrupt=self.interrupt)
        await self.tl(seg["id"], persist=False, kind="candidate", t=round(seg["start_ms"] / 1000, 2),
                      text=seg["text"].strip(), partial=True)
        self._schedule_eot(EOT_S)

    def _schedule_eot(self, delay: float):
        if self.eot_task and not self.eot_task.done():
            self.eot_task.cancel()
        self.eot_task = asyncio.create_task(self._eot_after(delay))

    async def _eot_after(self, delay: float):
        await asyncio.sleep(delay)
        seg = self.cand_seg
        if seg is not None:
            self.cand_seg = None
            text = seg["text"].strip()
            self.log.write("cand_seg", id=seg["id"], start_ms=seg["start_ms"], end_ms=seg["end_ms"], text=text)
            if not is_answerable(text):
                # coughs, breaths and fillers stay in the record, never answered
                self.pending.remove(seg["id"])
                await self.tl(seg["id"], kind="noise", t=round(seg["start_ms"] / 1000, 2), text=text, partial=False)
                return
            await self.tl(seg["id"], kind="candidate", t=round(seg["start_ms"] / 1000, 2), text=text, partial=False,
                          card=(self.state.current_card() or {}).get("id"))
        utterance = self._utterance()
        if not utterance:
            return
        if sounds_unfinished(utterance) and delay > 0 and time.monotonic() - self.last_in < EOT_S + HOLD_S:
            self.eot_task = asyncio.create_task(self._eot_after_hold())
            return
        await self._think()

    async def _eot_after_hold(self):
        await asyncio.sleep(HOLD_S)
        if self.cand_seg is None:
            await self._think()

    def _utterance(self) -> str:
        return " ".join(self.timeline.get(k, {}).get("text", "") for k in self.pending).strip()

    # ── the brain ────────────────────────────────────────────────────────────
    async def _think(self):
        if self.brain_lock.locked():
            self.rerun = True
            return
        async with self.brain_lock:
            while True:
                self.rerun = False
                done = await self._think_once()
                if done or not self.rerun or self.state.ended:
                    return

    async def _think_once(self) -> bool:
        """One graph run on a copy of the session; commit only if no newer candidate speech arrived."""
        pending = list(self.pending)
        utterance = self._utterance()
        if not pending or not utterance or self.state.ended:
            return True
        if not self.opening_heard:
            # Spoken before the opening question was heard (e.g. "Hello, hello" over the greeting): not an answer.
            for k in pending:
                self.pending.remove(k)
                await self.tl(k, intent="before_opening")
            self.log.write("before_opening", utterance=utterance)
            return True
        rev = self.rev
        # A commit stays provisional until its line is heard. If the candidate speaks
        # before the planner's last line has started, they are still answering the question they heard: roll back
        # to the state before that commit and decide again on the merged turn.
        base, rollback = self.state, None
        a = self.awaiting
        if (a and a.get("sent") and not a.get("started") and self.prev is not None and not self.closing
                and self.prev.get("turn") == a.get("turn")):
            rollback = self.prev
            base = rollback["state"]
            utterance = f"{rollback['utterance']} {utterance}".strip()
        self.log.write("brain_start", rev=rev, session_ms=round(self.state.elapsed * 1000), utterance=utterance,
                       rollback=rollback["turn"] if rollback else None)
        n_agent, n_flags = len(self.state.agent_turns), len(self.state.flags)
        work = copy.deepcopy(base)
        work.started_at = self.state.started_at
        last = self.timeline.get(pending[-1], {})
        eot_ms = self.last_in_ms
        work.candidate_turns.append((self.timeline.get(pending[0], {}).get("t", round(work.elapsed, 2)),
                                     work.stage, utterance))
        brain = Brain(session=work, recent=self._recent(), run_id=self.state.id, model=self.cfg["brain"]["model"],
                      max_tokens=self.cfg["brain"]["max_tokens"], system_extra=self.brain_extra, step=self.step,
                      allow_wait=self.allow_wait)
        try:
            out = await run_turn(brain, utterance, self.interrupt, deadline_s=self.cfg["brain"]["deadline_s"])
        except budget.BudgetExceeded as e:
            self.log.write("budget_exceeded", error=str(e))
            await self.broadcast({"type": "error", "error": {"message": str(e)}})
            await self.end("alt budget exhausted")
            return True
        except Exception:
            self.log.write("brain_error", traceback=traceback.format_exc())
            return True
        self.brain_cost = sum(e["usd"] for e in costs.entries()
                              if e.get("run_id") == self.state.id and e.get("provider") == "anthropic")
        if self.rev != rev or self.cand_seg is not None:
            # The candidate kept talking: drop this result; the next end of turn runs on the merged turn.
            self.log.write("brain_stale", utterance=utterance, trace=out.get("trace"))
            await self.tl(f"stale-{rev}", persist=False, kind="note", t=round(self.state.elapsed, 2),
                          text="(planner result dropped: the candidate kept talking)", intent="stale")
            return False
        if out.get("hold"):
            # The candidate stopped mid-thought: nothing is committed or said. If they stay silent, decide anyway.
            self.log.write("brain_hold", utterance=utterance, trace=out.get("trace"))
            await self.tl(f"hold-{rev}", persist=False, kind="note", t=round(self.state.elapsed, 2), intent="incomplete",
                          text="(planner: the candidate seems mid-thought; keep listening)", trace=out.get("trace"))
            asked = bool(TIME_REQUEST.search(utterance))
            # gpt-live-1 acknowledges "wait a sec" by itself ("Sure, take your time"); a planner line said it twice
            # (c49cf97735). The planner only waits longer.
            self.force_task = asyncio.create_task(
                self._decide_after_silence(rev, HOLD_ASKED_S if asked else HOLD_LONG_S))
            return True
        # Events that touched the live state while the brain ran are carried over to the committed copy.
        if rollback:
            work.agent_turns = list(self.state.agent_turns)
            work.flags += [f for f in self.state.flags if f.get("source") == "monitor" and f not in base.flags]
        else:
            work.agent_turns += self.state.agent_turns[n_agent:]
            work.flags += self.state.flags[n_flags:]
        work.force_close = work.force_close or self.state.force_close
        await self._commit(work, out, pending, utterance, eot_ms, last, base=base, rollback=rollback)
        return True

    async def _commit(self, work: SessionState, out: dict, pending: list[str], utterance: str, eot_ms: int,
                      last: dict, base: SessionState | None = None, rollback: dict | None = None):
        n_stages = len((base or self.state).stage_log)
        stage_before = (base or self.state).stage
        if rollback:
            self.log.write("rollback", turn=rollback["turn"], utterance=utterance)
            await self.tl(f"note-{rollback['turn']}", intent="superseded",
                          text="(superseded: the candidate answered before hearing this line)")
        self.state = work
        self.interrupt = None
        for k in pending:
            if k in self.pending:
                self.pending.remove(k)
        self.turn_n += 1
        tn = self.turn_n
        t = round(self.state.elapsed, 2)
        d = out.get("decision") or {}
        await self.tl(pending[-1], intent=d.get("kind"), specific=d.get("specific"))
        for i, a in enumerate(out.get("actions") or []):
            await self.tl(f"act-{tn}-{i}", kind="tool", t=t, stage=stage_before, name=a["name"],
                          arguments=a["arguments"], output=a["output"], text="")
            self.log.write("tool", t=t, name=a["name"], arguments=json.dumps(a["arguments"]), output=a["output"])
        for i, mark in enumerate(self.state.stage_log[n_stages:], start=n_stages):
            await self.tl(f"stage-{i}", kind="stage", t=t, stage=mark["stage"], text=mark["stage"])
        lines = out.get("lines") or []
        spoken = [ln for ln in lines if ln["channel"] in ("commentary", "exact")]
        trace = out.get("trace") or []
        note = " / ".join(f"[{ln['channel']}] {ln['text']}" for ln in spoken) or "(voice model handles this turn)"
        a = {"turn": tn, "eot_ms": eot_ms, "kind": d.get("kind"), "brain_ms": out.get("total_ms"),
             "llm_ms": next((x.get("llm_ms") for x in trace if x["node"] == "interpret"), None),
             "degraded": bool(d.get("degraded")), "sent": bool(spoken)}
        await self.tl(f"note-{tn}", kind="note", t=t, text=note, intent=d.get("kind"), trace=trace,
                      dropped=out.get("dropped"), problems=out.get("problems"),
                      followup_by_brain=any(ln.get("by") == "brain" for ln in lines),
                      brain_ms=out.get("total_ms"))
        self.log.write("decision", utterance=utterance, decision=d, lines=lines, end=out.get("end"), trace=trace,
                       dropped=out.get("dropped"), problems=out.get("problems"))
        if out.get("end"):
            self.closing = out["end"]
            self.closing_line = next((ln["text"] for ln in reversed(lines) if ln["channel"] == "exact"), "")
        prev = self.awaiting
        if prev is not None and not prev.get("reported"):
            # Its line was never spoken as a separate segment (e.g. merged into the model's own reply).
            prev["reported"] = True
            self.turn_metrics.append(self._metric(prev))
        self.awaiting = a
        bridge = next((g for g in self.agent_segs if g["start_ms"] >= eot_ms), None)
        if bridge is not None:                                     # the model already bridged the gap
            a["first_sound_ms"] = bridge["start_ms"]
            bridge["role"] = "bridge"
            await self.tl(bridge["id"], persist=bridge is not self.agent_seg, role="bridge")
        if spoken:
            self.last_question = spoken[-1]["text"]
        self.prev = {"state": base or self.state, "utterance": utterance, "turn": tn}
        if rollback:
            lines = [{"channel": "instructions", "text": DISCARD}, *lines]
        await self._send_lines(lines)
        if not spoken:
            a["reported"] = True
            self.turn_metrics.append(self._metric(a))
        if self.closing:
            asyncio.create_task(self._close_fallback())
        await self.push_state()

    async def _decide_after_silence(self, rev: int, delay: float = HOLD_LONG_S):
        await asyncio.sleep(delay)
        # If the voice model answered the fragment itself (repeated or explained the question), the candidate is
        # answering that: wait until the model has been quiet for the hold time before deciding (c49cf97735).
        while self.rev == rev and (self.speaking or time.monotonic() - self.last_loud < HOLD_LONG_S):
            await asyncio.sleep(0.5)
        if self.rev == rev and self.cand_seg is None and self.pending:
            self.allow_wait = False
            await self._think()

    async def _wait_quiet(self, max_s: float):
        """Let the model finish its own sentence before an exact line (otherwise warnings and closings cut it off
        mid-word)."""
        end = time.monotonic() + max_s
        while time.monotonic() < end and (self.speaking or time.monotonic() - self.last_loud < 0.4):
            await asyncio.sleep(0.1)

    async def _send_lines(self, lines: list[dict], first_turn: bool = False):
        did, self.delegation = self.delegation, None
        say_id = None
        if not first_turn and any(ln["channel"] == "exact" for ln in lines):
            await self._wait_quiet(QUIET_WAIT_S)
        for ln in lines:
            ch, text = ln["channel"], ln["text"]
            if ch == "thinking":
                await self.adapter.append("thinking", text, did)
            elif ch == "instructions":
                await self.adapter.append("instructions", text, did)
            elif ch == "exact":
                say_id = await self.adapter.append("instructions", EXACT.format(text=text), did)
            else:
                say_id = await self.adapter.append("commentary", COMMENTARY.format(text=text), did)
        if self.awaiting is not None and not first_turn:
            self.awaiting["say_id"] = say_id
            self.awaiting["sent_ms"] = round(self.state.elapsed * 1000)
            self.log.write("lines_sent", session_ms=self.awaiting["sent_ms"], turn=self.awaiting.get("turn"))

    def _metric(self, a: dict) -> dict:
        eot = a.get("eot_ms")
        m = {k: a.get(k) for k in ("turn", "kind", "brain_ms", "llm_ms", "degraded", "sent")}
        m["first_sound_s"] = round((a["first_sound_ms"] - eot) / 1000, 2) if eot and a.get("first_sound_ms") else None
        m["next_line_s"] = round((a["next_line_ms"] - eot) / 1000, 2) if eot and a.get("next_line_ms") else None
        return m

    async def _report_turn(self, a: dict):
        a["reported"] = True
        m = self._metric(a)
        self.turn_metrics.append(m)
        await self.tl(f"note-{a['turn']}", persist=True, first_sound_s=m["first_sound_s"],
                      next_line_s=m["next_line_s"])

    async def _close_fallback(self):
        """If the closing line is never confirmed (e.g. no transcript), end the call anyway."""
        await asyncio.sleep(25)
        await self.end(self.closing or "completed")

    def _recent(self) -> list[tuple[str, str]]:
        rows = [e for e in sorted(self.timeline.values(), key=lambda e: e["t"])
                if e["kind"] in ("agent", "candidate") and not e.get("partial")]
        return [("Interviewer" if e["kind"] == "agent" else "Candidate", e.get("text", "")) for e in rows[-8:]]

    async def _timer(self):
        hard = self.state.stages["time"]["hard_stop_s"]
        while not self.done.is_set():
            await asyncio.sleep(1)
            if self.state.elapsed >= hard:
                await self.end("hard_stop")
            elif self.cost >= self.cap * 1.25:
                await self.end("cost_cap_hard")
            elif int(self.state.elapsed) % 5 == 0:
                await self.push_state()


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _overlap(line: str, spoken: str) -> float:
    """Share of the line's content words that appear in what was spoken."""
    a = _stems(line or "")
    return _share(a, _stems(spoken or "")) if a else 0.0
