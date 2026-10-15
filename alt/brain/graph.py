"""The brain: one LangGraph run per candidate turn.

    START → prepare → interpret ─(error/timeout: degrade)─→ check ─route─┬→ record → advance ─┐
                                                                         ├→ clarify ──────────┤
                                                                         ├→ redirect ─────────┤→ speak → END
                                                                         ├→ defer ────────────┤
                                                                         └→ close ────────────┘
Code routes, the LLM judges (LangGraph pattern): `interpret` is the only LLM node; routers are pure functions;
`check` is the trust boundary; the card engine, `record_evidence` and the quota check are the shared core's code
(`backend/`). The graph never talks to the voice session: it returns spoken `lines`, and the orchestrator sends them
only if no newer candidate speech arrived (commit).
"""
from __future__ import annotations

import operator
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Callable, TypedDict

import yaml
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from alt.brain import llm
from alt.pipeline import cvcheck
from backend.controller import (END_AFTER_NON_ENGAGING, NON_ENGAGING, _close, _current_question, _decline,
                                _quota_check, _record, allowed_fields)
from backend.session import SessionState, render
from backend.tools.evidence import _share, _stems

ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def speech() -> dict:
    return yaml.safe_load((ROOT / "speech.yaml").read_text())


# ── state and context ────────────────────────────────────────────────────────

class Turn(TypedDict, total=False):
    utterance: str
    interrupt: dict | None          # {"heard": str, "unheard_question": bool} when the candidate cut in
    prompt: str                     # the brain's user message (kept for the trace)
    stage_block: str                # the stage's skill SOPs and rubric anchors (business logic for the LLM)
    decision: dict                  # Haiku's output after the trust-boundary check
    dropped: list[str]              # what `check` removed, and why
    lines: list[dict]               # [{"channel": commentary|exact|instructions|thinking, "text": ...}]
    actions: Annotated[list[dict], operator.add]   # tool calls for the timeline (record_evidence, stages, quota)
    end: str | None                 # end reason when the call must close after these lines
    strike: str | None              # "warn" or "end" from the engagement rule
    hold: bool                      # the candidate stopped mid-thought: say nothing, commit nothing, keep listening
    nudged: bool                    # the candidate stayed silent after a fragment: the question was offered again
    gave_up: bool                   # silent again after that: the card was given up (advance says a transition)
    requeued: bool                  # the candidate cut in before the question was heard: it was marked un-asked
    problems: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]


@dataclass
class Brain:
    """Dependencies of one run (LangGraph context): the session copy it may change, and the model settings."""
    session: SessionState
    recent: list[tuple[str, str]]
    run_id: str
    model: str = "claude-haiku-5-5"
    max_tokens: int = 600
    system_extra: str = ""
    step: str = "alt-dev"
    interpret_fn: Callable | None = None      # tests inject a fake; None = the real Haiku call
    allow_wait: bool = True                   # False once the candidate stayed silent after a held fragment
    t0: float = field(default_factory=time.monotonic)


def _trace(node: str, t0: float, **extra) -> list[dict]:
    return [{"node": node, "ms": round((time.monotonic() - t0) * 1000), **extra}]


# ── nodes ────────────────────────────────────────────────────────────────────

def prepare(state: Turn, runtime: Runtime[Brain]) -> dict:
    """Code: apply the interruption (an unheard question is asked again) and build the brain's input."""
    t0 = time.monotonic()
    s = runtime.context.session
    intr = state.get("interrupt") or {}
    undone = bool(intr.get("unheard_question")) and s.unhear_question()
    card = s.current_card() or {}
    question = (s.last_instruction or {}).get("question") or card.get("primary", "")
    missing = [f for f in card.get("must_capture", []) if not s.is_captured(f)]
    captured = {f: ev.value for f, ev in s.captured().items()}
    prompt = llm.build_user(stage=s.stage, question=question, missing=missing, fields=allowed_fields(s),
                            captured=captured, recent=runtime.context.recent, utterance=state["utterance"],
                            interrupt=intr.get("heard") if intr else None, hints=session_hints(s))
    return {"prompt": prompt, "stage_block": stage_block(s), "requeued": undone,
            "trace": _trace("prepare", t0, note="question re-queued" if undone else None)}


def session_hints(s: SessionState) -> dict[str, str]:
    """Field hints from the session's market preset and the role's requirement questions."""
    h = dict((s.role.get("market") or {}).get("field_hints") or {})
    for q in s.role.get("role_questions") or []:
        h[q["field"]] = f"their answer to \"{q['question']}\" (a strong answer: {q.get('look_for', '')})"
    return h


def stage_block(s: SessionState) -> str:
    """Business logic for the reasoning step, per stage: the stage's skill SOPs from
    content/skills, the 1/3/5 anchors of the competencies its fields feed (content/rubric.yaml), and the layers
    of this session: the market preset's notes and reference, the role's objection and references, the role's
    requirement questions. Cached per stage and role, so the content costs one cache write per stage."""
    dims = tuple(sorted(set(s.fields[f] for c in s.stage_cfg()["cards"] for f in c.get("must_capture", [])
                            if f in s.fields)))
    key = (s.stage, dims, s.role.get("role_id"), (s.role.get("source") or {}).get("role_md_sha"))
    if key not in _BLOCKS:
        _BLOCKS[key] = _stage_block(s.stage, dims) + _layers(s.stage, s.role)
    return _BLOCKS[key]


_BLOCKS: dict[tuple, str] = {}


def _layers(stage: str, role: dict) -> str:
    market = role.get("market") or {}
    parts = []
    if note := (market.get("brain_notes") or {}).get(stage):
        parts.append(f"## Market: {market.get('name')}\n{note}")
    if stage == "fit_and_logistics" and (ref := ROOT.parent / "content" / "markets" / f"{market.get('market_id')}.md").exists():
        parts.append(f"## Market reference (never read aloud)\n{ref.read_text().strip()}")
    if stage == "ai_objection_roleplay" and (ob := role.get("objection_scenario")):
        parts.append(f"## This role's objection\nBuyer: {ob.get('role_played')}. Concern: {ob.get('concern')}\n"
                     f"A strong answer: {ob.get('strong_answer', '')}")
        refs = ROOT.parent / "content" / "roles" / str(role.get("role_id")) / "references"
        for r in sorted(refs.glob("*.md")) if refs.exists() else []:
            parts.append(f"## Role reference: {r.stem} (never read aloud)\n{r.read_text().strip()}")
    if stage == "background" and role.get("role_questions"):
        parts.append("## This role's requirement questions\n" + "\n".join(
            f"- {q['field']}: {q['question']} A strong answer: {q.get('look_for', '')}" for q in role["role_questions"]))
    return "".join("\n\n" + p for p in parts)


@lru_cache
def _stage_block(stage: str, dims: tuple[str, ...]) -> str:
    from backend.content_store import ContentStore
    store = ContentStore()
    cfg = store.stages()["stages"][stage]
    names = cfg.get("skill") or []
    names = [names] if isinstance(names, str) else names
    parts = [f"# Current stage: {stage}\nGoal: {cfg.get('goal', '')}"]
    for n in names:
        sk = store.skill(n)
        if sk:
            parts.append(f"## Skill: {n}\n{sk['body']}")
    comps = store.rubric()["competencies"]
    anchors = [f"- {d} ({comps[d]['name']}): 1 = {comps[d]['anchors'][1]} 3 = {comps[d]['anchors'][3]} "
               f"5 = {comps[d]['anchors'][5]}" for d in dims if d in comps]
    if anchors:
        parts.append("## Rubric anchors for this stage (use them to judge specificity, follow-ups and red flags; "
                     "never tell the candidate)\n" + "\n".join(anchors))
    return "\n\n".join(parts)


async def interpret(state: Turn, runtime: Runtime[Brain]) -> dict:
    """LLM: Claude Haiku 5.5 judges the turn (strict JSON)."""
    t0 = time.monotonic()
    c = runtime.context
    fn = c.interpret_fn or llm.interpret
    out = await fn(c.system_extra, state["prompt"], model=c.model, max_tokens=c.max_tokens, run_id=c.run_id,
                   step=c.step, stage_block=state.get("stage_block", ""))
    return {"decision": out, "trace": _trace("interpret", t0, kind=out.get("kind"), llm_ms=out.get("_ms"),
                                            usage=out.get("_usage"))}


def interpret_failed(state: Turn, error: NodeError) -> Command:
    """Deadline or failure: degrade to "an answer with nothing extracted" so the card engine still moves on
    (as the shared controller does). Never surfaces to the candidate."""
    err = getattr(error, "error", error)
    return Command(goto="check", update={
        "decision": {"kind": "answer", "evidence": [], "specific": True, "followup_field": "",
                     "followup_question": "", "reply": "", "concern_type": "none", "concern_quote": "",
                     "degraded": True},
        "problems": [f"interpret degraded: {type(err).__name__}"],
        "trace": [{"node": "interpret", "ms": None, "status": "degraded", "error": type(err).__name__}]})


# Words that reveal the interviewer's internal material when spoken to a candidate.
INTERNAL = re.compile(r"\b(the|my|our) (brief|notes|documents?|instructions|prompt|system|screen configuration)\b|"
                      r"\brole brief\b", re.I)


def _in_turn(quote: str, utterance: str) -> bool:
    q = _stems(quote)
    return bool(q) and _share(q, _stems(utterance)) >= 0.8


def check(state: Turn, runtime: Runtime[Brain]) -> dict:
    """Code: the trust boundary. Keep only what the candidate literally said and what this stage allows."""
    t0 = time.monotonic()
    s = runtime.context.session
    d = dict(state["decision"])
    utt = state["utterance"]
    dropped: list[str] = []
    if d.get("kind") not in llm.KINDS:
        dropped.append(f"kind {d.get('kind')!r} → answer")
        d["kind"] = "answer"
    allowed = set(allowed_fields(s))
    keep = []
    for ev in d.get("evidence") or []:
        f, v, q = ev.get("field"), str(ev.get("value") or "").strip(), str(ev.get("quote") or "")
        if f not in allowed or f not in s.fields:
            dropped.append(f"{f}: not a field of this stage")
        elif not v:
            dropped.append(f"{f}: empty value")
        elif not _in_turn(q, utt):
            dropped.append(f"{f}: quote not in the candidate's words")
        else:
            keep.append({"field": f, "value": v, "quote": q})
    d["evidence"] = keep
    card = s.current_card() or {}
    ff, fq = d.get("followup_field") or "", (d.get("followup_question") or "").strip()
    if ff or fq:
        two_asks = re.search(r",?\s\b(and|also)\b\s+(who|what|how|when|which|where|why|did|do|was)\b", fq, re.I)
        ok = (ff in card.get("must_capture", []) and fq.endswith("?") and fq.count("?") == 1 and len(fq) <= 220
              and not two_asks
              and s.stage not in ("consent", "ai_objection_roleplay") and not card.get("strict"))
        if not ok:
            dropped.append(f"follow-up for {ff!r} not usable here")
            d["followup_field"], d["followup_question"] = "", ""
    if d.get("reply") and d["kind"] != "role_question":
        d["reply"] = ""
    d["reply"] = (d.get("reply") or "")[:400]
    if INTERNAL.search(d["reply"]):        # the prompt forbids it, but runs showed "The brief doesn't cover …"
        dropped.append("reply named internal material; replaced")
        d["reply"] = "I don't have those details, but the recruiter will follow up with you on that."
    if d.get("concern_type", "none") != "none" and not _in_turn(d.get("concern_quote") or "", utt):
        dropped.append(f"concern {d['concern_type']}: quote not in the candidate's words")
        d["concern_type"], d["concern_quote"] = "none", ""
    return {"decision": d, "dropped": dropped, "trace": _trace("check", t0, dropped=len(dropped))}


def route(state: Turn, runtime: Runtime[Brain]) -> str:
    """Pure router: turn kind → node."""
    return route_for(state["decision"]["kind"], runtime.context.session.stage, runtime.context.allow_wait)


def route_for(kind: str, stage: str, allow_wait: bool = True) -> str:
    if kind == "incomplete":
        return "wait" if allow_wait else "nudge"
    if kind in ("abusive", "end_request"):
        return "close"
    if kind in NON_ENGAGING:
        return "redirect"
    if stage == "candidate_questions" and kind in ("role_question", "clarification"):
        return "clarify"
    if stage in ("candidate_questions", "close") and kind == "end_request":
        return "record"           # "no, that's all" ends the questions stage normally (adversarial run)
    if kind == "role_question":
        return "clarify"
    if kind in ("repeat_request", "clarification"):
        return "defer"            # gpt-live-1 repeats or explains its own question; the brain stays silent
    return "record"


def record(state: Turn, runtime: Runtime[Brain]) -> dict:
    """Code: record evidence through the shared core's checks; red flags go to the recruiter."""
    t0 = time.monotonic()
    s = runtime.context.session
    d = state["decision"]
    acts: list[dict] = []
    engagement = d.get("engagement", "engaged")
    if engagement in ("low_effort", "not_engaging") and \
            len(re.findall(r"[A-Za-z']+", state["utterance"])) < MIN_WORDS_FOR_STRIKE:
        engagement = "weak_but_genuine"   # an answer this short is a fragment; explicit "next"/"skip" go via redirect
    strike = _engagement(s, engagement, state["utterance"])
    for ev in d["evidence"]:
        if ev["field"] in allowed_fields(s):
            _record(s, ev["field"], ev["value"], ev["quote"], acts)
    if d["kind"] == "end_request" and s.stage == "candidate_questions" and not s.is_captured("candidate_done"):
        _record(s, "candidate_done", "yes", state["utterance"], acts, force=True)
    if d["kind"] == "decline":
        card = s.current_card() or {}
        missing = [f for f in card.get("must_capture", []) if not s.is_captured(f)]
        if missing:
            _decline(s, missing[0], state["utterance"], acts)
    if state.get("requeued") and d["kind"] in ("answer", "partial_answer"):
        _heard_after_all(s)
    if d.get("concern_type", "none") != "none":
        s.flags.append({"t": round(s.elapsed, 1), "type": d["concern_type"], "detail": d["concern_quote"],
                        "source": "brain"})
    ok = sum(1 for a in acts if a["output"].get("status") == "recorded")
    return {"actions": acts, "strike": strike,
            "trace": _trace("record", t0, recorded=ok, rejected=len(acts) - ok,
                            engagement=d.get("engagement"), strike=strike, count=s.non_engaging)}


def _heard_after_all(s: SessionState):
    """The question was re-queued after an interruption, but the candidate answered it: gpt-live-1 had already
    repeated it by itself while the brain held. Mark it asked again, so the engine goes on to probe what is still
    missing instead of reading the whole question a third time."""
    card = s.current_card()
    if card and not any(e["card"] == card["id"] and e["kind"] == "asked" for e in s.card_log):
        s.log_card("asked", card["id"], note="answered after an interruption")


MIN_WORDS_FOR_STRIKE = 3      # "lost", "Ja": transcript fragments are noise, not a judgement of effort (cfee5dd7ab,
                              # c49cf97735 ended a call on two one-word fragments)


def _engagement(s: SessionState, engagement: str, utterance: str) -> str | None:
    """Strict non-cooperation rule: low-effort answers
    count like off-topic turns; only an engaged answer resets the count; a weak but genuine answer neither counts
    nor resets. 2nd strike: warning; 3rd: end."""
    if engagement == "engaged":
        s.non_engaging = 0
        return None
    if engagement not in ("low_effort", "not_engaging"):
        return None
    s.non_engaging += 1
    s.flags.append({"t": round(s.elapsed, 1), "type": engagement, "source": "brain",
                    "detail": f"strike {s.non_engaging}: {utterance[:160]}"})
    if s.non_engaging >= END_AFTER_NON_ENGAGING:
        return "end"
    return "warn" if s.non_engaging == 2 else None


def after_record(state: Turn) -> str:
    return "close" if state.get("strike") == "end" else "advance"


def nudge(state: Turn, runtime: Runtime[Brain]) -> dict:
    """The candidate stopped mid-thought and stayed silent (without this they were moved on without a word).
    Once per card, offer the question again, gently; nothing is recorded. The second time, or when the fragment
    did hold evidence, the turn is recorded and the interview moves on (with a spoken transition, see `advance`)."""
    t0 = time.monotonic()
    s = runtime.context.session
    card = s.current_card() or {}
    nudges = s.__dict__.setdefault("nudges", {})
    if state["decision"]["evidence"]:
        return {"nudged": False, "trace": _trace("nudge", t0, nudged=False)}
    if nudges.get(card.get("id")):            # silent again after the gentle repeat: no third ask of the same thing
        s.give_up_card("no answer after a pause")
        return {"nudged": False, "gave_up": True, "trace": _trace("nudge", t0, nudged=False, gave_up=card.get("id"))}
    nudges[card.get("id")] = 1
    q = _current_question(s)       # the pending follow-up if one was asked last, else the card's short form
    return {"nudged": True, "lines": [{"channel": "commentary", "text": f"No rush. {q}"}],
            "trace": _trace("nudge", t0, nudged=True)}


def after_nudge(state: Turn) -> str:
    return "speak" if state.get("nudged") else "record"


def wait(state: Turn, runtime: Runtime[Brain]) -> dict:
    """The candidate stopped mid-thought: commit nothing and stay silent; the orchestrator keeps listening and
    asks the brain again on the merged turn, or without this option once the candidate stays silent."""
    return {"lines": [], "hold": True, "trace": _trace("wait", time.monotonic())}


def _say(s: SessionState, text: str) -> str:
    first = (s.candidate.get("name") or "").replace(" (synthetic)", "").split(" ")[0]
    ctx = {**s.ctx, "candidate": {**s.candidate, "first_name": first}}
    return re.sub(r"\s+", " ", render(text, ctx)).strip()


def _card_line(s: SessionState, nxt: dict, d: dict) -> list[dict]:
    """The spoken line for an ask/probe instruction: the card's spoken wording, or the LLM's contextual
    follow-up when it targets a field the engine is probing for."""
    card = s.current_card() or {}
    sp = speech()["cards"].get(card.get("id"), {})
    lines: list[dict] = []
    if nxt["type"] == "ask":
        if sp.get("instructions"):
            lines.append({"channel": "instructions", "text": _say(s, sp["instructions"])})
        text = _say(s, sp["say"]) if sp.get("say") else nxt["question"]
        s.last_instruction = {**nxt, "question": text}      # what was actually said (for the brain and repeats)
        lines.append({"channel": sp.get("channel", "commentary"), "text": text})
        return lines
    probe_field = next((e.get("field") for e in reversed(s.card_log)
                        if e["card"] == card.get("id") and e["kind"] == "probe"), None)
    psp = (sp.get("probes") or {}).get(probe_field)
    if psp:
        text = _say(s, psp["say"])
        s.last_instruction = {**nxt, "question": text}
        lines.append({"channel": psp.get("channel", "commentary"), "text": text})
    elif d.get("followup_question") and d.get("followup_field") in nxt.get("capture", []):
        # The LLM chose which missing field matters most and asked for it in context.
        for e in reversed(s.card_log):
            if e["card"] == card.get("id") and e["kind"] == "probe":
                e["field"], e["by"] = d["followup_field"], "brain"
                break
        s.last_instruction = {**nxt, "question": d["followup_question"], "by": "brain"}
        lines.append({"channel": "commentary", "text": d["followup_question"], "by": "brain"})
    else:
        q = nxt["question"]
        if q.startswith("Ask briefly for:"):      # the engine's internal fallback is never spoken as such
            q = card.get("short") or card.get("primary", "")
        lines.append({"channel": "commentary", "text": q})
    return lines


def advance(state: Turn, runtime: Runtime[Brain]) -> dict:
    """Code: follow the card engine until there is something to say."""
    t0 = time.monotonic()
    s = runtime.context.session
    d = state["decision"]
    acts: list[dict] = []
    readback = ""
    n_log, n_over = len(s.card_log), len(s.overrides)
    for _ in range(20):
        nxt = s.next_instruction()
        t = nxt["type"]
        if t in ("ask", "probe"):
            lines = _card_line(s, nxt, d)
            gave_up = state.get("gave_up") or len(s.overrides) > n_over \
                or any(e["kind"] in ("exhausted", "gave_up") for e in s.card_log[n_log:])
            if gave_up and not readback:      # moving on without the answer: say so, never switch topics silently
                last = lines[-1]
                lines[-1] = {**last, "text": f"{speech()['transitions']['move_on']} {last['text']}"}
            if state.get("strike") == "warn":
                warning = _say(s, speech()["redirect"]["warning"])
                readback = f"{warning} {readback}".strip()
            if readback:            # numbers are read back word for word, then the next question
                last = lines[-1]
                lines[-1] = {"channel": "exact", "text": f"{readback} {last['text']}"}
            return {"lines": lines, "actions": acts,
                    "trace": _trace("advance", t0, next=f"{t}:{nxt.get('card')}", stage=s.stage)}
        if t == "action":
            card = s.current_card()
            out = _quota_check(s)
            acts.append({"name": "check_quota_consistency", "arguments": {}, "output": out})
            s.actions_done.add(card["id"])
            if out.get("status") in ("consistent", "inconsistent"):
                readback = out["readback"]
            _cv_check(s, acts)
            continue
        if t in ("stage_complete", "cards_exhausted", "wrap_up"):
            to = "candidate_questions" if t == "wrap_up" else s.next_stage()
            if t != "stage_complete":
                s.overrides.append({"t": round(s.elapsed, 1), "from": s.stage, "to": to, "reason": t,
                                    "missing": nxt.get("missing", [])})
            s.enter_stage(to)
            order = shuffle_cards(s)
            acts.append({"name": "advance_stage", "arguments": {"to": to, **({"order": order} if order else {})},
                         "output": {"status": "ok"}})
            if to == "close":
                return _closing(s, "completed", "completed", acts, t0, "advance")
            continue
        reason, key = {"must_close": ("completed: time limit", "time_limit"),
                       "consent_refused": ("ended: no consent", "consent_refused")}.get(t, ("completed", "completed"))
        return _closing(s, reason, key, acts, t0, "advance")
    return _closing(s, "completed", "completed", acts, t0, "advance")


def _cv_check(s: SessionState, acts: list[dict]):
    """Code: the performance figures the candidate just stated vs candidate.md's claims. Blind recall:
    only now, after the candidate has committed to a figure, may a neutral follow-up quote the CV (`perf_cv`).
    One clarification per call: skipped when the quota check already inserted `perf_clarify`."""
    claims = s.candidate.get("claims") or []
    if not claims:
        return
    said = {f: ev.value for f, ev in s.captured().items() if not ev.declined}
    rows = [r for r in cvcheck.compare(claims, said, s.role.get("currency")) if r["field"] in cvcheck.PERFORMANCE]
    acts.append({"name": "check_cv_claims", "arguments": {}, "output": {"rows": rows}})
    for r in rows:
        if r["status"] == "differs":
            s.flags.append({"t": round(s.elapsed, 1), "type": "cv_mismatch", "source": "code",
                            "detail": f"{r['field']}: said {r['said']!r}; CV: {r['cv']}"})
    q = cvcheck.followup(rows, claims, (s.role.get("market") or {}).get("amount_style", "western"))
    if q and not any(c["id"] in ("perf_clarify", "perf_cv") for c in s.cards):
        s.insert_card({"id": "perf_cv", "primary": q, "must_capture": ["quota_clarification"], "strict": True,
                       "probes": {"quota_clarification": "Could you walk me through how those figures fit together?"},
                       "max_probes": 1})       # one retry: a half answer ("let me explain…") is not an answer


def shuffle_cards(s: SessionState) -> list[str] | None:
    """Optional within-stage shuffle, seeded per session and stage so a run is reproducible.
    Same cards for every candidate (structured-interview fairness); only their order changes.
    Action and strict cards stay where they are and split the stage into segments, so dependencies hold: the quota
    check still follows the three number cards, and the role-play exit still follows the objection."""
    seed = getattr(s, "shuffle_seed", None)
    if seed is None or s.stage in ("consent", "close") or len(s.cards) < 2:
        return None
    rng = random.Random(f"{seed}:{s.stage}")
    out, seg = [], []
    for c in s.cards:
        if c.get("action") or c.get("strict"):
            rng.shuffle(seg)
            out += seg + [c]
            seg = []
        else:
            seg.append(c)
    rng.shuffle(seg)
    out += seg
    order = [c["id"] for c in out]
    if order == [c["id"] for c in s.cards]:
        return None
    s.cards, s.card_idx = out, 0
    for e in reversed(s.card_log):                 # enter_stage logged the old first card as issued
        if e["kind"] == "issued":
            e["card"] = out[0]["id"]
            break
    return order


def _closing(s: SessionState, reason: str, key: str, acts: list, t0: float, node: str) -> dict:
    _close(s, reason, "", acts)
    text = _say(s, speech()["closing"][key])
    return {"lines": [{"channel": "exact", "text": text}], "actions": acts, "end": reason,
            "trace": _trace(node, t0, end=reason)}


def clarify(state: Turn, runtime: Runtime[Brain]) -> dict:
    """The candidate asked about the role or process: the LLM's answer (from the brief only), then back on track."""
    t0 = time.monotonic()
    s = runtime.context.session
    d = state["decision"]
    acts: list[dict] = []
    reply = d.get("reply") or "That's a good question for the recruiter, who will follow up with you."
    if s.stage == "candidate_questions":
        _record(s, "candidate_question", state["utterance"], state["utterance"], acts, force=True)
        text = f"Answer the candidate: {reply} Then ask whether they have any other questions."
    else:
        text = f"Answer the candidate briefly: {reply} Then ask: {_current_question(s)}"
    return {"lines": [{"channel": "commentary", "text": text}], "actions": acts, "trace": _trace("clarify", t0)}


def redirect(state: Turn, runtime: Runtime[Brain]) -> dict:
    """Strict non-cooperation rules: bring it back; warn on the second; end on the third."""
    t0 = time.monotonic()
    s = runtime.context.session
    strike = _engagement(s, "not_engaging", state["utterance"])
    if strike == "end":
        return _closing(s, "ended: non-cooperative", "non_cooperative", [], t0, "redirect")
    sp = speech()["redirect"]
    q = _current_question(s)
    if strike == "warn":
        return {"lines": [{"channel": "exact", "text": f"{_say(s, sp['warning'])} {q}"}],
                "trace": _trace("redirect", t0, level="warning")}
    # First time: gpt-live-1 already declines and returns to its question by itself, so the
    # brain only counts it; speaking too would ask the question twice.
    return {"lines": [], "trace": _trace("redirect", t0, level="first (voice model handles)")}


def defer(state: Turn, runtime: Runtime[Brain]) -> dict:
    """Repeat or clarification of the current question: gpt-live-1 handles it itself. Endless clarifications on
    one card count as not engaging."""
    t0 = time.monotonic()
    s = runtime.context.session
    cid = (s.current_card() or {}).get("id", "")
    s.clarifications[cid] = s.clarifications.get(cid, 0) + 1
    if s.clarifications[cid] > 2:
        s.non_engaging += 1
        if s.non_engaging >= END_AFTER_NON_ENGAGING:
            return _closing(s, "ended: non-cooperative", "non_cooperative", [], t0, "defer")
    return {"lines": [], "trace": _trace("defer", t0, clarifications=s.clarifications[cid])}


def close(state: Turn, runtime: Runtime[Brain]) -> dict:
    t0 = time.monotonic()
    s = runtime.context.session
    if state.get("strike") == "end":
        return _closing(s, "ended: non-cooperative", "non_cooperative", [], t0, "close")
    if state["decision"]["kind"] == "abusive":
        return _closing(s, "ended: abusive language", "abusive", [], t0, "close")
    return _closing(s, "ended: candidate asked to stop", "end_request", [], t0, "close")


def speak(state: Turn, runtime: Runtime[Brain]) -> dict:
    """Code: add silent state for the voice model, so it can repeat or explain the current question itself."""
    t0 = time.monotonic()
    s = runtime.context.session
    lines = list(state.get("lines") or [])
    if state.get("hold"):
        return {"lines": [], "trace": _trace("speak", t0, n=0)}
    if not state.get("end"):
        q = (s.last_instruction or {}).get("question") or (s.current_card() or {}).get("primary", "")
        rp = s.stage == "ai_objection_roleplay" and (s.current_card() or {}).get("id") == "rp_objection"
        lines.insert(0, {"channel": "thinking", "text": (
            f"Interview state: stage {s.stage}; the current question is: {q} "
            f"{'You are in the role-play, playing the customer.' if rp else 'You are the interviewer.'}")})
    return {"lines": lines, "trace": _trace("speak", t0, n=len(lines))}


# ── graph ────────────────────────────────────────────────────────────────────

def build(deadline_s: float = 3.5):
    g = StateGraph(Turn, context_schema=Brain)
    g.add_node("prepare", prepare)
    g.add_node("interpret", interpret, timeout=deadline_s, error_handler=interpret_failed, destinations=("check",))
    g.add_node("check", check)
    for name, fn in (("record", record), ("advance", advance), ("clarify", clarify), ("redirect", redirect),
                     ("defer", defer), ("close", close), ("speak", speak)):
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "interpret")
    g.add_edge("interpret", "check")
    g.add_node("wait", wait)
    g.add_node("nudge", nudge)
    g.add_conditional_edges("check", route, ["record", "clarify", "redirect", "defer", "close", "wait", "nudge"])
    g.add_conditional_edges("nudge", after_nudge, ["speak", "record"])
    g.add_conditional_edges("record", after_record, ["advance", "close"])
    g.add_edge("wait", "speak")
    for n in ("advance", "clarify", "redirect", "defer", "close"):
        g.add_edge(n, "speak")
    g.add_edge("speak", END)
    return g.compile()


@lru_cache
def graph(deadline_s: float = 3.5):
    return build(deadline_s)


async def run_turn(brain: Brain, utterance: str, interrupt: dict | None = None, deadline_s: float = 3.5) -> dict:
    out = await graph(deadline_s).ainvoke({"utterance": utterance, "interrupt": interrupt, "problems": [],
                                           "trace": [], "actions": []}, context=brain)
    out["total_ms"] = round((time.monotonic() - brain.t0) * 1000)
    return out


def mermaid() -> str:
    return graph().get_graph().draw_mermaid()
