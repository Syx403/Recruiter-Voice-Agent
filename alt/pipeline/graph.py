"""Durable LangGraph pipelines (docs/ARCHITECTURE.md), one thread each, in SQLite.

Role pipeline (one thread per role.md version; the role step of the question generator):

    role_intake (parse role.md) → role_compile (Haiku proposes) → role_validate (code) → role_review ⏸
    → role_publish (content/roles/<id>/screen.yaml)

Candidate pipeline (one thread per candidate and role; the candidate step):

    intake (parse candidate.md) → plan (Haiku: CV × JD) → validate (code) → approve_plan ⏸ → invite
    → await_call ⏸ → brief_score → brief_ground → (brief_repair, at most once) → brief_compute → review ⏸ → finalise

⏸ = `interrupt()`: the thread is saved in SQLite (runs/alt/pipeline.sqlite) and resumed by the recruiter (plan,
review) or by the orchestrator when the call ends. On resume LangGraph re-runs the interrupted node, so nodes with an
interrupt have no side effects before it, and `invite` is idempotent (LangGraph interrupt rule).
The in-call per-turn graph (alt/brain/graph.py) runs inside `await_call`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TypedDict

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlmodel import Field, Session, SQLModel, select

from alt.pipeline import brief as B
from alt.pipeline import cvcheck
from alt.pipeline import inputs as I
from alt.pipeline import plan as P
from alt.pipeline import screen as S
from backend import export, store
from backend.content_store import ContentStore, candidate

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("uvicorn.error")


class PipelineRow(SQLModel, table=True):
    """Index of pipeline threads (the state itself lives in the LangGraph checkpointer)."""
    id: str = Field(primary_key=True)
    candidate_id: str
    role_id: str
    persona_id: str
    token: str | None = Field(default=None, index=True)
    session_id: str | None = Field(default=None, index=True)
    status: str = "intake"   # intake | plan_review | invited | in_call | briefing | brief_review | approved | rejected | verify
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class PState(TypedDict, total=False):
    pipeline_id: str
    candidate_id: str
    role_id: str
    persona_id: str
    auto_approve_plan: bool
    parsed: dict                 # candidate.md, parsed in code
    input_issues: list[str]      # candidate.md or role.md broke docs/INPUT_SPEC.md: the pipeline stops
    plan: dict
    plan_issues: list[str]
    token: str
    session_id: str
    draft: dict
    unmatched: list[str]
    repaired: bool
    brief: dict
    computed: dict
    brief_error: str
    decision: dict


def _row_update(pid: str, **fields):
    with Session(store.engine()) as s:
        row = s.get(PipelineRow, pid)
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)
        row.updated_at = time.time()
        s.add(row)
        s.commit()


# ── nodes ────────────────────────────────────────────────────────────────────

def intake(state: PState) -> dict:
    """candidate.md (docs/INPUT_SPEC.md), parsed in code; a document that breaks the spec stops the pipeline."""
    if state.get("parsed"):
        return {}
    if not I.candidate_path(state["candidate_id"]).exists():
        c = candidate(state["candidate_id"])           # an older synthetic profile: no plan, standard cards
        return {"parsed": {"candidate_id": state["candidate_id"], "name": c.get("name"), "claims": []}}
    try:
        parsed, _ = I.load_candidate(state["candidate_id"])
        I.load_role(state["role_id"])
    except I.InputError as e:
        _row_update(state["pipeline_id"], status="input_error")
        return {"input_issues": [f"{e.kind}: {i}" for i in e.issues]}
    return {"parsed": parsed}


def after_intake(state: PState) -> str:
    if state.get("input_issues"):
        return END
    return "brief_score" if state.get("session_id") else "plan"


async def plan(state: PState) -> dict:
    if not state["parsed"].get("claims") and not I.candidate_path(state["candidate_id"]).exists():
        return {"plan": P.empty(state["candidate_id"], state["role_id"])}
    return {"plan": await P.generate(state["candidate_id"], state["role_id"], run_id=state["pipeline_id"])}


def _plan_inputs(state: PState) -> tuple[dict, str, str]:
    screen = S.approved(state["role_id"]) or {"requirements": []}
    cmd = I.candidate_path(state["candidate_id"]).read_text() if I.candidate_path(state["candidate_id"]).exists() else ""
    rmd = I.role_path(state["role_id"]).read_text() if I.role_path(state["role_id"]).exists() else ""
    return screen, cmd, rmd


def validate(state: PState) -> dict:
    clean, issues = P.validate(state.get("plan") or {}, ContentStore().stages(), *_plan_inputs(state),
                               cand=state.get("parsed"))
    return {"plan": clean, "plan_issues": issues}


def approve_plan(state: PState) -> dict:
    if state.get("auto_approve_plan") or state.get("session_id"):
        return {}
    _row_update(state["pipeline_id"], status="plan_review")
    answer = interrupt({"kind": "plan", "plan": state.get("plan"), "issues": state.get("plan_issues")})
    if isinstance(answer, dict) and answer.get("plan"):         # the recruiter edited the plan: validate again
        clean, issues = P.validate({**state["plan"], **answer["plan"]}, ContentStore().stages(), *_plan_inputs(state),
                                   cand=state.get("parsed"))
        return {"plan": clean, "plan_issues": issues}
    return {}


def invite(state: PState) -> dict:
    if state.get("token") or state.get("session_id"):
        return {}
    inv = store.open_invite(state["role_id"], state["persona_id"], state["candidate_id"])
    _row_update(state["pipeline_id"], token=inv.token, status="invited")
    return {"token": inv.token}


def await_call(state: PState) -> dict:
    if state.get("session_id"):
        return {}
    answer = interrupt({"kind": "call", "token": state.get("token")})
    sid = answer["session_id"]
    _row_update(state["pipeline_id"], session_id=sid, status="briefing")
    return {"session_id": sid}


def _role(role_id: str) -> dict:
    """The approved screen configuration; older roles without one fall back to the hand-written brief.yaml."""
    return S.approved(role_id) or ContentStore().role(role_id)


def _record(sid: str, state: PState | None = None) -> dict:
    r = export.record(sid)
    role = _role(r["session"]["role_id"])
    plan = (state or {}).get("plan") or {}
    claims = plan.get("claims") or ((state or {}).get("parsed") or {}).get("claims") or []
    said = {e["field"]: e["value"] for e in r["evidence"] if not e.get("declined")}
    return {"role": role, "rubric": ContentStore().rubric(),
            "ledger": r["evidence"], "flags": r["flags"], "consistency": r["consistency"],
            "timeline": r["timeline"], "overrides": r["overrides"],
            "requirement_fit": plan.get("requirement_fit") or [],
            "cv_check": cvcheck.compare(claims, said, role.get("currency"))}


async def brief_score(state: PState) -> dict:
    _row_update(state["pipeline_id"], status="briefing")
    try:
        rec = _record(state["session_id"], state)
        user = B.build_input(**rec)
        draft = await B.call([{"role": "user", "content": user}], run_id=state["session_id"])
        return {"draft": draft, "brief_error": ""}
    except Exception as e:  # the recruiter still gets a review step, with the error
        log.error("brief failed: %s", e)
        return {"brief_error": f"{type(e).__name__}: {e}"}


def brief_ground(state: PState) -> dict:
    if state.get("brief_error"):
        return {}
    said = B.candidate_text(_record(state["session_id"])["timeline"])
    clean, unmatched = B.ground(state["draft"], said)
    return {"brief": clean, "unmatched": unmatched}


def after_ground(state: PState) -> str:
    return "brief_repair" if state.get("unmatched") and not state.get("repaired") else "brief_compute"


async def brief_repair(state: PState) -> dict:
    """At most once: send unmatched quotes back; lines that pass stay as they are (repair before rewrite)."""
    rec = _record(state["session_id"], state)
    user = B.build_input(**rec)
    fix = ("These quotes are not the candidate's exact words: " + json.dumps(state["unmatched"], ensure_ascii=False)
           + ". Return the whole brief again, replacing each with an exact quote from the candidate's turns, or "
             "removing it. Change nothing else.")
    try:
        draft = await B.call([{"role": "user", "content": user},
                              {"role": "assistant", "content": json.dumps(state["draft"], ensure_ascii=False)},
                              {"role": "user", "content": fix}], run_id=state["session_id"])
    except Exception as e:
        return {"repaired": True, "brief_error": "", "unmatched": [f"repair failed: {type(e).__name__}"]}
    clean, unmatched = B.ground(draft, B.candidate_text(rec["timeline"]))
    for c in clean["competencies"]:
        if not c["evidence"] and c["score"]:
            c["confidence"] = "low"
    return {"draft": draft, "brief": clean, "unmatched": unmatched, "repaired": True}


def brief_compute(state: PState) -> dict:
    if state.get("brief_error"):
        return {"computed": {}}
    rec = _record(state["session_id"], state)
    computed = B.compute(state["brief"], rec["role"], rec["rubric"], rec["consistency"], rec["ledger"],
                         rec["cv_check"], rec["requirement_fit"])
    path = ROOT / "runs" / state["session_id"] / "brief.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"brief": state["brief"], "computed": computed, "dropped_quotes": state.get("unmatched")},
                               indent=1, ensure_ascii=False))
    return {"computed": computed}


def review(state: PState) -> dict:
    _row_update(state["pipeline_id"], status="brief_review")
    answer = interrupt({"kind": "brief", "brief": state.get("brief"), "computed": state.get("computed"),
                        "error": state.get("brief_error")})
    return {"decision": {"decision": answer.get("decision"), "note": answer.get("note", ""), "t": time.time()}}


def finalise(state: PState) -> dict:
    """Idempotent status update in the mock ATS."""
    d = state["decision"]["decision"]
    _row_update(state["pipeline_id"], status={"approve": "approved", "reject": "rejected"}.get(d, "verify"))
    return {}


def build(checkpointer=None):
    g = StateGraph(PState)
    for name, fn in (("intake", intake), ("plan", plan), ("validate", validate), ("approve_plan", approve_plan),
                     ("invite", invite), ("await_call", await_call), ("brief_score", brief_score),
                     ("brief_ground", brief_ground), ("brief_repair", brief_repair), ("brief_compute", brief_compute),
                     ("review", review), ("finalise", finalise)):
        g.add_node(name, fn)
    for a, b in (("plan", "validate"), ("validate", "approve_plan"), ("approve_plan", "invite"),
                 ("invite", "await_call"), ("await_call", "brief_score"), ("brief_score", "brief_ground"),
                 ("brief_repair", "brief_compute"), ("brief_compute", "review"), ("review", "finalise")):
        g.add_edge(a, b)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", after_intake, ["plan", "brief_score", END])
    g.add_conditional_edges("brief_ground", after_ground, ["brief_repair", "brief_compute"])
    g.add_edge("finalise", END)
    return g.compile(checkpointer=checkpointer)


# ── runtime ──────────────────────────────────────────────────────────────────

class Pipelines:
    graph = None
    _conn = None
    _lock = asyncio.Lock()

    @classmethod
    async def ensure(cls):
        async with cls._lock:
            if cls.graph is None:
                path = ROOT / "runs" / "alt" / "pipeline.sqlite"
                path.parent.mkdir(parents=True, exist_ok=True)
                cls._conn = await aiosqlite.connect(str(path))
                cls.graph = build(AsyncSqliteSaver(cls._conn))
        return cls.graph

    @staticmethod
    def cfg(pid: str) -> dict:
        return {"configurable": {"thread_id": pid}}

    @classmethod
    async def start(cls, candidate_id: str, role_id: str, persona_id: str, *, session_id: str | None = None,
                    auto_approve_plan: bool = True) -> str:
        g = await cls.ensure()
        pid = "p" + uuid.uuid4().hex[:9]
        store.add(PipelineRow(id=pid, candidate_id=candidate_id, role_id=role_id, persona_id=persona_id,
                              session_id=session_id, status="briefing" if session_id else "intake"))
        state = {"pipeline_id": pid, "candidate_id": candidate_id, "role_id": role_id, "persona_id": persona_id,
                 "auto_approve_plan": auto_approve_plan}
        if session_id:
            state["session_id"] = session_id
        await g.ainvoke(state, cls.cfg(pid))
        return pid

    @classmethod
    async def resume(cls, pid: str, value) -> dict:
        g = await cls.ensure()
        return await g.ainvoke(Command(resume=value), cls.cfg(pid))

    @classmethod
    async def state(cls, pid: str) -> dict:
        g = await cls.ensure()
        snap = await g.aget_state(cls.cfg(pid))
        waiting = [i.value for t in snap.tasks for i in t.interrupts]
        return {"values": snap.values, "next": list(snap.next), "waiting": waiting}

    @staticmethod
    def row(**where) -> PipelineRow | None:
        with Session(store.engine()) as s:
            q = select(PipelineRow)
            for k, v in where.items():
                q = q.where(getattr(PipelineRow, k) == v)
            return s.exec(q.order_by(PipelineRow.created_at.desc())).first()

    @staticmethod
    def rows(limit: int = 100) -> list[PipelineRow]:
        with Session(store.engine()) as s:
            return list(s.exec(select(PipelineRow).order_by(PipelineRow.created_at.desc()).limit(limit)))

    @classmethod
    async def call_ended(cls, token: str | None, session_id: str):
        """Called by the orchestrator when a call ends: resume `await_call`, which writes the brief."""
        row = cls.row(token=token) if token else None
        try:
            if row is None:
                return
            await cls.resume(row.id, {"session_id": session_id})
        except Exception:
            log.exception("pipeline resume failed for session %s", session_id)


# ── role pipeline (the role step of the question generator) ──────────────────

class RoleRow(SQLModel, table=True):
    """Index of role-pipeline threads (state in the checkpointer)."""
    id: str = Field(primary_key=True)
    role_id: str = Field(index=True)
    status: str = "intake"      # intake | input_error | review | approved | rejected | error
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class RState(TypedDict, total=False):
    thread_id: str
    role_id: str
    role: dict                  # role.md, parsed in code
    role_md: str
    input_issues: list[str]
    proposal: dict              # the model's raw proposal (kept for the recruiter and the trace)
    repaired: bool
    screen: dict                # the validated draft configuration
    issues: list[str]           # what validation dropped or repaired
    error: str
    decision: dict


def _role_update(tid: str, **fields):
    with Session(store.engine()) as s:
        row = s.get(RoleRow, tid)
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)
        row.updated_at = time.time()
        s.add(row)
        s.commit()


def role_intake(state: RState) -> dict:
    try:
        role, md = I.load_role(state["role_id"])
    except I.InputError as e:
        _role_update(state["thread_id"], status="input_error")
        return {"input_issues": e.issues}
    return {"role": role, "role_md": md}


async def role_compile(state: RState) -> dict:
    try:
        market = I.market(state["role"]["market"])
        return {"proposal": await S.propose(state["role_md"], market, run_id=state["thread_id"]), "error": ""}
    except Exception as e:
        log.error("role compile failed: %s", e)
        _role_update(state["thread_id"], status="error")
        return {"error": f"{type(e).__name__}: {e}"}


def role_validate(state: RState) -> dict:
    clean, issues = S.validate(state["proposal"], state["role"], state["role_md"], ContentStore().stages())
    if S.repair_needed(issues) and not state.get("repaired"):
        return {"issues": issues}                    # → role_repair, once
    screen = S.compose(state["role"], I.market(state["role"]["market"]), clean, state["role_md"])
    S.save_draft(state["role_id"], {**screen, "validation_issues": issues})
    return {"screen": screen, "issues": issues}


def after_role_validate(state: RState) -> str:
    return "role_repair" if S.repair_needed(state.get("issues") or []) and not state.get("repaired") else "role_review"


async def role_repair(state: RState) -> dict:
    """At most once: the rejected parts go back to the model with the reasons; the rest stays."""
    try:
        market = I.market(state["role"]["market"])
        fixed = await S.propose(state["role_md"], market, run_id=state["thread_id"], previous=state["proposal"],
                                fix=S.repair_needed(state["issues"]))
        return {"proposal": fixed, "repaired": True}
    except Exception as e:
        log.error("role repair failed: %s", e)
        return {"repaired": True}


def role_review(state: RState) -> dict:
    _role_update(state["thread_id"], status="review")
    answer = interrupt({"kind": "screen", "screen": state.get("screen"), "issues": state.get("issues")})
    return {"decision": {"decision": answer.get("decision"), "note": answer.get("note", ""),
                         "by": answer.get("by", "recruiter"), "t": time.time()}}


def role_publish(state: RState) -> dict:
    d = state["decision"]
    if d["decision"] == "approve":
        S.approve(state["role_id"], state["screen"], by=d["by"], note=d["note"])
        _role_update(state["thread_id"], status="approved")
    else:
        _role_update(state["thread_id"], status="rejected")
    return {}


def build_role(checkpointer=None):
    g = StateGraph(RState)
    for name, fn in (("role_intake", role_intake), ("role_compile", role_compile), ("role_validate", role_validate),
                     ("role_repair", role_repair), ("role_review", role_review), ("role_publish", role_publish)):
        g.add_node(name, fn)
    g.add_edge(START, "role_intake")
    g.add_conditional_edges("role_intake", lambda s: END if s.get("input_issues") else "role_compile",
                            ["role_compile", END])
    g.add_conditional_edges("role_compile", lambda s: END if s.get("error") else "role_validate",
                            ["role_validate", END])
    g.add_conditional_edges("role_validate", after_role_validate, ["role_repair", "role_review"])
    g.add_edge("role_repair", "role_validate")
    g.add_edge("role_review", "role_publish")
    g.add_edge("role_publish", END)
    return g.compile(checkpointer=checkpointer)


class Roles:
    graph = None
    _lock = asyncio.Lock()

    @classmethod
    async def ensure(cls):
        async with cls._lock:
            if cls.graph is None:
                await Pipelines.ensure()
                cls.graph = build_role(AsyncSqliteSaver(Pipelines._conn))
        return cls.graph

    @classmethod
    async def start(cls, role_id: str) -> str:
        g = await cls.ensure()
        tid = "r" + uuid.uuid4().hex[:9]
        store.add(RoleRow(id=tid, role_id=role_id))
        await g.ainvoke({"thread_id": tid, "role_id": role_id}, Pipelines.cfg(tid))
        return tid

    @classmethod
    async def resume(cls, tid: str, value) -> dict:
        g = await cls.ensure()
        return await g.ainvoke(Command(resume=value), Pipelines.cfg(tid))

    @classmethod
    async def state(cls, tid: str) -> dict:
        g = await cls.ensure()
        snap = await g.aget_state(Pipelines.cfg(tid))
        return {"values": snap.values, "next": list(snap.next)}

    @staticmethod
    def latest(role_id: str) -> RoleRow | None:
        with Session(store.engine()) as s:
            return s.exec(select(RoleRow).where(RoleRow.role_id == role_id)
                          .order_by(RoleRow.created_at.desc())).first()
