"""FastAPI app (gpt-live-1 + LangGraph brain): recruiter console API, candidate call socket, live monitor:

    uv run --group alt uvicorn alt.app:app --port 8100

Sessions, invites and timelines go to runs/alt/screener.sqlite; recordings and event logs to runs/<id>/ (ids start
with "c"); spend goes to the shared ledger under the step "alt-*".
"""
from __future__ import annotations

from pathlib import Path

from backend import store

store.DB_PATH = store.ROOT / "runs" / "alt" / "screener.sqlite"     # before the first engine() call

import json  # noqa: E402

from alt.pipeline.graph import Pipelines, PipelineRow, RoleRow, Roles  # noqa: E402,F401  (registers the tables)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from alt import budget  # noqa: E402
from alt.brain.graph import mermaid  # noqa: E402
from alt.orchestrator import AltOrchestrator, voice_info  # noqa: E402
from alt.pipeline import inputs as I  # noqa: E402
from alt.pipeline import screen as S  # noqa: E402
from backend import export  # noqa: E402
from backend.content_store import ContentStore, candidate  # noqa: E402
from backend.orchestrator import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "alt" / "frontend" / "dist"

app = FastAPI(title="AE Voice Screener")
content = ContentStore()


class NewInvite(BaseModel):
    role_id: str | None = None
    persona_id: str | None = None
    candidate_id: str


@app.get("/api/config")
def config():
    cfg = load_config()
    return {"defaults": cfg["defaults"], "voice": budget.cfg()["voice"], "voice_info": voice_info(budget.cfg()),
            "roles": _roles(), "personas": content.list("personas"),
            "candidates": _candidates(),
            "budget": {"total": budget.cfg()["budget"]["alt_cap_usd"], "spent": round(budget.spent(), 4)}}


@app.get("/api/graph")
def graph():
    return {"mermaid": mermaid()}


@app.get("/api/content/{kind}")
def list_content(kind: str):
    if kind not in ("core", "personas", "roles", "skills"):
        raise HTTPException(404)
    return content.list(kind)


def _roles() -> list[dict]:
    """Roles from role.md, with the state of their screen configuration."""
    out = []
    for rid in I.list_roles():
        try:
            r, _ = I.load_role(rid)
        except I.InputError as e:
            out.append({"id": rid, "title": rid, "status": "input_error", "issues": e.issues})
            continue
        status = "approved" if S.approved(rid) else "draft" if S.draft(rid) else "none"
        row = Roles.latest(rid)
        if row is not None and row.status == "review":
            status = "review"
        out.append({"id": rid, "title": r["title"], "company": r["company"], "location": r["location"],
                    "market": r["market"], "status": status, "thread": row.id if row else None})
    return out


def _candidates() -> list[dict]:
    out = []
    for cid in I.list_candidates():
        try:
            c, _ = I.load_candidate(cid)
            out.append({"id": cid, "name": c["name"], "headline": c["headline"], "location": c["location"]})
        except I.InputError as e:
            out.append({"id": cid, "name": cid, "issues": e.issues})
    return out


def _role_title(role_id: str) -> str:
    sc = S.current(role_id, allow_draft=True)
    return sc["title"] if sc else content.role(role_id).get("title")


def _name(candidate_id: str) -> str:
    if I.candidate_path(candidate_id).exists():
        try:
            return I.load_candidate(candidate_id)[0]["name"]
        except I.InputError:
            return candidate_id
    try:
        return candidate(candidate_id).get("name", candidate_id).replace(" (synthetic)", "")
    except FileNotFoundError:
        return candidate_id


def _invite_view(i: store.Invite) -> dict:
    o = AltOrchestrator.registry.get(i.session_id or "")
    status = "in_call" if o is not None and o.status == "live" else i.status
    return {**i.model_dump(), "status": status, "url": f"/interview/{i.token}",
            "candidate": _name(i.candidate_id), "role": _role_title(i.role_id)}


@app.post("/api/invites")
async def create_invite(body: NewInvite):
    """Starts a candidate pipeline: intake (candidate.md) → plan (CV × JD) → validate → plan review ⏸. The interview
    link is issued only after the recruiter approves the plan (POST /api/pipelines/{pid}/plan). The role needs an
    approved screen configuration first."""
    d = load_config()["defaults"]
    role, persona = body.role_id or d["role"], body.persona_id or d["persona"]
    if S.approved(role) is None:
        raise HTTPException(409, "approve this role's screen configuration first")
    for row in Pipelines.rows():                       # reuse an open pipeline for the same candidate and role
        if row.candidate_id == body.candidate_id and row.role_id == role and row.persona_id == persona \
                and row.status in ("plan_review", "invited"):
            return await pipeline(row.id)
    pid = await Pipelines.start(body.candidate_id, role, persona, auto_approve_plan=False)
    return await pipeline(pid)


# ── roles: the role step of the question generator ──────────────────────────

@app.get("/api/roles")
def roles():
    return _roles()


@app.get("/api/roles/{rid}")
async def role_detail(rid: str):
    if not I.role_path(rid).exists():
        raise HTTPException(404)
    text = I.role_path(rid).read_text()
    try:
        parsed, issues = I.parse_role(text), []
    except I.InputError as e:
        parsed, issues = None, e.issues
    row = Roles.latest(rid)
    st = await Roles.state(row.id) if row else {"values": {}, "next": []}
    return {"id": rid, "role_md": text, "parsed": parsed, "input_issues": issues or st["values"].get("input_issues"),
            "approved": S.approved(rid), "draft": st["values"].get("screen") or S.draft(rid),
            "issues": st["values"].get("issues"), "error": st["values"].get("error"),
            "thread": row.id if row else None, "status": row.status if row else "none", "next": st["next"]}


@app.post("/api/roles/{rid}/compile")
async def role_compile(rid: str):
    if not I.role_path(rid).exists():
        raise HTTPException(404)
    await Roles.start(rid)
    return await role_detail(rid)


class RoleDecision(BaseModel):
    decision: str            # approve | reject
    note: str = ""


@app.post("/api/roles/{rid}/decision")
async def role_decision(rid: str, body: RoleDecision):
    if body.decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be approve or reject")
    row = Roles.latest(rid)
    if row is None or "role_review" not in (await Roles.state(row.id))["next"]:
        raise HTTPException(409, "this role is not waiting for a decision")
    await Roles.resume(row.id, {**body.model_dump(), "by": "recruiter"})
    return await role_detail(rid)


def _pipeline_view(row) -> dict:
    inv = store.get_invite(row.token) if row.token else None
    return {"id": row.id, "candidate_id": row.candidate_id, "candidate": _name(row.candidate_id),
            "role_id": row.role_id, "role": _role_title(row.role_id), "persona_id": row.persona_id,
            "status": row.status, "token": row.token, "url": f"/interview/{row.token}" if row.token else None,
            "invite_status": inv.status if inv else None, "session_id": row.session_id, "updated_at": row.updated_at}


@app.get("/api/pipelines")
def pipelines():
    return [_pipeline_view(r) for r in Pipelines.rows()]


@app.get("/api/pipelines/{pid}")
async def pipeline(pid: str):
    row = Pipelines.row(id=pid)
    if row is None:
        raise HTTPException(404)
    st = await Pipelines.state(pid)
    v = st["values"]
    return {**_pipeline_view(row), "next": st["next"], "waiting": st["waiting"],
            **{k: v.get(k) for k in ("parsed", "plan", "plan_issues", "input_issues", "brief", "computed",
                                     "brief_error", "decision", "unmatched")}}


class PlanDecision(BaseModel):
    plan: dict | None = None     # optional recruiter edits: {"cards": [...], "extra_probes": [...]}


class Decision(BaseModel):
    decision: str            # approve | reject | verify
    note: str = ""


@app.post("/api/pipelines/{pid}/decision")
async def decide(pid: str, body: Decision):
    if body.decision not in ("approve", "reject", "verify"):
        raise HTTPException(400, "decision must be approve, reject or verify")
    st = await Pipelines.state(pid)
    if "review" not in st["next"]:
        raise HTTPException(409, "this pipeline is not waiting for a recruiter decision")
    await Pipelines.resume(pid, body.model_dump())
    return await pipeline(pid)


@app.post("/api/pipelines/{pid}/cancel")
async def cancel_pipeline(pid: str):
    """The recruiter withdraws an invitation before the call: the plan is dropped and an issued link stops working."""
    row = Pipelines.row(id=pid)
    if row is None:
        raise HTTPException(404)
    if row.status not in ("plan_review", "invited", "intake", "input_error"):
        raise HTTPException(409, "only an interview that has not started can be cancelled")
    if row.token:
        inv = store.get_invite(row.token)
        if inv is not None and inv.status != "invited":
            raise HTTPException(409, "the candidate has already joined")
        store.update_invite(row.token, status="cancelled")
    from alt.pipeline.graph import _row_update
    _row_update(pid, status="cancelled")
    return await pipeline(pid)


@app.post("/api/pipelines/{pid}/plan")
async def approve_plan(pid: str, body: PlanDecision | None = None):
    """The recruiter approves the question plan (optionally edited; edits are validated again); the interview link
    is issued."""
    st = await Pipelines.state(pid)
    if "approve_plan" not in st["next"]:
        raise HTTPException(409, "this pipeline is not waiting for plan approval")
    await Pipelines.resume(pid, {"approved": True, **({"plan": body.plan} if body and body.plan else {})})
    return await pipeline(pid)


@app.post("/api/sessions/{sid}/brief")
async def brief_for_session(sid: str):
    """Write a brief for a call that had no pipeline (e.g. older sessions): a thread that starts after the call."""
    if (row := Pipelines.row(session_id=sid)) is not None:
        return await pipeline(row.id)
    s = store.get_session(sid)
    if s is None or s.status != "ended":
        raise HTTPException(409, "the call has not ended")
    pid = await Pipelines.start(s.candidate_id, s.role_id, s.persona_id, session_id=sid)
    return await pipeline(pid)


@app.get("/api/graph/pipeline")
async def pipeline_graph():
    g = await Pipelines.ensure()
    return {"mermaid": g.get_graph().draw_mermaid()}


@app.get("/api/invites")
def invites():
    return [_invite_view(i) for i in store.list_invites()]


@app.get("/api/interview/{token}")
def interview_info(token: str):
    i = store.get_invite(token)
    if i is None:
        raise HTTPException(404)
    role = S.current(i.role_id, allow_draft=True) or content.role(i.role_id)
    persona = content.persona(i.persona_id)
    return {"status": i.status, "role": role.get("title"), "company": role.get("company"),
            "interviewer": persona.get("name"), "candidate": _name(i.candidate_id).split(" ")[0]}


@app.get("/api/sessions")
def sessions():
    out = []
    for s in store.list_sessions(200):
        if s.status == "created" and s.id not in AltOrchestrator.registry:
            continue
        d = s.model_dump()
        summ = d.pop("summary") or {}
        d.update(names=summ.get("names", {}), elapsed_s=summ.get("elapsed_s"))
        bp = ROOT / "runs" / s.id / "brief.json"          # the brief's outcome, for the recruiter's list
        if bp.exists():
            c = json.loads(bp.read_text()).get("computed") or {}
            d.update(recommendation=c.get("recommendation"), total_25=c.get("total_25"))
        out.append(d)
    return out


def _record(sid: str) -> dict:
    o = AltOrchestrator.registry.get(sid)
    live = o.snapshot() if o is not None and o.status == "live" else None
    try:
        r = export.record(sid, live)
    except KeyError:
        raise HTTPException(404)
    sess = store.get_session(sid)
    summ = (sess.summary if sess else {}) or {}
    r["turns"] = (live or {}).get("turns") or summ.get("turns") or []
    r["latency"] = summ.get("latency")
    r["cost_split"] = (live or {}).get("cost_split") or summ.get("cost_split")
    r["layout"] = (live or {}).get("layout") or summ.get("layout")
    inv = next((i for i in store.list_invites() if i.session_id == sid), None)
    row = Pipelines.row(session_id=sid) or (Pipelines.row(token=inv.token) if inv else None)
    r["pipeline"] = _pipeline_view(row) if row else None
    return r


@app.get("/api/sessions/{sid}")
async def session(sid: str):
    r = _record(sid)
    if r["pipeline"]:
        st = await Pipelines.state(r["pipeline"]["id"])
        v = st["values"]
        r["pipeline"].update(next=st["next"], **{k: v.get(k) for k in ("plan", "brief", "computed", "brief_error",
                                                                      "decision", "unmatched")})
    return r


@app.get("/api/sessions/{sid}/export")
def export_session(sid: str, format: str = "md"):
    r = _record(sid)
    name = f"screen_{r['session']['candidate_id']}_{sid}"
    if format == "json":
        return Response(json.dumps(r, indent=2, ensure_ascii=False, default=str), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
    return PlainTextResponse(export.to_markdown(r), media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="{name}.md"'})


@app.get("/api/sessions/{sid}/recording")
def recording(sid: str):
    path = ROOT / "runs" / sid / "recording.wav"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.websocket("/ws/interview/{token}")
async def interview(ws: WebSocket, token: str):
    await ws.accept()
    i = store.get_invite(token)
    if i is None or i.status != "invited":
        await ws.send_json({"type": "error", "error": "This interview link is no longer valid."
                            if i is not None else "Interview link not found."})
        await ws.close()
        return
    try:
        budget.require(budget.cfg()["budget"]["session_cap_usd"], "new session")
    except budget.BudgetExceeded as e:
        await ws.send_json({"type": "error", "error": str(e)})
        await ws.close()
        return
    row = Pipelines.row(token=token)
    plan = (await Pipelines.state(row.id))["values"].get("plan") if row else None
    o = AltOrchestrator(i.role_id, i.persona_id, i.candidate_id, invite=token, plan=plan)
    store.update_invite(token, status="in_call", session_id=o.state.id)
    await o.run_call(ws)
    try:
        await ws.close()
    except RuntimeError:
        pass


@app.websocket("/ws/monitor/{sid}")
async def monitor(ws: WebSocket, sid: str):
    o = AltOrchestrator.registry.get(sid)
    await ws.accept()
    if o is None:
        await ws.send_json({"type": "error", "error": "session not found"})
        await ws.close()
        return
    o.add_monitor(ws)
    await o.push_state()
    try:
        while True:
            msg = json.loads(await ws.receive_text() or "{}")
            if msg.get("type") == "mirror":
                o.set_mirror(ws, bool(msg.get("on")))
    except WebSocketDisconnect:
        o.remove_monitor(ws)
    except (ValueError, RuntimeError):
        o.remove_monitor(ws)


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/interview/{token}")
    def candidate_page(token: str):
        return FileResponse(DIST / "interview.html")

    @app.get("/{path:path}")
    def spa(path: str):
        return FileResponse(DIST / "index.html")
