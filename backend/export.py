"""Session record and export (Markdown for people, JSON for downstream processing)."""
from __future__ import annotations

import json
import time

from backend import store
from backend.content_store import ContentStore, candidate

AGENT, CAND = "Interviewer", "Candidate"


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def heard_text(e: dict) -> str:
    """For an interrupted interviewer turn, the part the candidate heard (proportional to audio played)."""
    text = e.get("text", "")
    if not e.get("interrupted") or not e.get("audio_ms"):
        return text
    cut = int(len(text) * min(1.0, (e.get("heard_ms") or 0) / e["audio_ms"]))
    return text[:cut].rsplit(" ", 1)[0] if cut < len(text) else text


def record(session_id: str, live: dict | None = None) -> dict:
    """Everything about one session: metadata, timeline, evidence, flags, checks. `live` overlays the
    in-memory state of a session that is still running."""
    row = store.get_session(session_id)
    if row is None:
        raise KeyError(session_id)
    summ = row.summary or {}
    out = {
        "session": {k: v for k, v in row.model_dump().items() if k != "summary"},
        "names": summ.get("names", {}), "voice": summ.get("voice", {}),
        "elapsed_s": summ.get("elapsed_s"), "stage_log": summ.get("stage_log", []),
        "order": summ.get("order", []), "stage_targets": summ.get("stage_targets", {}),
        "cap_usd": summ.get("cap_usd"), "recording": summ.get("recording", False),
        "timeline": store.timeline(session_id), "evidence": store.evidence(session_id),
        "flags": store.flags(session_id), "consistency": summ.get("consistency", []),
        "overrides": summ.get("overrides", []), "refusals": summ.get("refusals", []),
        "rejections": summ.get("rejections", []), "card_log": summ.get("card_log", []),
    }
    out["candidate"] = candidate_profile(row.candidate_id)
    out["role"] = role_profile(row.role_id)
    if live:
        out.update(evidence=live["ledger"], flags=live["flags"], consistency=live["consistency"],
                   overrides=live["overrides"], refusals=live["refusals"], rejections=live.get("rejections", []),
                   stage_log=live["stage_log"], order=live["order"], stage_targets=live["stage_targets"],
                   elapsed_s=live["clock"]["elapsed_s"], cap_usd=live["cap_usd"], names=live["names"],
                   voice=live["voice"])
        out["session"]["cost_usd"] = live["cost_usd"]
        out["session"]["status"] = live["status"]
    return out


def candidate_profile(candidate_id: str) -> dict:
    try:
        c = candidate(candidate_id)
    except FileNotFoundError:
        return {"id": candidate_id}
    return {"id": candidate_id, "name": c.get("name", candidate_id), "cv_summary": (c.get("cv_summary") or "").strip(),
            "locale": c.get("locale"), "synthetic": True}


def role_profile(role_id: str) -> dict:
    try:
        r = ContentStore().role(role_id)
    except FileNotFoundError:
        return {"id": role_id}
    return {"id": role_id, "title": r.get("title"), "company": r.get("company"), "location": r.get("location"),
            "context": r.get("company_context")}


def to_markdown(r: dict) -> str:
    s, n, v = r["session"], r["names"], r["voice"]
    started = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("started_at") or s["created_at"]))
    iv, tr = v.get("interviewer", {}), v.get("transcriber", {})
    lines = [
        f"# Screening call: {n.get('candidate', s['candidate_id'])}",
        "",
        f"- **Role:** {n.get('role', s['role_id'])} ({n.get('company', '')}, {r.get('role', {}).get('location', '')})",
        f"- **Candidate background (synthetic):** {r.get('candidate', {}).get('cv_summary', '')}",
        f"- **Interviewer persona:** {n.get('persona', s['persona_id'])}",
        f"- **Session:** `{s['id']}`, started {started}, duration {mmss(r.get('elapsed_s') or 0)}, "
        f"status {s['status']} ({s.get('end_reason') or '—'})",
        f"- **Interviewer model:** {iv.get('model')} (reasoning {iv.get('reasoning_effort')}, voice {iv.get('voice')})",
        f"- **Transcription:** {tr.get('model')} (language {tr.get('language')})",
        f"- **Cost:** USD {s.get('cost_usd', 0):.3f}",
        "",
        "> Candidate lines are the transcription model's text, not what the interviewer model heard.",
        "",
        "## Transcript",
        "",
    ]
    for e in r["timeline"]:
        t = mmss(e["t"])
        if e["kind"] == "stage":
            lines += [f"### {e.get('stage', '').replace('_', ' ').title()}  ({t})", ""]
        elif e["kind"] == "agent":
            txt = heard_text(e)
            tag = " *(interrupted)*" if e.get("interrupted") else ""
            lines += [f"**{AGENT}** [{t}]: {txt}{tag}", ""]
        elif e["kind"] == "candidate":
            tag = f" *({e['intent'].replace('_', ' ')})*" if e.get("intent") not in (None, "answer") else ""
            lines += [f"**{CAND}** [{t}]: {e.get('text', '')}{tag}", ""]
        elif e["kind"] == "note":
            lines += [f"> *Steering ({e.get('intent', '')}):* {e.get('text', '').split(': ', 1)[-1]}", ""]
        elif e["kind"] == "tool":
            out = e.get("output") or {}
            lines += [f"> `{e.get('name')}` {json.dumps(e.get('arguments'), ensure_ascii=False)} → "
                      f"{out.get('status', '')}", ""]
    lines += ["## Evidence ledger", "", "| Time | Dimension | Field | Value | Quote |", "|---|---|---|---|---|"]
    for ev in r["evidence"]:
        val = "declined" if ev.get("declined") else ev.get("value", "")
        lines.append(f"| {mmss(ev['t'])} | {ev['dimension']} | {ev['field']} | {val} | {ev.get('quote', '')} |")
    lines += ["", "## Flags", ""]
    lines += [f"- [{mmss(f['t'])}] **{f['type']}** ({f.get('source')}): {f.get('detail')}" for f in r["flags"]] or ["- None"]
    lines += ["", "## Quota consistency", ""]
    for c in r["consistency"]:
        lines.append(f"- **{c.get('status')}**: {c.get('readback', '')} "
                     f"(claimed USD {c.get('claimed_closed_usd', '—')}, implied USD {c.get('implied_closed_usd', '—')}, "
                     f"ratio {c.get('ratio', '—')})")
    if not r["consistency"]:
        lines.append("- Not run")
    lines += ["", "## Stage discipline", "",
              f"- Overrides: {len(r['overrides'])}", f"- Refused transitions: {len(r['refusals'])}",
              f"- Rejected evidence: {len(r['rejections'])}", ""]
    return "\n".join(lines)
