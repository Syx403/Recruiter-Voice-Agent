"""The hiring-manager brief: the LLM scores with quotes, code grounds the quotes and computes the result.

score (Claude Haiku 5.5, strict JSON) → ground (code: every quote must be the candidate's words) → repair (at most
once: unmatched quotes go back to the LLM) → compute (code: weighted total out of 25, caps, recommendation).
The LLM never computes the total or the recommendation.
"""
from __future__ import annotations

import json

from alt import budget
from alt.brain import llm
from backend import costs
from backend.tools.evidence import _share, _stems

MODEL = "claude-haiku-5-5"
COMPETENCIES = ["deal_execution", "performance_credibility", "technical_credibility", "outcome_orientation",
                "ownership_learning"]
FIT = ["must_haves", "motion", "segment_and_cycle", "territory_and_languages", "ote_expectation", "notice_period",
       "work_authorisation"]

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "competencies", "fit_checks", "key_evidence", "risks", "next_round_focus",
                 "missing_information"],
    "properties": {
        "summary": {"type": "string"},
        "competencies": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "score", "confidence", "evidence", "rationale"],
            "properties": {
                "id": {"type": "string", "enum": COMPETENCIES},
                "score": {"type": "integer", "description": "1-5, or 0 when not observed"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"}}}},
        "fit_checks": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "status", "note"],
            "properties": {"id": {"type": "string", "enum": FIT},
                           "status": {"type": "string", "enum": ["pass", "flag", "unknown"]},
                           "note": {"type": "string"}}}},
        "key_evidence": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "next_round_focus": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM = """You write the hiring-manager brief after a structured first-round voice screen for an Account Executive.
You score; code computes the total and the recommendation, so never state a total or a recommendation.

Rules:
- Score each competency 1-5 against its anchors, using only the transcript and the evidence ledger. If there is no
  evidence for a competency, score 0 (not observed) — never a middle score by default.
- Every competency with a score above 0 needs 1-3 evidence quotes copied exactly from the CANDIDATE's words
  (not the interviewer's), each at most 25 words.
- outcome_orientation needs evidence of the CUSTOMER's business result (cost, downtime, release speed, risk, revenue).
  Contract value and deal size are the seller's numbers, not outcomes: without customer-outcome evidence score 0.
- Fit checks compare the candidate's facts with the role's screen configuration: pass, flag (with the reason), or
  unknown. must_haves: the role's [must] requirements, using the pre-call CV assessment and what the candidate said
  (requirement_1/requirement_2 answer the role's requirement questions); flag names the requirement that is not
  met, unknown when neither the CV nor the screen shows it. A range marked "inferred" is not the employer's: never
  flag on it alone. work_authorisation covers location and arrangement (and eligibility only where it was asked).
- CV vs screen: rows marked "differs" are code facts. Say which figure differs in risks or next_round_focus,
  neutrally (a unit or window difference is common); never call it dishonesty.
- key_evidence: 2-3 exact candidate quotes that matter most. risks: 1-2. next_round_focus: what the hiring manager
  should verify. missing_information: facts the screen did not obtain.
- Quota consistency results, flags and declined fields are facts from code: use them. If the quota check was
  inconsistent and the clarification did not resolve it, say so in the rationale.
- Never infer protected attributes. Neutral, specific wording. summary: one sentence for the hiring manager."""


def candidate_text(timeline: list[dict]) -> str:
    return " ".join(e.get("text", "") for e in timeline if e.get("kind") == "candidate")


def build_input(*, role: dict, rubric: dict, ledger: list[dict], flags: list[dict], consistency: list[dict],
                timeline: list[dict], overrides: list[dict], requirement_fit: list[dict] | None = None,
                cv_check: list[dict] | None = None) -> str:
    comps = rubric["competencies"]
    weights = role.get("competency_weights") or {k: v["default_weight"] for k, v in comps.items()}
    anchors = "\n".join(f"- {k} (weight {weights.get(k)}): 1 = {v['anchors'][1]} 3 = {v['anchors'][3]} "
                        f"5 = {v['anchors'][5]}" for k, v in comps.items())
    fit = json.dumps(role.get("fit", {}), ensure_ascii=False)
    led = "\n".join(f"- [{e.get('stage')}] {e['field']} = {e['value']}{' (declined)' if e.get('declined') else ''}"
                    f" — \"{e.get('quote', '')}\"" for e in ledger)
    conv = "\n".join(f"[{e.get('t', 0):.0f}s] {'Interviewer' if e['kind'] == 'agent' else 'Candidate'}: {e.get('text', '')}"
                     for e in sorted(timeline, key=lambda e: e.get("t", 0))
                     if e.get("kind") in ("agent", "candidate") and e.get("text"))
    reqs = {r["id"]: r for r in role.get("requirements") or []}
    pre = {r["requirement_id"]: r for r in requirement_fit or []}
    req_lines = "\n".join(
        f"- {rid} ({r.get('check')}): {r.get('text')} | look for: {r.get('look_for', '')} | CV before the call: "
        f"{pre.get(rid, {}).get('status', 'n/a')} — {pre.get(rid, {}).get('note', '')}" for rid, r in reqs.items())
    cv_lines = "\n".join(f"- {c['claim']} {c['field']}: CV \"{c['cv']}\" | said: {c.get('said')} | {c['status']}"
                          for c in cv_check or [])
    return "\n\n".join([
        f"# Role\n{role.get('title')} at {role.get('company')} ({role.get('location')}). Fit targets: {fit}",
        f"# Must-have requirements (role.md) and the CV assessment before the call\n{req_lines or '(none)'}",
        f"# CV claims vs what the candidate said (code)\n{cv_lines or '(no comparable claims)'}",
        f"# Competency anchors\n{anchors}",
        f"# Evidence ledger (code-checked)\n{led or '(empty)'}",
        f"# Quota consistency (code)\n{json.dumps(consistency, ensure_ascii=False) or '[]'}",
        f"# Flags\n{json.dumps([{k: f[k] for k in ('type', 'detail', 'source') if k in f} for f in flags], ensure_ascii=False)}",
        f"# Stage overrides (cards exhausted)\n{json.dumps(overrides, ensure_ascii=False)}",
        f"# Transcript\n{conv}"])


async def call(messages: list[dict], run_id: str, step: str = "alt-dev") -> dict:
    budget.require(0.02, "brief")
    msg = await llm.client().messages.create(
        model=MODEL, max_tokens=2500, thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}], messages=messages)
    usage = msg.usage.model_dump() if msg.usage else {}
    costs.log("anthropic", MODEL, costs.anthropic_usd(MODEL, usage), usage, run_id, step, "brief")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if msg.stop_reason != "end_turn" or not text:
        raise ValueError(f"brief stopped with {msg.stop_reason}")
    return json.loads(text)


def grounded(quote: str, said: str) -> bool:
    q = _stems(quote)
    return bool(q) and _share(q, _stems(said)) >= 0.8


def ground(draft: dict, said: str) -> tuple[dict, list[str]]:
    """Code: keep only quotes that are the candidate's words. Returns the cleaned brief and the unmatched quotes."""
    out = json.loads(json.dumps(draft))
    unmatched: list[str] = []
    for c in out["competencies"]:
        keep = [q for q in c["evidence"] if grounded(q, said)]
        unmatched += [q for q in c["evidence"] if q not in keep]
        c["evidence"] = keep
    keep = [q for q in out["key_evidence"] if grounded(q, said)]
    unmatched += [q for q in out["key_evidence"] if q not in keep]
    out["key_evidence"] = keep
    return out, unmatched


def compute(brief: dict, role: dict, rubric: dict, consistency: list[dict], ledger: list[dict],
            cv_check: list[dict] | None = None, requirement_fit: list[dict] | None = None) -> dict:
    """Code: weighted total out of 25 over scored competencies, caps, recommendation. Knockouts are
    listed for the recruiter, never applied automatically."""
    comps = rubric["competencies"]
    weights = role.get("competency_weights") or {k: v["default_weight"] for k, v in comps.items()}
    notes: list[str] = []
    scores: dict[str, int | None] = {}
    for c in brief["competencies"]:
        s = c["score"] if 1 <= c["score"] <= 5 else None
        if s is not None and not c["evidence"]:
            notes.append(f"{c['id']}: score {s} had no grounded quote, so it is treated as not observed")
            s = None
        scores[c["id"]] = s
    inconsistent = any(x.get("status") == "inconsistent" for x in consistency)
    if inconsistent and (scores.get("performance_credibility") or 0) > 3:
        scores["performance_credibility"] = 3
        notes.append("performance_credibility capped at 3: the quota check was inconsistent")
    scored = {k: v for k, v in scores.items() if v is not None and k in weights}
    total = round(5 * sum(weights[k] * v for k, v in scored.items()) / sum(weights[k] for k in scored), 1) \
        if scored else None
    nulls = [k for k in COMPETENCIES if scores.get(k) is None]
    if len(nulls) >= rubric["recommendation"]["needs_verification_if_null_count"] or \
            any(k in nulls for k in rubric["recommendation"]["needs_verification_if_null"]):
        rec = "needs_verification"
    elif total is not None and total >= rubric["recommendation"]["recommend_min"]:
        rec = "recommend"
    elif total is not None and total >= rubric["recommendation"]["conditional_min"]:
        rec = "conditional"
    else:
        rec = "not_recommended"
    fields = {e["field"] for e in ledger if not e.get("declined")}
    knockouts = [k["id"] for k in role.get("knockouts", [])
                 if k["id"] == "no_quota_history" and "quota" not in fields]
    return {"scores": scores, "total_25": total, "recommendation": rec, "not_observed": nulls,
            "knockouts_to_review": knockouts, "notes": notes, "weights": weights,
            "cv_check": cv_check or [], "cv_differs": [c for c in cv_check or [] if c["status"] == "differs"],
            "requirement_fit": requirement_fit or []}
