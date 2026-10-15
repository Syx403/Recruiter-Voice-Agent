"""Tool declarations (identical on every provider) and the dispatcher.

The dispatcher runs every function call of one model response, then attaches the next instruction
(chosen by the card engine), any monitor correction and the clock to the last result. Computing the
instruction once per batch keeps probe counts right when the model records several fields at once.
"""
from __future__ import annotations

import json
from typing import Any

from backend.content_store import ContentStore
from backend.session import SessionState
from backend.tools.concerns import flag_concern
from backend.tools.consistency import check_quota_consistency
from backend.tools.coverage import get_coverage
from backend.tools.evidence import record_evidence
from backend.tools.stages import advance_stage, end_interview

_AMOUNT = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "description": "The number as said, e.g. 1.2 for '1.2 crore'"},
        "unit": {"type": "string", "enum": ["none", "k", "M", "lakh", "crore", "B"]},
        "currency": {"type": "string", "description": "ISO code as said or implied, e.g. INR, USD, SGD"},
    },
    "required": ["amount", "unit", "currency"],
}


def declarations(stages: dict, rubric: dict) -> list[dict]:
    dims = list(rubric["competencies"]) + list(rubric["fit_checks"]) + ["context"]
    return [
        {"name": "advance_stage",
         "description": "Move to another interview stage. Code checks the exit condition. On success the "
                        "result holds the stage goal, its SOP and the first question card.",
         "parameters": {"type": "object", "properties": {
             "to": {"type": "string", "enum": stages["order"]},
             "override_reason": {"type": "string", "description": "Only when told to (time, or declined)."}},
             "required": ["to"]}},
        {"name": "record_evidence",
         "description": "Record one piece of evidence the candidate gave, once per field, after each "
                        "substantive answer. The result says what to ask next.",
         "parameters": {"type": "object", "properties": {
             "dimension": {"type": "string", "enum": dims},
             "field": {"type": "string", "enum": list(stages["fields"])},
             "value": {"type": "string", "description": "The value, keeping units and currency, or 'declined'."},
             "quote": {"type": "string", "description": "The candidate's words as you heard them."},
             "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
             "required": ["dimension", "field", "value", "quote", "confidence"]}},
        {"name": "check_quota_consistency",
         "description": "Deterministic check that quota × attainment matches average deal × deals closed "
                        "for one period. Returns normalised values to read back.",
         "parameters": {"type": "object", "properties": {
             "quota": _AMOUNT, "period_months": {"type": "number"},
             "attainment_pct": {"type": "number", "description": "0-100+ scale"},
             "avg_deal": _AMOUNT, "deals_closed": {"type": "number"},
             "self_sourced_pct": {"type": "number"}, "team_avg_pct": {"type": "number"}},
             "required": ["quota", "period_months", "attainment_pct", "avg_deal", "deals_closed"]}},
        {"name": "get_coverage",
         "description": "Covered and missing competencies and fit checks, and the time left.",
         "parameters": {"type": "object", "properties": {}}},
        {"name": "flag_concern",
         "description": "Record a red flag for the recruiter (never mention it to the candidate).",
         "parameters": {"type": "object", "properties": {
             "type": {"type": "string", "enum": ["inconsistent_numbers", "account_management_as_new_logo",
                                                 "over_promising", "blame", "evasive", "other"]},
             "detail": {"type": "string"}}, "required": ["type", "detail"]}},
        {"name": "end_interview",
         "description": "End the call. Only after your closing line.",
         "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
    ]


def _unit_none(d: Any) -> Any:
    if isinstance(d, dict) and d.get("unit") == "none":
        return {**d, "unit": None}
    return d


def run_one(state: SessionState, store: ContentStore, name: str, args: dict) -> dict:
    try:
        if name == "advance_stage":
            return advance_stage(state, store, args.get("to", ""), args.get("override_reason"))
        if name == "record_evidence":
            return record_evidence(state, args.get("dimension", ""), args.get("field", ""),
                                   args.get("value", ""), args.get("quote", ""), args.get("confidence", "medium"))
        if name == "check_quota_consistency":
            return check_quota_consistency(
                state, _unit_none(args.get("quota")), args.get("period_months"), args.get("attainment_pct"),
                _unit_none(args.get("avg_deal")), args.get("deals_closed"), args.get("self_sourced_pct"),
                args.get("team_avg_pct"))
        if name == "get_coverage":
            return get_coverage(state)
        if name == "flag_concern":
            return flag_concern(state, args.get("type", "other"), args.get("detail", ""))
        if name == "end_interview":
            return end_interview(state, args.get("reason", ""))
        return {"status": "error", "reason": f"unknown tool {name}"}
    except Exception as e:  # a tool bug must not kill the call; the error is logged by the orchestrator
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}


def run_batch(state: SessionState, store: ContentStore, calls: list[dict]) -> list[tuple[str, dict]]:
    """calls: [{call_id, name, arguments(json str)}] → [(call_id, output)]."""
    outputs: list[tuple[str, dict]] = []
    needs_next = False
    advanced = False
    accepted = 0
    before = set(state.captured())
    last = state.last_instruction or {}
    probed = set(last.get("capture", [])) if last.get("type") == "probe" else set()   # e.g. the role-play follow-up
    for c in calls:
        try:
            args = json.loads(c.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        out = run_one(state, store, c["name"], args)
        outputs.append((c["call_id"], out))
        if c["name"] in ("record_evidence", "check_quota_consistency"):
            needs_next = True
            if c["name"] == "check_quota_consistency":
                accepted += out.get("status") in ("consistent", "inconsistent")
            elif out.get("status") == "recorded":
                # Re-recording a captured field is not progress, unless it answers the probe just issued.
                accepted += out.get("field") not in before or out.get("field") in probed
        if c["name"] == "advance_stage" and out.get("status") == "ok":
            advanced = True      # the stage block already carries the first card
    if outputs and not state.ended:
        last = outputs[-1][1]
        if needs_next and not advanced:
            # Nothing new accepted (rejected, insufficient, or a field already captured): repeat the
            # instruction, so such a call neither uses up a probe nor exhausts a card.
            last["next"] = state.next_instruction() if accepted else state.repeat_instruction()
        if state.corrections:
            last["correction"] = " ".join(state.corrections)
            state.corrections.clear()
        last["clock"] = state.clock()
    return outputs
