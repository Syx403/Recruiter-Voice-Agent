"""advance_stage and end_interview."""
from __future__ import annotations

import re

from backend.content_store import ContentStore
from backend.session import SessionState


def stage_block(state: SessionState, store: ContentStore) -> dict:
    """The stage block: goal, skill SOP, persona delivery notes and the stage's first card."""
    cfg = state.stage_cfg()
    skills = cfg.get("skill") or []
    skills = [skills] if isinstance(skills, str) else skills
    sops, notes = [], []
    for name in skills:
        sk = store.skill(name, state.content_hashes)
        if sk:
            sops.append(f"## Skill: {name}\n{sk['body']}")
        dn = store.delivery_notes(name, state.persona["id"], state.content_hashes)
        if dn:
            notes.append(dn)
    block = {"stage": state.stage, "goal": cfg["goal"], "target_minutes": round(cfg["target_s"] / 60, 1)}
    if sops:
        block["skill_sop"] = "\n\n".join(sops)
    if notes:
        block["delivery_notes"] = "\n\n".join(notes)
    block["next"] = state.card_payload()
    return block


def advance_stage(state: SessionState, store: ContentStore, to: str, override_reason: str | None = None) -> dict:
    order = state.stages["order"]
    if to not in order:
        return {"status": "refused", "reason": f"unknown stage {to!r} (card ids are not stages)",
                "allowed": [state.next_stage(), "close"], "next": state.repeat_instruction()}
    if to == state.stage:
        return {"status": "refused", "reason": f"already in {to}", "next": state.repeat_instruction()}
    clk = state.clock()
    nxt = state.next_stage()
    skip_ok = to == "close" or (clk["wrap_up"] and to == "candidate_questions")
    if to != nxt and not skip_ok:
        state.refusals.append({"t": round(state.elapsed, 1), "from": state.stage, "to": to, "reason": "order"})
        return {"status": "refused", "reason": f"stages run in order; the next stage is {nxt!r}",
                "allowed": [nxt, "close"], "next": state.repeat_instruction()}
    if state.stage == "consent" and state.consent_refused() and to != "close":
        state.refusals.append({"t": round(state.elapsed, 1), "from": state.stage, "to": to, "reason": "no consent"})
        return {"status": "refused", "reason": "consent was refused; only close is allowed", "allowed": ["close"]}
    ok, missing = state.exit_status()
    if not ok and to != "close":
        remaining_short = clk["remaining_s"] < state.stage_cfg()["target_s"] or clk["wrap_up"]
        declined_ok = override_reason and state.cards_exhausted()
        if override_reason and (remaining_short or declined_ok):
            state.overrides.append({"t": round(state.elapsed, 1), "from": state.stage, "to": to,
                                    "reason": override_reason, "missing": missing})
        else:
            state.refusals.append({"t": round(state.elapsed, 1), "from": state.stage, "to": to,
                                   "reason": "exit condition", "missing": missing})
            return {"status": "refused", "reason": "exit condition not met", "missing": missing,
                    "next": state.repeat_instruction()}
    elif to == "close" and state.stage != "close" and not ok:
        state.overrides.append({"t": round(state.elapsed, 1), "from": state.stage, "to": to,
                                "reason": override_reason or "close", "missing": missing})
    state.enter_stage(to)
    return {"status": "ok", **stage_block(state, store)}


CLOSING = re.compile(r"\b(recruiter|business days?|next steps?)\b", re.I)


def end_interview(state: SessionState, reason: str) -> dict:
    """Requires a closing line first. Code checks the agent actually said the next steps."""
    if state.stage != "close":
        state.refusals.append({"t": round(state.elapsed, 1), "from": state.stage, "to": "end", "reason": "not in close"})
        return {"status": "refused", "reason": "Call advance_stage(to='close') first, say the closing line, "
                                               "then call end_interview.", "next": state.repeat_instruction()}
    said = " ".join(t for _, st, t in state.agent_turns if st == "close")
    if not CLOSING.search(said) and not state.force_close:
        state.end_requested = True     # the orchestrator ends the call once the closing line has been spoken
        state.refusals.append({"t": round(state.elapsed, 1), "from": "close", "to": "end", "reason": "no closing line"})
        return {"status": "refused", "reason": "Say the closing line aloud first: thank the candidate and say a "
                                               "recruiter will review the screen and get back within two business "
                                               "days. Then call end_interview."}
    state.ended = True
    state.end_reason = reason
    return {"status": "ended"}
