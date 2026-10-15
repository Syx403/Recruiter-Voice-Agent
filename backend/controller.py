"""Turn controller: one decision per complete candidate turn.

The voice model only listens and speaks. After each complete candidate turn:
1. `decide` (Claude Haiku 5.5) classifies the turn and extracts evidence for the current question, with the
   candidate's words as quotes;
2. `plan` (code) records the evidence, advances cards and stages, runs the quota check, applies the
   strict non-cooperation rules, and returns the directive for the interviewer's next turn;
3. the orchestrator sends the directive as per-response instructions and speaks it.
`plan` is deterministic and testable with text only.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import anthropic

from backend import costs
from backend.keystore import get_secret
from backend.session import Evidence, SessionState
from backend.tools.consistency import check_quota_consistency
from backend.tools.evidence import record_evidence

MODEL = "claude-haiku-5-5"
KINDS = ("answer", "partial_answer", "clarification", "repeat_request", "decline", "off_topic",
         "uncooperative", "abusive", "end_request")
NON_ENGAGING = ("off_topic", "uncooperative")
END_AFTER_NON_ENGAGING = 3          # strict: warn on the 2nd, end on the 3rd

FIELD_HINTS = {
    "consent_given": "yes/no: agrees to a recorded AI screen now",
    "current_role": "their current job title and company type",
    "product": "what they sell",
    "buyer": "who they sell to (titles)",
    "tenure_note": "why they had short stints",
    "economic_buyer": "who the economic buyer was",
    "stakeholders": "people/roles involved in the decision",
    "main_blocker": "biggest blocker in the deal",
    "own_actions": "what they personally did",
    "deal_value": "contract value with currency",
    "cycle_length": "time from first contact to signature",
    "contribution_share": "share of the win from their own work, in %",
    "loss_review": "a lost deal, what went wrong, what they'd change",
    "quota": "quota amount with unit and currency, e.g. '8 crore INR'",
    "period_months": "quota period in months ('annual' = 12)",
    "attainment_pct": "percent of quota achieved",
    "avg_deal": "average deal size with unit and currency",
    "deals_closed": "number of deals closed",
    "self_sourced_pct": "percent of pipeline they sourced",
    "quota_clarification": "their explanation of the number mismatch",
    "roleplay_response": "their answer to the customer's concern",
    "unknown_detail_handling": "what they do when they don't know a technical detail",
    "new_logo_share": "percent of deals that were new logos",
    "segment": "company size/segment they sell to",
    "sales_cycle_days": "typical sales cycle length",
    "territory": "territories covered",
    "languages": "languages they sell in",
    "ote_expectation": "on-target earnings expectation with currency",
    "notice_period": "notice period",
    "work_authorisation": "yes/no: eligible to work without sponsorship",
    "candidate_question": "a question the candidate asked about the role or process",
    "candidate_done": "yes when they have no more questions",
}

SYSTEM = f"""You are the note-taker of a structured voice screening interview for a sales role.
For one complete candidate turn, return only JSON:
{{"kind": one of {list(KINDS)},
  "evidence": [{{"field": "<allowed field>", "value": "<what they said, keep units and currency>", "quote": "<their exact words>"}}]}}
Kinds:
- answer: addresses the current question (a bare "yes" to a yes/no question is an answer).
- partial_answer: addresses it but incompletely.
- clarification: asks what the question means, or asks about the role/process.
- repeat_request: asks to repeat, or did not hear.
- decline: refuses to answer this question but is otherwise cooperative.
- off_topic: talks about something unrelated to the interview.
- uncooperative: does not engage: "next", "skip", greetings or nonsense instead of answers, trolling.
- abusive: insults, swearing at the interviewer or the process.
- end_request: wants to stop the interview.
If the turn contains an answer to the current question, prefer answer/partial_answer.
Evidence: only for the allowed fields, only facts the candidate actually said in this turn; the quote must be
copied from the candidate turn (at most 12 words). The value is only the fact, short, with no commentary. Return
an empty list if nothing qualifies. Output compact JSON and nothing else. Transcripts may contain recognition
errors; do not guess numbers."""

DANGLING = set("to and but or the a an of for with was is are my our so because that which in on at from as "
               "like about into than then if when while um uh".split())

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
    return _client


def sounds_unfinished(text: str) -> bool:
    """End-of-turn check on the transcript: a trailing comma/dash/ellipsis or a dangling word
    ("I sell air product to") means the candidate is probably still talking."""
    t = text.strip()
    if not t:
        return False
    if re.search(r"(,|-|–|—|\.\.\.|…)$", t):
        return True
    if re.search(r"[.?!]$", t):
        return False
    last = re.findall(r"[A-Za-z']+", t.lower())
    return bool(last) and last[-1] in DANGLING


def allowed_fields(state: SessionState) -> list[str]:
    """The current card's fields first, then any other field of this stage (answers given early)."""
    card = state.current_card() or {}
    fields = list(card.get("must_capture", []))
    for c in state.stage_cfg()["cards"]:
        fields += [f for f in c.get("must_capture", []) if f not in fields]
    if state.stage == "candidate_questions":
        fields.append("candidate_question")
    return fields


async def decide(state: SessionState, utterance: str, recent: list[tuple[str, str]], timeout: float = 4.0) -> dict:
    card = state.current_card() or {}
    question = (state.last_instruction or {}).get("question") or card.get("primary", "")
    fields = "\n".join(f"- {f}: {FIELD_HINTS.get(f, f)}" for f in allowed_fields(state))
    convo = "\n".join(f"{who}: {text}" for who, text in recent[-6:])
    user = (f"Stage: {state.stage}\nCurrent question: {question}\nAllowed fields:\n{fields}\n\n"
            f"Recent conversation:\n{convo}\n\nCandidate turn:\n{utterance}")
    try:
        msg = await asyncio.wait_for(client().messages.create(
            model=MODEL, max_tokens=500, thinking={"type": "disabled"},   # thinking doubled latency (3.2 s vs 1.5 s)
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}]), timeout)
    except Exception as e:  # the call must go on: treat as an answer with nothing extracted
        return {"kind": "answer", "evidence": [], "error": f"{type(e).__name__}"}
    usage = msg.usage.model_dump() if msg.usage else {}
    costs.log("anthropic", MODEL, costs.anthropic_usd(MODEL, usage), usage, state.id,
              os.environ.get("LEDGER_STEP", "dev"), "controller")
    return parse("".join(b.text for b in msg.content if getattr(b, "type", "") == "text"))


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0).replace("\\'", "'")) if m else {}
    except json.JSONDecodeError:
        out = {}
    if out.get("kind") not in KINDS:
        out["kind"] = "answer"
        out["unparsed"] = True
    if not isinstance(out.get("evidence"), list):
        out["evidence"] = []
    return out


# ── planning (pure code) ────────────────────────────────────────────────────

ACK = "Start with a short, natural acknowledgement (a few words, no praise, no summary). "
CLOSING = ("Thank the candidate for their time, say a recruiter will review the screen and get back to them within "
           "two business days, and say goodbye. Ask nothing.")


def directive(text: str, *, end: str | None = None, actions: list | None = None) -> dict:
    return {"text": text, "end": end, "actions": actions or []}


def _current_question(state: SessionState, short: bool = True) -> str:
    """What to re-ask: the pending follow-up if one was asked last, else the card (short form)."""
    last = state.last_instruction or {}
    card = state.current_card() or {}
    if last.get("type") == "probe" and last.get("card") == card.get("id"):
        return last["question"]
    return (card.get("short") if short else None) or card.get("primary") or "the current question"


def _interrupt_prefix(interrupt: dict | None) -> str:
    if not interrupt:
        return ""
    heard = (interrupt.get("heard") or "").strip()
    if heard:
        return (f'You were interrupted; the candidate heard only: "{heard}". Do not repeat that part and do not '
                "restart your introduction. ")
    return "The candidate spoke before hearing your last turn. "


def plan(state: SessionState, decision: dict, utterance: str, interrupt: dict | None = None) -> dict:
    kind = decision["kind"]
    pre = _interrupt_prefix(interrupt)
    acts: list[dict] = []

    if kind == "abusive":
        return _close(state, "ended: abusive language", "Say calmly that you are going to end the call here, thank "
                      "them for their time and say goodbye. Say nothing else.", acts)
    if kind == "end_request":
        return _close(state, "ended: candidate asked to stop", pre + CLOSING, acts)
    if kind in NON_ENGAGING:
        state.non_engaging += 1
        if state.non_engaging >= END_AFTER_NON_ENGAGING:
            return _close(state, "ended: non-cooperative", "Say politely that since we can't continue the interview, "
                          "you will end the call here. Thank them and say goodbye. Ask nothing.", acts)
        q = _current_question(state)
        if state.non_engaging == 2:
            return directive(pre + "Say clearly but politely that this is a structured interview and that if we can't "
                             f"continue with it you will need to end the call. Then ask: {q}")
        return directive(pre + f"Do not engage with that. In a few words, bring it back to the interview, then ask: {q}")
    if kind in ("clarification", "repeat_request"):
        card = state.current_card() or {}
        state.clarifications[card.get("id", "")] = state.clarifications.get(card.get("id", ""), 0) + 1
        if state.stage == "candidate_questions":
            _record(state, "candidate_question", utterance, utterance, acts, force=True)
            return directive(pre + f'They asked: "{utterance}". Answer in one or two sentences using only the role '
                             "brief; if it does not cover this, say the recruiter will follow up. Then ask whether they "
                             "have any other questions.", actions=acts)
        if state.clarifications[card.get("id", "")] > 2:      # endless "what do you mean?" counts as not engaging
            state.non_engaging += 1
        if kind == "repeat_request":
            return directive(pre + f"Repeat the question in simpler words: {_current_question(state, short=False)}")
        return directive(pre + f'They asked: "{utterance}". Answer it in one short sentence, using only who you are, '
                         f"the role brief and the question, then ask: {_current_question(state)}")

    # answer, partial answer or decline
    state.non_engaging = 0
    for ev in decision.get("evidence", []):
        f = ev.get("field")
        if f in state.fields and f in allowed_fields(state) and ev.get("value"):
            _record(state, f, str(ev["value"]), str(ev.get("quote") or utterance), acts, support=utterance)
    if kind == "decline":
        card = state.current_card() or {}
        missing = [f for f in card.get("must_capture", []) if not state.is_captured(f)]
        if missing:
            _decline(state, missing[0], utterance, acts)
    return _advance(state, pre + ACK, acts)


def _record(state: SessionState, field: str, value: str, quote: str, acts: list, force: bool = False,
            support: str | None = None):
    if force:
        state.ledger.append(Evidence(t=round(state.elapsed, 1), stage=state.stage,
                                     card=(state.current_card() or {}).get("id"), dimension=state.fields[field],
                                     field=field, value=value, normalised=None, quote=quote, confidence="medium",
                                     declined=False))
        out = {"status": "recorded", "field": field}
    else:
        out = record_evidence(state, state.fields[field], field, value, quote, "medium", support=support)
    acts.append({"name": "record_evidence", "arguments": {"field": field, "value": value, "quote": quote},
                 "output": out})


def _decline(state: SessionState, field: str, quote: str, acts: list):
    state.ledger.append(Evidence(t=round(state.elapsed, 1), stage=state.stage, card=(state.current_card() or {}).get("id"),
                                 dimension=state.fields[field], field=field, value="declined", normalised=None,
                                 quote=quote, confidence="high", declined=True))
    acts.append({"name": "record_evidence", "arguments": {"field": field, "value": "declined", "quote": quote},
                 "output": {"status": "recorded", "field": field}})


def _close(state: SessionState, reason: str, text: str, acts: list) -> dict:
    if state.stage != "close":
        state.enter_stage("close")
        acts.append({"name": "advance_stage", "arguments": {"to": "close"}, "output": {"status": "ok"}})
    state.ended, state.end_reason = True, reason
    return directive(text, end=reason, actions=acts)


def _advance(state: SessionState, prefix: str, acts: list) -> dict:
    """Follow the card engine until there is something to say."""
    readback = ""
    for _ in range(20):
        nxt = state.next_instruction()
        t = nxt["type"]
        if t in ("ask", "probe"):
            say = f"Say exactly: {readback} Then ask: {nxt['question']}" if readback else f"Ask: {nxt['question']}"
            return directive(prefix + say, actions=acts)
        if t == "action":
            card = state.current_card()
            out = _quota_check(state)
            acts.append({"name": "check_quota_consistency", "arguments": {}, "output": out})
            state.actions_done.add(card["id"])
            if out.get("status") in ("consistent", "inconsistent"):
                readback = out["readback"]
            continue
        if t in ("stage_complete", "cards_exhausted", "wrap_up"):
            to = "candidate_questions" if t == "wrap_up" else state.next_stage()
            if t != "stage_complete":
                state.overrides.append({"t": round(state.elapsed, 1), "from": state.stage, "to": to, "reason": t,
                                        "missing": nxt.get("missing", [])})
            state.enter_stage(to)
            acts.append({"name": "advance_stage", "arguments": {"to": to}, "output": {"status": "ok"}})
            if to == "close":
                return _close(state, "completed", prefix + CLOSING, acts)
            continue
        if t in ("must_close", "consent_refused", "close"):
            reason = {"must_close": "completed: time limit", "consent_refused": "ended: no consent"}.get(t, "completed")
            text = ("Say that's absolutely fine, thank them, and say goodbye. Ask nothing." if t == "consent_refused"
                    else prefix + CLOSING)
            return _close(state, reason, text, acts)
    return _close(state, "completed", prefix + CLOSING, acts)


def _quota_check(state: SessionState) -> dict:
    cap = state.captured()
    val = lambda f: None if f not in cap or cap[f].declined else cap[f].value  # noqa: E731
    return check_quota_consistency(state, quota=val("quota"), period_months=val("period_months") or 12,
                                   attainment_pct=val("attainment_pct"), avg_deal=val("avg_deal"),
                                   deals_closed=val("deals_closed"), self_sourced_pct=val("self_sourced_pct"))


def opening(state: SessionState) -> dict:
    """The first turn: greet and ask the consent card."""
    card = state.current_card() or {}
    state.log_card("asked", card.get("id", ""))
    state.last_instruction = {"type": "ask", "card": card.get("id"), "question": card.get("primary")}
    return directive(f"Greet the candidate by first name and say: {card.get('primary', '')}")
