"""The thinking step of the brain: one Claude Haiku 5.5 call per candidate turn, strict JSON output.

Beyond classifying and extracting, Haiku here also judges the answer, writes
a contextual follow-up for the most important missing field, drafts answers to questions about the role (from the
brief only), and raises red flags. Code checks all of it before anything is used (graph.check).
"""
from __future__ import annotations

import json
import time

import anthropic

from alt import budget
from backend import costs
from backend.controller import FIELD_HINTS as _A_HINTS

# Field hints for the stage machine (content/stages.yaml). Generic: the market preset
# adds its own wording (currency units, financial year, compensation terms) per session, and the role questions add
# what each requirement field should contain (alt/brain/graph.prepare).
FIELD_HINTS = {**_A_HINTS,
    "time_in_seat": "how long they have been in the current role",
    "motivation": "why they are open to a new role, in their words",
    "requirement_1": "their answer to the role's first requirement question, in their words",
    "requirement_2": "their answer to the role's second requirement question, in their words",
    "first_meeting_source": "how the first meeting happened: own outbound, SDR, inbound or partner",
    "economic_buyer": "who could say no after the main contact said yes (or: one person decided, with deal size)",
    "deal_value": "contract value with currency, and whether annual or full contract",
    "team_on_deal": "who else worked the deal (SE, manager, SDR, partner) and what needed the candidate",
    "customer_outcome": "the customer's business result in numbers (cost, downtime, release speed, risk)",
    "loss_review": "a lost deal: why (competitor, no decision, price, product gap, champion left) and what they'd change",
    "quota": "quota amount with unit and currency as said",
    "quota_unit": "what the quota was measured on: ARR, ACV, TCV or bookings",
    "quota_owner": "individual or team quota",
    "quota_window": "the period's window: a financial year, calendar year, or last four quarters",
    "self_sourced_pct": "percent of closed deals that started from a meeting they booked themselves",
    "new_logo_share": "percent of closed deals that were new paying customers (not expansions or renewals)",
    "segment": "company size of customers, as an employee band or a deal-size band",
    "current_comp": "current annual pay in the market's usual unit, with currency",
    "comp_split": "fixed vs variable share of current pay",
    "ote_expectation": "expected on-target earnings, base plus variable at plan, with currency",
    "notice_status": "whether the notice period has started and whether it can be bought out",
    "work_location": "where they are based and whether the role's location and work arrangement work for them",
    "work_eligibility": "whether they can already work in the role's country or would need a work pass (never nationality)"}
from backend.keystore import get_secret

KINDS = ["answer", "partial_answer", "incomplete", "clarification", "repeat_request", "role_question", "decline",
         "off_topic", "uncooperative", "abusive", "end_request"]
ENGAGEMENT = ["engaged", "weak_but_genuine", "low_effort", "not_engaging"]
# Numeric contradictions are checked in code (check_quota_consistency), not judged by the LLM: an LLM
# "contradicts_earlier" label fired on consistent answers in testing.
CONCERNS = ["none", "over_promising", "blames_others", "account_management_as_new_logo", "evasive_on_numbers"]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "engagement", "evidence", "specific", "followup_field", "followup_question", "reply", "concern_type",
                 "concern_quote"],
    "properties": {
        "kind": {"type": "string", "enum": KINDS},
        "engagement": {"type": "string", "enum": ENGAGEMENT},
        "evidence": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["field", "value", "quote"],
            "properties": {"field": {"type": "string"}, "value": {"type": "string"}, "quote": {"type": "string"}}}},
        "specific": {"type": "boolean"},
        "followup_field": {"type": "string"},
        "followup_question": {"type": "string"},
        "reply": {"type": "string"},
        "concern_type": {"type": "string", "enum": CONCERNS},
        "concern_quote": {"type": "string"},
    },
}

SYSTEM = """You are the reasoning step behind a voice interviewer running a structured first-round screen for an \
Account Executive role. A separate voice model does all the talking. For one complete candidate turn you return \
JSON only.

kind:
- answer: addresses the current question (a bare "yes" to a yes/no question is an answer).
- partial_answer: addresses it but leaves out a part that the question itself asked for. Judge against the \
question as spoken: fields still missing that the question did not ask for are not a reason (the interviewer \
follows up on those).
- incomplete: the candidate stopped mid-thought and is very likely to go on: an unfinished sentence or a fragment \
("my title is senior", "I'm working in a", "the most complex one was, uh"). Only when the turn is clearly cut off; \
a short but complete answer is answer/partial_answer.
- clarification: asks what the current question means, or asks about the interview itself or whose turn it is \
("should I answer now?", "is it my turn?", "do you mean annual?").
- repeat_request: asks to repeat, or did not hear.
- role_question: asks about the role, company, team, pay band, process or next steps.
- decline: refuses this question but is otherwise cooperative.
- off_topic: talks about something unrelated to the interview.
- uncooperative: does not engage ("next", "skip", nonsense, trolling).
- abusive: insults or swears at the interviewer or the process.
- end_request: wants to stop the interview. (In the candidate_questions stage, "no, that's all" means they have no
  more questions: that is an answer with candidate_done, not end_request.)
If the turn answers the current question, prefer answer/partial_answer. A late answer to an earlier interview \
question is engaged with the interview: classify it answer/partial_answer, never off_topic. "I already answered that" \
after a question was asked again is an answer with no new evidence (the interviewer moves on), never \
clarification.

engagement (judge this turn in the context of the recent conversation, for a recruiter deciding whether the \
candidate is taking the interview seriously):
- engaged: a real attempt that moves the interview forward.
- weak_but_genuine: short or vague but plausible and sincere (nervous, unsure, or simply weak), e.g. not \
remembering who signed a deal two years ago. Hedged or apologetic answers that still offer something relevant \
("I'm not sure who had the final say, maybe the CRO"), saying "we" instead of "I", rough numbers, and not \
remembering details are weak_but_genuine even when they repeat: a weak candidate is scored low in the brief, never \
ended. Not a reason to end a call; it is evidence for scoring.
- low_effort: technically on topic but not a serious attempt (the test: is the candidate trying? a poor answer \
from someone trying is weak_but_genuine): implausible or throwaway answers ("I closed it in \
one day" for a complex deal), dismissive one-liners ("That's all. Thanks." after a question that needs detail), \
answers that dodge after being asked again, and implausible "I don't know"s (a candidate not knowing their own \
job title, product or quota). Judge plausibility for this question, not the words alone. Fillers, hesitation, accents and short \
acknowledgements ("okay, that's fine", "fair catch") are never low effort by themselves: judge the content.
- not_engaging: off topic, trolling, refusing to engage.

evidence: only allowed fields, only what the candidate actually said in this turn. Facts (roles, numbers, \
periods, names) must be concrete; a vague mention is not evidence. Story fields (own_actions, main_blocker, \
loss_review, roleplay_response, unknown_detail_handling, quota_clarification, tenure_note, motivation, \
requirement_1, requirement_2) count whenever the \
candidate gives a substantive account. quote = the candidate's exact words from this turn that support the value \
(up to 12 words for facts, up to 25 for story fields and lists). value = short, written with the candidate's \
own words taken from the quote (every key word of value must appear in the quote), keeping units and currency. \
Never guess numbers; transcripts may contain recognition errors.

specific: true if the answer is concrete; false if it is vague, generic or evasive.

followup_field / followup_question: if a field listed under "Still missing on this question" is still missing \
after this turn, pick the single most important one and write ONE short follow-up question (at most 25 words) \
that refers to what the candidate just said and asks only for that field. Neutral, no praise, no judgement, \
never two questions. Otherwise both are "".

reply: only for role_question: answer in one or two short sentences using only the role brief below; if the \
brief does not cover it, say the recruiter will follow up with those details. The reply is spoken to the candidate: \
never mention "the brief", notes, documents, the system or what you were given. Otherwise "".

concern_type / concern_quote: a red flag a recruiter should see, with the candidate's exact words: \
over_promising (absolute claims about AI accuracy, security or outcomes, e.g. "our AI never makes mistakes", \
"mistakes are really rare", "guaranteed"; a plain factual answer is not over-promising), blames_others (no \
ownership of a loss), account_management_as_new_logo, evasive_on_numbers. Otherwise "none" and ""."""

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=get_secret("ANTHROPIC_API_KEY"), max_retries=0)
    return _client


def build_user(*, stage: str, question: str, missing: list[str], fields: list[str], captured: dict[str, str],
               recent: list[tuple[str, str]], utterance: str, interrupt: str | None,
               hints: dict[str, str] | None = None) -> str:
    """`hints`: per-session field hints (market preset, role questions) over the generic FIELD_HINTS."""
    table = {**FIELD_HINTS, **(hints or {})}
    hints = "\n".join(f"- {f}: {table.get(f, f)}" for f in fields)
    have = "\n".join(f"- {k}: {v}" for k, v in captured.items()) or "- (nothing yet)"
    convo = "\n".join(f"{who}: {text}" for who, text in recent[-6:])
    parts = [f"Stage: {stage}", f"Current question: {question}",
             f"Still missing on this question: {', '.join(missing) or '(none)'}",
             f"Allowed fields:\n{hints}", f"Captured so far in the interview:\n{have}",
             f"Recent conversation:\n{convo}"]
    if interrupt:
        parts.append(f"Note: the candidate interrupted the interviewer, who had only said: \"{interrupt}\"")
    parts.append(f"Candidate turn:\n{utterance}")
    return "\n\n".join(parts)


async def interpret(system_extra: str, user: str, *, model: str, max_tokens: int, run_id: str,
                    step: str = "alt-dev", stage_block: str = "") -> dict:
    """One structured call. Raises on any failure; the graph's error handler degrades.
    System prompt: the stable part (cached), then the current stage's skill SOP and rubric anchors (cached per
    stage, so the business logic costs one cache write per stage)."""
    budget.require(0.003, "brain call")
    t0 = time.monotonic()
    system = [{"type": "text", "text": SYSTEM + "\n\n" + system_extra, "cache_control": {"type": "ephemeral"}}]
    if stage_block:
        system.append({"type": "text", "text": stage_block, "cache_control": {"type": "ephemeral"}})
    msg = await client().messages.create(
        model=model, max_tokens=max_tokens, thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        system=system,
        messages=[{"role": "user", "content": user}])
    usage = msg.usage.model_dump() if msg.usage else {}
    costs.log("anthropic", model, costs.anthropic_usd(model, usage), usage, run_id, step, "brain")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if msg.stop_reason != "end_turn" or not text:
        raise ValueError(f"brain stopped with {msg.stop_reason}")
    out = json.loads(text)
    out["_ms"] = round((time.monotonic() - t0) * 1000)
    out["_usage"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")}
    return out


async def warm(system_extra: str, *, model: str, run_id: str, step: str = "alt-dev") -> None:
    """Write the cached prefix at session start, while the greeting plays, so the first real turn (consent) does
    not pay the cache write (3.8 s cold vs about 2 s warm in the replays). One output token; failures are ignored."""
    try:
        budget.require(0.001, "brain warm-up")
        msg = await client().messages.create(
            model=model, max_tokens=1, thinking={"type": "disabled"},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            system=[{"type": "text", "text": SYSTEM + "\n\n" + system_extra, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "(warm-up)"}])
        usage = msg.usage.model_dump() if msg.usage else {}
        costs.log("anthropic", model, costs.anthropic_usd(model, usage), usage, run_id, step, "brain_warmup")
    except Exception:
        pass

