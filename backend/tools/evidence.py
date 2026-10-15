"""record_evidence: append to the ledger with values normalised by the shared normaliser."""
from __future__ import annotations

import re

from backend.normalise import parse_amount, parse_count, parse_percent, words_to_number
from backend.session import Evidence, SessionState

AMOUNT_FIELDS = {"quota", "avg_deal", "deal_value", "ote_expectation"}
PERCENT_FIELDS = {"attainment_pct", "self_sourced_pct", "new_logo_share", "contribution_share"}
COUNT_FIELDS = {"deals_closed", "period_months"}
BOOL_FIELDS = {"consent_given", "candidate_done", "work_authorisation"}
YES = ("yes", "true", "y", "eligible", "ok", "okay", "sure", "agreed", "consent")
NO = ("no", "false", "n", "not eligible", "refused", "declined consent")


def _bool(value: str):
    v = str(value).strip().lower()
    if v in NO or v.startswith("no ") or v.startswith("not "):
        return False
    if v in YES or v.startswith(YES):
        return True
    return None


SUPPORT_MIN = 0.4   # text fields: share of the value's content words that must appear in the quote
GROUND_MIN = 0.5    # share of the quote's content words that must appear in the candidate's transcribed speech
_STOP = set("""a an the and or but of to in on at for with about from by as is are was were be been it its this that
they them their we our us i me my you your he she his her who which what was had has have did do does not no so very
just really also around about roughly approximately""".split())
_DECLINE_VALUE = re.compile(r"^\s*(declined|not provided|refused|prefers? not)", re.I)
_REFUSAL = re.compile(r"\b(rather not|prefer not|can't share|cannot share|can't disclose|cannot disclose|"
                      r"confidential|not comfortable|don't want to|won't share|pass on that|skip that)\b", re.I)


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", words_to_number(str(text)).lower())
    return {w[:5] for w in words if w not in _STOP and (len(w) > 2 or w.isdigit())}


def _numbers(text: str) -> set[float]:
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", words_to_number(str(text)).replace(",", ""))}


def _share(part: set[str], whole: set[str]) -> float:
    return 1.0 if not part else len(part & whole) / len(part)


def is_decline(value: str) -> bool:
    return bool(_DECLINE_VALUE.match(str(value)))


def check_support(field: str, value: str, quote: str, candidate_text: str | None,
                  support: str | None = None) -> str | None:
    """Return a rejection reason, or None. Cheap lexical guards against values the candidate never stated;
    semantic support is checked after the call."""
    if field == "candidate_question":
        return None
    if not quote.strip():
        return "quote is empty: record only what the candidate said, in their words"
    q = _stems(quote)
    need = 1.0 if len(q) <= 3 else GROUND_MIN      # short quotes must match fully
    if candidate_text is not None and _share(q, _stems(candidate_text)) < need:
        return "quote does not match what the candidate said; record only what you heard"
    if field in BOOL_FIELDS:
        return None
    if is_decline(value):
        if not _REFUSAL.search(quote):
            return ("the candidate did not refuse to answer; record `declined` only on an explicit refusal, "
                    "otherwise ask the question")
        return None
    ref = support or quote      # the controller's quotes are short; the value is checked against the whole turn
    if field == "period_months" and period_from_words(ref) is not None:
        return None                     # "full financial year" supports 12 months
    if field in AMOUNT_FIELDS | PERCENT_FIELDS | COUNT_FIELDS:
        if not _numbers(value) or not (_numbers(value) & _numbers(ref)):
            return "the number in value does not appear in the quote; record only numbers the candidate said"
        return None
    if _share(_stems(value), _stems(ref)) < SUPPORT_MIN:
        return ("value is not supported by the quote. Record only facts the candidate stated; "
                "if they have not addressed this field yet, do not record it")
    return None


PERIOD_WORDS = [(re.compile(r"\b(annual|annually|yearly|a year|one year|full year|financial year|fiscal year|fy)\b", re.I), 12),
                (re.compile(r"\b(half[- ]year|six months|h1|h2)\b", re.I), 6),
                (re.compile(r"\b(quarter|quarterly|q[1-4])\b", re.I), 3)]


def period_from_words(text: str) -> float | None:
    for pat, months in PERIOD_WORDS:
        if pat.search(str(text)):
            return months
    return None


def normalise(field: str, value: str, currency: str | None):
    if field == "period_months" and parse_count(value) is None:
        return period_from_words(value)
    if field in AMOUNT_FIELDS:
        a = parse_amount(value, currency)
        return a.to_dict() if a else None
    if field in PERCENT_FIELDS:
        return parse_percent(value)
    if field in COUNT_FIELDS:
        return parse_count(value)
    if field in BOOL_FIELDS:
        return _bool(value)
    return None


def record_evidence(state: SessionState, dimension: str, field: str, value: str, quote: str,
                    confidence: str = "medium", support: str | None = None) -> dict:
    fields = state.fields
    if field not in fields:
        card = state.current_card()
        return {"status": "rejected", "reason": f"unknown field {field!r}",
                "fields_for_current_card": card.get("must_capture", []) if card else []}
    if not state.candidate_turns:
        state.rejections.append({"t": round(state.elapsed, 1), "field": field, "value": str(value),
                                 "quote": quote, "reason": "no candidate speech yet"})
        return {"status": "rejected", "field": field,
                "reason": "the candidate has not said anything yet; ask the current card's question and wait"}
    owner = next((c for c in state.cards if field in c.get("must_capture", [])), None)
    if owner and owner.get("strict") and not any(
            e["card"] == owner["id"] and e["kind"] == "asked" for e in state.card_log):
        state.rejections.append({"t": round(state.elapsed, 1), "field": field, "value": str(value),
                                 "quote": quote, "reason": "strict card not asked yet"})
        return {"status": "rejected", "field": field,
                "reason": f"{field} belongs to a later card; ask that card's question first"}
    expected_dim = fields[field]
    note = None
    if dimension != expected_dim:
        note = f"dimension corrected from {dimension!r} to {expected_dim!r}"
    declined = is_decline(value)
    if reason := check_support(field, value, quote, state.candidate_text(), support):
        state.rejections.append({"t": round(state.elapsed, 1), "field": field, "value": str(value),
                                 "quote": quote, "reason": reason})
        return {"status": "rejected", "field": field, "reason": reason}
    norm = None if declined else normalise(field, value, state.role.get("currency"))
    card = state.current_card()
    state.ledger.append(Evidence(
        t=round(state.elapsed, 1), stage=state.stage, card=card["id"] if card else None,
        dimension=expected_dim, field=field, value=str(value), normalised=norm, quote=quote,
        confidence=confidence, declined=declined))
    out = {"status": "recorded", "field": field}
    if norm is not None and field not in BOOL_FIELDS:
        out["normalised"] = norm
    if note:
        out["note"] = note
    return out
