"""The candidate step of the question generator: candidate.md + role.md (+ the role's approved screen
configuration and market preset) → a question plan, validated in code before a recruiter sees it.

The plan maps the CV onto the JD's requirements, personalises the wording of some standard cards, and adds at most
three extra probes. It never adds or removes a field, card or competency, so every candidate for a role is assessed
on the same facts. Claims come from candidate.md's claims table (code), never from the model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from alt import budget
from alt.brain import llm
from alt.pipeline import inputs as I
from alt.pipeline import screen as S
from backend import costs
from backend.content_store import ContentStore
from backend.monitor import PROTECTED
from backend.normalise import words_to_number
from backend.tools.evidence import _share, _stems

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "content" / "question_generator" / "prompt.md"
MODEL = "claude-haiku-5-5"
MAX_WORDS = 35
MAX_EXTRA = 3
# Cards whose wording a plan may personalise. Not: the disclosure, role questions (same for every candidate of the
# role), the quota check, the role-play (same scenario for everyone), compensation and eligibility (sensitive),
# candidate questions and the close.
PERSONALISABLE = ["bg_current", "bg_motivation", "deal_open", "deal_loss", "perf_quota", "perf_attainment",
                  "perf_deals", "perf_self_sourced", "fit_motion", "fit_segment", "fit_territory", "fit_notice",
                  "fit_location"]
NO_PROBE_STAGES = {"consent", "close", "candidate_questions", "ai_objection_roleplay"}
NO_PROBE_FIELDS = {"current_comp", "comp_split", "ote_expectation", "work_eligibility", "consent_given",
                   "candidate_question", "candidate_done", "roleplay_response", "unknown_detail_handling",
                   "requirement_1", "requirement_2", "quota_clarification"}
CLOSED = re.compile(r"^(is|are|was|were|did|do|does|have|has|had|can|could|would|will|should)\b(?!\s+you\s+"
                    r"(tell|walk|describe|explain|share|talk|give|take|outline|summari[sz]e)\b)", re.I)
STATUSES = ["shown", "partial", "not_shown", "unclear"]

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "requirement_fit", "cards", "extra_probes"],
    "properties": {
        "summary": {"type": "string"},
        "requirement_fit": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["requirement_id", "cv_basis", "note", "status"],
            "properties": {"requirement_id": {"type": "string"}, "cv_basis": {"type": "string"},
                           "note": {"type": "string"},          # reasoning first, then the status (order matters)
                           "status": {"type": "string", "enum": STATUSES}}}},
        "cards": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["card", "primary", "rationale", "verifies"],
            "properties": {"card": {"type": "string", "enum": PERSONALISABLE}, "primary": {"type": "string"},
                           "rationale": {"type": "string"},
                           "verifies": {"type": "array", "items": {"type": "string"}}}}},
        "extra_probes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["stage", "field", "question", "requirement_id", "verifies", "rationale"],
            "properties": {"stage": {"type": "string"}, "field": {"type": "string"}, "question": {"type": "string"},
                           "requirement_id": {"type": "string"},
                           "verifies": {"type": "array", "items": {"type": "string"}},
                           "rationale": {"type": "string"}}}},
    },
}


def empty(candidate_id: str = "", role_id: str = "", claims: list | None = None) -> dict:
    return {"candidate_id": candidate_id, "role_id": role_id, "summary": "", "claims": claims or [],
            "requirement_fit": [], "cards": [], "extra_probes": [], "source": "none"}


# ── inputs ───────────────────────────────────────────────────────────────────

def _screen_text(screen: dict) -> str:
    keep = {"title": screen["title"], "company": screen["company"], "arrangement": screen["arrangement_phrase"],
            "fit": screen["fit"], "requirements": screen["requirements"], "role_questions": screen["role_questions"],
            "objection_scenario": {k: screen["objection_scenario"][k] for k in ("role_played", "concern")}}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def _cards_text(stages: dict) -> str:
    lines = []
    for stage, cfg in stages["stages"].items():
        for c in cfg["cards"]:
            if c["id"] in PERSONALISABLE:
                std = (c.get("variants") or [c.get("primary", "")])[0]
                lines.append(f"- {c['id']} (stage {stage}) — must capture {c['must_capture']}: {' '.join(std.split())}")
    probe_fields = {s: [f for c in cfg["cards"] for f in c.get("must_capture", []) if f not in NO_PROBE_FIELDS]
                    for s, cfg in stages["stages"].items() if s not in NO_PROBE_STAGES}
    lines.append("\nFields an extra probe may target, by stage:")
    lines += [f"- {s}: {fs}" for s, fs in probe_fields.items() if fs]
    return "\n".join(lines)


def computed_facts(cand: dict) -> str:
    """Facts code computes from candidate.md, so the model does not do date arithmetic (it miscounted years in the
    first contrast run)."""
    lines = []
    for e in cand["experience"]:
        kind = "closing or account seat" if e["closing"] else "pipeline generation (SDR/BDR), not closing"
        length = f"{e['months']} months" if e["months"] is not None else "length unknown"
        lines.append(f"- {e['title']} at {e['company']}: {length} ({e['start']} to {e['end']}), {kind}")
    closing = cand["quota_carrying_months"]
    total = sum(e["months"] or 0 for e in cand["experience"])
    lines.append(f"- Closing or account seats in total: {closing} months ({closing / 12:.1f} years)")
    lines.append(f"- All sales and business-development seats in total: {total} months ({total / 12:.1f} years)")
    lines.append(f"- Two or more stints under twelve months: {'yes' if cand['short_tenure'] else 'no'}")
    return "\n".join(lines)


def user_message(role_md: str, screen: dict, market: dict, candidate_md: str, stages: dict,
                 cand: dict | None = None) -> str:
    return "\n\n".join([
        f"# role.md\n{role_md}",
        f"# Screen configuration (approved for this role)\n{_screen_text(screen)}",
        f"# Market preset\n{S._market_text(market)}",
        f"# candidate.md\n{candidate_md}",
        *([f"# Computed from candidate.md by code (use these numbers; do not recount dates)\n{computed_facts(cand)}"]
          if cand else []),
        f"# Cards you may personalise\n{_cards_text(stages)}"])


async def generate(candidate_id: str, role_id: str, run_id: str, step: str = "alt-dev",
                   screen: dict | None = None) -> dict:
    cand, candidate_md = I.load_candidate(candidate_id)
    role, role_md = I.load_role(role_id)
    screen = screen or S.approved(role_id)
    if screen is None:
        raise ValueError(f"role {role_id} has no approved screen configuration")
    if not PROMPT.exists():
        return empty(candidate_id, role_id, cand["claims"])
    budget.require(0.02, "question plan")
    stages = ContentStore().stages()
    msg = await llm.client().messages.create(
        model=MODEL, max_tokens=3000, thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        system=[{"type": "text", "text": PROMPT.read_text(), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message(role_md, screen, I.market(role["market"]), candidate_md,
                                                           stages, cand)}])
    usage = msg.usage.model_dump() if msg.usage else {}
    costs.log("anthropic", MODEL, costs.anthropic_usd(MODEL, usage), usage, run_id, step, "question_plan")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if msg.stop_reason != "end_turn" or not text:
        raise ValueError(f"question plan stopped with {msg.stop_reason}")
    out = json.loads(text)
    return {"candidate_id": candidate_id, "role_id": role_id, "claims": cand["claims"], **out, "source": "generated"}


# ── validation (code) ────────────────────────────────────────────────────────

def question_issue(q: str, max_words: int = MAX_WORDS) -> str | None:
    q = (q or "").strip()
    if not q.endswith("?") or q.count("?") != 1:
        return "must be one question ending in '?'"
    if len(q.split()) > max_words:
        return f"longer than {max_words} words"
    asked = re.split(r"(?<=[.!])\s+", q)[-1]               # the sentence that carries the question mark
    if CLOSED.match(asked):
        return "a closed (yes/no) question; it must be open"
    if m := PROTECTED.search(q):
        return f"protected-attribute wording {m.group(0)!r}"
    if re.search(r"\b(that deal|that one|you just (described|mentioned|said)|earlier you|as you said)\b", q, re.I):
        return "refers to an earlier answer; every question must stand alone (order is shuffled)"
    return None


TIME_UNIT = r"\s*(?:-\s*)?(?:day|week|month|quarter|year)s?\b"


def _split_nums(text: str) -> tuple[set[float], set[float]]:
    """(numbers that name a period: "four quarters", "60 days"; all other numbers)."""
    t = words_to_number(str(text)).replace(",", "")
    timed = {float(n) for n in re.findall(r"(\d+(?:\.\d+)?)" + TIME_UNIT, t, re.I)}
    return timed, {float(n) for n in re.findall(r"\d+(?:\.\d+)?", t)} - timed


def _nums(text: str) -> set[float]:
    timed, other = _split_nums(text)
    return timed | other


DURATION_UNITS = {"days", "months", "month"}


def claim_numbers(claims: list[dict]) -> tuple[set[float], set[float]]:
    """Numbers a question must not contain (blind recall), as (periods, other): duration and date claims' values,
    the other numeric claims' values, and every number in a claim's CV wording ("about two days to same day"),
    except the numbers that only name its window (FY2025-26)."""
    periods, other = set(), set()
    for c in claims:
        if c["check"] == "numeric" or c["unit"] == "month":
            (periods if c["unit"] in DURATION_UNITS else other).update(_nums(c["value"]))
        t_timed, t_other = _split_nums(c["text"])
        window = _nums(c["window"])
        periods |= t_timed - window
        other |= t_other - window
    return periods, other


SCALE = None


def readback_issue(q: str, claims: list[dict], already_said: str = "") -> str | None:
    """`already_said`: what the candidate has said so far (live check): repeating their own figure is not a
    CV read-back."""
    global SCALE
    SCALE = SCALE or I.amount_scale_words()
    scaled = re.search(r"\d[\d.,]*\s*(?:" + "|".join(sorted(map(re.escape, SCALE), key=len, reverse=True)) + r")\b",
                       words_to_number(q), re.I)
    if (re.search(r"[%₹$€£¥]", q) or scaled) and not already_said:
        return "states an amount or a percentage (blind recall: the candidate states the number)"
    periods, other = claim_numbers(claims)
    q_timed, q_other = _split_nums(q)
    hit = ((q_timed & periods) | (q_other & other)) - _nums(already_said)
    if hit:
        return f"reads a CV number back ({', '.join(f'{n:g}' for n in sorted(hit))}); the candidate must state it"
    return None


# Capitalised words a question may use without them appearing in the CV or the JD: job-title and business-function
# vocabulary of B2B sales (AE domain knowledge), calendar words, and acronyms. Anything else capitalised is treated
# as a proper noun (an employer, logo, person or place) and must come from the documents.
GENERIC_CAPS = set("""Head Heads Chief Officer Officers Director Directors Manager Managers President Vice VP SVP EVP
Founder Founders Partner Partners Lead Leads Engineering Engineer Engineers Sales Marketing Finance Financial
Operations Security Information Technology Risk Procurement Legal Product Products Platform Data Revenue Customer
Customers Success Account Accounts Executive Executives Board Committee SaaS Cloud CV JD AI Monday Tuesday Wednesday
Thursday Friday January February March April May June July August September October November December Q1 Q2 Q3 Q4
""".split())


def _names_ok(q: str, sources: str) -> str | None:
    """Capitalised names inside a question must appear in candidate.md or role.md (no borrowed logos)."""
    for sentence in re.split(r"(?<=[.?!:])\s+", q.strip()):
        for t in sentence.split()[1:]:                    # the first word of a sentence is capitalised anyway
            w = re.sub(r"[^\w-]", "", t)
            if not w[:1].isupper() or w in GENERIC_CAPS or re.fullmatch(r"[A-Z0-9]{2,6}s?", w):
                continue
            if w.lower() not in sources.lower():
                return f"names {w!r}, which is not in the CV or the JD"
    return None


YEARS_REQ = re.compile(r"(\d+)\s*(?:[–-]\s*\d+\s*)?\+?\s*years?", re.I)
CLOSING_REQ = re.compile(r"clos|full[- ]cycle|quota|account executive|carr(y|ied) a (number|target)", re.I)


def years_status(text: str, cand: dict) -> dict | None:
    """Code decides "N+ years of …" requirements from the CV's dates:
    closing seats for a closing/quota requirement, every sales and business-development seat otherwise."""
    m = YEARS_REQ.search(text)
    if not m or not cand.get("experience"):
        return None
    need = int(m.group(1)) * 12
    closing = bool(CLOSING_REQ.search(text))
    have = cand["quota_carrying_months"] if closing else sum(e["months"] or 0 for e in cand["experience"])
    unknown = any(e["months"] is None for e in cand["experience"])
    status = ("shown" if have >= need else "unclear" if unknown else "partial" if have >= 0.75 * need
              else "not_shown")
    kind = "closing seats" if closing else "sales and business-development seats"
    return {"status": status, "note": f"Code: {have / 12:.1f} years in {kind} from the CV's dates, against "
                                      f"{m.group(1)}+ required" + (" (some dates unreadable)" if unknown else "") + "."}


def validate(plan: dict, stages: dict, screen: dict, candidate_md: str = "", role_md: str = "",
             cand: dict | None = None) -> tuple[dict, list[str]]:
    """Drop every item that breaks the rules; return the clean plan and the reasons."""
    issues: list[str] = []
    claims = plan.get("claims") or []
    ids = {c["id"] for c in claims}
    musts = [r["id"] for r in screen["requirements"]]
    sources = candidate_md + "\n" + role_md
    stage_fields = {s: {f for c in cfg["cards"] for f in c.get("must_capture", [])} - NO_PROBE_FIELDS
                    for s, cfg in stages["stages"].items() if s not in NO_PROBE_STAGES}
    card_fields = {c["id"]: c.get("must_capture", []) for cfg in stages["stages"].values() for c in cfg["cards"]}
    # Standard wordings per stage: an extra probe that only rephrases one of them would ask the same thing twice.
    standard = {s: [_stems(v) for c in cfg["cards"] for v in (c.get("variants") or [c.get("primary", "")])]
                for s, cfg in stages["stages"].items()}
    asked = {r["id"] for r in screen["requirements"] if r.get("questions")}      # served by a role question
    gaps = {r.get("requirement_id") for r in plan.get("requirement_fit") or [] if r.get("status") in ("partial",
                                                                                                    "not_shown", "unclear")}

    def wording(q: str) -> str | None:
        return question_issue(q) or readback_issue(q, claims) or (_names_ok(q, sources) if sources.strip() else None)

    fit, seen = [], set()
    for r in plan.get("requirement_fit") or []:
        rid = r.get("requirement_id")
        if rid not in musts or rid in seen:
            issues.append(f"requirement fit {rid!r}: not a [must] requirement, or listed twice")
            continue
        seen.add(rid)
        basis = (r.get("cv_basis") or "").strip()
        if basis and candidate_md and _share(_stems(basis), _stems(candidate_md)) < 0.8:
            issues.append(f"requirement fit {rid}: cv_basis is not the CV's wording; status set to unclear")
            r = {**r, "status": "unclear", "cv_basis": "", "note": r.get("note", "") + " (basis not verified)"}
        req = next((x for x in screen["requirements"] if x["id"] == rid), {})
        if cand and (ys := years_status(req.get("text", ""), cand)):
            if ys["status"] != r.get("status"):
                issues.append(f"requirement fit {rid}: status {r.get('status')!r} replaced by code's {ys['status']!r}")
            r = {**r, "status": ys["status"], "note": f"{ys['note']} {r.get('note', '')}".strip(), "by": "code"}
        fit.append(r)
    for rid in musts:
        if rid not in seen:
            issues.append(f"requirement fit {rid}: missing; added as unclear")
            fit.append({"requirement_id": rid, "status": "unclear", "cv_basis": "", "note": "not assessed"})

    out_cards, seen = [], set()
    for c in plan.get("cards") or []:
        cid = c.get("card")
        why = ("not a personalisable card" if cid not in PERSONALISABLE else
               "second entry for the same card" if cid in seen else wording(c.get("primary", "")))
        if why:
            issues.append(f"card {cid}: {why}")
            continue
        seen.add(cid)
        out_cards.append({**c, "verifies": [v for v in c.get("verifies") or [] if v in ids],
                          "fields": card_fields.get(cid, [])})
    out_extra = []
    for p in plan.get("extra_probes") or []:
        st, f = p.get("stage"), p.get("field")
        q_stems = _stems(p.get("question", ""))
        why = (f"stage {st!r} takes no extra probes" if st not in stage_fields else
               f"field {f!r} is not a probe-able field of {st}" if f not in stage_fields[st] else
               f"requirement {p.get('requirement_id')!r} is unknown" if p.get("requirement_id") and
               p["requirement_id"] not in musts else
               f"a role question already asks about {p['requirement_id']}" if p.get("requirement_id") in asked else
               "serves no CV claim and no requirement gap" if not set(p.get("verifies") or []) & ids and
               p.get("requirement_id") not in gaps else
               "repeats a standard card of the stage (asked anyway)" if any(
                   _share(q_stems, std) >= 0.6 for std in standard.get(st, [])) else wording(p.get("question", "")))
        if not why and len(out_extra) >= MAX_EXTRA:
            why = f"over the limit of {MAX_EXTRA} extra probes"
        if why:
            issues.append(f"extra probe ({st}/{f}): {why}")
            continue
        out_extra.append({**p, "verifies": [v for v in p.get("verifies") or [] if v in ids]})
    clean = {**{k: v for k, v in plan.items() if k not in ("requirement_fit", "cards", "extra_probes")},
             "requirement_fit": fit, "cards": out_cards, "extra_probes": out_extra}
    return clean, issues


def card_wording(plan: dict | None) -> dict[str, str]:
    return {c["card"]: c["primary"] for c in (plan or {}).get("cards") or []}
