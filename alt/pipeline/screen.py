"""The role step of the question generator: role.md + market preset → a screen configuration.

The deterministic part comes from role.md in code (title, company, location, arrangement, motion, segment, buyers,
requirements). Claude Haiku 5.5 proposes only the judgement calls (weights, the role-play objection, how each
requirement is checked, at most two role questions, FAQ, inferred ranges, next-round suggestions); code validates
them and a recruiter approves the result. Every candidate for the role then gets the same configuration.

`session_stages` builds one session's cards from the generic layer (content/stages.yaml), the market preset, the
screen configuration and the candidate's approved plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path

import yaml

from alt import budget
from alt.brain import llm
from alt.pipeline import inputs as I
from backend import costs
from backend.content_store import ContentStore
from backend.tools.evidence import _share, _stems

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "content" / "question_generator" / "role_prompt.md"
MODEL = "claude-haiku-5-5"
COMPETENCIES = ["deal_execution", "performance_credibility", "technical_credibility", "outcome_orientation",
                "ownership_learning"]
MAX_ROLE_QUESTIONS = 2
YEARS = re.compile(r"\b\d+\s*(?:[–-]\s*\d+\s*)?\+?\s*years?\b", re.I)


def r_text(role: dict, rid: str) -> str:
    return next((x["text"] for x in role["requirements"] if x["id"] == rid), "")
NEW_LOGO_FLOOR = {"new_logo": 50, "mixed": 30}          # % of closed deals that are new paying accounts
DEFAULT_KNOCKOUTS = [{"id": "no_quota_history", "text": "never carried an individual quota"}]


def _range_schema() -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["lo", "hi", "basis"],
            "properties": {"lo": {"type": ["number", "null"]}, "hi": {"type": ["number", "null"]},
                           "basis": {"type": "string", "enum": ["jd", "inferred", "unknown"]}}}


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["weights", "weights_rationale", "objection", "requirements", "role_questions", "sales_cycle_days",
                 "deal_size_usd", "faq", "next_round_suggestions"],
    "properties": {
        "weights": {"type": "object", "additionalProperties": False, "required": COMPETENCIES,
                    "properties": {k: {"type": "number"} for k in COMPETENCIES}},
        "weights_rationale": {"type": "string"},
        "objection": {"type": "object", "additionalProperties": False,
                      "required": ["role_played", "concern", "probe_if_weak", "strong_answer", "rationale"],
                      "properties": {k: {"type": "string"} for k in
                                     ("role_played", "concern", "probe_if_weak", "strong_answer", "rationale")}},
        "requirements": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "check", "evidence_fields", "question_index", "look_for"],
            "properties": {"id": {"type": "string"}, "check": {"type": "string", "enum": ["screen", "cv", "question"]},
                           "evidence_fields": {"type": "array", "items": {"type": "string"}},
                           "question_index": {"type": "integer"}, "look_for": {"type": "string"}}}},
        "role_questions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["question", "requirement_ids", "look_for"],
            "properties": {"question": {"type": "string"},
                           "requirement_ids": {"type": "array", "items": {"type": "string"}},
                           "look_for": {"type": "string"}}}},
        "sales_cycle_days": _range_schema(),
        "deal_size_usd": _range_schema(),
        "faq": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["q", "a"],
                                           "properties": {"q": {"type": "string"}, "a": {"type": "string"}}}},
        "next_round_suggestions": {"type": "array", "items": {"type": "string"}},
    },
}


# ── inputs for the model ─────────────────────────────────────────────────────

def card_catalog(stages: dict) -> str:
    """The generic interview as the generators see it: stage → card → fields (and standard wording)."""
    lines = []
    for stage, cfg in stages["stages"].items():
        lines.append(f"## {stage}: {cfg.get('goal', '')}")
        for c in cfg["cards"]:
            text = (c.get("variants") or [c.get("primary", "")])[0]
            if c.get("market_card"):
                text = f"(wording from the market preset: {c['market_card']})"
            if c.get("role_question") is not None:
                text = f"(role question {c['role_question']} from the screen configuration, if any)"
            lines.append(f"- {c['id']} — fields {c.get('must_capture', [])}: {' '.join(str(text).split())}")
    return "\n".join(lines)


def _market_text(m: dict) -> str:
    keep = {k: m[k] for k in ("name", "currency", "units", "financial_year", "notice_norms") if k in m}
    keep["asks_work_eligibility"] = bool((m.get("work_eligibility") or {}).get("ask"))
    return yaml.safe_dump(keep, sort_keys=False, allow_unicode=True)


def role_user_message(role_md: str, market: dict, stages: dict, rubric: dict) -> str:
    rub = "\n".join(f"- {k} (default {v['default_weight']}): {v['name']}" for k, v in rubric["competencies"].items())
    return "\n\n".join([f"# role.md\n{role_md}", f"# Market preset\n{_market_text(market)}",
                        f"# Generic interview (stages, cards, fields)\n{card_catalog(stages)}",
                        f"# Rubric competencies\n{rub}"])


# ── the model call ───────────────────────────────────────────────────────────

REPAIRABLE = ("role question", "objection", "weights")


def repair_needed(issues: list[str]) -> list[str]:
    """Issues worth one repair round: wording rules the model can fix (not grounding or field-name noise)."""
    return [i for i in issues if i.startswith(REPAIRABLE)]


async def propose(role_md: str, market: dict, run_id: str, step: str = "alt-dev",
                  previous: dict | None = None, fix: list[str] | None = None) -> dict:
    """One proposal; with `previous` and `fix`, a repair round that keeps everything else (repair before rewrite)."""
    budget.require(0.03, "screen configuration")
    store = ContentStore()
    messages = [{"role": "user", "content": role_user_message(role_md, market, store.stages(), store.rubric())}]
    if previous is not None:
        messages += [{"role": "assistant", "content": json.dumps(previous, ensure_ascii=False)},
                     {"role": "user", "content": "Code rejected these parts: " + json.dumps(fix, ensure_ascii=False)
                      + ". Return the whole proposal again with those parts fixed by the rules. Change nothing else."}]
    msg = await llm.client().messages.create(
        model=MODEL, max_tokens=3000, thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        system=[{"type": "text", "text": PROMPT.read_text(), "cache_control": {"type": "ephemeral"}}],
        messages=messages)
    usage = msg.usage.model_dump() if msg.usage else {}
    costs.log("anthropic", MODEL, costs.anthropic_usd(MODEL, usage), usage, run_id, step, "screen_config")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if msg.stop_reason != "end_turn" or not text:
        raise ValueError(f"screen configuration stopped with {msg.stop_reason}")
    return json.loads(text)


# ── validation (code) ────────────────────────────────────────────────────────

def question_issue(q: str, max_words: int = 35) -> str | None:
    from alt.pipeline.plan import question_issue as qi       # one rule set for every generated question
    return qi(q, max_words)


def _grounded(text: str, source: str, share: float = 0.6) -> bool:
    s = _stems(text)
    return bool(s) and _share(s, _stems(source)) >= share


def validate(proposal: dict, role: dict, role_md: str, stages: dict) -> tuple[dict, list[str]]:
    """Drop or repair every part that breaks a rule; return the clean proposal and the reasons."""
    issues: list[str] = []
    p = copy.deepcopy(proposal)
    fields = set(stages["fields"])
    w = p.get("weights") or {}
    if set(w) != set(COMPETENCIES) or any(not 0.05 <= float(v) <= 0.40 for v in w.values()) \
            or abs(sum(w.values()) - 1) > 0.011:
        issues.append(f"weights {w} break the rule (0.05–0.40 each, sum 1.00): rubric defaults used")
        rub = ContentStore().rubric()["competencies"]
        w = {k: rub[k]["default_weight"] for k in COMPETENCIES}
    p["weights"] = {k: round(float(v), 2) for k, v in w.items()}

    ob = p.get("objection") or {}
    for k, limit in (("concern", 40), ("probe_if_weak", 30)):
        if len((ob.get(k) or "").split()) > limit + 5:
            issues.append(f"objection {k} longer than {limit} words")
    if any(not (ob.get(k) or "").strip() for k in ("role_played", "concern", "probe_if_weak")):
        issues.append("objection incomplete")

    musts = [r["id"] for r in role["requirements"] if r["kind"] == "must"]
    reqs, seen = [], set()
    for r in p.get("requirements") or []:
        if r.get("id") not in musts or r["id"] in seen:
            issues.append(f"requirement {r.get('id')!r}: not a [must] id of role.md, or listed twice")
            continue
        bad = [f for f in r.get("evidence_fields") or [] if f not in fields]
        if bad:
            issues.append(f"requirement {r['id']}: unknown fields {bad} dropped")
        r["evidence_fields"] = [f for f in r.get("evidence_fields") or [] if f in fields]
        seen.add(r["id"])
        reqs.append(r)
    for rid in musts:
        if rid not in seen:
            issues.append(f"requirement {rid}: missing; marked as checked from the CV")
            reqs.append({"id": rid, "check": "cv", "evidence_fields": [], "question_index": -1,
                         "look_for": "not configured"})

    rqs = []
    for i, q in enumerate(p.get("role_questions") or []):
        why = question_issue(q.get("question", ""))
        if not why and not set(q.get("requirement_ids") or []) & set(musts):
            why = "serves no [must] requirement"
        if not why and len(rqs) >= MAX_ROLE_QUESTIONS:
            why = f"more than {MAX_ROLE_QUESTIONS} role questions"
        if why:
            issues.append(f"role question {i}: {why}")
            continue
        rqs.append({**q, "field": f"requirement_{len(rqs) + 1}", "index": i})
    for q in rqs:
        q.pop("index", None)
    # A role question must serve a requirement the generic cards do not cover (`check: question`); a question that
    # only repeats a card costs screen time (15 minutes) and is dropped.
    checks = {r["id"]: r["check"] for r in reqs}
    keep = [q for q in rqs if any(checks.get(rid) == "question" for rid in q.get("requirement_ids") or [])]
    for n, q in enumerate(rqs):
        if q not in keep:
            issues.append(f"role question {n}: serves only requirements the cards already cover")
    rqs = [{**q, "field": f"requirement_{n + 1}"} for n, q in enumerate(keep)]
    # Code decides the links, from the role questions' own requirement ids (the model's question_index was
    # inconsistent in the first compile): a requirement is checked by every role question that names it.
    for r in reqs:
        r["questions"] = [q["field"] for q in rqs if r["id"] in (q.get("requirement_ids") or [])]
        if r["check"] == "question" and not r["questions"]:
            issues.append(f"requirement {r['id']}: no role question serves it; checked from the CV")
            r["check"] = "cv"
        if YEARS.search(r_text(role, r["id"])):        # years of experience: counted from the CV's dates
            r["years_from_cv"] = True
        r.pop("question_index", None)
    p["requirements"], p["role_questions"] = reqs, rqs

    faq = []
    for f in p.get("faq") or []:
        if _grounded(f.get("a", ""), role_md) and not question_issue(f"{f.get('q', '')}?".rstrip("?") + "?", 60):
            faq.append(f)
        else:
            issues.append(f"faq {f.get('q')!r}: the answer is not grounded in role.md")
    faq.append({"q": "process", "a": "A recruiter reviews this screen and follows up about the next steps."})
    p["faq"] = faq
    for k in ("sales_cycle_days", "deal_size_usd"):
        r = p.get(k) or {}
        if r.get("basis") == "jd" and k == "sales_cycle_days" and not role.get("sales_cycle_days"):
            issues.append(f"{k}: basis 'jd' but role.md states none; marked inferred")
            r["basis"] = "inferred"
        p[k] = r
    p["next_round_suggestions"] = (p.get("next_round_suggestions") or [])[:3]
    return p, issues


# ── the configuration ────────────────────────────────────────────────────────

def arrangement_phrase(role: dict) -> str:
    city = role.get("city") or role.get("location")
    return {"hybrid": f"hybrid, based in {city}", "on-site": f"on-site in {city}",
            "remote": f"remote, covering {role.get('territory') or role.get('location')}"}[role["work_arrangement"]]


def compose(role: dict, market: dict, proposal: dict, role_md: str) -> dict:
    """The screen configuration: role.md facts (code) + the validated proposal. Keeps the brief.yaml keys the
    shared code reads (title, company, fit, currency, faq, objection_scenario, competency_weights, …)."""
    cyc = role.get("sales_cycle_days")
    cyc_basis = "jd" if cyc else proposal["sales_cycle_days"].get("basis", "unknown")
    if not cyc and proposal["sales_cycle_days"].get("lo") is not None:
        cyc = [proposal["sales_cycle_days"]["lo"], proposal["sales_cycle_days"]["hi"]]
    deal = proposal["deal_size_usd"]
    ob = proposal["objection"]
    return {
        "role_id": role["role_id"], "title": role["title"], "company": role["company"],
        "company_context": role["company_context"], "location": role["location"], "city": role["city"],
        "work_arrangement": role["work_arrangement"], "arrangement_phrase": arrangement_phrase(role),
        "market_id": market["market_id"], "country": market["name"], "currency": market["currency"],
        "buyer_personas": role["buyers"], "products": role["products"], "team_leadership": role["team_leadership"],
        "fit": {
            "motion": role["motion"], "min_new_logo_share": NEW_LOGO_FLOOR.get(role["motion"]),
            "segment": role["segment"], "sales_cycle_days": cyc, "sales_cycle_basis": cyc_basis,
            "territory": role.get("territory"), "languages": role.get("languages") or [],
            "max_notice_days": market.get("default_max_notice_days"),
            "ote": role.get("compensation"),          # only what the posting states; never spoken
            "work_eligibility_asked": bool((market.get("work_eligibility") or {}).get("ask")),
        },
        "expected_ranges": ({"avg_deal_usd": [deal["lo"], deal["hi"]], "basis": deal["basis"]}
                            if deal.get("lo") is not None and deal.get("hi") is not None else {}),
        "requirements": [{**r, "text": next(x["text"] for x in role["requirements"] if x["id"] == r["id"])}
                         for r in proposal["requirements"]],
        "nice_to_have": [x for x in role["requirements"] if x["kind"] == "nice"],
        "role_questions": proposal["role_questions"],
        "competency_weights": proposal["weights"], "weights_rationale": proposal["weights_rationale"],
        "objection_scenario": {k: ob[k] for k in ("role_played", "concern", "probe_if_weak", "strong_answer")},
        "objection_rationale": ob.get("rationale", ""),
        "faq": proposal["faq"], "next_round_suggestions": proposal["next_round_suggestions"],
        "knockouts": DEFAULT_KNOCKOUTS,
        "source": {"role_md_sha": hashlib.sha256(role_md.encode()).hexdigest()[:12],
                   "source_url": role.get("source_url"), "compiled_at": time.time(), "model": MODEL},
    }


async def compile_role(role_id: str, run_id: str) -> dict:
    """role.md → validated draft configuration (no approval)."""
    role, role_md = I.load_role(role_id)
    market = I.market(role["market"])
    proposal = await propose(role_md, market, run_id)
    clean, issues = validate(proposal, role, role_md, ContentStore().stages())
    return {"screen": compose(role, market, clean, role_md), "issues": issues, "proposal": proposal}


# ── storage: draft (reviewable) and approved ─────────────────────────────────

def _dir(role_id: str) -> Path:
    return ROOT / "content" / "roles" / role_id


def save_draft(role_id: str, draft: dict):
    (_dir(role_id) / "screen.draft.yaml").write_text(
        "# Draft screen configuration (role step of the question generator); awaiting recruiter approval.\n"
        + yaml.safe_dump(draft, sort_keys=False, allow_unicode=True, width=110))


def approve(role_id: str, screen: dict, by: str, note: str = "") -> dict:
    out = {**screen, "approval": {"by": by, "at": time.time(), "note": note}}
    (_dir(role_id) / "screen.yaml").write_text(
        "# Approved screen configuration: every candidate for this role is screened with it.\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))
    return out


def approved(role_id: str) -> dict | None:
    p = _dir(role_id) / "screen.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None


def draft(role_id: str) -> dict | None:
    p = _dir(role_id) / "screen.draft.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None


def current(role_id: str, allow_draft: bool = False) -> dict | None:
    return approved(role_id) or (draft(role_id) if allow_draft else None)


def session_role(screen: dict) -> dict:
    """The role dict a session renders cards with: the configuration plus its market preset."""
    return {**screen, "market": I.market(screen["market_id"])}


# ── one session's cards ──────────────────────────────────────────────────────

def session_stages(stages: dict, role: dict, plan: dict | None = None, seed: int | None = None,
                   permute: bool = False) -> dict:
    """Generic cards + market wording + role questions + one variant per card + the candidate's approved plan.
    The fields each card must capture never change (structured-interview comparability)."""
    st = copy.deepcopy(stages)
    market = role["market"]
    rqs = role.get("role_questions") or []
    rng = random.Random(f"{seed}:variants") if seed is not None else None
    plan_cards = {c["card"]: c for c in (plan or {}).get("cards") or []}
    for stage, cfg in st["stages"].items():
        out = []
        for c in cfg["cards"]:
            c = dict(c)
            if c.get("market_card") == "work_eligibility":
                we = market.get("work_eligibility") or {}
                if not we.get("ask"):
                    continue
                c.update(variants=we["variants"], short=we.get("short", ""))
            elif c.get("market_card"):
                mc = market["wording"][c["market_card"]]
                c.update(variants=mc["variants"], short=mc.get("short", ""), probes=dict(mc.get("probes") or {}))
            if c.get("role_question") is not None:
                i = c["role_question"]
                if i >= len(rqs):
                    continue
                c.update(primary=rqs[i]["question"], short=rqs[i]["question"], look_for=rqs[i].get("look_for"),
                         requirement_ids=rqs[i].get("requirement_ids"),
                         probes={rqs[i]["field"]: "Could you give me one concrete example of that?"})
            motion = (role.get("fit") or {}).get("motion")
            if motion in (c.get("variants_by_motion") or {}):
                c["variants"] = c["variants_by_motion"][motion]
            if c.get("variants"):
                k = rng.randrange(len(c["variants"])) if rng else 0
                c["primary"], c["variant"] = c["variants"][k], "AB"[k] if k < 2 else str(k)
            if c["id"] in plan_cards:
                c["standard"], c["primary"] = c.get("primary"), plan_cards[c["id"]]["primary"]
                c["personalised"], c["verifies"] = True, plan_cards[c["id"]].get("verifies", [])
            c.setdefault("probes", {})
            out.append(c)
        for n, x in enumerate(p for p in (plan or {}).get("extra_probes") or [] if p["stage"] == stage):
            out.append({"id": f"xp_{stage[:4]}_{n + 1}", "extra": True, "primary": x["question"],
                        "short": x["question"], "must_capture": [], "min_probes": 1, "max_probes": 1,
                        "probes": {x["field"]: x["question"]}, "verifies": x.get("verifies", []),
                        "requirement_id": x.get("requirement_id", "")})
        cfg["cards"] = out
    if permute and seed is not None:
        mid = list(st.get("middle_stages") or [])
        order = list(st["order"])
        idx = [order.index(m) for m in mid]
        random.Random(f"{seed}:stages").shuffle(mid)
        for i, m in zip(sorted(idx), mid):
            order[i] = m
        st["order"] = order
    return st
