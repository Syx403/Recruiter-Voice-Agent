"""Pipelines, question generator validation, session layout, CV check and brief.
No API calls (the LLM is faked; checkpointers are in memory)."""
import asyncio
import copy
from datetime import date
from pathlib import Path

import pytest
import yaml
from langgraph.checkpoint.memory import InMemorySaver

from alt.pipeline import brief as B
from alt.pipeline import cvcheck
from alt.pipeline import graph as G
from alt.pipeline import inputs as I
from alt.pipeline import plan as P
from alt.pipeline import screen as S
from backend import store
from backend.content_store import ContentStore

STAGES = ContentStore().stages()
RUBRIC = ContentStore().rubric()
FIX = Path(__file__).parent / "fixtures"
SCREEN = yaml.safe_load((FIX / "screen_mongodb_senior_ae_india.yaml").read_text())
SCREEN_SG = yaml.safe_load((FIX / "screen_sg_sales_lead_ai_fintech.yaml").read_text())
KABIR, KABIR_MD = I.load_candidate("kabir_new_logo")
ROLE_MD = I.role_path("mongodb_senior_ae_india").read_text()


# ── inputs (docs/INPUT_SPEC.md) ──────────────────────────────────────────────

def test_all_repo_inputs_parse():
    assert set(I.list_roles()) >= {"mongodb_senior_ae_india", "sg_sales_lead_ai_fintech"}
    for rid in I.list_roles():
        I.load_role(rid)
    for cid in I.list_candidates():
        c = I.parse_candidate(I.candidate_path(cid).read_text(), date(2026, 10, 15))
        assert c["claims"] and c["experience"]


def test_candidate_parser_degrades_instead_of_rejecting():
    bad = KABIR_MD.replace("| cl_quota_fy26 | quota | 40000000 | INR |", "| cl_quota_fy26 | quota | 4 Cr | INR |")
    bad = bad.replace("| cl_buyer | buyer |", "| cl_buyer | favourite_food |")
    bad = bad.replace("## Profile\n", "## Profile\n- Marital status: married\n")
    bad = bad.replace("- Motion: new_logo\n- Segment: commercial\n- Territory: India, from", "- Motion: hunting\n"
                      "- Segment: commercial\n- Territory: India, from")
    c = I.parse_candidate(bad, date(2026, 10, 15))
    w = " | ".join(c["warnings"])
    assert "not a plain number" in w and "favourite_food" in w and "protected attribute" in w and "hunting" in w
    q = next(x for x in c["claims"] if x["id"] == "cl_quota_fy26")
    assert q["check"] == "factual"                                   # kept, but never compared as a number
    assert "cl_buyer" not in {x["id"] for x in c["claims"]}
    assert c["experience"][0]["motion"] == "not stated"
    _, text = I.load_candidate("kabir_new_logo")
    assert "married" not in I._redact(bad, [])                       # the model never reads the line


def test_only_unusable_documents_are_rejected():
    with pytest.raises(I.InputError) as e:
        I.parse_candidate("---\ncandidate_id: x\n---\n# X\n## Profile\n- Location: Tokyo\n")
    assert "Experience" in e.value.issues[0]
    with pytest.raises(I.InputError):
        I.parse_role(ROLE_MD.replace("- Location: Gurugram, India\n", ""))


def test_unseen_inputs_parse_with_generic_fallbacks():
    """Documents unlike the test set: a market with no preset, loose formatting, no claims table."""
    role = I.parse_role("""---
role_id: jp_enterprise_ae_hr_saas
market: japan
---
# Enterprise Account Executive

## Company
- Name: Undisclosed HR SaaS company
- Context: Sells an HR and payroll platform to large Japanese employers

## Seat
- Location: Tokyo, Japan
- Work arrangement: Onsite
- Motion: hunter
- Segment: Enterprise
- Sales cycle: 6-9 months
- Team leadership: none

## Requirements
- 5+ years of enterprise SaaS sales
- [must] Business-level Japanese
- [nice] Experience selling to CHROs
""")
    assert role["market"] == "generic" and role["work_arrangement"] == "on-site" and role["segment"] == "enterprise"
    assert role["motion"] == "not stated" and role["sales_cycle_days"] == [180, 270]
    assert [r["id"] for r in role["requirements"]] == ["R1", "R2", "N1"]
    assert I.market(role["market"])["work_eligibility"]["ask"]
    cand = I.parse_candidate("""---
candidate_id: ken_sato
---
# Ken Sato
## Profile
- Location: Tokyo
## Experience
### Enterprise AE | Some HR Tech Co | 2021 – present
- Highlights:
  - Closed 6 enterprise deals in 2025
### Business Development Representative | Another Co | SaaS | 2019-04 – 2020-12
""", date(2026, 10, 15))
    assert cand["claims"] == [] and any("Claims" in w for w in cand["warnings"])
    assert [e["closing"] for e in cand["experience"]] == [True, False]
    assert cand["experience"][0]["months"] is not None


def test_derived_facts():
    c = I.parse_candidate(KABIR_MD, date(2026, 10, 15))
    assert c["current"]["company"] == "Arcbridge Labs" and not c["short_tenure"]
    assert c["quota_carrying_months"] == 43 + 25         # the SDR seat is not a closing seat
    claim = next(x for x in c["claims"] if x["id"] == "cl_quota_fy26")
    assert claim["value"] == 40000000 and claim["check"] == "numeric"


# ── the role step: validation ────────────────────────────────────────────────

def proposal():
    return {"weights": {"deal_execution": 0.3, "performance_credibility": 0.3, "technical_credibility": 0.15,
                        "outcome_orientation": 0.15, "ownership_learning": 0.1},
            "weights_rationale": "hunter seat",
            "objection": {"role_played": "a CTO", "concern": "Where does our data sit?", "probe_if_weak": "Which region?",
                          "strong_answer": "clarifies", "rationale": ""},
            "requirements": [{"id": "R1", "check": "cv", "evidence_fields": [], "question_index": -1, "look_for": ""},
                             {"id": "R2", "check": "screen", "evidence_fields": ["new_logo_share", "deal_open"],
                              "question_index": -1, "look_for": ""},
                             {"id": "R4", "check": "question", "evidence_fields": [], "question_index": 0,
                              "look_for": ""}],
            "role_questions": [
                {"question": "How do you build a new-account pipeline from scratch?", "requirement_ids": ["R4"],
                 "look_for": "channels"},
                {"question": "Do you have four years of closing experience?", "requirement_ids": ["R1"],
                 "look_for": ""},
                {"question": "How do you run a complex deal?", "requirement_ids": ["R2"], "look_for": ""}],
            "sales_cycle_days": {"lo": None, "hi": None, "basis": "unknown"},
            "deal_size_usd": {"lo": 10000, "hi": 150000, "basis": "inferred"},
            "faq": [{"q": "Where is it?", "a": "The role is in Gurugram, hybrid."},
                    {"q": "Pay?", "a": "The OTE band is 90 lakh."}],
            "next_round_suggestions": ["a", "b", "c", "d"]}


def test_screen_validation():
    role, md = I.load_role("mongodb_senior_ae_india")
    clean, issues = S.validate(proposal(), role, md, STAGES)
    reqs = {r["id"]: r for r in clean["requirements"]}
    assert set(reqs) == {"R1", "R2", "R3", "R4", "R5", "R6"}                  # missing ones added as CV checks
    assert reqs["R2"]["evidence_fields"] == ["new_logo_share"] and reqs["R1"]["years_from_cv"]
    assert [q["field"] for q in clean["role_questions"]] == ["requirement_1"]    # closed and redundant ones dropped
    assert reqs["R4"]["questions"] == ["requirement_1"]
    assert any("closed" in i for i in issues) and any("already cover" in i for i in issues)
    assert not any("90 lakh" in f["a"] for f in clean["faq"])                  # an ungrounded answer is dropped
    assert len(clean["next_round_suggestions"]) == 3
    bad = proposal()
    bad["weights"]["deal_execution"] = 0.9
    assert S.validate(bad, role, md, STAGES)[0]["weights"]["deal_execution"] == RUBRIC["competencies"]["deal_execution"]["default_weight"]
    assert S.repair_needed(issues)


def test_compose_keeps_the_brief_keys_and_jd_facts():
    role, md = I.load_role("sg_sales_lead_ai_fintech")
    clean, _ = S.validate(proposal() | {"requirements": [], "role_questions": []}, role, md, STAGES)
    sc = S.compose(role, I.market("singapore"), clean, md)
    for k in ("title", "company", "fit", "currency", "faq", "objection_scenario", "competency_weights", "knockouts"):
        assert k in sc
    assert sc["currency"] == "SGD" and sc["arrangement_phrase"] == "on-site in Singapore"
    assert sc["fit"]["min_new_logo_share"] == 30 and sc["fit"]["work_eligibility_asked"]
    assert sc["expected_ranges"]["basis"] == "inferred"


# ── one session's cards ──────────────────────────────────────────────────────

def ids(st, stage):
    return [c["id"] for c in st["stages"][stage]["cards"]]


def test_session_layers_by_market_and_role():
    india = S.session_stages(STAGES, S.session_role(SCREEN), seed=1)
    sg = S.session_stages(STAGES, S.session_role(SCREEN_SG), seed=1)
    assert "fit_eligibility" not in ids(india, "fit_and_logistics") and "fit_eligibility" in ids(sg, "fit_and_logistics")
    comp_in = next(c for c in india["stages"]["fit_and_logistics"]["cards"] if c["id"] == "fit_comp")
    comp_sg = next(c for c in sg["stages"]["fit_and_logistics"]["cards"] if c["id"] == "fit_comp")
    assert "CTC" in comp_in["primary"] and "base salary" in comp_sg["primary"]
    assert comp_in["must_capture"] == comp_sg["must_capture"]                  # same fields in every market
    n_rq = len(SCREEN_SG["role_questions"])
    assert [i for i in ids(sg, "background") if i.startswith("rq_")] == [f"rq_{k + 1}" for k in range(n_rq)]
    for st in (india, sg):
        for cfg in st["stages"].values():
            assert all(c.get("primary") for c in cfg["cards"])


def test_variants_are_seeded_and_plan_wording_applies():
    a = S.session_stages(STAGES, S.session_role(SCREEN), seed=3)
    b = S.session_stages(STAGES, S.session_role(SCREEN), seed=3)
    assert a == b
    variants = {S.session_stages(STAGES, S.session_role(SCREEN), seed=k)["stages"]["background"]["cards"][0]["variant"]
                for k in range(10)}
    assert variants == {"A", "B"}
    plan = {"cards": [{"card": "deal_open", "primary": "Walk me through the Jaipur logo: who decided?",
                       "verifies": ["cl_outcome"]}],
            "extra_probes": [{"stage": "deal_deep_dive", "field": "customer_outcome", "question": "What changed?",
                              "verifies": [], "requirement_id": ""}]}
    st = S.session_stages(STAGES, S.session_role(SCREEN), plan, seed=3)
    deal = st["stages"]["deal_deep_dive"]["cards"]
    assert deal[0]["primary"].startswith("Walk me through the Jaipur") and deal[0]["personalised"]
    assert deal[0]["must_capture"] == STAGES["stages"]["deal_deep_dive"]["cards"][0]["must_capture"]
    assert deal[-1]["extra"] and deal[-1]["min_probes"] == 1 and deal[-1]["must_capture"] == []


def test_middle_stage_permutation():
    orders = {tuple(S.session_stages(STAGES, S.session_role(SCREEN), seed=k, permute=True)["order"]) for k in range(20)}
    assert len(orders) > 3
    for o in orders:
        assert o[:2] == ("consent", "background") and o[-2:] == ("candidate_questions", "close")


# ── the candidate step: validation ───────────────────────────────────────────

def test_plan_validation_enforces_blind_recall_and_open_questions():
    plan = {"claims": KABIR["claims"],
            "requirement_fit": [{"requirement_id": "R1", "status": "shown",
                                 "cv_basis": "Account Executive at Arcbridge Labs since Mar 2023", "note": ""},
                                {"requirement_id": "R2", "status": "shown", "cv_basis": "won every deal in Asia",
                                 "note": ""},
                                {"requirement_id": "R9", "status": "shown", "cv_basis": "", "note": ""}],
            "cards": [
                {"card": "perf_attainment", "primary": "Your CV says 110% in FY2025-26. What did you finish at?",
                 "rationale": "", "verifies": ["cl_attain_fy26"]},
                {"card": "fit_notice", "primary": "Your CV says your notice is ninety days. Is that still the case?",
                 "rationale": "", "verifies": []},
                {"card": "deal_open", "primary": "On the Jaipur logo, who decided, and what did you do yourself?",
                 "rationale": "", "verifies": ["cl_outcome", "nope"]},
                {"card": "bg_current", "primary": "What do you sell at Zenith Corp, and who signs?", "rationale": "",
                 "verifies": []},
                {"card": "deal_loss", "primary": "Unlike that deal, which one did you lose?", "rationale": "",
                 "verifies": []},
                {"card": "rp_objection", "primary": "What would you say?", "rationale": "", "verifies": []}],
            "extra_probes": [
                {"stage": "performance_verification", "field": "quota_window",
                 "question": "Which months does your financial year at Arcbridge Labs cover?", "requirement_id": "",
                 "verifies": ["cl_quota_fy26"], "rationale": ""},
                {"stage": "fit_and_logistics", "field": "current_comp", "question": "What do you earn?",
                 "requirement_id": "", "verifies": [], "rationale": ""},
                {"stage": "background", "field": "quota", "question": "What was your quota?", "requirement_id": "",
                 "verifies": [], "rationale": ""}]}
    clean, issues = P.validate(plan, STAGES, SCREEN, KABIR_MD, ROLE_MD)
    assert [c["card"] for c in clean["cards"]] == ["deal_open"] and clean["cards"][0]["verifies"] == ["cl_outcome"]
    assert [p["field"] for p in clean["extra_probes"]] == ["quota_window"]
    text = " | ".join(issues)
    for expected in ("blind recall", "closed", "Zenith", "stand alone", "not a personalisable card",
                     "not a probe-able field", "R9"):
        assert expected in text, expected
    fit = {r["requirement_id"]: r for r in clean["requirement_fit"]}
    assert fit["R1"]["status"] == "shown" and fit["R2"]["status"] == "unclear"      # ungrounded basis
    assert set(fit) == {r["id"] for r in SCREEN["requirements"]}


# ── CV check ─────────────────────────────────────────────────────────────────

def test_cv_check_and_neutral_followup():
    arjun, _ = I.load_candidate("arjun_inflated_attainment")
    said = {"quota": "4 crore rupees", "attainment_pct": "150 percent", "avg_deal": "60 lakh", "deals_closed": "10",
            "notice_period": "two months"}
    rows = cvcheck.compare(arjun["claims"], said, "INR")
    st = {r["field"]: r["status"] for r in rows}
    assert st["quota"] == "match" and st["attainment_pct"] == "match" and st["avg_deal"] == "differs"
    assert st["notice_period"] == "match"
    q = cvcheck.followup(rows, arjun["claims"], "indian")
    assert q == ("Your CV lists an average deal of 20 lakh rupees for FY2025-26. "
                 "How does that fit with the figures you just gave me?")
    assert cvcheck.followup([r for r in rows if r["field"] != "avg_deal"], arjun["claims"]) is None


# ── pipelines ────────────────────────────────────────────────────────────────

def comp(id, score, ev=("I opened it myself through outbound",)):
    return {"id": id, "score": score, "confidence": "medium", "evidence": list(ev), "rationale": ""}


def test_compute_caps_and_recommendation():
    brief = {"competencies": [comp("deal_execution", 4), comp("performance_credibility", 5),
                              comp("technical_credibility", 3), comp("outcome_orientation", 0, ()),
                              comp("ownership_learning", 3)]}
    role = {"competency_weights": {"deal_execution": 0.3, "performance_credibility": 0.3, "technical_credibility": 0.2,
                                   "outcome_orientation": 0.1, "ownership_learning": 0.1}, "knockouts": []}
    c = B.compute(brief, role, RUBRIC, [{"status": "inconsistent"}], [{"field": "quota", "declined": False}],
                  [{"field": "avg_deal", "status": "differs"}], [])
    assert c["scores"]["performance_credibility"] == 3 and c["scores"]["outcome_orientation"] is None
    assert c["total_25"] == 16.7 and c["recommendation"] == "conditional"
    assert c["cv_differs"] == [{"field": "avg_deal", "status": "differs"}]


def test_compute_needs_verification_and_ungrounded_scores():
    brief = {"competencies": [comp("deal_execution", 4, ()), comp("performance_credibility", 0, ()),
                              comp("technical_credibility", 3), comp("outcome_orientation", 2),
                              comp("ownership_learning", 3)]}
    c = B.compute(brief, SCREEN, RUBRIC, [], [])
    assert c["recommendation"] == "needs_verification" and c["scores"]["deal_execution"] is None
    assert "no_quota_history" in c["knockouts_to_review"]


def test_grounding_drops_interviewer_and_invented_quotes():
    draft = {"competencies": [comp("deal_execution", 4, ("I opened it myself through outbound",
                                                         "I always exceed every quota by 300 percent"))],
             "key_evidence": ["I opened it myself through outbound"]}
    clean, unmatched = B.ground(draft, "Well, I opened it myself through outbound, cold calling the CTO.")
    assert clean["competencies"][0]["evidence"] == ["I opened it myself through outbound"]
    assert unmatched == ["I always exceed every quota by 300 percent"]


@pytest.fixture
def pipes(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.sqlite")
    store.engine.cache_clear()
    saver = InMemorySaver()
    monkeypatch.setattr(G.Pipelines, "graph", G.build(saver))
    monkeypatch.setattr(G.Roles, "graph", G.build_role(saver))
    monkeypatch.setattr(S, "approved", lambda rid: SCREEN)
    monkeypatch.setattr(S, "save_draft", lambda rid, d: None)
    published = []
    monkeypatch.setattr(S, "approve", lambda rid, sc, by, note="": published.append((rid, by)))
    rec = {"role": SCREEN, "rubric": RUBRIC, "ledger": [{"field": "quota", "value": "8 crore", "declined": False}],
           "flags": [], "consistency": [{"status": "inconsistent"}], "overrides": [], "requirement_fit": [],
           "cv_check": [],
           "timeline": [{"kind": "candidate", "t": 1, "text": "I opened it myself through outbound to the CTO."}]}
    monkeypatch.setattr(G, "_record", lambda sid, state=None: rec)
    calls = []

    async def fake_call(messages, run_id, step="alt-dev"):
        calls.append(messages)
        ev = ("I invented this" if len(calls) == 1 else "I opened it myself through outbound",)
        return {"summary": "s", "competencies": [comp(k, 4, ev) for k in B.COMPETENCIES],
                "fit_checks": [], "key_evidence": [], "risks": [], "next_round_focus": [], "missing_information": []}
    monkeypatch.setattr(B, "call", fake_call)

    async def fake_plan(cid, rid, run_id, step="alt-dev", screen=None):
        return {"candidate_id": cid, "role_id": rid, "claims": KABIR["claims"], "summary": "fits",
                "requirement_fit": [], "cards": [{"card": "deal_open", "primary": "Who decided on the Jaipur logo?",
                                                  "rationale": "", "verifies": []}],
                "extra_probes": [], "source": "generated"}
    monkeypatch.setattr(P, "generate", fake_plan)
    proposals = []

    async def fake_propose(role_md, market, run_id, step="alt-dev", previous=None, fix=None):
        proposals.append(fix)
        p = proposal()
        if previous is not None:
            p["role_questions"] = p["role_questions"][:1]
        return p
    monkeypatch.setattr(S, "propose", fake_propose)
    monkeypatch.setattr(G, "ROOT", tmp_path)
    yield {"calls": calls, "proposals": proposals, "published": published}
    store.engine.cache_clear()


def test_candidate_pipeline_plan_review_call_brief(pipes):
    async def run():
        pid = await G.Pipelines.start("kabir_new_logo", "mongodb_senior_ae_india", "rigorous_verifier",
                                      auto_approve_plan=False)
        s = await G.Pipelines.state(pid)
        assert s["next"] == ["approve_plan"] and s["waiting"][0]["kind"] == "plan"
        assert s["values"]["plan"]["cards"][0]["card"] == "deal_open" and not s["values"].get("token")
        await G.Pipelines.resume(pid, {"approved": True})
        s = await G.Pipelines.state(pid)
        assert s["next"] == ["await_call"] and s["values"]["token"]
        await G.Pipelines.call_ended(s["values"]["token"], "c123")
        s = await G.Pipelines.state(pid)
        assert s["next"] == ["review"]
        v = s["values"]
        assert v["repaired"] and len(pipes["calls"]) == 2
        assert v["computed"]["scores"]["performance_credibility"] == 3
        await G.Pipelines.resume(pid, {"decision": "approve", "note": "ok"})
        assert G.Pipelines.row(id=pid).status == "approved"
    asyncio.run(run())


def test_candidate_pipeline_stops_on_a_bad_input(pipes, monkeypatch):
    def bad(cid):
        raise I.InputError("candidate.md", ["Claims: a table is required"])
    monkeypatch.setattr(I, "load_candidate", bad)

    async def run():
        pid = await G.Pipelines.start("kabir_new_logo", "mongodb_senior_ae_india", "rigorous_verifier",
                                      auto_approve_plan=False)
        s = await G.Pipelines.state(pid)
        assert s["next"] == [] and s["values"]["input_issues"] == ["candidate.md: Claims: a table is required"]
        assert G.Pipelines.row(id=pid).status == "input_error"
    asyncio.run(run())


def test_role_pipeline_repairs_once_then_waits_for_the_recruiter(pipes):
    async def run():
        tid = await G.Roles.start("mongodb_senior_ae_india")
        s = await G.Roles.state(tid)
        assert s["next"] == ["role_review"]
        assert len(pipes["proposals"]) == 2 and pipes["proposals"][0] is None and pipes["proposals"][1]
        assert [q["field"] for q in s["values"]["screen"]["role_questions"]] == ["requirement_1"]
        await G.Roles.resume(tid, {"decision": "approve", "note": "", "by": "recruiter"})
        assert pipes["published"] == [("mongodb_senior_ae_india", "recruiter")]
        assert G.Roles.latest("mongodb_senior_ae_india").status == "approved"
    asyncio.run(run())


def test_deal_card_follows_the_role_motion_with_the_same_fields():
    farmer = copy.deepcopy(SCREEN)
    farmer["fit"]["motion"] = "expansion"
    a = S.session_stages(STAGES, S.session_role(SCREEN), seed=4)["stages"]["deal_deep_dive"]["cards"][0]
    b = S.session_stages(STAGES, S.session_role(farmer), seed=4)["stages"]["deal_deep_dive"]["cards"][0]
    assert "new logo" in a["primary"] and "new logo" not in b["primary"]
    assert a["must_capture"] == b["must_capture"]


def test_years_requirements_are_decided_in_code():
    cand = {"experience": [{"months": 40, "closing": True}, {"months": 34, "closing": False}],
            "quota_carrying_months": 40}
    assert P.years_status("4+ years of full-cycle closing experience", cand)["status"] == "partial"   # 3.3 years
    assert P.years_status("6-10 years of sales or business development", cand)["status"] == "shown"   # 6.2 years
    assert P.years_status("10+ years in SaaS sales", cand)["status"] == "not_shown"
    assert P.years_status("Strong communication", cand) is None
    cand["experience"].append({"months": None, "closing": True})
    assert P.years_status("8+ years in SaaS sales", cand)["status"] == "unclear"
