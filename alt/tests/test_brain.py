"""Configuration C brain (LangGraph): text only, the LLM is faked. No API calls.

    uv run --group alt pytest alt/tests -q
"""
import asyncio
from pathlib import Path

import yaml

from alt.brain.graph import Brain, route_for, run_turn
from alt.pipeline import inputs as I
from alt.pipeline import screen as S
from backend.content_store import ContentStore
from backend.controller import opening
from backend.session import SessionState

store = ContentStore()
FIX = Path(__file__).parent / "fixtures"


def role(role_id="mongodb_senior_ae_india"):
    """A fixed copy of a compiled screen configuration (the live one changes when a role is recompiled)."""
    return S.session_role(yaml.safe_load((FIX / f"screen_{role_id}.yaml").read_text()))


def new(role_id="mongodb_senior_ae_india", candidate_id="arjun_inflated_attainment", plan=None, seed=None):
    r = role(role_id)
    s = SessionState(role=r, persona=store.persona("rigorous_verifier"), candidate=I.load_candidate(candidate_id)[0],
                     stages=S.session_stages(store.stages(), r, plan, seed=seed), rubric=store.rubric())
    s.enter_stage("consent")
    s.start_clock()
    opening(s)
    return s


def fake(kind="answer", evidence=(), followup=("", ""), reply="", concern=("none", ""), delay=0.0, fail=False,
         engagement="engaged"):
    async def fn(system_extra, prompt, **kw):
        fn.stage_blocks.append(kw.get("stage_block", ""))
        if delay:
            await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("boom")
        return {"kind": kind, "engagement": engagement, "evidence": [dict(field=f, value=v, quote=q) for f, v, q in evidence],
                "specific": True, "followup_field": followup[0], "followup_question": followup[1],
                "reply": reply, "concern_type": concern[0], "concern_quote": concern[1], "_ms": 1}
    fn.stage_blocks = []
    return fn


def turn(s, text, deadline=3.5, interrupt=None, allow_wait=True, **kw):
    s.candidate_turns.append((s.elapsed, s.stage, text))
    return asyncio.run(run_turn(Brain(session=s, recent=[], run_id="test", interpret_fn=fake(**kw),
                                      allow_wait=allow_wait), text, interrupt, deadline_s=deadline))


def say(out):
    return " ".join(l["text"] for l in out["lines"] if l["channel"] != "thinking")


def test_router_is_pure():
    assert route_for("abusive", "background") == "close"
    assert route_for("off_topic", "background") == "redirect"
    assert route_for("repeat_request", "background") == "defer"
    assert route_for("clarification", "candidate_questions") == "clarify"
    assert route_for("role_question", "deal_deep_dive") == "clarify"
    assert route_for("partial_answer", "deal_deep_dive") == "record"
    assert route_for("incomplete", "background") == "wait"
    assert route_for("incomplete", "background", allow_wait=False) == "nudge"


def test_consent_then_background_question():
    s = new()
    out = turn(s, "Yes, that's fine, I have fifteen minutes.",
               evidence=[("consent_given", "yes", "Yes, that's fine")])
    assert s.stage == "background"
    assert "who usually signs" in say(out)
    assert out["lines"][0]["channel"] == "thinking"
    assert [t["node"] for t in out["trace"]] == ["prepare", "interpret", "check", "record", "advance", "speak"]


def test_quote_not_in_turn_is_dropped():
    s = new()
    out = turn(s, "Yes, that's fine.", evidence=[("consent_given", "yes", "Yes, that's fine"),
                                                 ("current_role", "CTO", "I am the CTO of Google")])
    assert any("current_role" in d for d in out["dropped"])
    assert not s.is_captured("current_role")


def test_llm_followup_targets_most_important_missing_field():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    turn(s, "I'm an AE at a SaaS company selling observability to CTOs.",
         evidence=[("current_role", "AE at a SaaS company", "I'm an AE at a SaaS company"),
                   ("product", "observability", "selling observability"), ("buyer", "CTOs", "to CTOs")])
    turn(s, "Two years now, and I want a bigger territory.",
         evidence=[("time_in_seat", "two years", "Two years now"), ("motivation", "a bigger territory", "I want a bigger territory")])
    assert s.stage == "deal_deep_dive"
    out = turn(s, "It was a fintech deal, the CFO signed it in the end.",
               evidence=[("economic_buyer", "the CFO", "the CFO signed it")],
               followup=("main_blocker", "With the CFO signing, what nearly stopped that fintech deal?"))
    assert "nearly stopped that fintech deal" in say(out)
    probe = [e for e in s.card_log if e["kind"] == "probe"][-1]
    assert probe["field"] == "main_blocker" and probe["by"] == "brain"


def test_invalid_followup_falls_back_to_card_probe():
    s = new()
    out = turn(s, "Yes.", evidence=[], followup=("consent_given", "Tell me more? And more?"))
    assert any("follow-up" in d for d in out["dropped"])
    assert "Just to confirm" in say(out)


def test_repeat_request_is_left_to_the_voice_model():
    s = new()
    out = turn(s, "Sorry, can you say that again?", kind="repeat_request")
    assert say(out) == "" and s.stage == "consent"


def test_strict_non_cooperation_warn_then_end():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    o1 = turn(s, "What's the weather?", kind="off_topic")
    assert say(o1) == "" and s.non_engaging == 1          # the voice model redirects by itself
    o2 = turn(s, "Next.", kind="uncooperative")
    assert "end the call" in say(o2) and not o2.get("end")
    o3 = turn(s, "Next.", kind="uncooperative")
    assert o3["end"] == "ended: non-cooperative" and o3["lines"][0]["channel"] == "exact"


def test_role_question_answered_then_back_on_track():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    out = turn(s, "Is the role hybrid?", kind="role_question", reply="It's hybrid, based in Gurgaon.")
    assert "hybrid" in say(out) and "who signs" in say(out).lower()


def test_deadline_degrades_without_crashing():
    s = new()
    out = turn(s, "Yes, fine.", deadline=0.2, delay=1.0)
    assert out["problems"] and "degraded" in out["problems"][0]
    assert any(t.get("status") == "degraded" for t in out["trace"])
    assert say(out)          # the card engine still produced the next line


def test_concern_needs_literal_quote():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    turn(s, "Our AI never makes mistakes, honestly.", concern=("over_promising", "Our AI never makes mistakes"))
    assert any(f["type"] == "over_promising" and f["source"] == "brain" for f in s.flags)
    turn(s, "We sell to banks.", concern=("blames_others", "It was the SE's fault"))
    assert not any(f["type"] == "blames_others" for f in s.flags)


def test_abuse_closes_with_exact_line():
    s = new()
    out = turn(s, "This is stupid, f*** off.", kind="abusive")
    assert out["end"] == "ended: abusive language" and s.stage == "close"
    assert out["lines"] == [{"channel": "exact", "text": "I'm going to end the call here. Thank you for your time. Goodbye."}]


def test_incomplete_fragment_is_held_without_side_effects():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    out = turn(s, "Ah, my title is senior", kind="incomplete", evidence=[("current_role", "senior", "my title is senior")])
    assert out.get("hold") and out["lines"] == [] and not s.is_captured("current_role")
    assert [t["node"] for t in out["trace"]][-2:] == ["wait", "speak"]


def test_low_effort_answers_count_and_end_the_call():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    o1 = turn(s, "I sell stuff. That's all. Thanks.", engagement="low_effort")
    assert s.non_engaging == 1 and not o1.get("end")
    o2 = turn(s, "I don't know.", engagement="weak_but_genuine")        # neither counts nor resets
    assert s.non_engaging == 1
    o3 = turn(s, "Let's say one day.", engagement="low_effort")
    assert s.non_engaging == 2 and "end the call" in say(o3) and o3["lines"][-1]["channel"] == "exact"
    o4 = turn(s, "What's the name of your father?", kind="off_topic")
    assert o4["end"] == "ended: non-cooperative"
    assert sum(f["source"] == "brain" and f["type"] in ("low_effort", "not_engaging") for f in s.flags) == 3


def test_engaged_answer_resets_the_count():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    turn(s, "Whatever.", engagement="low_effort")
    turn(s, "I'm an AE at a SaaS company selling observability to CTOs.", engagement="engaged")
    assert s.non_engaging == 0


def test_followup_with_two_asks_is_dropped():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    turn(s, "I'm an AE at a SaaS company selling observability to CTOs.",
         evidence=[("current_role", "AE at a SaaS company", "I'm an AE at a SaaS company"),
                   ("product", "observability", "selling observability"), ("buyer", "CTOs", "to CTOs")])
    out = turn(s, "It was one million.", evidence=[("deal_value", "one million", "It was one million")],
               followup=("economic_buyer", "What currency was that in, and who was the economic buyer?"))
    assert any("follow-up" in d for d in out["dropped"])


def test_stage_block_carries_skill_and_anchors():
    from alt.brain.graph import stage_block
    s = new()
    s.enter_stage("deal_deep_dive")
    b = stage_block(s)
    assert "Skill: complex-deal-deconstruction" in b and "deal_execution" in b and "ownership_learning" in b
    s.enter_stage("consent")
    assert "Skill:" not in stage_block(s)


def test_shuffle_keeps_dependencies():
    from alt.brain.graph import shuffle_cards
    orders = set()
    for seed in range(12):
        s = new()
        s.shuffle_seed = seed
        s.enter_stage("performance_verification")
        shuffle_cards(s)
        ids = [c["id"] for c in s.cards]
        assert sorted(ids) == sorted(["perf_quota", "perf_attainment", "perf_deals", "perf_check", "perf_self_sourced"])
        assert ids.index("perf_check") == 3 and ids[-1] == "perf_self_sourced"
        s.enter_stage("ai_objection_roleplay")
        shuffle_cards(s)
        assert [c["id"] for c in s.cards] == ["rp_objection", "rp_exit"]
        s.enter_stage("fit_and_logistics")
        shuffle_cards(s)
        orders.add(tuple(c["id"] for c in s.cards))
    assert len(orders) > 3


def test_reply_naming_internal_material_is_replaced():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    out = turn(s, "How big is the team?", kind="role_question", reply="The brief doesn't cover team size.")
    assert "the brief" not in say(out).lower() and "recruiter will follow up" in say(out)


def test_silence_after_a_fragment_nudges_once_then_moves_on_with_a_transition():
    """Live call cfee5dd7ab: a thinking candidate was moved on silently."""
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    card = s.current_card()["id"]
    o1 = turn(s, "Oh, wait, wait a sec", kind="incomplete", allow_wait=False)
    assert o1.get("nudged") and say(o1).startswith("No rush.") and s.current_card()["id"] == card
    o2 = turn(s, "Uh", kind="incomplete", allow_wait=False)          # still nothing: probe, then move on
    o3 = turn(s, "Hmm", kind="incomplete", allow_wait=False)
    moved = [o for o in (o2, o3) if "let's move on" in say(o)]
    assert moved and s.current_card()["id"] != card


def test_time_request_is_recognised():
    from alt.orchestrator import TIME_REQUEST
    for t in ("oh, wait, wait a sec", "Let me think", "give me a moment", "hold on"):
        assert TIME_REQUEST.search(t)
    assert not TIME_REQUEST.search("I waited for the budget approval")


def test_role_question_card_has_a_spoken_probe():
    s = new("sg_sales_lead_ai_fintech", "nadia_bank_ai_enterprise")
    rq = next(c for c in s.stages["stages"]["background"]["cards"] if c["id"] == "rq_1")
    assert rq["probes"] and not any(p.startswith("Ask briefly") for p in rq["probes"].values())


def test_one_word_fragments_are_never_strikes():
    """c49cf97735: two transcript fragments ("lost") ended the call."""
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    for frag in ("lost", "Ja", "uh huh"):
        turn(s, frag, kind="partial_answer", engagement="low_effort")
    assert s.non_engaging == 0


def test_answer_after_an_interrupted_question_is_not_asked_again():
    """Live call cb6cb64210: the candidate cut into the deal question, gpt-live-1 repeated it, the candidate answered
    in full, and the brain read the whole question a third time instead of probing what was still missing."""
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    turn(s, "I'm an AE at a SaaS company selling observability to CTOs.",
         evidence=[("current_role", "AE at a SaaS company", "I'm an AE at a SaaS company"),
                   ("product", "observability", "selling observability"), ("buyer", "CTOs", "to CTOs")])
    turn(s, "Two years now, and I want a bigger territory.",
         evidence=[("time_in_seat", "two years", "Two years now"), ("motivation", "a bigger territory", "I want a bigger territory")])
    while s.stage != "deal_deep_dive":
        turn(s, "I built a list of eighty accounts and booked meetings myself.",
             evidence=[(f, "built a list", "I built a list of eighty accounts") for f in s.current_card()["must_capture"]])
    card = s.current_card()
    assert card["id"] == "deal_open"
    intr = {"heard": "Walk me through a new logo you opened yourself", "unheard_question": True}
    out = turn(s, "A logistics SaaS company. The CTO signed and the security lead nearly stopped it.", interrupt=intr,
               kind="partial_answer",
               evidence=[("economic_buyer", "the CTO", "The CTO signed"),
                         ("main_blocker", "security lead", "the security lead nearly stopped it")])
    assert s.current_card()["id"] == "deal_open"
    assert "Walk me through a new logo" not in say(out)       # a probe for a missing field, not the question again
    assert [e for e in s.card_log if e["card"] == "deal_open" and e["kind"] == "probe"]


def test_unheard_question_is_still_asked_again_when_not_answered():
    s = new()
    turn(s, "Yes.", evidence=[("consent_given", "yes", "Yes")])
    asked = s.current_card()["id"]
    turn(s, "Sorry, what?", kind="repeat_request", interrupt={"heard": "Could you", "unheard_question": True})
    assert not any(e["card"] == asked and e["kind"] == "asked" for e in s.card_log)
