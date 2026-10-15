"""Configuration C orchestrator: provisional commits (P1) and the closing match (P2). No API calls: the voice
adapter is faked and the brain runs the real graph with a fake LLM.

    uv run --group alt pytest alt/tests -q
"""
import asyncio
import shutil

import pytest

from alt import orchestrator as orch
from alt.brain import graph as brain_graph
from alt.tests.test_brain import fake
from backend import store
from backend.controller import opening


class FakeAdapter:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def append(self, kind, content, delegation_id=None):
        self.sent.append((kind, content))
        return f"id{len(self.sent)}"


@pytest.fixture
def o(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.sqlite")
    store.engine.cache_clear()
    calls: list[dict] = []
    script: list[dict] = []

    async def run_turn(brain, utterance, interrupt=None, deadline_s=4.5):
        calls.append({"utterance": utterance, "stage": brain.session.stage})
        brain.interpret_fn = fake(**script.pop(0))
        return await brain_graph.run_turn(brain, utterance, interrupt, deadline_s)

    monkeypatch.setattr(orch, "run_turn", run_turn)
    x = orch.AltOrchestrator("mongodb_senior_ae_india", "rigorous_verifier", "arjun_inflated_attainment")
    x.adapter = FakeAdapter()
    x.state.start_clock()
    opening(x.state)
    x.opening_heard = True
    x.calls, x.script = calls, script
    yield x
    x.log.close()
    shutil.rmtree(orch.ROOT / "runs" / x.state.id, ignore_errors=True)
    store.engine.cache_clear()


def say(o, cid, text, t=1.0):
    o.timeline[cid] = {"id": cid, "kind": "candidate", "t": t, "text": text}
    o.pending.append(cid)


def test_answer_before_the_line_is_heard_rolls_back_and_merges(o):
    o.script += [dict(evidence=[("consent_given", "yes", "Yes")]),
                 dict(evidence=[("consent_given", "yes", "Yes"), ("current_role", "AE at a SaaS company",
                                                                   "I'm an AE at a SaaS company")])]
    say(o, "cand-1", "Yes.")
    asyncio.run(o._think_once())
    assert o.state.stage == "background" and o.awaiting["sent"] and not o.awaiting.get("started")
    # The candidate goes on before the background question has started playing.
    say(o, "cand-2", "I'm an AE at a SaaS company.", t=3.0)
    asyncio.run(o._think_once())
    assert o.calls[1]["utterance"] == "Yes. I'm an AE at a SaaS company."
    assert o.calls[1]["stage"] == "consent"                  # decided again from the state before the commit
    assert any(kind == "instructions" and "outdated" in text for kind, text in o.adapter.sent)
    assert o.state.stage == "background" and o.state.is_captured("consent_given")
    assert o.timeline["note-1"]["intent"] == "superseded"


def test_no_rollback_once_the_line_was_spoken(o):
    o.script += [dict(evidence=[("consent_given", "yes", "Yes")]), dict()]
    say(o, "cand-1", "Yes.")
    asyncio.run(o._think_once())
    o.awaiting["started"] = True
    say(o, "cand-2", "Sure, my role is AE.", t=8.0)
    asyncio.run(o._think_once())
    assert o.calls[1]["utterance"] == "Sure, my role is AE." and o.calls[1]["stage"] == "background"
    assert not any("outdated" in text for _, text in o.adapter.sent)


def test_hang_up_only_after_the_closing_line(o):
    o.closing, o.closing_line = "ended: abusive language", "I'm going to end the call here. Thank you for your time."
    ended = []

    async def end_soon():
        ended.append(True)
    o._end_soon = end_soon

    async def run():
        o.agent_seg = {"id": "agent-1", "start_ms": 0, "end_ms": 400, "text": "I hear you."}
        await o._agent_final()
        await asyncio.sleep(0)
        assert not ended
        o.agent_seg = {"id": "agent-2", "start_ms": 900, "end_ms": 3000, "text": "I'm going to end the call here."}
        await o._agent_final()
        await asyncio.sleep(0)
        assert ended
    asyncio.run(run())


def test_session_is_built_from_screen_market_and_plan(tmp_path, monkeypatch):
    """The orchestrator builds the call from the role's screen configuration, the market preset, the
    candidate's candidate.md and the approved plan, and exposes the layout to the recruiter."""
    import yaml
    from pathlib import Path
    from alt.pipeline import screen as S
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.sqlite")
    store.engine.cache_clear()
    sg = yaml.safe_load((Path(__file__).parent / "fixtures" / "screen_sg_sales_lead_ai_fintech.yaml").read_text())
    monkeypatch.setattr(S, "current", lambda rid, allow_draft=False: sg)
    plan = {"summary": "s", "cards": [{"card": "deal_open", "primary": "Walk me through the Singapore digital bank "
                                       "logo: who decided, and what did you do yourself?", "verifies": ["cl_outcome"]}],
            "extra_probes": [{"stage": "deal_deep_dive", "field": "customer_outcome", "question": "How was the "
                              "outcome measured?", "verifies": ["cl_outcome"], "requirement_id": ""}]}
    x = orch.AltOrchestrator("sg_sales_lead_ai_fintech", "rigorous_verifier", "nadia_bank_ai_enterprise", plan=plan,
                             seed=11)
    lay = x.layout()
    assert x.state.role["currency"] == "SGD" and x.state.candidate["name"] == "Nadia Lim"
    assert lay["seed"] == 11 and lay["market"] == "Singapore"
    deal = {c["id"]: c for c in lay["cards"]["deal_deep_dive"]}
    assert deal["deal_open"]["personalised"] and deal["deal_open"]["standard"]
    assert any(c["extra"] for c in deal.values())
    assert any(c["id"] == "fit_eligibility" for c in lay["cards"]["fit_and_logistics"])
    assert any(c["role_question"] for c in lay["cards"]["background"])
    store.engine.cache_clear()
