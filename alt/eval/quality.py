"""Does the screen tell a good answer from a poor one and from nonsense?

For each role × candidate, three simulated candidates with the same facts but different answer quality go through
the real brain (text only) and then the real brief (score → ground → compute). Code checks:
- the nonsense candidate is ended by the strict non-cooperation rule;
- the good candidate's total is above the poor candidate's, by at least MIN_GAP points out of 25;
- the good candidate is not "not recommended", and the poor one is not "recommend".

    uv run --group alt python -m alt.eval.quality          # ≈ USD 0.25
Results: runs/alt/quality/<timestamp>.json
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from alt import budget
from alt.eval.adversarial import Scenario, load_plan, run_scenario
from alt.pipeline import brief as B
from alt.pipeline import cvcheck

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "alt" / "quality"
PAIRS = [("mongodb_senior_ae_india", "arjun_inflated_attainment"), ("sg_sales_lead_ai_fintech", "nadia_bank_ai_enterprise")]
MIN_GAP = 3.0
LEVELS = [
    Scenario("good", "You are a strong candidate. Answer every question specifically with the facts you know: name "
             "the people and their roles, say what YOU did, give exact numbers with units and windows, and the "
             "customer's measured result. In the role-play, first ask a clarifying question, never over-promise, "
             "offer to bring in the right specialist and propose a concrete next step.", "completed"),
    Scenario("poor", "You are a weak but sincere candidate. Ignore most details in your facts: you remember only "
             "your title, your company and roughly what you sell. Answer briefly and vaguely, say 'we' instead of "
             "'I', cannot name who decided or what blocked a deal, give rough or no numbers ('around a hundred "
             "percent, I think'), never mention a customer result, blame others for a loss, and in the role-play "
             "promise that everything is completely safe and guaranteed. You always try to answer and stay polite.",
             None),
    Scenario("nonsense", "You are not taking it seriously and never give your real facts: every answer is "
             "throwaway, implausible or dismissive ('I closed it in one day', 'stuff, to people, that's all', 'no "
             "idea what my title is', 'a billion dollars, obviously').", "non_cooperative"),
]


async def brief_for(r: dict, plan: dict | None) -> dict:
    s = r["_session"]
    timeline = [{"kind": "agent" if who == "Interviewer" else "candidate", "t": k, "text": text}
                for k, (who, text) in enumerate(r["_convo"])]
    claims = s.candidate.get("claims") or []
    said = {ev.field: ev.value for ev in s.ledger if not ev.declined}
    cv = cvcheck.compare(claims, said, s.role.get("currency"))
    rec = {"role": s.role, "rubric": s.rubric, "ledger": [ev.__dict__ for ev in s.ledger], "flags": s.flags,
           "consistency": s.consistency, "timeline": timeline, "overrides": s.overrides,
           "requirement_fit": (plan or {}).get("requirement_fit") or [], "cv_check": cv}
    run_id = f"quality-{r['name']}-{int(time.time())}"
    draft = await B.call([{"role": "user", "content": B.build_input(**rec)}], run_id=run_id, step="alt-eval")
    clean, unmatched = B.ground(draft, B.candidate_text(timeline))
    computed = B.compute(clean, s.role, s.rubric, s.consistency, rec["ledger"], cv, rec["requirement_fit"])
    return {"total_25": computed["total_25"], "recommendation": computed["recommendation"],
            "scores": computed["scores"], "summary": clean["summary"], "dropped_quotes": unmatched}


async def one(role_id: str, cid: str, sc: Scenario) -> dict:
    plan = load_plan(role_id, cid)
    r = await run_scenario(sc, None, role_id, cid, plan)
    out = {"role": role_id, "candidate": cid, "level": sc.name, "end": r["end"], "turns": r["turns"],
           "strikes": r["max_strikes"], "flags": r["flags"], "harness_fails": r["fails"]}
    if sc.name != "nonsense":
        out["brief"] = await brief_for(r, plan)
    return out


async def main():
    budget.require(0.3, "quality check")
    results = await asyncio.gather(*(one(rid, cid, sc) for rid, cid in PAIRS for sc in LEVELS))
    ok, bad = [], []
    for rid, cid in PAIRS:
        by = {r["level"]: r for r in results if r["role"] == rid and r["candidate"] == cid}
        g, p, n = by["good"]["brief"], by["poor"]["brief"], by["nonsense"]
        checks = [
            (f"{cid}: nonsense ended as non-cooperative ({n['end']})", "non-cooperative" in str(n["end"])),
            (f"{cid}: good {g['total_25']} > poor {p['total_25']} by ≥ {MIN_GAP}",
             (g["total_25"] or 0) - (p["total_25"] or 0) >= MIN_GAP),
            (f"{cid}: good is {g['recommendation']}", g["recommendation"] != "not_recommended"),
            (f"{cid}: poor is {p['recommendation']}", p["recommendation"] != "recommend"),
        ]
        for line, passed in checks:
            (ok if passed else bad).append(line)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str))
    for r in results:
        b = r.get("brief") or {}
        print(f"{r['candidate'][:14]:14} {r['level']:9} end={str(r['end'])[:28]:28} strikes={r['strikes']} "
              f"total={b.get('total_25')} rec={b.get('recommendation')} scores={b.get('scores')}")
    print(f"\n{len(ok)} passed, {len(bad)} failed · {path.relative_to(ROOT)}")
    for b in bad:
        print("FAIL", b)
    print(f"alt spend: USD {budget.spent():.3f}")


if __name__ == "__main__":
    asyncio.run(main())
