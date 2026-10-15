"""check_quota_consistency: deterministic quota maths, never done by the LLM.

quota × attainment should roughly equal average deal × deals closed (±20%), for one period.
"""
from __future__ import annotations

from backend.normalise import Amount, parse_amount, parse_count, parse_percent
from backend.session import SessionState

TOLERANCE = 0.20


CURRENCY_WORDS = {"INR": "rupees", "USD": "US dollars", "SGD": "Singapore dollars", "AUD": "Australian dollars",
                  "PHP": "pesos", "IDR": "rupiah", "MYR": "ringgit", "HKD": "Hong Kong dollars", "JPY": "yen",
                  "EUR": "euros", "GBP": "pounds"}


def _native(a: Amount) -> str:
    """The amount as the candidate would say it: '8 crore rupees', '1.2 million US dollars'."""
    unit = {"k": "thousand", "M": "million", "mil": "million", "B": "billion"}.get(a.unit or "", a.unit or "")
    cur = CURRENCY_WORDS.get(a.currency or "", a.currency or "")
    return f"{a.amount:g} {unit} {cur}".replace("  ", " ").strip()


def _fmt_amount(a: Amount) -> str:
    native = _native(a)
    if a.usd is None or a.currency == "USD":
        return native
    return f"{native} (about USD {a.usd:,.0f})"


def _fmt_usd_native(usd: float, like: Amount) -> str:
    """Express a USD figure in the candidate's own currency and unit, for the clarification."""
    if like.usd and like.amount:
        native = usd / (like.usd / like.amount)
        unit = {"k": "thousand", "M": "million", "mil": "million"}.get(like.unit or "", like.unit or "")
        return f"about {native:.3g} {unit}".replace("  ", " ").strip()
    return f"about USD {usd:,.0f}"


def check_quota_consistency(state: SessionState, quota: dict, period_months, attainment_pct, avg_deal: dict,
                            deals_closed, self_sourced_pct=None, team_avg_pct=None) -> dict:
    if state.stage != "performance_verification":
        return {"status": "refused", "reason": "only in the performance_verification stage, once quota, "
                                               "attainment, average deal and deal count are captured"}
    cur = state.role.get("currency")
    q = parse_amount(quota, cur)
    d = parse_amount(avg_deal, cur)
    att = parse_percent(attainment_pct)
    n = parse_count(deals_closed)
    months = parse_count(period_months) or 12
    state.tools_called.setdefault(state.stage, set()).add("check_quota_consistency")
    action_idx = next((i for i, c in enumerate(state.cards) if c.get("action") == "check_quota_consistency"), None)

    missing = [k for k, v in {"quota": q and q.usd, "attainment_pct": att, "avg_deal": d and d.usd,
                              "deals_closed": n}.items() if not v]
    result: dict = {"period_months": months}
    if missing:
        result.update(status="insufficient", missing=missing,
                      instruction="Ask for the missing values first, one at a time; then call this tool again.")
        state.consistency.append({"t": round(state.elapsed, 1), **result})
        return result

    if action_idx is not None:
        state.actions_done.add(state.cards[action_idx]["id"])
    claimed = q.usd * att["value"] / 100
    implied = d.usd * n
    ratio = claimed / implied if implied else float("inf")
    status = "consistent" if abs(ratio - 1) <= TOLERANCE else "inconsistent"
    notes = []
    ranges = state.role.get("expected_ranges", {})
    for key, val in (("quota_usd", q.usd * 12 / months), ("avg_deal_usd", d.usd)):
        lo_hi = ranges.get(key)
        if lo_hi and not (lo_hi[0] <= val <= lo_hi[1]):
            notes.append(f"{key} {val:,.0f} outside the role's expected range {lo_hi}")
    result.update(
        status=status,
        normalised={"quota": _fmt_amount(q), "attainment_pct": att["value"], "avg_deal": _fmt_amount(d),
                    "deals_closed": n},
        claimed_closed_usd=round(claimed), implied_closed_usd=round(implied), ratio=round(ratio, 2),
        readback=f"So that's a quota of {_native(q)}, {att['value']:g} percent attainment, "
                 f"an average deal of {_native(d)}, and {n:g} deals.",
        range_notes=notes,
        instruction="Say the readback sentence exactly as written. Never tell the candidate this result.",
    )
    if status == "inconsistent":
        clar = (f"Just so I have it right: {n:g} deals at around {_native(d)} comes to "
                f"{_fmt_usd_native(implied, q)}, and {att['value']:g} percent of the quota would be "
                f"{_fmt_usd_native(claimed, q)}. How do those two fit together?")
        result["suggested_clarification"] = clar
        result["instruction"] = ("Say the readback sentence exactly as written, then ask the suggested "
                                 "clarification once, neutrally. Never accuse, and never tell the candidate the result.")
        if not any(c["id"] == "perf_clarify" for c in state.cards):
            state.insert_card({"id": "perf_clarify", "primary": clar, "must_capture": ["quota_clarification"],
                               "probes": {}, "max_probes": 0, "strict": True}, after=action_idx)
        state.flags.append({"t": round(state.elapsed, 1), "type": "inconsistent_numbers", "source": "code",
                            "detail": f"claimed {claimed:,.0f} USD vs implied {implied:,.0f} USD (ratio {ratio:.2f})"})
    if self_sourced_pct is not None:
        result["self_sourced_pct"] = (parse_percent(self_sourced_pct) or {}).get("value")
    state.consistency.append({"t": round(state.elapsed, 1), **{k: v for k, v in result.items() if k != "instruction"}})
    return result
