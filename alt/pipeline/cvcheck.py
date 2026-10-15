"""CV check: numbers the candidate states are compared with candidate.md's claims in code, never by a
model. Blind recall: questions never state the CV's numbers; only after the candidate has given a figure may a
neutral follow-up quote the CV.

Used in the call (after the quota check: a performance figure that differs from the CV inserts `perf_cv`) and in the
brief (every comparable claim).
"""
from __future__ import annotations

from backend.normalise import parse_amount, parse_count, parse_percent
from backend.tools.consistency import CURRENCY_WORDS

AMOUNT = {"quota", "avg_deal", "deal_value"}
PERCENT = {"attainment_pct", "self_sourced_pct", "new_logo_share"}
COUNT = {"deals_closed", "notice_period", "sales_cycle_days", "period_months"}
COMPARABLE = AMOUNT | PERCENT | COUNT
PERFORMANCE = ["quota", "attainment_pct", "avg_deal", "deals_closed"]
AMOUNT_TOL, PERCENT_TOL, COUNT_TOL = 0.15, 5.0, 0.15


def _usd(value, currency: str | None) -> float | None:
    a = parse_amount(value, currency)
    return a.usd if a else None


def _claim_usd(c: dict) -> float | None:
    cur = c["unit"].split("_")[0]
    a = parse_amount({"amount": float(c["value"]), "unit": "none", "currency": cur})
    return a.usd if a else None


def _said_number(field: str, value: str) -> float | None:
    if field in PERCENT:
        p = parse_percent(value)
        return p["value"] if p else None
    n = parse_count(value)
    if n is not None and field in ("notice_period", "sales_cycle_days") and "month" in str(value).lower():
        n *= 30
    return n


def _match(field: str, cv: float, said: float) -> bool:
    if field in PERCENT:
        return abs(cv - said) <= PERCENT_TOL
    tol = AMOUNT_TOL if field in AMOUNT else COUNT_TOL
    return abs(cv - said) <= max(tol * abs(cv), 1 if field in COUNT else 0)


def compare(claims: list[dict], said: dict[str, str], currency: str | None) -> list[dict]:
    """claims: candidate.md claims; said: field → the candidate's value (declined fields excluded).
    One row per comparable claim: match | differs | not_obtained."""
    rows = []
    for c in claims:
        f = c["field"]
        if f not in COMPARABLE or c["check"] != "numeric":
            continue
        row = {"claim": c["id"], "field": f, "cv": c["text"], "window": c["window"], "said": said.get(f)}
        if f not in said:
            rows.append({**row, "status": "not_obtained"})
            continue
        if f in AMOUNT:
            cv, sv = _claim_usd(c), _usd(said[f], currency)
        else:
            cv, sv = float(c["value"]), _said_number(f, said[f])
        if cv is None or sv is None:
            rows.append({**row, "status": "not_comparable"})
            continue
        rows.append({**row, "status": "match" if _match(f, cv, sv) else "differs"})
    # A field with several claims (two years of attainment) matches if any claim matches.
    matched = {r["field"] for r in rows if r["status"] == "match"}
    return [r for r in rows if not (r["status"] == "differs" and r["field"] in matched)]


def amount_words(value: float, currency: str, style: str = "western") -> str:
    """How a CV amount is read back after recall, in the market's convention (content/markets/*.yaml amount_style)."""
    cur = currency.split("_")[0]
    words = CURRENCY_WORDS.get(cur, cur)
    if style == "indian":
        if value >= 1e7:
            return f"{value / 1e7:g} crore {words}"
        if value >= 1e5:
            return f"{value / 1e5:g} lakh {words}"
    if value >= 1e6:
        return f"{value / 1e6:g} million {words}"
    if value >= 1e3:
        return f"{value / 1e3:g} thousand {words}"
    return f"{value:g} {words}"


def cv_phrase(c: dict, style: str = "western") -> str:
    v = float(c["value"])
    return {"quota": lambda: f"a quota of {amount_words(v, c['unit'], style)}",
            "attainment_pct": lambda: f"{v:g} percent attainment",
            "avg_deal": lambda: f"an average deal of {amount_words(v, c['unit'], style)}",
            "deals_closed": lambda: f"{v:g} deals closed"}[c["field"]]()


def followup(rows: list[dict], claims: list[dict], style: str = "western") -> str | None:
    """The neutral follow-up after recall, when a performance figure differs from the CV (never an accusation)."""
    differs = [r for r in rows if r["status"] == "differs" and r["field"] in PERFORMANCE]
    if not differs:
        return None
    by_id = {c["id"]: c for c in claims}
    window = by_id[differs[0]["claim"]]["window"]
    same = [by_id[r["claim"]] for r in differs if by_id[r["claim"]]["window"] == window][:3]
    parts = [cv_phrase(c, style) for c in same]
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"Your CV lists {listed} for {window}. How does that fit with the figures you just gave me?"
