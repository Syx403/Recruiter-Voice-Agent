"""Shared normaliser for spoken or written amounts and percentages.

Used by check_quota_consistency, the post-call grounding check and the eval scoring, so all three
agree on what "1.2 crore" or "about a hundred and forty percent" means.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_APPROX = re.compile(r"\b(about|around|roughly|approximately|approx|nearly|almost|close to|over|under|~)\b|~")
_CURRENCY_WORDS = {
    "usd": "USD", "us dollars": "USD", "dollars": "USD", "dollar": "USD", "$": "USD",
    "inr": "INR", "rupees": "INR", "rupee": "INR", "rs": "INR", "₹": "INR",
    "sgd": "SGD", "singapore dollars": "SGD", "aud": "AUD", "australian dollars": "AUD",
    "php": "PHP", "pesos": "PHP", "idr": "IDR", "rupiah": "IDR", "myr": "MYR", "ringgit": "MYR",
    "hkd": "HKD", "jpy": "JPY", "yen": "JPY", "eur": "EUR", "euros": "EUR", "gbp": "GBP", "pounds": "GBP",
}


@lru_cache
def fx() -> dict:
    return yaml.safe_load((ROOT / "config/fx.yaml").read_text())


@dataclass
class Amount:
    amount: float               # in `unit`
    unit: str | None
    currency: str | None
    usd: float | None           # None if the currency is unknown
    approx: bool = False
    range_usd: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def words_to_number(text: str) -> str:
    """Replace spelled-out numbers ("one point two", "a hundred and forty") with digits."""
    # gpt-live-1 transcribes decimals as "1 point 2".
    text = re.sub(r"(\d+)\s+point\s+(\d+)", r"\1.\2", text)
    # A homophone before a scale word: "for crore" (seen in gpt-live-1 transcripts). "to" is left alone:
    # "up to a million" is common.
    text = re.sub(r"\b(?:for|fore)(?=\s+(?:crores?|lakhs?|million|thousand|hundred|percent)\b)", "four", text,
                  flags=re.I)
    toks = re.findall(r"[A-Za-z]+|[^A-Za-z\s]+|\s+", text)
    words = [t.lower() for t in toks]

    def is_num(w):
        return w in _SMALL or w in _TENS or w == "hundred"

    out, i = [], 0
    while i < len(toks):
        w = words[i]
        # A bare "hundred and forty" (no "a") starts a number too (otherwise it parses as 40).
        starts = is_num(w) or (w == "a" and _peek(words, i) == "hundred")
        if not starts:
            out.append(toks[i]); i += 1; continue
        cur, frac, j, last = 0, "", i, i
        if w == "a":
            cur, j = 1, _skip(words, i)
        while j < len(words):
            w = words[j]
            if w in _SMALL or w in _TENS:
                cur += _SMALL.get(w, 0) + _TENS.get(w, 0)
            elif w == "hundred":
                cur = (cur or 1) * 100
            elif w == "and" and is_num(_peek(words, j) or "") and cur >= 100:
                pass
            elif w == "-" and words[last] in _TENS and (_peek(words, j) or "") in _SMALL:
                pass
            elif w == "point" and (_peek(words, j) or "") in _SMALL:
                k = _skip(words, j)
                while k < len(words) and words[k] in _SMALL:
                    frac += str(_SMALL[words[k]]); last = k; k = _skip(words, k)
                j = k
                break
            else:
                break
            last = j
            j = _skip(words, j)
        out.append(f"{cur}.{frac}" if frac else str(cur))
        i = last + 1
    return "".join(out)


def _skip(words, i):
    """Index of the next non-space token after i."""
    j = i + 1
    while j < len(words) and words[j].isspace():
        j += 1
    return j


def _peek(words, i):
    j = _skip(words, i)
    return words[j] if j < len(words) else None


def _currency(text: str, default: str | None) -> str | None:
    t = text.lower()
    for k in sorted(_CURRENCY_WORDS, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", t):
            return _CURRENCY_WORDS[k]
    return default


def _unit(text: str) -> str | None:
    mult = fx()["multipliers"]
    m = re.search(r"(\d)\s*(k|m|b)\b", text, re.I)
    if m:
        return {"k": "k", "m": "M", "b": "B"}[m.group(2).lower()]
    for u in ("crore", "lakh", "billion", "million", "mil", "thousand"):
        if re.search(rf"\b{u}s?\b", text, re.I):
            return u if u in mult else None
    return None


def to_usd(amount: float, unit: str | None, currency: str | None) -> float | None:
    if currency is None:
        return None
    rate = fx()["usd_per_unit"].get(currency.upper())
    if rate is None:
        return None
    mult = fx()["multipliers"].get(unit, 1) if unit else 1
    return amount * mult * rate


def parse_amount(value, default_currency: str | None = None) -> Amount | None:
    """Accepts {amount, unit, currency} or free text ("1.2 crore", "500k SGD", "around forty lakh")."""
    if value is None:
        return None
    if isinstance(value, dict):
        amt = value.get("amount")
        if amt is None:
            return None
        unit = value.get("unit") or None
        if unit and unit.lower() in ("m", "mn", "million", "mil"):
            unit = "M"
        elif unit and unit.lower() in ("k", "thousand"):
            unit = "k"
        elif unit and unit.lower() in ("lakh", "lakhs", "lac"):
            unit = "lakh"
        elif unit and unit.lower() in ("crore", "crores", "cr"):
            unit = "crore"
        cur = (value.get("currency") or default_currency)
        cur = cur.upper() if cur else None
        return Amount(float(amt), unit, cur, to_usd(float(amt), unit, cur), bool(value.get("approx")))
    if isinstance(value, (int, float)):
        cur = default_currency
        return Amount(float(value), None, cur, to_usd(float(value), None, cur))
    text = _after_correction(words_to_number(str(value)))
    approx = bool(_APPROX.search(str(value).lower()))
    nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
    if not nums:
        return None
    unit = _unit(text) or _unit(str(value))
    explicit = _currency(str(value), None)
    cur = explicit or ("INR" if unit in ("lakh", "crore") else default_currency)
    if len(nums) >= 2 and re.search(r"\d\s*(-|to|–)\s*\d", text):
        lo, hi = to_usd(nums[0], unit, cur), to_usd(nums[1], unit, cur)
        mid = (nums[0] + nums[1]) / 2
        return Amount(mid, unit, cur, to_usd(mid, unit, cur), True, [lo, hi] if lo is not None else None)
    return Amount(nums[0], unit, cur, to_usd(nums[0], unit, cur), approx)


def _after_correction(text: str) -> str:
    """Self-corrections ("1.2, sorry, 1.4 million") keep only what follows the last correction marker."""
    parts = re.split(r"\b(?:sorry|actually|i mean|correction|no wait)\b", text, flags=re.I)
    tail = parts[-1]
    return tail if re.search(r"\d", tail) else text


def parse_percent(value) -> dict | None:
    """Returns {value (0-100), approx, range} or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return {"value": v * 100 if 0 < v <= 1.5 and isinstance(value, float) and v != int(v) else v,
                "approx": False, "range": None}
    raw = str(value)
    text = _after_correction(words_to_number(raw))
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return None
    approx = bool(_APPROX.search(raw.lower()))
    if len(nums) >= 2 and re.search(r"\d\s*%?\s*(-|to|–)\s*\d", text):
        return {"value": (nums[0] + nums[1]) / 2, "approx": True, "range": [nums[0], nums[1]]}
    return {"value": nums[0], "approx": approx, "range": None}


def parse_count(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    nums = re.findall(r"\d+(?:\.\d+)?", words_to_number(str(value)))
    return float(nums[0]) if nums else None


def matches(observed: float | None, truth: float, rng: list[float] | None = None, tol: float = 0.05) -> bool:
    """A value is correct if within ±5% of the truth, or if its range contains the truth."""
    if rng and rng[0] is not None and rng[0] <= truth <= rng[1]:
        return True
    if observed is None:
        return False
    if truth == 0:
        return observed == 0
    return abs(observed - truth) / abs(truth) <= tol
