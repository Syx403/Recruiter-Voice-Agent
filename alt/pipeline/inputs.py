"""Parsers for the two structured Markdown inputs (docs/INPUT_SPEC.md): `role.md` and `candidate.md`.

Tolerant by design. Upstream extractors will vary, so only a
document the screener cannot use at all is rejected (no id, no name or title, no readable experience). Everything else
degrades with a warning the recruiter sees: an unreadable claim row is dropped, an unknown enum value becomes
"not stated", an unknown market falls back to the generic preset, a line naming a protected attribute is removed
before any model reads it. The parser never guesses a value.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MARKETS = ROOT / "content" / "markets"
GENERIC_MARKET = "generic"

ROLE_SCHEMA, CANDIDATE_SCHEMA = "role-md/1", "candidate-md/1"
ARRANGEMENTS = {"on-site", "hybrid", "remote"}
MOTIONS = {"new_logo", "expansion", "mixed"}
CV_MOTIONS = MOTIONS | {"outbound_meetings"}
SEGMENTS = {"smb", "commercial", "mid_market", "enterprise"}
LEADERSHIP = {"none", "later", "now"}
CLAIM_FIELDS = {"current_role", "product", "buyer", "time_in_seat", "deal_value", "customer_outcome", "quota",
                "quota_unit", "quota_owner", "quota_window", "period_months", "attainment_pct", "avg_deal",
                "deals_closed", "self_sourced_pct", "new_logo_share", "segment", "sales_cycle_days", "territory",
                "languages", "current_comp", "notice_period", "notice_status", "work_location"}
NUMERIC_UNITS = {"percent", "count", "days", "months"}
TEXT_UNITS = {"month", "category", "text"}
CURRENCY = re.compile(r"^[A-Z]{3}(_[A-Z]+)?$")        # INR, USD, SGD; INR_ARR is a currency with a basis
# Pipeline-generation seats that do not close: AE domain knowledge, not a property of any test CV.
NON_CLOSING_TITLE = re.compile(r"\b(sales|business) development (rep|representative)\b|\b[SB]DR\b|"
                               r"\blead generation\b|\bappointment sett", re.I)
# Protected attributes in a document. Narrower than the interviewer's spoken-line regex (backend/monitor.PROTECTED):
# a CV may name a medical-devices product or a children's-education customer.
PROTECTED_DOCS = re.compile(
    r"\b(age:|aged \d+|\d+ years old|date of birth|dob|born (in|on)|gender|sex:|marital|married|spouse|"
    r"religio\w*|caste|ethnic\w*|nationality|citizen\w*|permanent resident|disabilit\w*|health condition|"
    r"pregnan\w*|photo(graph)?:)", re.I)


class InputError(ValueError):
    """The document cannot be used at all."""
    def __init__(self, kind: str, issues: list[str]):
        super().__init__(f"{kind}: " + "; ".join(issues))
        self.kind, self.issues = kind, issues


# ── shared ───────────────────────────────────────────────────────────────────

def _front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    try:
        return yaml.safe_load(m.group(1)) or {}, m.group(2)
    except yaml.YAMLError:
        return {}, m.group(2)


def _sections(body: str) -> tuple[str, dict[str, str]]:
    title = ""
    if m := re.search(r"^# (.+)$", body, re.M):
        title = m.group(1).strip()
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    return title, {parts[i].strip().lower(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _kv(section: str) -> dict[str, str]:
    out = {}
    for line in section.splitlines():
        if m := re.match(r"^- ([A-Za-z][A-Za-z /]*?):\s*(.*)$", line.strip()):
            out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


def _bullets(section: str) -> list[str]:
    return [line.strip()[2:].strip() for line in section.splitlines() if line.startswith("- ")]


def _stated(v: str | None) -> str | None:
    return None if v is None or v.strip().lower() in ("", "not stated", "n/a", "none stated", "unknown") else v.strip()


def _enum(v: str | None, allowed: set[str], what: str, warnings: list[str]) -> str:
    v = (_stated(v) or "").lower().replace("-", "_").replace(" ", "_")
    v = {"onsite": "on-site", "on_site": "on-site"}.get(v, v)
    if v and v not in allowed:
        warnings.append(f"{what} {v!r} is not one of {sorted(allowed)}: treated as not stated")
    return v if v in allowed else "not stated"


def _redact(text: str, warnings: list[str]) -> str:
    """Remove every line that names a protected attribute; the model never reads it."""
    out = []
    for line in text.split("\n"):
        if (m := PROTECTED_DOCS.search(line)) and not line.startswith(("schema:", "role_id:", "candidate_id:")):
            warnings.append(f"removed a line naming a protected attribute ({m.group(0)!r}); upstream should not send it")
            continue
        out.append(line)
    return "\n".join(out)


def markets() -> set[str]:
    return {p.stem for p in MARKETS.glob("*.yaml")}


# ── role.md ──────────────────────────────────────────────────────────────────

def _range(text: str | None) -> list[float] | None:
    """'30-120 days' → [30, 120]; 'USD 10k-150k' → [10000, 150000]."""
    if not text:
        return None
    nums = [float(n) * {"k": 1e3, "m": 1e6}.get(suf.lower(), 1)
            for n, suf in re.findall(r"(\d+(?:\.\d+)?)\s*([kKmM]?)(?![A-Za-z])", text)]
    return nums[:2] if len(nums) >= 2 else None


def _cycle_days(text: str | None) -> list[float] | None:
    r = _range(text)
    if not r:
        return None
    unit = (text or "").lower()
    return [x * 30 for x in r] if "month" in unit else [x * 7 for x in r] if "week" in unit else r if "day" in unit \
        else None


def parse_role(text: str) -> dict:
    fatal, warnings = [], []
    text = _redact(text, warnings)
    fm, body = _front_matter(text)
    title, sec = _sections(body)
    if fm.get("schema") != ROLE_SCHEMA:
        warnings.append(f"front matter: schema is not {ROLE_SCHEMA}; read as that version")
    rid = str(fm.get("role_id") or "")
    if not re.fullmatch(r"[a-z0-9_]+", rid):
        fatal.append("front matter: role_id must be lowercase letters, digits and underscores")
    if not title:
        fatal.append("missing the '# <job title>' line")
    company, seat = _kv(sec.get("company", "")), _kv(sec.get("seat", ""))
    if not company.get("name"):
        fatal.append("Company: Name is required")
    if not _stated(seat.get("location")):
        fatal.append("Seat: Location is required")
    if fatal:
        raise InputError("role.md", fatal)
    market = str(fm.get("market") or "")
    if market not in markets():
        warnings.append(f"market {market!r} has no preset in content/markets/: the generic preset is used")
        market = GENERIC_MARKET
    for s in ("product", "buyers", "requirements"):
        if s not in sec:
            warnings.append(f"no '## {s.title()}' section")
    reqs, seen = [], set()
    for b in _bullets(sec.get("requirements", "")):
        m = re.match(r"^\[(must|nice)\]\s+(?:([RN]\d+):\s*)?(.+)$", b, re.I)
        kind, rq, txt = (m.group(1).lower(), m.group(2), m.group(3)) if m else ("must", None, b)
        if not m:
            warnings.append(f"requirement without [must]/[nice] read as must: {b[:60]!r}")
        if not rq or rq in seen:
            n = sum(1 for r in reqs if r["kind"] == kind) + 1
            rq = f"{'R' if kind == 'must' else 'N'}{n}"
            while rq in seen:
                n += 1
                rq = f"{'R' if kind == 'must' else 'N'}{n}"
        seen.add(rq)
        reqs.append({"id": rq, "kind": kind, "text": txt.strip()})
    if not any(r["kind"] == "must" for r in reqs):
        warnings.append("no [must] requirement: the screen uses the generic AE cards only")
    loc = seat["location"]
    cycle = _stated(seat.get("sales cycle"))
    return {
        "role_id": rid, "market": market, "source_url": fm.get("source_url"),
        "source_date": str(fm.get("source_date") or ""), "title": title,
        "company": company["name"], "company_context": company.get("context", ""),
        "location": loc, "city": loc.split(",")[0].strip(),
        "work_arrangement": _enum(seat.get("work arrangement"), ARRANGEMENTS, "Work arrangement", warnings),
        "territory": _stated(seat.get("territory")),
        "languages": [x.strip() for x in (_stated(seat.get("languages")) or "").split(",") if x.strip()],
        "motion": _enum(seat.get("motion"), MOTIONS, "Motion", warnings),
        "segment": _enum(seat.get("segment"), SEGMENTS, "Segment", warnings),
        "sales_cycle": cycle, "sales_cycle_days": _cycle_days(cycle),
        "deal_size": _stated(seat.get("deal size")),
        "team_leadership": _enum(seat.get("team leadership"), LEADERSHIP, "Team leadership", warnings),
        "compensation": _stated(seat.get("compensation")),
        "products": _bullets(sec.get("product", "")), "buyers": _bullets(sec.get("buyers", "")),
        "responsibilities": _bullets(sec.get("responsibilities", "")), "requirements": reqs,
        "warnings": warnings,
    }


# ── candidate.md ─────────────────────────────────────────────────────────────

def _ym(s: str) -> date | None:
    s = s.strip()
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})", s):
        return date(int(m.group(1)), int(m.group(2)), 1)
    if m := re.fullmatch(r"(\d{4})", s):
        return date(int(m.group(1)), 7, 1)            # a year alone: mid-year, so months are roughly right
    return None


def _months(a: date, b: date) -> int:
    return max(0, (b.year - a.year) * 12 + b.month - a.month)


def _experience(section: str, warnings: list[str], today: date) -> list[dict]:
    out = []
    for block in re.split(r"^### ", section, flags=re.M)[1:]:
        head, *rest = block.split("\n")
        parts = [p.strip() for p in head.split("|")]
        if len(parts) == 3:
            parts.insert(2, "not stated")                # no company type
        if len(parts) != 4:
            warnings.append(f"Experience: heading {head[:60]!r} needs 'title | company | company type | dates'; skipped")
            continue
        title, company, ctype, dates = parts
        dm = re.fullmatch(r"(\d{4}(?:-\d{1,2})?)\s*(?:[–—-]+|to)\s*(\d{4}(?:-\d{1,2})?|present|current|now)",
                          dates.strip(), re.I)
        current = bool(dm) and dm.group(2).lower() in ("present", "current", "now")
        start = _ym(dm.group(1)) if dm else None
        end = today.replace(day=1) if current else (_ym(dm.group(2)) if dm else None)
        if not dm:
            warnings.append(f"Experience ({company}): dates {dates!r} not readable; length unknown")
        kv = _kv("\n".join(rest))
        hi = re.search(r"^- Highlights:\s*\n((?:\s+- .*\n?)*)", "\n".join(rest), re.M)
        highlights = [h.strip()[2:] for h in (hi.group(1).splitlines() if hi else []) if h.strip().startswith("- ")]
        motion = _enum(kv.get("motion"), CV_MOTIONS, f"Experience ({company}): Motion", warnings)
        out.append({"title": title, "company": company, "company_type": ctype,
                    "start": dm.group(1) if dm else None, "end": dm.group(2).lower() if dm else None,
                    "months": _months(start, end) if start and end else None, "current": current,
                    "closing": motion != "outbound_meetings" and not NON_CLOSING_TITLE.search(title),
                    "motion": motion, "segment": _enum(kv.get("segment"), SEGMENTS, f"Experience ({company}): Segment",
                                                       warnings),
                    "territory": _stated(kv.get("territory")), "products": _stated(kv.get("products")),
                    "buyers": _stated(kv.get("buyers")), "highlights": highlights})
    return out


def _claims(section: str | None, warnings: list[str]) -> list[dict]:
    if section is None:
        warnings.append("no '## Claims' table: nothing on the CV can be checked against the call")
        return []
    rows = [r for r in section.splitlines() if r.strip().startswith("|")]
    if len(rows) < 3:
        warnings.append("Claims: the table is empty")
        return []
    head = [h.strip().lower() for h in rows[0].strip().strip("|").split("|")]
    if head != ["id", "field", "value", "unit", "window", "text"]:
        warnings.append("Claims: columns are not id | field | value | unit | window | text; table ignored")
        return []
    out, seen = [], set()
    for r in rows[2:]:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if len(cells) != 6:
            warnings.append(f"Claims: row {r[:50]!r} does not have 6 cells; dropped")
            continue
        cid, fld, val, unit, window, txt = cells
        if not cid or cid in seen:
            warnings.append(f"Claims {cid!r}: missing or duplicate id; dropped")
            continue
        if fld not in CLAIM_FIELDS:
            warnings.append(f"Claims {cid}: field {fld!r} is not a screener field; dropped (stays a CV highlight)")
            continue
        numeric = unit in NUMERIC_UNITS or bool(CURRENCY.match(unit))
        if numeric:
            try:
                v = float(val.replace(",", ""))
                val = int(v) if v.is_integer() else v
            except ValueError:
                warnings.append(f"Claims {cid}: value {val!r} is not a plain number; kept as text, not compared")
                numeric, unit = False, "text"
        elif unit not in TEXT_UNITS:
            warnings.append(f"Claims {cid}: unit {unit!r} unknown; kept as text, not compared")
            unit = "text"
        seen.add(cid)
        out.append({"id": cid, "field": fld, "value": val, "unit": unit, "window": window, "text": txt,
                    "check": "numeric" if numeric else "factual"})
    return out


def parse_candidate(text: str, today: date | None = None) -> dict:
    fatal, warnings = [], []
    today = today or date.today()
    text = _redact(text, warnings)
    fm, body = _front_matter(text)
    name, sec = _sections(body)
    if fm.get("schema") != CANDIDATE_SCHEMA:
        warnings.append(f"front matter: schema is not {CANDIDATE_SCHEMA}; read as that version")
    cid = str(fm.get("candidate_id") or "")
    if not re.fullmatch(r"[a-z0-9_]+", cid):
        fatal.append("front matter: candidate_id must be lowercase letters, digits and underscores")
    if not name:
        fatal.append("missing the '# <name>' line")
    exp = _experience(sec.get("experience", ""), warnings, today)
    if not exp:
        fatal.append("Experience: no readable '### title | company | type | dates' entries")
    if fatal:
        raise InputError("candidate.md", fatal)
    profile = _kv(sec.get("profile", ""))
    claims = _claims(sec.get("claims"), warnings)
    finished = [e for e in exp if not e["current"]]
    short = [e for e in finished[:-1] if e["months"] is not None and e["months"] < 12]   # the first job is not a stint
    return {
        "candidate_id": cid, "source": fm.get("source", ""), "name": name,
        "headline": profile.get("headline", ""), "location": _stated(profile.get("location")),
        "languages": [x.strip() for x in (_stated(profile.get("languages")) or "").split(",") if x.strip()],
        "notice": _stated(profile.get("notice")), "current_comp": _stated(profile.get("current compensation")),
        "experience": exp, "current": next((e for e in exp if e["current"]), None), "claims": claims,
        "education": _bullets(sec.get("education", "")), "skills": _bullets(sec.get("skills", "")),
        "quota_carrying_months": sum(e["months"] or 0 for e in exp if e["closing"]),
        "short_tenure": len(short) >= 2, "warnings": warnings,
    }


# ── loading ──────────────────────────────────────────────────────────────────

def role_path(role_id: str) -> Path:
    return ROOT / "content" / "roles" / role_id / "role.md"


def candidate_path(candidate_id: str) -> Path:
    return ROOT / "data" / "candidates" / candidate_id / "candidate.md"


def load_role(role_id: str) -> tuple[dict, str]:
    """The parsed role and the text the models read (protected lines removed)."""
    text = role_path(role_id).read_text()
    parsed = parse_role(text)
    return parsed, _redact(text, [])


def load_candidate(candidate_id: str) -> tuple[dict, str]:
    text = candidate_path(candidate_id).read_text()
    parsed = parse_candidate(text)
    return parsed, _redact(text, [])


def list_roles() -> list[str]:
    return sorted(p.parent.name for p in (ROOT / "content" / "roles").glob("*/role.md"))


def list_candidates() -> list[str]:
    return sorted(p.parent.name for p in (ROOT / "data" / "candidates").glob("*/candidate.md"))


def market(market_id: str) -> dict:
    p = MARKETS / f"{market_id}.yaml"
    return yaml.safe_load((p if p.exists() else MARKETS / f"{GENERIC_MARKET}.yaml").read_text())


def amount_scale_words() -> set[str]:
    """Every scale word any market preset names (thousand, million, lakh, crore, …): a question that states an
    amount uses one of them."""
    words = {"thousand", "million", "billion", "k", "m", "mn", "bn"}
    for p in MARKETS.glob("*.yaml"):
        words |= {w.lower() for w in (yaml.safe_load(p.read_text()) or {}).get("scale_words", [])}
    return words
