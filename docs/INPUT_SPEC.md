# Input specification: the two structured Markdown documents

Audience: whoever builds the upstream extractors, i.e. the JD extractor and the sourcing agent that parses CVs.

The screener does not parse raw JDs or CVs. Its question generator reads exactly two documents per screen:

| Document | Produced by (upstream, not built here) | Path in this repo |
|---|---|---|
| `role.md` — the structured JD | a JD extractor, from a public job posting | `content/roles/<role_id>/role.md` |
| `candidate.md` — the structured CV | the sourcing agent, from a CV or profile | `data/candidates/<candidate_id>/candidate.md` |

Code parses both (`alt/pipeline/inputs.py`). The parser is **tolerant** (generality over precise extraction):
upstream extractors will vary, so it rejects only a document the screener cannot use at all (no id, no
title or name, no location for a role, no readable experience for a candidate). Everything else degrades with a
warning that the recruiter sees: an unreadable claim row is dropped or kept as text (never compared as a number), an
unknown enum value becomes `not stated`, a requirement without `[must]`/`[nice]` is read as must, a market without a
preset falls back to `content/markets/generic.yaml`, and a line that names a protected attribute is removed before
any model reads it. The parser never guesses a value. The test documents in this repo were written by hand,
standing in for the upstream extractors.

General rules for both documents:
- UTF-8 Markdown. YAML front matter between `---` lines. One `#` title. Sections are `##` headings with the exact
  names below. Unknown sections are kept as text and ignored by code.
- `- Key: value` lines inside a section. Keys are case-insensitive. Write `not stated` when the source does not say.
  Never infer a value the source does not state.
- Never include protected attributes: age, date of birth, graduation years, gender, marital status, family, religion,
  caste, ethnicity, nationality, health, photos. The parser rejects documents that mention them.

## 1. `role.md` (schema `role-md/1`)

```markdown
---
schema: role-md/1
role_id: sg_sales_lead_ai_fintech        # lowercase, digits, underscores
market: singapore                        # a preset in content/markets/<market>.yaml
source_url: https://example.com/jobs/...
source_date: 2026-09-18                  # posting date, if shown
extracted: 2026-10-15
---

# Sales Lead – AI Fintech                 (the job title)

## Company
- Name: MongoDB                          (or "Undisclosed (Cygnify client)")
- Context: one sentence on what the company sells

## Seat
- Location: Singapore                    (city, country)
- Work arrangement: on-site | hybrid | remote
- Territory: free text, or not stated
- Languages: comma-separated selling languages, or not stated
- Motion: new_logo | expansion | mixed | not stated
- Segment: smb | commercial | mid_market | enterprise | not stated
- Sales cycle: e.g. "30-120 days", or not stated
- Deal size: e.g. "USD 10k-150k", or not stated
- Team leadership: none | later | now
- Compensation: what the posting says, or not stated

## Product
- one bullet per product or offering

## Buyers
- one bullet per buyer title or persona

## Responsibilities
- one bullet per responsibility

## Requirements
- [must] R1: one requirement per bullet, as the posting states it
- [nice] N1: nice-to-haves
```

Rules:
- Required (or the document is rejected): `role_id`, the title, `Company: Name`, `Seat: Location`.
- Expected (a warning when missing): `Product`, `Buyers`, `Requirements`, the enumerated `Seat` keys with the listed
  values. Without `[must]` requirements the screen uses the generic AE cards only.
- Requirement ids are `R<n>` for must and `N<n>` for nice; missing or duplicate ids are numbered by the parser.
  One requirement per bullet: split "6–10 years and banking knowledge" into two.
- `market` names a preset in `content/markets/` (what is local and never in a JD: currency and scale words, the
  financial-year convention, how compensation is asked, whether work eligibility is asked, notice norms). An
  unknown market uses the generic preset, which asks for currency and window instead of assuming them.
- "N+ years" requirements are decided by code from the CV's dates (closing seats for a closing or quota
  requirement, all sales and business-development seats otherwise).

## 2. `candidate.md` (schema `candidate-md/1`)

```markdown
---
schema: candidate-md/1
candidate_id: kabir_new_logo
source: synthetic                        # synthetic | sourcing_agent
extracted: 2026-10-15
---

# Kabir Sen                               (display name)

## Profile
- Headline: one line
- Location: Gurugram, India
- Languages: English
- Notice: 90 days, not serving           (or not stated)
- Current compensation: not stated       (or what the CV says)

## Experience
### Account Executive | Arcbridge Labs | developer tools | 2023-03 – present
- Motion: new_logo | expansion | mixed | outbound_meetings | not stated
- Segment: smb | commercial | mid_market | enterprise | not stated
- Territory: India
- Products: API platform
- Buyers: engineering leaders
- Highlights:
  - one bullet per CV bullet, numbers kept as written

## Claims
| id | field | value | unit | window | text |
|---|---|---|---|---|---|
| c1 | quota | 40000000 | INR | FY2025-26 | ₹4.0 crore individual ARR quota in FY2025-26 |

## Education
- Bachelor of Technology | private university

## Skills
- free text
```

Rules:
- Required (or the document is rejected): `candidate_id`, the name, at least one readable experience entry.
- Experience heading: `### <title> | <company> | <company type> | <YYYY-MM> – <YYYY-MM or present>`, most recent first
  (the company type may be left out; a year alone is read as mid-year; unreadable dates leave the length unknown).
- SDR/BDR and lead-generation seats are pipeline generation, not closing (AE domain rule in code).
- `Claims` is optional: without it nothing can be checked against the call, and the recruiter sees a warning.
- Claims are the verifiable statements in the CV, one per row:
  - `field` should be one of the screener's evidence fields (a row with another field is dropped with a warning): `current_role`, `product`, `buyer`, `time_in_seat`,
    `deal_value`, `customer_outcome`, `quota`, `quota_unit`, `quota_owner`, `quota_window`, `period_months`,
    `attainment_pct`, `avg_deal`, `deals_closed`, `self_sourced_pct`, `new_logo_share`, `segment`,
    `sales_cycle_days`, `territory`, `languages`, `current_comp`, `notice_period`, `notice_status` or
    `work_location`. A statement that fits no field stays a highlight, not a claim.
  - `unit` is a currency code (INR, USD, SGD, …), `percent`, `count`, `days`, `months`, `month` (a YYYY-MM date),
    `category` or `text`. Numeric units need a plain number as the value: `40000000`, not "4 Cr".
  - `window` is the period the claim covers, as the CV states it (`FY2025-26`, `2025`, `current`).
  - `text` is the CV's own wording, so a recruiter can trace it.
- Ids are unique.

## 3. What the screener derives in code

- From `candidate.md`: current seat, years in quota-carrying roles, short stints (two or more roles under twelve
  months), and the claims table, which the question plan cites and the call checks against what the candidate says.
- From `role.md`: the deterministic part of the screen configuration (title, company, location, arrangement,
  market, motion, segment, buyers, requirements).
- The model only proposes the judgement calls: competency weights, the role-play objection, how each requirement
  is checked, and the candidate's question plan. Code validates every proposal, and a recruiter approves it.
