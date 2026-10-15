---
name: performance-verification
description: "Performance stage: quota with unit, owner and window; attainment; deal size and count; self-sourced; the consistency check."
---
# Performance verification

## Objective
Capture the numbers one at a time, with their unit, owner and window, so code can check that quota x attainment
roughly equals deal size x deal count. The check, the read-back and the clarification are run by code; the
reasoning step must capture the numbers cleanly and never do the arithmetic itself.

## Evidence to capture
- quota (amount, scale, currency), quota_unit (ARR, ACV, TCV or bookings), quota_owner (individual or team),
  quota_window (a financial year, a calendar year, or the trailing four quarters; the market note says which
  financial year is usual).
- attainment_pct; avg_deal (amount, scale, currency); deals_closed.
- self_sourced_pct: deals that started from a meeting the candidate booked, not an SDR or marketing.
- quota_clarification: their explanation if the numbers did not reconcile, or did not match the CV.
- The CV check is code's: after the candidate states a figure, code compares it with the CV and may ask once, neutrally.
  Never quote a CV figure in a follow-up of your own.

## Units
Store amount, scale and currency as said (the market note lists local scale words); never add amounts in different
currencies or scales. Approximate answers and ranges are fine.

## Follow-up priority
1. Missing currency or scale ("eight" of what?). 2. quota_unit: TCV credited against an ARR quota is the most common
reason the math does not reconcile. 3. quota_owner ("we did eight million" is often a team number). 4. quota_window.

## What good looks like (rubric anchors, performance_credibility)
1: vague or badly irreconcilable. 3: mostly clear; fit middling. 5: specific, self-consistent, fits the role. Capped
at 3 if the check is inconsistent and the clarification does not resolve it.

## Notes
- No formal quota at an early-stage company is not the same fact as never owning a number: record what target they
  had; the recruiter applies the knockout.
- One strong year can be a territory event; "was that year typical?" is a fair single follow-up.

## Red flags
- Refusing or dodging numbers after being asked twice: evasive_on_numbers. Defensiveness is evidence, not a reason
  to argue. Never accuse; never reveal the check result.
