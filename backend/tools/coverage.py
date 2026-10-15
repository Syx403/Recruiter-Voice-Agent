"""get_coverage: weighted coverage of competencies and fit checks, plus the clock."""
from __future__ import annotations

from backend.session import SessionState


def get_coverage(state: SessionState) -> dict:
    weights = state.role.get("competency_weights") or {
        k: v["default_weight"] for k, v in state.rubric["competencies"].items()}
    captured = state.captured()
    covered_dims = {ev.dimension for ev in captured.values() if not ev.declined}
    comps = [{"id": k, "weight": w, "covered": k in covered_dims} for k, w in weights.items()]
    fits = [{"id": k, "covered": k in covered_dims} for k in state.rubric["fit_checks"]]
    gaps = sorted((c for c in comps if not c["covered"]), key=lambda c: -c["weight"])
    out = {"stage": state.stage, "competencies": comps, "fit_checks": fits,
           "highest_weight_gaps": [g["id"] for g in gaps], "clock": state.clock()}
    return out
