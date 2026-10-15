"""flag_concern: red flags for the recruiter, never shown to the candidate."""
from __future__ import annotations

from backend.session import SessionState


def flag_concern(state: SessionState, type: str, detail: str) -> dict:
    state.flags.append({"t": round(state.elapsed, 1), "type": type, "detail": detail, "source": "agent",
                        "stage": state.stage})
    return {"status": "flagged"}
