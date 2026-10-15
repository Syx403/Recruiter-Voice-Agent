"""System prompt assembly.

Only stable content goes into the instructions: core rules, the persona, a short role summary, the
candidate's pre-call context and the consent stage block. Everything else arrives through tool
results, identically on every provider (parity rule).
"""
from __future__ import annotations


from backend.content_store import ContentStore
from backend.session import SessionState


def role_summary(role: dict) -> str:
    fit = role.get("fit", {})
    lines = [
        f"- Title: {role['title']} at {role['company']} ({role.get('location', '')})",
        f"- Context: {role.get('company_context', '')}",
        f"- Buyers: {', '.join(role.get('buyer_personas', []))}",
        f"- Motion: {fit.get('motion')}; segment: {fit.get('segment')}; territory: {fit.get('territory')}",
        f"- Candidates usually quote amounts in {role.get('currency', 'USD')}",
    ]
    faq = role.get("faq") or []
    if faq:
        lines.append("- What you may tell the candidate about the role:")
        lines += [f"  - {f['q']}: {f['a']}" for f in faq]
    return "\n".join(lines)


def candidate_context(c: dict) -> str:
    name = c.get("name", "the candidate").replace(" (synthetic)", "")
    lines = [f"- Name: {name}", f"- CV summary: {c.get('cv_summary', '').strip()}"]
    hist = c.get("open_items")
    if hist:
        lines.append("- Historical (may be stale), from a previous call: " + "; ".join(hist))
    return "\n".join(lines)


def build_instructions(state: SessionState, store: ContentStore) -> str:
    """The stable prefix of every turn's instructions (cache-friendly); the turn directive is appended."""
    core = store.core(state.content_hashes)
    return "\n\n".join([
        core.strip(),
        f"# Interviewer persona (how you ask, never what is assessed)\n{state.persona['body']}",
        f"# Role (authoritative)\n{role_summary(state.role)}",
        f"# Candidate (pre-call context)\n{candidate_context(state.candidate)}",
    ])


def turn_instructions(base: str, directive: str) -> str:
    return f"{base}\n\n# Your next turn\n{directive}"


def turn_note(directive: str, n: int) -> str:
    """The directive as a system message appended to the conversation. The session instructions stay
    byte-stable, so the provider's prefix cache covers the whole conversation; the label makes older notes stale."""
    return f"# Your next turn (turn note {n}; it replaces every earlier turn note)\n{directive}"
