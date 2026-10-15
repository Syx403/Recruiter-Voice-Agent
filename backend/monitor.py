"""Runtime adherence monitor.

Reads each agent turn's transcript after it is spoken and detects: two questions in one turn, an
off-card question, and protected-attribute wording. What was said cannot be unsaid; the correction
rides on the next tool result and a flag is logged. Heuristics only; the eval judge does the rest.
"""
from __future__ import annotations

import re

PROTECTED = re.compile(
    r"\b(how old|your age|age are you|married|marital|spouse|husband|wife|children|kids|family plans|"
    r"pregnan\w*|religio\w*|ethnic\w*|caste|nationality|disabilit\w*|health condition|medical)\b", re.I)
CLARIFY = re.compile(r"\b(repeat|say that again|clarify|did you mean|just to confirm|could you confirm|"
                     r"sorry|i didn't catch|correct me|have it right|is that right)\b", re.I)
STOP = set("""a an the and or but if of to in on at for with about from by as is are was were be been do did
does you your yours i me my we our us it its this that these those what which who whom how when where why
can could would should will just so then there their they them have has had any some much many more most
tell walk me please okay ok also really very""".split())
EXEMPT_STAGES = {"candidate_questions", "close", "ai_objection_roleplay"}
NARRATION = re.compile(r"\blet me (capture|jot|note|record|log|think|check|unpack)\b", re.I)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2}


def questions(text: str) -> list[str]:
    return [q.strip() for q in re.findall(r"[^.?!]*\?", text) if q.strip()]


def check_turn(text: str, stage: str, card_texts: list[str]) -> list[dict]:
    issues: list[dict] = []
    if m := PROTECTED.search(text):
        issues.append({"type": "protected_attribute", "detail": f"agent said {m.group(0)!r}",
                       "correction": "Never ask about or comment on protected attributes. Move on."})
    if m := NARRATION.search(text):
        issues.append({"type": "narration", "detail": m.group(0),
                       "correction": "Do not narrate what you are doing. Acknowledge briefly, or say nothing."})
    qs = questions(text)
    if len(qs) >= 2:
        issues.append({"type": "double_question", "detail": " | ".join(qs)[:300],
                       "correction": "Ask exactly one question per turn."})
    if stage not in EXEMPT_STAGES and card_texts:
        allowed = set().union(*(_content_words(t) for t in card_texts))
        for q in qs:
            words = _content_words(q)
            if len(words) < 3 or CLARIFY.search(q):
                continue
            overlap = len(words & allowed) / len(words)
            if overlap < 0.3:
                issues.append({"type": "off_card_question", "detail": q[:200],
                               "correction": "Return to the current card; ask nothing outside it except clarifications."})
                break
    return issues
