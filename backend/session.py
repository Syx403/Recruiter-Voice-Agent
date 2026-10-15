"""Per-session working memory: stage, question cards, evidence ledger, flags, clock.

This state is never dumped into the model context; the model sees it only through tool results.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def render(text: str, ctx: dict) -> str:
    """Fill `{role.title}`-style placeholders; unknown keys are left as they are."""
    def sub(m):
        obj: Any = ctx
        for part in m.group(1).split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
            if obj is None:
                return m.group(0)
        return str(obj)
    return re.sub(r"\{([a-z_]+(?:\.[a-z_]+)+)\}", sub, text)


@dataclass
class Evidence:
    t: float
    stage: str
    card: str | None
    dimension: str
    field: str
    value: str
    normalised: Any
    quote: str
    confidence: str
    declined: bool


@dataclass
class SessionState:
    role: dict
    persona: dict
    candidate: dict
    stages: dict
    rubric: dict
    content_hashes: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    stage: str = "consent"
    cards: list[dict] = field(default_factory=list)
    card_idx: int = 0
    probes_used: dict[str, int] = field(default_factory=dict)
    actions_done: set[str] = field(default_factory=set)
    ledger: list[Evidence] = field(default_factory=list)
    tools_called: dict[str, set[str]] = field(default_factory=dict)   # stage → tool names
    flags: list[dict] = field(default_factory=list)
    consistency: list[dict] = field(default_factory=list)
    overrides: list[dict] = field(default_factory=list)
    refusals: list[dict] = field(default_factory=list)
    card_log: list[dict] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    candidate_turns: list[tuple[float, str, str]] = field(default_factory=list)   # (t, stage, text)
    agent_turns: list[tuple[float, str, str]] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    started_at: float | None = None
    ended: bool = False
    end_reason: str | None = None
    force_close: str | None = None    # reason, e.g. the cost cap
    last_instruction: dict | None = None
    stage_log: list[dict] = field(default_factory=list)                         # [{stage, t}] entries
    non_engaging: int = 0                         # off-topic/uncooperative turns since the last real answer
    clarifications: dict[str, int] = field(default_factory=dict)                # card id → clarification turns
    end_requested: bool = False       # end_interview was called before the closing line was spoken
    close_entered_at: float | None = None

    # ── context for templates ──────────────────────────────────────────────
    @property
    def ctx(self) -> dict:
        return {"role": self.role, "persona": self.persona, "candidate": self.candidate,
                "objection": self.role.get("objection_scenario", {})}

    # ── clock ─────────────────────────────────────────────────────────────
    def start_clock(self):
        if self.started_at is None:
            self.started_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def clock(self) -> dict:
        t = self.stages["time"]
        e = self.elapsed
        return {"elapsed_s": round(e), "remaining_s": max(0, round(t["must_close_s"] - e)),
                "wrap_up": e >= t["wrap_up_s"], "must_close": e >= t["must_close_s"] or bool(self.force_close)}

    # ── fields and ledger ─────────────────────────────────────────────────
    @property
    def fields(self) -> dict[str, str]:
        return self.stages["fields"]

    def captured(self) -> dict[str, Evidence]:
        """Latest evidence per field."""
        out: dict[str, Evidence] = {}
        for ev in self.ledger:
            out[ev.field] = ev
        return out

    def candidate_text(self) -> str | None:
        """What the candidate was transcribed saying in this stage and the one before (None if nothing yet)."""
        order = self.stages["order"]
        i = order.index(self.stage)
        recent = {self.stage, order[i - 1] if i > 0 else self.stage}
        texts = [t for _, st, t in self.candidate_turns if st in recent]
        return " ".join(texts) if texts else None

    def is_captured(self, f: str) -> bool:
        return f in self.captured()

    def consent_refused(self) -> bool:
        ev = self.captured().get("consent_given")
        return ev is not None and ev.normalised is False

    # ── stages and cards ──────────────────────────────────────────────────
    def stage_cfg(self, stage: str | None = None) -> dict:
        return self.stages["stages"][stage or self.stage]

    def next_stage(self) -> str | None:
        order = self.stages["order"]
        i = order.index(self.stage)
        return order[i + 1] if i + 1 < len(order) else None

    def enter_stage(self, stage: str):
        self.stage = stage
        self.stage_log.append({"stage": stage, "t": round(self.elapsed, 1)})
        if stage == "close":
            self.close_entered_at = self.elapsed
        cards = []
        for c in self.stage_cfg(stage)["cards"]:
            cond = c.get("when")
            if cond and not self.candidate.get(cond.split(".", 1)[1]):
                continue
            cards.append(self.render_card(c))
        self.cards, self.card_idx = cards, 0
        if self.cards:
            self.log_card("issued", self.cards[0]["id"])

    def render_card(self, c: dict) -> dict:
        card = {**c, "primary": render(c["primary"], self.ctx), "short": render(c.get("short") or "", self.ctx)}
        card["probes"] = {k: render(v, self.ctx) for k, v in (c.get("probes") or {}).items()}
        return card

    def current_card(self) -> dict | None:
        return self.cards[self.card_idx] if self.card_idx < len(self.cards) else None

    def insert_card(self, card: dict, after: int | None = None):
        """Insert a card after `after` (default: the current card), e.g. the consistency clarification."""
        idx = (self.card_idx if after is None else after) + 1
        self.cards.insert(max(idx, self.card_idx), card)

    def max_probes(self, card: dict) -> int:
        return min(card.get("max_probes", 0), int(self.persona.get("max_followups", 99)))

    def log_card(self, kind: str, card_id: str, **extra):
        self.card_log.append({"t": round(self.elapsed, 2), "stage": self.stage, "kind": kind, "card": card_id, **extra})

    def give_up_card(self, reason: str) -> list[str]:
        """Stall guard: the candidate has not supplied the card's missing fields after repeated turns.
        Mark them "not obtained" (counts like a decline for exit conditions, flagged for the recruiter) and move on."""
        card = self.current_card()
        if card is None or card.get("action"):
            return []
        missing = [f for f in card.get("must_capture", []) if not self.is_captured(f)]
        for f in missing:
            self.ledger.append(Evidence(t=round(self.elapsed, 1), stage=self.stage, card=card["id"],
                                        dimension=self.fields[f], field=f, value="not obtained", normalised=None,
                                        quote="", confidence="low", declined=True))
        if missing:
            self.flags.append({"t": round(self.elapsed, 1), "type": "not_obtained", "source": "code",
                               "detail": f"{', '.join(missing)} ({reason})"})
        self.log_card("gave_up", card["id"], missing=missing)
        self._advance_card()
        return missing

    def unhear_question(self) -> bool:
        """The interviewer was cut off before its question was heard: undo the last 'asked'/'probe' mark of the
        current card, so the engine asks it again. Returns True if something was undone."""
        card = self.current_card()
        if card is None:
            return False
        for i in range(len(self.card_log) - 1, -1, -1):
            e = self.card_log[i]
            if e["card"] != card["id"]:
                continue
            if e["kind"] == "asked":
                del self.card_log[i]
                return True
            if e["kind"] == "probe":
                del self.card_log[i]
                self.probes_used[card["id"]] = max(0, self.probes_used.get(card["id"], 0) - 1)
                return True
            if e["kind"] == "issued":
                return False
        return False

    def card_done(self, card: dict) -> bool:
        if card.get("action"):
            return card["id"] in self.actions_done
        return all(self.is_captured(f) for f in card.get("must_capture", []))

    def exit_status(self, stage: str | None = None) -> tuple[bool, list[str]]:
        spec = self.stage_cfg(stage).get("exit") or {}
        missing: list[str] = []
        for f in spec.get("all", []):
            if not self.is_captured(f):
                missing.append(f)
        if "min_of" in spec:
            n, of = spec["min_of"]["n"], spec["min_of"]["of"]
            have = sum(self.is_captured(f) for f in of)
            if have < n:
                missing.append(f"{n - have} more of {[f for f in of if not self.is_captured(f)]}")
        for tool in spec.get("tool_called", []):
            if tool not in self.tools_called.get(stage or self.stage, set()):
                missing.append(f"call {tool}")
        if (stage or self.stage) == "consent" and self.consent_refused():
            missing.append("consent refused: only `close` is allowed")
        return not missing, missing

    def cards_exhausted(self) -> bool:
        return self.current_card() is None

    def next_instruction(self) -> dict:
        """What the agent asks next, chosen by code. Called once per tool batch."""
        self.last_instruction = self._next_instruction()
        return self.last_instruction

    def repeat_instruction(self) -> dict | None:
        """The last instruction again, without consuming probes (used by refusals)."""
        return self.last_instruction

    def _next_instruction(self) -> dict:
        clk = self.clock()
        if clk["must_close"] and self.stage != "close":
            return {"type": "must_close", "say": "Time is up. Call advance_stage(to='close') now and close politely."}
        if self.stage == "consent" and self.consent_refused():
            return {"type": "consent_refused", "say": "The candidate declined. Call advance_stage(to='close')."}
        if clk["wrap_up"] and self.stage in self.stages["order"][:6]:
            return {"type": "wrap_up",
                    "say": "Wrap-up time. Call advance_stage(to='candidate_questions', override_reason='time')."}
        while (card := self.current_card()) is not None:
            if card.get("action"):
                if card["id"] in self.actions_done:
                    self._advance_card()
                    continue
                return {"type": "action", "card": card["id"], "say": card["primary"]}
            missing = [f for f in card.get("must_capture", []) if not self.is_captured(f)]
            used = self.probes_used.get(card["id"], 0)
            if not missing and used >= card.get("min_probes", 0):
                self._advance_card()
                continue
            if not missing:   # a follow-up that is always asked once the field is captured
                f = next(iter(card["probes"]))
                self.probes_used[card["id"]] = used + 1
                self.log_card("probe", card["id"], field=f)
                return {"type": "probe", "card": card["id"], "question": card["probes"][f], "capture": [f]}
            asked = any(e["card"] == card["id"] and e["kind"] == "asked" for e in self.card_log)
            if not asked:
                self.log_card("asked", card["id"])
                return {"type": "ask", "card": card["id"], "question": card["primary"], "capture": missing}
            if used < self.max_probes(card):
                f = next((m for m in missing if m in card["probes"]), missing[0])
                self.probes_used[card["id"]] = used + 1
                probe = card["probes"].get(f) or f"Ask briefly for: {f.replace('_', ' ')}."
                self.log_card("probe", card["id"], field=f)
                return {"type": "probe", "card": card["id"], "question": probe, "capture": missing}
            self.log_card("exhausted", card["id"], missing=missing)
            self._advance_card()
        ok, missing = self.exit_status()
        nxt = self.next_stage()
        if self.stage == "close":
            return {"type": "close", "say": "Say your closing line, then call end_interview."}
        if ok:
            return {"type": "stage_complete", "say": f"Stage complete. Call advance_stage(to='{nxt}')."}
        return {"type": "cards_exhausted", "missing": missing,
                "say": f"All cards asked; still missing {missing}. "
                       f"Call advance_stage(to='{nxt}', override_reason='declined')."}

    def _advance_card(self):
        self.card_idx += 1
        if (c := self.current_card()) is not None:
            self.log_card("issued", c["id"])

    def card_payload(self) -> dict | None:
        """The current card as the model sees it. Marks it as asked."""
        card = self.current_card()
        if card is None:
            return None
        if card.get("action"):
            return {"type": "action", "card": card["id"], "say": card["primary"]}
        self.log_card("asked", card["id"])
        self.last_instruction = {"type": "ask", "card": card["id"], "question": card["primary"],
                                 "capture": [f for f in card.get("must_capture", []) if not self.is_captured(f)]}
        return self.last_instruction

    # ── snapshot for the monitor ──────────────────────────────────────────
    def snapshot(self) -> dict:
        card = self.current_card()
        return {
            "id": self.id, "stage": self.stage, "order": self.stages["order"], "stage_log": self.stage_log,
            "stage_targets": {k: v["target_s"] for k, v in self.stages["stages"].items()},
            "card": {"id": card["id"], "primary": card["primary"],
                     "probes_used": self.probes_used.get(card["id"], 0),
                     "max_probes": self.max_probes(card),
                     "last_probe": next((e.get("field") for e in reversed(self.card_log)
                                         if e["card"] == card["id"] and e["kind"] == "probe"), None)}
            if card else None,
            "clock": self.clock(),
            "ledger": [ev.__dict__ for ev in self.ledger],
            "flags": self.flags, "consistency": self.consistency,
            "overrides": self.overrides, "refusals": self.refusals, "rejections": self.rejections,
            "ended": self.ended, "end_reason": self.end_reason,
        }
