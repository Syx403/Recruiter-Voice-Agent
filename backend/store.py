"""SQLite store (mock ATS): sessions, evidence, transcript turns and flags.

Raw provider events (audio stripped) go to runs/<session_id>/events.jsonl, not to the database.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any
from pathlib import Path

from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine, select

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "runs" / "screener.sqlite"


class ScreenSession(SQLModel, table=True):
    id: str = Field(primary_key=True)
    role_id: str
    persona_id: str
    candidate_id: str
    provider: str
    model: str
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    status: str = "created"          # created | live | ended | error
    end_reason: str | None = None
    cost_usd: float = 0.0
    content_hashes: dict = Field(default_factory=dict, sa_column=Column(JSON))
    summary: dict = Field(default_factory=dict, sa_column=Column(JSON))   # overrides, refusals, card log, clock


class EvidenceRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    t: float
    stage: str
    card: str | None
    dimension: str
    field: str
    value: str
    normalised: Any = Field(default=None, sa_column=Column(JSON))
    quote: str
    confidence: str
    declined: bool


class TurnRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    t: float
    speaker: str                     # agent | candidate
    text: str
    stage: str
    card: str | None = None


class FlagRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    t: float
    type: str
    detail: str
    source: str                      # agent | code | monitor


class Invite(SQLModel, table=True):
    """An interview link for one candidate and role. The session is created when the candidate joins;
    a link is single-use."""
    token: str = Field(primary_key=True)
    role_id: str
    persona_id: str
    candidate_id: str
    created_at: float = Field(default_factory=time.time)
    status: str = "invited"           # invited | in_call | completed
    session_id: str | None = None


class TimelineRow(SQLModel, table=True):
    """One entry of a session's conversation view: an agent or candidate turn, a tool call, a stage
    change, or a non-speech sound that was ignored. Upserted as it changes (streaming text, truncation)."""
    key: str = Field(primary_key=True)          # f"{session_id}:{entry_id}"
    session_id: str = Field(index=True)
    entry_id: str
    kind: str                                   # agent | candidate | noise | tool | stage
    t: float                                    # seconds from call start (when the speech began)
    stage: str | None = None
    text: str = ""
    data: dict = Field(default_factory=dict, sa_column=Column(JSON))


@lru_cache
def engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(f"sqlite:///{DB_PATH}")
    SQLModel.metadata.create_all(eng)
    return eng


def add(*rows):
    with Session(engine()) as s:
        for r in rows:
            s.add(r)
        s.commit()


def get_session(session_id: str) -> ScreenSession | None:
    with Session(engine()) as s:
        return s.get(ScreenSession, session_id)


def update_session(session_id: str, **fields):
    with Session(engine()) as s:
        row = s.get(ScreenSession, session_id)
        for k, v in fields.items():
            setattr(row, k, v)
        s.add(row)
        s.commit()


def list_sessions(limit: int = 50) -> list[ScreenSession]:
    with Session(engine()) as s:
        return list(s.exec(select(ScreenSession).order_by(ScreenSession.created_at.desc()).limit(limit)))


def transcript(session_id: str) -> list[TurnRow]:
    with Session(engine()) as s:
        return list(s.exec(select(TurnRow).where(TurnRow.session_id == session_id).order_by(TurnRow.t)))


class EventLog:
    """Append-only JSONL of provider and orchestrator events for one session."""

    def __init__(self, session_id: str):
        self.path = ROOT / "runs" / session_id / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a")

    def write(self, _event: str, **data):
        self.f.write(json.dumps({"ts": time.time(), "event": _event, **data}, default=str) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def upsert_timeline(session_id: str, entry: dict):
    with Session(engine()) as s:
        key = f"{session_id}:{entry['id']}"
        row = s.get(TimelineRow, key) or TimelineRow(key=key, session_id=session_id, entry_id=entry["id"],
                                                       kind=entry["kind"], t=entry["t"])
        row.kind, row.t, row.stage, row.text = entry["kind"], entry["t"], entry.get("stage"), entry.get("text", "")
        row.data = {k: v for k, v in entry.items() if k not in ("id", "kind", "t", "stage", "text")}
        s.add(row)
        s.commit()


def timeline(session_id: str) -> list[dict]:
    with Session(engine()) as s:
        rows = s.exec(select(TimelineRow).where(TimelineRow.session_id == session_id).order_by(TimelineRow.t))
        return [{"id": r.entry_id, "kind": r.kind, "t": r.t, "stage": r.stage, "text": r.text, **(r.data or {})}
                for r in rows]


def evidence(session_id: str) -> list[dict]:
    with Session(engine()) as s:
        rows = s.exec(select(EvidenceRow).where(EvidenceRow.session_id == session_id).order_by(EvidenceRow.t))
        return [r.model_dump(exclude={"id", "session_id"}) for r in rows]


def flags(session_id: str) -> list[dict]:
    with Session(engine()) as s:
        rows = s.exec(select(FlagRow).where(FlagRow.session_id == session_id).order_by(FlagRow.t))
        return [r.model_dump(exclude={"id", "session_id"}) for r in rows]


def open_invite(role_id: str, persona_id: str, candidate_id: str) -> Invite:
    """Reuse the unused link for this candidate and role (same persona), or create one."""
    import secrets
    with Session(engine()) as s:
        row = s.exec(select(Invite).where(Invite.role_id == role_id, Invite.candidate_id == candidate_id,
                                          Invite.persona_id == persona_id, Invite.status == "invited")).first()
        if row is None:
            row = Invite(token=secrets.token_urlsafe(9), role_id=role_id, persona_id=persona_id,
                         candidate_id=candidate_id)
            s.add(row)
            s.commit()
            s.refresh(row)
        return row


def get_invite(token: str) -> Invite | None:
    with Session(engine()) as s:
        return s.get(Invite, token)


def update_invite(token: str, **fields):
    with Session(engine()) as s:
        row = s.get(Invite, token)
        for k, v in fields.items():
            setattr(row, k, v)
        s.add(row)
        s.commit()


def list_invites(limit: int = 200) -> list[Invite]:
    with Session(engine()) as s:
        return list(s.exec(select(Invite).order_by(Invite.created_at.desc()).limit(limit)))
