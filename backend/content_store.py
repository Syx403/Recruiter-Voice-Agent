"""Content store: list and read the specialist content through a storage interface.

Now: the local filesystem. Production: a per-tenant store. Every read returns the content hash, so a
session can record exactly which versions it used.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
DATA = ROOT / "data"

KINDS = {
    "core": ("core", "*.md"),
    "personas": ("interviewer_personas", "*.md"),
    "roles": ("roles", "*/brief.yaml"),
    "skills": ("skills", "*/SKILL.md"),
}


class Storage(Protocol):
    def read(self, rel: str) -> str: ...
    def exists(self, rel: str) -> bool: ...
    def glob(self, rel_dir: str, pattern: str) -> list[str]: ...


class LocalStorage:
    def __init__(self, base: Path = CONTENT):
        self.base = base

    def read(self, rel: str) -> str:
        return (self.base / rel).read_text()

    def exists(self, rel: str) -> bool:
        return (self.base / rel).is_file()

    def glob(self, rel_dir: str, pattern: str) -> list[str]:
        return sorted(str(p.relative_to(self.base)) for p in (self.base / rel_dir).glob(pattern))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def split_front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


class ContentStore:
    def __init__(self, storage: Storage | None = None, stages_file: str = "stages.yaml"):
        """`stages_file`: an alternative stage file under content/ (default `stages.yaml`)."""
        self.storage = storage or LocalStorage()
        self.stages_file = stages_file

    def _read(self, rel: str, hashes: dict | None) -> str:
        text = self.storage.read(rel)
        if hashes is not None:
            hashes[rel] = sha(text)
        return text

    def core(self, hashes=None) -> str:
        return self._read("core/core.md", hashes)

    def rubric(self, hashes=None) -> dict:
        return yaml.safe_load(self._read("rubric.yaml", hashes))

    def stages(self, hashes=None) -> dict:
        return yaml.safe_load(self._read(self.stages_file, hashes))

    def persona(self, persona_id: str, hashes=None) -> dict:
        meta, body = split_front_matter(self._read(f"interviewer_personas/{persona_id}.md", hashes))
        return {**meta, "id": persona_id, "body": body.strip()}

    def role(self, role_id: str, hashes=None) -> dict:
        return yaml.safe_load(self._read(f"roles/{role_id}/brief.yaml", hashes))

    def skill(self, name: str, hashes=None) -> dict | None:
        rel = f"skills/{name}/SKILL.md"
        if not self.storage.exists(rel):
            return None
        meta, body = split_front_matter(self._read(rel, hashes))
        return {**meta, "body": body.strip()}

    def delivery_notes(self, skill: str, persona_id: str, hashes=None) -> str | None:
        rel = f"skills/{skill}/references/delivery-{persona_id}.md"
        return self._read(rel, hashes).strip() if self.storage.exists(rel) else None

    def list(self, kind: str) -> list[dict]:
        rel_dir, pattern = KINDS[kind]
        items = []
        for rel in self.storage.glob(rel_dir, pattern):
            text = self.storage.read(rel)
            ident = Path(rel).parent.name if kind in ("roles", "skills") else Path(rel).stem
            meta = yaml.safe_load(text) if rel.endswith(".yaml") else split_front_matter(text)[0]
            title = (meta or {}).get("title") or (meta or {}).get("name") or ident
            items.append({"id": ident, "title": title, "path": rel, "hash": sha(text)})
        return items

    def read_raw(self, kind: str, ident: str) -> dict:
        rel = {
            "core": f"core/{ident}.md",
            "personas": f"interviewer_personas/{ident}.md",
            "roles": f"roles/{ident}/brief.yaml",
            "skills": f"skills/{ident}/SKILL.md",
            "rubric": "rubric.yaml",
            "stages": "stages.yaml",
        }[kind]
        text = self.storage.read(rel)
        return {"id": ident, "path": rel, "hash": sha(text), "text": text}


def candidate(candidate_id: str) -> dict:
    return yaml.safe_load((DATA / "sim_candidates" / f"{candidate_id}.yaml").read_text())


def list_candidates() -> list[dict]:
    out = []
    for p in sorted((DATA / "sim_candidates").glob("*.yaml")):
        c = yaml.safe_load(p.read_text())
        out.append({"id": p.stem, "name": c.get("name", p.stem), "role_id": c.get("role_id")})
    return out
