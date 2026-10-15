"""Cost ledger: usage events → USD, appended to runs/costs.jsonl.

    python -m backend.costs status      # spend by provider, step and run
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "runs" / "costs.jsonl"


@lru_cache
def prices() -> dict:
    return yaml.safe_load((ROOT / "config/prices.yaml").read_text())


@lru_cache
def budget_total() -> float:
    return yaml.safe_load((ROOT / "config/config.yaml").read_text())["budget"]["total_usd"]


def openai_response_usd(model: str, usage: dict) -> float:
    """Price one Realtime `response.done` usage block, splitting cached and uncached input."""
    p = prices()[model]
    inp = usage.get("input_token_details") or {}
    out = usage.get("output_token_details") or {}
    cached = inp.get("cached_tokens_details") or {}
    a_in, t_in = inp.get("audio_tokens", 0), inp.get("text_tokens", 0)
    a_c, t_c = cached.get("audio_tokens", 0), cached.get("text_tokens", 0)
    usd = ((a_in - a_c) * p["audio_in"] + a_c * p["audio_in_cached"]
           + (t_in - t_c) * p["text_in"] + t_c * p["text_in_cached"]
           + out.get("audio_tokens", 0) * p["audio_out"] + out.get("text_tokens", 0) * p["text_out"]) / 1e6
    return usd


def anthropic_usd(model: str, usage: dict) -> float:
    p = prices()[model]
    cached = usage.get("cache_read_input_tokens", 0) or 0
    fresh = (usage.get("input_tokens", 0) or 0) + (usage.get("cache_creation_input_tokens", 0) or 0)
    return (fresh * p["text_in"] + cached * p.get("text_in_cached", p["text_in"])
            + (usage.get("output_tokens", 0) or 0) * p["text_out"]) / 1e6


def transcription_usd(model: str, seconds: float) -> float:
    return seconds / 60 * prices()[model]["per_minute"]


def log(provider: str, model: str, usd: float, usage: dict, run_id: str, step: str, kind: str):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "provider": provider, "model": model, "run_id": run_id,
                            "step": step, "kind": kind, "usd": round(usd, 6), "usage": usage}) + "\n")


def entries() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def spent() -> float:
    return sum(e["usd"] for e in entries())


def remaining() -> float:
    return budget_total() - spent()


def status() -> str:
    by_provider, by_step, by_run = defaultdict(float), defaultdict(float), defaultdict(float)
    for e in entries():
        by_provider[f"{e['provider']}/{e['model']}"] += e["usd"]
        by_step[e["step"]] += e["usd"]
        by_run[e["run_id"]] += e["usd"]
    lines = [f"Spent USD {spent():.4f} of {budget_total():.2f} (remaining {remaining():.4f})", "", "By model:"]
    lines += [f"  {k:40s} {v:9.4f}" for k, v in sorted(by_provider.items())]
    lines += ["By step:"] + [f"  {k:40s} {v:9.4f}" for k, v in sorted(by_step.items())]
    lines += ["By run (last 10):"] + [f"  {k:40s} {v:9.4f}" for k, v in list(by_run.items())[-10:]]
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1:] == ["status"]:
        print(status())
    else:
        print(__doc__)
