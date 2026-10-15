"""Hard spend cap for the app (alt/config.yaml budget.alt_cap_usd). Spend is read from the shared cost ledger
(runs/costs.jsonl), counting only entries whose step starts with "alt"."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from backend import costs

ROOT = Path(__file__).resolve().parent


@lru_cache
def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


class BudgetExceeded(RuntimeError):
    pass


def spent() -> float:
    return sum(e["usd"] for e in costs.entries() if str(e.get("step", "")).startswith("alt"))


def remaining() -> float:
    return cfg()["budget"]["alt_cap_usd"] - spent()


def require(usd: float, what: str) -> None:
    """Refuse before sending if this spend could take alt work past the cap."""
    left = remaining()
    if usd > left:
        raise BudgetExceeded(f"{what}: needs up to USD {usd:.3f}, only USD {left:.3f} of the alt cap left")


if __name__ == "__main__":
    print(f"alt spend USD {spent():.4f} of {cfg()['budget']['alt_cap_usd']:.2f} (remaining {remaining():.4f})")
