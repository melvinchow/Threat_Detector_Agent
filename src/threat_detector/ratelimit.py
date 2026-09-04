"""Per-day API budgets for every real (external) source.

The analyst wants to test many times per day without burning through free-tier
quotas, so limits are STRICT by default and every real call must pass through
``spend()`` first. When a budget is exhausted the tool falls back to its
synthetic fixture — the run degrades honestly instead of failing, and the trace
says so.

Counters persist in a small JSON file (gitignored) keyed by date, so the budget
is per-calendar-day across restarts, not per-process.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .config import Config

_DEFAULT_LIMITS = {
    "reddit": 20,       # reddit search calls (PRAW)
    "exa": 10,          # exa.ai news searches
    "nominatim": 40,    # OSM geocoding lookups
    "overpass": 8,      # OSM hotel/POI queries (heavier; be extra polite)
    "osrm": 20,         # routing
    "llm_calls": 300,   # model calls/day in LLM mode (~15-25 per assessment)
}


def _path(cfg: Config) -> Path:
    return cfg.root / "data" / "rate_limits.json"


def _load(cfg: Config) -> dict:
    p = _path(cfg)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def limit_for(cfg: Config, key: str) -> int:
    return int(cfg.raw.get("limits", {}).get(key, _DEFAULT_LIMITS.get(key, 10)))


def used_today(cfg: Config, key: str) -> int:
    return int(_load(cfg).get(date.today().isoformat(), {}).get(key, 0))


def spend(cfg: Config, key: str, n: int = 1) -> bool:
    """Try to spend ``n`` call(s) from today's budget for ``key``.

    Returns True (and records the spend) if allowed; False if the budget is
    exhausted — in which case the caller must fall back to synthetic.
    """
    data = _load(cfg)
    today = date.today().isoformat()
    day = data.setdefault(today, {})
    if day.get(key, 0) + n > limit_for(cfg, key):
        return False
    day[key] = day.get(key, 0) + n
    # Keep only today's counters; yesterday's budget is irrelevant.
    p = _path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({today: day}, indent=2))
    return True


def status(cfg: Config) -> dict[str, tuple[int, int]]:
    """{key: (used_today, limit)} for every budgeted source — for the GUI."""
    return {k: (used_today(cfg, k), limit_for(cfg, k))
            for k in sorted(set(_DEFAULT_LIMITS) | set(cfg.raw.get("limits", {})))}


__all__ = ["spend", "status", "used_today", "limit_for"]
