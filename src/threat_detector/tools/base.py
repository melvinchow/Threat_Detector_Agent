"""Shared helpers for tools.

The important idea lives here: a **tool contract** has two implementations
(synthetic + real), and ``config.source_is_real(name)`` picks between them. Tool
modules import ``load_fixture`` for their synthetic path and hit real APIs for
their real path — but always return the same ``Finding`` schema.
"""

from __future__ import annotations

import json

from ..config import Config, load_config


def load_fixture(name: str, cfg: Config | None = None) -> dict:
    """Read one JSON fixture from data/fixtures/."""
    cfg = cfg or load_config()
    return json.loads(cfg.fixture(name).read_text())


def mark_synthetic(findings: list) -> list:
    """Tag findings that came from a synthetic fixture (Module 6 honesty rule).

    The analyst must always be able to tell demo data from live collection —
    especially when a real-configured source silently fell back to fixtures
    (missing key, spent rate budget, network error). The GUI surfaces the tag.
    """
    for f in findings:
        f.detail["synthetic_fixture"] = True
    return findings


def mark_fallback(findings: list, reason: str | None) -> list:
    """Record WHY a source configured `real` returned synthetic data.

    ``synthetic_fixture`` says *that* it happened; without this the analyst
    cannot tell "Reddit had nothing" from "the Reddit client was never
    installed" — which is exactly the confusion that let live social
    collection sit broken while the console looked healthy.
    """
    if reason:
        for f in findings:
            f.detail["fallback_reason"] = reason
    return findings
