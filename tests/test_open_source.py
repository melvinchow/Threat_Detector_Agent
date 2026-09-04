"""The real-source preconditions for Open-Source Monitoring.

These pin one bug and one silence, both of which let live collection sit broken
while the console looked healthy:

* the Reddit daily budget was charged *before* anyone checked whether the Reddit
  client was importable, so a missing `praw` drained the quota with calls that
  never left the process;
* the fallback to fixtures said only *that* it happened, never *why*.
"""

from __future__ import annotations

import dataclasses

import pytest

from threat_detector import ratelimit
from threat_detector.tools import open_source


@pytest.fixture
def social_cfg(synthetic_cfg):
    """`synthetic_cfg`, but with social configured REAL — the interesting case."""
    raw = dict(synthetic_cfg.raw)
    raw["source"] = dict(raw["source"], social="real")
    raw["limits"] = {"reddit": 5}
    return dataclasses.replace(synthetic_cfg, raw=raw)


def test_a_missing_praw_does_not_spend_the_reddit_budget(social_cfg, monkeypatch):
    monkeypatch.setattr(open_source, "_praw_available", lambda: False)
    monkeypatch.setenv("REDDIT_CLIENT_ID", "x")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "y")

    before = ratelimit.used_today(social_cfg, "reddit")
    open_source.search_social("keynote", social_cfg, entity_id="jordan_vale",
                              city="Berlin")
    assert ratelimit.used_today(social_cfg, "reddit") == before, (
        "the budget was charged for a call that never left the process")


def test_the_fallback_says_which_precondition_failed(social_cfg, monkeypatch):
    monkeypatch.setattr(open_source, "_praw_available", lambda: False)
    findings = open_source.search_social("keynote", social_cfg,
                                         entity_id="jordan_vale", city="Berlin")
    assert findings, "the synthetic fallback returned nothing to explain"
    reasons = {f.detail.get("fallback_reason") for f in findings}
    assert len(reasons) == 1
    assert "praw" in reasons.pop()


def test_synthetic_by_configuration_is_not_reported_as_a_fallback(synthetic_cfg):
    """`social: synthetic` is a choice, not a degradation. Don't cry wolf."""
    findings = open_source.search_social("keynote", synthetic_cfg,
                                         entity_id="jordan_vale", city="Berlin")
    assert findings
    assert not any(f.detail.get("fallback_reason") for f in findings)
