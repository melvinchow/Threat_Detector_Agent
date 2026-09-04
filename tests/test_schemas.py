"""Contract + anti-hallucination tests."""

import pytest
from pydantic import ValidationError

from threat_detector.schemas import Finding, SourceType, ThreatChannel
from threat_detector.tools import open_source, risk


def test_finding_rejects_out_of_range_severity():
    with pytest.raises(ValidationError):
        Finding(
            channel=ThreatChannel.REPUTATION,
            source_type=SourceType.NEWS,
            source_id="x",
            summary="bad",
            severity=1.5,  # > 1.0
        )


def test_risk_agent_scores_only_given_findings():
    # With no findings, the risk score is exactly zero — the agent has no way to
    # invent a threat, because it has no retrieval tools.
    score = risk.score_findings([])
    assert score.overall == 0.0
    assert score.by_channel == {}


def test_reliability_downweights_score():
    strong = Finding(channel=ThreatChannel.HOSTILE_ACTOR, source_type=SourceType.SOCIAL,
                     source_id="a", summary="threat", severity=0.8, reliability=1.0)
    weak = Finding(channel=ThreatChannel.HOSTILE_ACTOR, source_type=SourceType.SOCIAL,
                   source_id="b", summary="threat", severity=0.8, reliability=0.2)
    assert risk.score_findings([strong]).overall > risk.score_findings([weak]).overall


def test_entity_resolution_drops_namesake(synthetic_cfg):
    # The marathon-runner article names "Jordan Vale" but has no company/role
    # context, so it must be dropped; the CEO articles must survive.
    news = open_source.search_news("Jordan Vale", synthetic_cfg)
    resolved = open_source.resolve_entity(
        news, "Jordan Vale", role_terms=["CEO", "Vantage", "Robotics"]
    )
    summaries = " ".join(f.summary for f in resolved)
    assert "marathon" not in summaries.lower()
    assert "Vantage" in summaries
