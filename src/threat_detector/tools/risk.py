"""Risk Assessment agent: scoring. **No retrieval tools at all.**

This agent is deliberately powerless to fetch anything. It receives the list of
``Finding`` objects the collection agents returned and produces a ``RiskScore``.
Because it cannot call out to the world, it structurally *cannot* invent a source
or a threat mid-analysis — it can only score what was actually collected. That is
the architectural feature that makes hallucinated threats impossible rather than
merely discouraged.

The score is a transparent, auditable rubric — reliability-weighted severity,
aggregated per channel — not a black box. Every number can be traced to the
findings that produced it.
"""

from __future__ import annotations

from collections import defaultdict

from ..schemas import Finding, RiskScore, ThreatChannel


def score_findings(findings: list[Finding], recency_boost: bool = True) -> RiskScore:
    """Blend findings into an overall 0-1 risk score, broken out by channel.

    Weighting rules (each is a rubric line you can defend in the writeup):
      * A finding contributes ``severity * reliability`` — a scary claim from an
        untrusted source counts for little.
      * Findings with a future ``event_date`` get a small boost: threats are
        time-relative, and an imminent, dated event is more actionable than an
        old one.
      * Per channel we take a saturating aggregate so ten weak findings don't
        outweigh one strong, credible one.
    """
    by_channel_contribs: dict[str, list[float]] = defaultdict(list)
    rationale: list[str] = []

    for f in findings:
        weight = f.severity * f.reliability
        if recency_boost and f.event_date is not None:
            weight = min(1.0, weight * 1.15)
        by_channel_contribs[f.channel.value].append(weight)

    by_channel: dict[str, float] = {}
    for channel, weights in by_channel_contribs.items():
        by_channel[channel] = _saturating_aggregate(weights)

    overall = _saturating_aggregate(list(by_channel.values())) if by_channel else 0.0

    # Build a short human-readable rationale from the top contributors.
    ranked = sorted(findings, key=lambda f: f.severity * f.reliability, reverse=True)
    for f in ranked[:5]:
        rationale.append(
            f"[{f.channel.value}] {f.summary} "
            f"(severity {f.severity:.2f} x reliability {f.reliability:.2f})"
        )

    return RiskScore(
        overall=round(overall, 3),
        by_channel={k: round(v, 3) for k, v in by_channel.items()},
        rationale=rationale,
        top_findings=ranked[:5],
    )


def _saturating_aggregate(weights: list[float]) -> float:
    """Combine weights so they accumulate but never exceed 1.0.

    Uses noisy-OR: 1 - prod(1 - w). Two independent 0.5 signals combine to 0.75,
    not 1.0 — evidence stacks up but with diminishing returns.
    """
    product = 1.0
    for w in weights:
        product *= (1.0 - max(0.0, min(1.0, w)))
    return 1.0 - product


def band(overall: float) -> str:
    """Human label for the numeric score."""
    if overall >= 0.75:
        return "HIGH"
    if overall >= 0.45:
        return "ELEVATED"
    if overall >= 0.2:
        return "GUARDED"
    return "LOW"


__all__ = ["score_findings", "band"]
