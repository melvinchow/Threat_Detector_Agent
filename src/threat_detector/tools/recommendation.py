"""Recommendation agent: proposes actions. **Never executes.**

Maps the scored findings to playbook actions. Every action that would change the
world (submit a takedown, move a calendar event, contact law enforcement) is
returned with ``requires_approval=True``. The agent is a proposer behind a human
approval gate — the collection agents are read-only, this agent is
propose-only, and only a human analyst turns a recommendation into an action.
"""

from __future__ import annotations

from ..schemas import Finding, Recommendation, RiskScore, ThreatChannel


def recommend(score: RiskScore, findings: list[Finding]) -> list[Recommendation]:
    recs: list[Recommendation] = []
    channels = set(score.by_channel.keys())

    if ThreatChannel.PHYSICAL_LOCATION.value in channels and score.by_channel.get(
        ThreatChannel.PHYSICAL_LOCATION.value, 0
    ) >= 0.3:
        recs.append(
            Recommendation(
                action="Advance venue survey + plan an alternate arrival route.",
                channel=ThreatChannel.PHYSICAL_LOCATION,
                rationale="Location-specific signal near the venue (proximate protest/incident activity).",
                estimated_cost="1 advance team, 1 day",
            )
        )
        recs.append(
            Recommendation(
                action="Pre-stage contact details for nearest hospital and police station.",
                channel=ThreatChannel.PHYSICAL_LOCATION,
                rationale="Reduce response time if an incident occurs on-site.",
                requires_approval=False,  # gathering contacts is read-only prep
                estimated_cost="minimal",
            )
        )

    if ThreatChannel.EXPOSURE.value in channels:
        exposure_findings = [f for f in findings if f.channel == ThreatChannel.EXPOSURE]
        broker = [f for f in exposure_findings if f.detail.get("takedown_url")]
        if broker:
            recs.append(
                Recommendation(
                    action=f"Submit takedown requests on {len(broker)} data-broker listing(s).",
                    channel=ThreatChannel.EXPOSURE,
                    rationale="Protectee PII (address/phone) publicly discoverable via brokers.",
                    estimated_cost="staff time; automatable",
                )
            )
        if any(f.source_type.value == "dark_web" for f in exposure_findings):
            recs.append(
                Recommendation(
                    action="Escalate: itinerary/plans referenced in dark-web chatter — "
                    "consider adjusting or concealing travel details.",
                    channel=ThreatChannel.EXPOSURE,
                    rationale="A leaked itinerary combined with negative sentiment is the highest-risk pattern.",
                )
            )

    if ThreatChannel.HOSTILE_ACTOR.value in channels and score.by_channel.get(
        ThreatChannel.HOSTILE_ACTOR.value, 0
    ) >= 0.4:
        recs.append(
            Recommendation(
                action="Open a monitoring file on the named group/individual; brief the protective detail.",
                channel=ThreatChannel.HOSTILE_ACTOR,
                rationale="A specific actor has expressed intent tied to a known event.",
                estimated_cost="analyst time",
            )
        )

    if score.overall >= 0.75:
        recs.append(
            Recommendation(
                action="Recommend reviewing whether the appearance should proceed as planned.",
                channel=ThreatChannel.PHYSICAL_LOCATION,
                rationale="Overall risk is HIGH across multiple channels.",
            )
        )

    return recs


__all__ = ["recommend"]
