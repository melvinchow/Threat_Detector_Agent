"""Module 6 evaluation metrics.

A guardrail is a per-item check that passes or fails; a METRIC is a number
aggregated across runs that tells you whether the checks are calibrated. These
are the checkpoint's metrics, computed from a finished Briefing (plus one
seeded-corpus test for retrieval recall):

* **Groundedness rate** — share of claims traceable to a real source
  (source_id present, provenance attached). Target 95–100%; below that is
  hallucination. The deterministic pipeline earns ~100% *by construction* —
  the metric exists to catch regressions when an LLM harness is enabled.
* **Escalation rate** — share of proposed actions gated behind a human.
  Too high → alert fatigue (the analyst starts rubber-stamping, which
  manufactures a false audit trail); too low → the gates catch nothing.
* **Corroboration rate** — share of high-severity findings with 2+ independent
  sources (the uncorroborated ones were confidence-capped, not dropped).
* **Fallback success** — when a loop hit its cap, did the run still return a
  usable, explicitly-labeled product rather than nothing?
* **Retrieval recall (seeded)** — plant known threat documents in the corpus
  and measure how many the standard query surfaces. The only honest way to
  measure the Module 3 false-negative risk.

If escalation rate climbs, tighten collection filters — don't loosen the gate.
"""

from __future__ import annotations

from .config import Config
from .schemas import Briefing


def evaluate_briefing(b: Briefing, cfg: Config) -> dict:
    """Per-run metric snapshot from a finished briefing."""
    findings = b.findings

    grounded = [f for f in findings if f.source_id and (f.event_date or f.detail)]
    groundedness = len(grounded) / len(findings) if findings else 1.0

    gated = [r for r in b.recommendations if r.requires_approval]
    escalation = len(gated) / len(b.recommendations) if b.recommendations else 0.0

    high = [f for f in findings if f.severity >= cfg.corroboration_severity
            or f.detail.get("uncorroborated")]
    uncorro = [f for f in high if f.detail.get("uncorroborated")]
    corroboration = 1.0 - (len(uncorro) / len(high)) if high else 1.0

    trace = "\n".join(b.trace)
    caps_hit = sum(1 for marker in
                   ("hard cap", "INCOMPLETE", "no supporting evidence",
                    "stayed single-sourced")
                   if marker in trace)
    # Fallback succeeded if every cap event still left a labeled artifact:
    # the briefing exists, incomplete plans carry `uncovered`, capped claims
    # carry the uncorroborated flag. In this pipeline that is structural.
    fallback_ok = all(p.uncovered for p in b.plans if not p.complete)

    return {
        "groundedness_rate": round(groundedness, 3),
        "escalation_rate": round(escalation, 3),
        "human_gate_hit": bool(gated),
        "corroboration_rate": round(corroboration, 3),
        "high_severity_claims": len(high),
        "confidence_capped": len(uncorro),
        "stale_flagged": sum(1 for f in findings if f.detail.get("stale")),
        "geo_flagged": sum(1 for f in findings
                           if f.detail.get("geo_invalid") or f.detail.get("geo_far")),
        "cap_events_in_trace": caps_hit,
        "fallback_success": fallback_ok,
        "findings": len(findings),
        "trace_steps": len(b.trace),
    }


# Documents that the standard surveillance query MUST surface — the planted
# threats for the recall test. If a redesign of chunking/reranking loses one of
# these, this metric (and its test) goes red before an analyst ever misses it.
SEEDED_THREATS = {"forum:redwing", "social:handle-probe", "incident:munich-2025"}


def seeded_recall(cfg: Config, entity_id: str = "jordan_vale") -> dict:
    """Retrieval recall on the seeded corpus: plant known threats, run the
    standard query, count how many surface."""
    from .retrieval import RetrievalFilter, build_index

    index = build_index(cfg)
    result = index.retrieve(
        "surveillance of the executive's route and schedule at the venue",
        RetrievalFilter(entity_id=entity_id),
    )
    got = {rc.chunk.source_id for rc in result.results}
    hit = SEEDED_THREATS & got
    return {
        "expected": sorted(SEEDED_THREATS),
        "surfaced": sorted(hit),
        "missed": sorted(SEEDED_THREATS - hit),
        "recall": round(len(hit) / len(SEEDED_THREATS), 3),
    }


__all__ = ["evaluate_briefing", "seeded_recall", "SEEDED_THREATS"]
