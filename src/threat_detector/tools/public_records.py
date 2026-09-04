"""Public Records agent tools: court dockets, incident/CAD logs, protest permits.

All synthetic — no free public API serves this data. Protest permits are the
valuable one: they are *forward-looking*. A permit filed for a demonstration
near the venue on the event date is a signal you get *before* anything happens,
unlike news which reports after the fact.

Two synthetic sources, in strict order of preference:

1. the **curated** fixtures in ``data/fixtures/public_records.json`` — the fixed
   Berlin scenario the evals and the graded baseline are written against;
2. **generated** records for any other city (``synthetic.py``), so a tasking
   outside the curated scenario returns something plausible instead of a blank
   panel.

Curated always wins on a city match, so the baseline is bit-for-bit unchanged.
Generated records are tagged ``generated_for_city`` on top of the usual
``synthetic_fixture`` flag — the analyst can always tell the three apart.
"""

from __future__ import annotations

from datetime import date

from ..config import Config, load_config
from ..schemas import Finding, SourceType, ThreatChannel
from . import synthetic
from .base import load_fixture, mark_synthetic


def protest_permits_near(city: str, cfg: Config | None = None,
                         event_date: date | None = None,
                         entity_id: str = "") -> list[Finding]:
    """Permits filed for demonstrations in/near a city."""
    cfg = cfg or load_config()
    permits = load_fixture("public_records.json", cfg)["protest_permits"]
    if not synthetic.has_curated_data(city, {p["city"] for p in permits}):
        return synthetic.generate_permits(city, cfg, event_date, entity_id)
    findings = []
    for p in permits:
        if city.lower() not in p["city"].lower():
            continue
        findings.append(
            Finding(
                channel=ThreatChannel.PHYSICAL_LOCATION,
                source_type=SourceType.PROTEST_PERMIT,
                source_id=p["source_id"],
                summary=f"Protest permit: {p['group']} at {p['location']} on {p['date']}.",
                event_date=date.fromisoformat(p["date"]),
                severity=float(p.get("severity", 0.4)),
                detail={"group": p["group"], "location": p["location"], "expected_size": p.get("expected_size")},
            )
        )
    return mark_synthetic(findings)


def recent_incidents(city: str, cfg: Config | None = None,
                     entity_id: str = "") -> list[Finding]:
    """CAD/incident-log entries in a city, optionally tied to a known group."""
    cfg = cfg or load_config()
    incidents = load_fixture("public_records.json", cfg)["incidents"]
    if not synthetic.has_curated_data(city, {i["city"] for i in incidents}):
        return synthetic.generate_incidents(city, cfg, entity_id)
    findings = []
    for i in incidents:
        if city.lower() not in i["city"].lower():
            continue
        findings.append(
            Finding(
                channel=ThreatChannel.HOSTILE_ACTOR if i.get("group") else ThreatChannel.PHYSICAL_LOCATION,
                source_type=SourceType.INCIDENT_LOG,
                source_id=i["source_id"],
                summary=i["summary"],
                event_date=date.fromisoformat(i["date"]) if i.get("date") else None,
                severity=float(i.get("severity", 0.4)),
                detail={"group": i.get("group"), "location": i.get("location")},
            )
        )
    return mark_synthetic(findings)


def court_dockets(name: str, cfg: Config | None = None,
                  city: str = "") -> list[Finding]:
    """Court cases naming the protectee or their company."""
    cfg = cfg or load_config()
    dockets = load_fixture("public_records.json", cfg)["court_dockets"]
    findings = []
    for d in dockets:
        if name.lower() not in (d["parties"] + " " + d["summary"]).lower():
            continue
        findings.append(
            Finding(
                channel=ThreatChannel.REPUTATION,
                source_type=SourceType.COURT_DOCKET,
                source_id=d["source_id"],
                summary=d["summary"],
                event_date=date.fromisoformat(d["date"]) if d.get("date") else None,
                severity=float(d.get("severity", 0.25)),
                detail={"parties": d["parties"], "status": d.get("status")},
            )
        )
    if not findings:
        # No curated docket names this protectee — generate one rather than
        # returning a bare [], which the loop cannot distinguish from "clean".
        return synthetic.generate_dockets(name, cfg, city)
    return mark_synthetic(findings)


__all__ = ["protest_permits_near", "recent_incidents", "court_dockets"]
