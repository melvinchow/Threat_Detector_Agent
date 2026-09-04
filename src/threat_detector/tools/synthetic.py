"""City-generalized synthetic records.

The curated fixtures in ``data/fixtures/`` describe one scenario — a conference
in Berlin — and every tool filters them by city. That is correct and it stays
the gradeable baseline, but it means a tasking anywhere else returns nothing
from Public Records and Exposure, and the console looks broken when it is
merely empty.

So: for a city with no curated data, generate records for it. Two rules make
that safe rather than dishonest:

1. **Deterministic.** Everything is seeded from ``(city, entity_id)``, so the
   same tasking produces the same records on every run — the suite stays
   reproducible and a demo repeats exactly.
2. **Labeled, twice.** Findings carry ``synthetic_fixture`` (already surfaced by
   the GUI and by ``evals``) *and* ``generated_for_city``, so a generated record
   can never be mistaken for a curated one or for live collection.

Coordinates are the one real thing here: they come from the geocoder, so a
generated permit pins to a place that actually exists in that city. An invented
record at invented coordinates would fail the Module 6 geo-sanity check, and it
should.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

from ..config import Config
from ..schemas import Finding, SourceType, ThreatChannel

# Deliberately generic, obviously-invented group names. Nothing here should
# resemble a real organisation: a synthetic record naming a real group is a
# defamation problem, not a demo.
_GROUPS = [
    "Open Sky Collective", "Civic Action Front", "Riverside Assembly",
    "The Common Ground Network", "Workers' Voice Coalition",
    "Transparency Now", "Citizens for Fair Automation",
]
_INCIDENT_KINDS = [
    ("trespass at a corporate venue", 0.40),
    ("unauthorised drone flight near a conference site", 0.45),
    ("doxxing of a visiting executive on a local forum", 0.50),
    ("disruption of a public speaking event", 0.42),
]
_DOCKET_KINDS = [
    ("labour dispute over automation rollout", "pending", 0.22),
    ("defamation claim brought by a former contractor", "dismissed", 0.15),
    ("regulatory complaint over data handling", "pending", 0.28),
]


def _rng(*parts: str) -> int:
    """A stable integer seed. hash() is salted per process; this is not."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16)


def _pick(pool: list, *seed_parts: str, offset: int = 0):
    return pool[(_rng(*seed_parts) + offset) % len(pool)]


def _slug(city: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in city).strip("-")


def has_curated_data(city: str, curated_cities: set[str]) -> bool:
    return any(city.lower() in c.lower() or c.lower() in city.lower()
               for c in curated_cities if c)


def _coords(place: str, cfg: Config) -> tuple[float, float] | None:
    """Real coordinates for an invented record — see the module docstring."""
    from . import geospatial
    try:
        return geospatial.geocode(place, cfg)
    except Exception:
        return None


def _mark(findings: list[Finding], city: str) -> list[Finding]:
    for f in findings:
        f.detail["synthetic_fixture"] = True
        f.detail["generated_for_city"] = city
    return findings


# ---------------------------------------------------------------------------
# GENERATORS
# ---------------------------------------------------------------------------

def generate_permits(city: str, cfg: Config, event_date: date | None = None,
                     entity_id: str = "") -> list[Finding]:
    """A forward-looking protest permit near the tasking city."""
    if not city:
        return []
    group = _pick(_GROUPS, city, entity_id)
    when = (event_date or date.today() + timedelta(days=21))
    coords = _coords(city, cfg)
    return _mark([Finding(
        channel=ThreatChannel.PHYSICAL_LOCATION,
        source_type=SourceType.PROTEST_PERMIT,
        source_id=f"permit:{_slug(city)}-city-authority",
        summary=(f"Protest permit: {group} has filed for a demonstration in "
                 f"central {city} on {when.isoformat()}."),
        event_date=when,
        severity=0.42,
        detail={"group": group, "location": f"central {city}",
                "expected_size": 150 + (_rng(city, "size") % 8) * 50,
                **({"venue_coords": [coords[0], coords[1]]} if coords else {})},
    )], city)


def generate_incidents(city: str, cfg: Config, entity_id: str = "") -> list[Finding]:
    """One or two recent incident-log entries for the tasking city."""
    if not city:
        return []
    out = []
    for i in range(1 + _rng(city, entity_id, "n") % 2):
        kind, sev = _pick(_INCIDENT_KINDS, city, entity_id, offset=i)
        group = _pick(_GROUPS, city, entity_id, offset=i + 1)
        when = date.today() - timedelta(days=14 + (_rng(city, str(i)) % 90))
        out.append(Finding(
            channel=ThreatChannel.HOSTILE_ACTOR,
            source_type=SourceType.INCIDENT_LOG,
            source_id=f"cad:{_slug(city)}-police",
            summary=f"Reported {kind} in {city}, linked to {group}.",
            event_date=when,
            severity=sev,
            detail={"group": group, "location": city},
        ))
    return _mark(out, city)


def generate_dockets(name: str, cfg: Config, city: str = "") -> list[Finding]:
    """A court docket naming the protectee. Reputation channel, low severity."""
    if not name:
        return []
    summary_kind, status, sev = _pick(_DOCKET_KINDS, name, city)
    when = date.today() - timedelta(days=60 + (_rng(name, "d") % 300))
    return _mark([Finding(
        channel=ThreatChannel.REPUTATION,
        source_type=SourceType.COURT_DOCKET,
        source_id=f"docket:{_slug(city or 'district')}-civil",
        summary=f"Civil filing naming {name}: {summary_kind}.",
        event_date=when,
        severity=sev,
        detail={"parties": name, "status": status},
    )], city or "unspecified")


def generate_social(query: str, city: str, cfg: Config,
                    entity_id: str = "") -> list[Finding]:
    """Chatter about the event, for a scenario the curated posts don't cover."""
    if not city:
        return []
    group = _pick(_GROUPS, city, entity_id)
    return _mark([Finding(
        channel=ThreatChannel.HOSTILE_ACTOR,
        source_type=SourceType.SOCIAL,
        source_id=f"social:{_slug(city)}-forum",
        summary=(f"Local forum thread discussing {group}'s plans to demonstrate "
                 f"around the {city} event; no specific individual named."),
        severity=0.30,
        detail={"group": group, "text": "aggregate thread, no named target"},
    )], city)


def generate_news(query: str, city: str, cfg: Config,
                  entity_id: str = "") -> list[Finding]:
    """Coverage of the event's context, scoped to the tasked city."""
    if not city:
        return []
    group = _pick(_GROUPS, city, entity_id, offset=2)
    return _mark([Finding(
        channel=ThreatChannel.REPUTATION,
        source_type=SourceType.NEWS,
        source_id=f"news:{_slug(city)}-local",
        summary=(f"Local coverage: {group} announces a season of protests "
                 f"targeting corporate events in {city}."),
        severity=0.28,
        detail={"text": f"regional reporting on activity in {city}",
                "sentiment": "negative"},
    )], city)


def in_scope(fixture: dict, entity_id: str, city: str) -> bool:
    """Does a curated fixture describe the scenario we are actually assessing?

    Unscoped callers (the deterministic Orchestrator's baseline run, the test
    suite) pass nothing and get the curated data as before. A caller that DOES
    know the scenario gets the curated fixtures only when they match it.
    """
    scope = fixture.get("scope")
    if not scope or not (entity_id or city):
        return True
    if entity_id and scope.get("entity_id") and scope["entity_id"] != entity_id:
        return False
    if city and scope.get("city") and scope["city"].lower() not in city.lower():
        return False
    return True


__all__ = ["generate_permits", "generate_incidents", "generate_dockets",
           "generate_social", "generate_news", "has_curated_data", "in_scope"]
