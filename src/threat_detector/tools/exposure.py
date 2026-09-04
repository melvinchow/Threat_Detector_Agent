"""Exposure agent tools: data-broker scan, breach lookup, dark-web, doxxing.

**This is the only agent that receives the VIP's PII.** It takes the sensitive
``PII`` block and scans corpora for places that data appears where it shouldn't —
a broker listing the home address, a breach containing a known email, chatter
that references the itinerary. Everything here is synthetic; there is no
legitimate free source for this data, and putting real PII in the repo would be
the exact harm the system exists to prevent.

The tools take the PII as an argument rather than reading a global, so the trust
boundary is visible in every call site: if a function doesn't receive PII, it
can't leak it.
"""

from __future__ import annotations

from ..config import Config, load_config
from ..profile import PII
from ..schemas import Finding, SourceType, ThreatChannel
from .base import load_fixture, mark_synthetic


def scan_data_brokers(pii: PII, cfg: Config | None = None) -> list[Finding]:
    """Find data-broker listings exposing the protectee's address/phone."""
    cfg = cfg or load_config()
    listings = load_fixture("exposure.json", cfg)["broker_listings"]
    findings = []
    for entry in listings:
        if _pii_hit(entry["exposes"], pii):
            findings.append(
                Finding(
                    channel=ThreatChannel.EXPOSURE,
                    source_type=SourceType.DATA_BROKER,
                    source_id=entry["source_id"],
                    summary=f"Data broker {entry['broker']} lists {entry['exposes']}.",
                    severity=float(entry.get("severity", 0.5)),
                    detail={"broker": entry["broker"], "takedown_url": entry.get("takedown_url")},
                )
            )
    return mark_synthetic(findings)


def breach_lookup(pii: PII, cfg: Config | None = None) -> list[Finding]:
    """Find the protectee's emails/usernames in breach corpora."""
    cfg = cfg or load_config()
    breaches = load_fixture("exposure.json", cfg)["breaches"]
    findings = []
    identifiers = set(pii.emails) | set(pii.known_usernames)
    for b in breaches:
        if b["identifier"] in identifiers:
            findings.append(
                Finding(
                    channel=ThreatChannel.EXPOSURE,
                    source_type=SourceType.BREACH_CORPUS,
                    source_id=b["source_id"],
                    summary=f"{b['identifier']} appears in breach '{b['breach']}' ({b['year']}).",
                    severity=float(b.get("severity", 0.5)),
                    detail={"breach": b["breach"], "leaked_fields": b.get("leaked_fields", [])},
                )
            )
    return mark_synthetic(findings)


def darkweb_itinerary_scan(event_terms: list[str], cfg: Config | None = None) -> list[Finding]:
    """Scan dark-web chatter for references to the protectee's plans/event.

    This is the escalation the Module 2 trace reaches once a named actor plus a
    specific event surface: has the itinerary leaked?
    """
    cfg = cfg or load_config()
    chatter = load_fixture("exposure.json", cfg)["darkweb_chatter"]
    terms = [t.lower() for t in event_terms if t]
    findings = []
    for c in chatter:
        text = c["text"].lower()
        if any(t in text for t in terms):
            findings.append(
                Finding(
                    channel=ThreatChannel.EXPOSURE,
                    source_type=SourceType.DARK_WEB,
                    source_id=c["source_id"],
                    summary=c["summary"],
                    severity=float(c.get("severity", 0.6)),
                    detail={"text": c["text"], "forum": c.get("forum")},
                )
            )
    return mark_synthetic(findings)


def _pii_hit(exposes: str, pii: PII) -> bool:
    haystack = exposes.lower()
    for value in (*pii.home_addresses, *pii.phone_numbers, *pii.emails):
        if value and value.lower() in haystack:
            return True
    # Fixtures may describe the *category* rather than the literal value.
    return any(word in haystack for word in ("home address", "phone", "email")) and _has_any(pii)


def _has_any(pii: PII) -> bool:
    return bool(pii.home_addresses or pii.phone_numbers or pii.emails)


__all__ = ["scan_data_brokers", "breach_lookup", "darkweb_itinerary_scan"]
