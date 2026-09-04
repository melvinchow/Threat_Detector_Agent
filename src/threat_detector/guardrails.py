"""Module 6 guardrails: input checks, staleness, geo sanity, source vetting.

The Module 6 checkpoint's rule set, enforced in code (a rule that matters is
never left as prompt text):

* **Intake check (pre-generation).** Does the system have what it needs —
  from the profile (long-term memory), the config, or the tasking itself?
  Major gaps become follow-up QUESTIONS to the analyst (who/where/when);
  minor gaps become stated ASSUMPTIONS the run proceeds under. The system
  never silently substitutes the demo scenario for what the analyst typed.
* **Staleness check (post-collection).** A months-old post presented as
  current signal (the karma-farming repost problem) is flagged and
  down-weighted. Deliberately-historical material from case memory is exempt —
  it is *labeled* historical, which is the difference between recall and
  staleness.
* **Geo sanity check.** Coordinates must be plausible: on the globe, not the
  null island (0,0), and actually near the target city — false pins get added
  to online maps, and a "protest site" 800 km from the venue is a data error,
  not a threat.
* **Source vetting.** A source the reliability memory has never seen cannot be
  the sole support of a high-severity claim — new sources need an analyst
  promotion before they can independently drive an escalation. (Checking "online
  sentiment" about a new source would be circular: using the untrusted internet
  to vet the internet.)

Everything here is deterministic and offline; the checks run identically in
tests, the CLI, and the GUI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .config import Config
from .memory import ReliabilityMemory
from .profile import Profile
from .schemas import Finding, SourceType

# ---------------------------------------------------------------------------
# INTAKE CHECK
# ---------------------------------------------------------------------------

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_ROLE_WORDS = ("ceo", "cfo", "cto", "coo", "president", "chairman", "chairwoman",
               "executive", "founder", "senator", "governor", "mayor", "celebrity",
               "vip", "protectee", "principal")
_VENUE_WORDS = ("arena", "hall", "center", "centre", "stadium", "hotel",
                "kongress", "messe", "convention", "theater", "theatre",
                "auditorium", "campus", "plaza", "expo", "pavilion")
# Capitalized words that are never a city or a person in a tasking.
_NOT_A_NAME = set(w.title() for w in _MONTHS) | set(_ROLE_WORDS) | {
    "The", "A", "An", "Our", "My", "Their", "This", "That", "Next", "Last",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}


@dataclass
class IntakeReport:
    """What the tasking gave us, what we assumed, and what we must ask."""

    raw: str = ""
    city: str | None = None
    event_date: date | None = None
    venue: str | None = None
    mentioned_names: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)     # MAJOR gaps -> ask
    assumptions: list[str] = field(default_factory=list)   # minor gaps -> state

    @property
    def ready(self) -> bool:
        return not self.questions


# Fast path for offline/deterministic mode; real mode can geocode-verify any city.
_KNOWN_CITIES = ["Berlin", "Munich", "New York", "NYC", "London", "Paris",
                 "San Francisco", "Tokyo", "Chicago", "Los Angeles", "Boston",
                 "Seattle", "Austin", "Miami", "Prague", "Vienna", "Amsterdam",
                 "Madrid", "Rome", "Singapore", "Hong Kong", "Sydney", "Toronto"]


def extract_city(raw: str) -> str | None:
    """Find the location. Known-city fast path first, then a general
    'in <Proper Noun>' pattern so ANY city the analyst types is honored —
    never silently replaced by the demo scenario's."""
    for c in _KNOWN_CITIES:
        if re.search(rf"\b{re.escape(c)}\b", raw, re.IGNORECASE):
            return "New York" if c.upper() == "NYC" else c
    m = re.search(r"\b(?:in|to|at)\s+([A-Z][a-zA-Z-]+(?:\s+[A-Z][a-zA-Z-]+)?)", raw)
    if m and m.group(1).split()[0] not in _NOT_A_NAME:
        return m.group(1)
    return None


def extract_event_date(raw: str, today: date | None = None) -> date | None:
    """Parse an event date: ISO, 'in 3 weeks', month-day, tomorrow/next week."""
    today = today or date.today()
    low = raw.lower()

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = re.search(r"\bin\s+(\d+)\s+(day|week|month)s?\b", low)
    if m:
        n = int(m.group(1))
        days = {"day": 1, "week": 7, "month": 30}[m.group(2)]
        return today + timedelta(days=n * days)

    m = re.search(rf"\b({'|'.join(_MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?",
                  low)
    if m:
        month = _MONTHS.index(m.group(1)) + 1
        year = int(m.group(3)) if m.group(3) else today.year
        try:
            d = date(year, month, int(m.group(2)))
        except ValueError:
            return None
        if d < today and not m.group(3):
            d = date(year + 1, month, int(m.group(2)))
        return d

    if "tomorrow" in low:
        return today + timedelta(days=1)
    if "next week" in low:
        return today + timedelta(days=7)
    if "next month" in low:
        return today + timedelta(days=30)
    return None


def _extract_names(raw: str, city: str | None, venue: str | None) -> list[str]:
    """Capitalized multi-word sequences that look like person names —
    excluding the extracted city and venue (a place is not a protectee)."""
    names = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", raw):
        cand = m.group(1)
        if city and (cand in city or city in cand):
            continue
        if venue and (cand in venue or venue in cand):
            continue
        if any(w in _NOT_A_NAME for w in cand.split()):
            continue
        if any(w in cand.lower() for w in _VENUE_WORDS):
            continue
        names.append(cand)
    return names


def _extract_venue(raw: str) -> str | None:
    for w in _VENUE_WORDS:
        # Case-insensitive on the venue stem only; the preceding words must be
        # genuinely capitalized (part of the proper name), so lowercase verbs
        # like 'attending' can't be swallowed into the venue.
        m = re.search(rf"\b((?:[A-Z][\w-]*\s+){{0,3}}\w*(?i:{w})\w*)\b", raw)
        if m and any(ch.isupper() for ch in m.group(1)):
            venue = m.group(1).strip()
            venue = re.sub(r"^(?:at|the|in|a)\s+", "", venue, flags=re.IGNORECASE)
            venue = re.sub(r"^(?:at|the|in|a)\s+", "", venue, flags=re.IGNORECASE)
            return venue
    return None


def intake_check(raw: str, profile: Profile, cfg: Config,
                 today: date | None = None) -> IntakeReport:
    """The pre-generation input guardrail.

    Major gaps (who / where / when) produce questions and block the run until
    the analyst answers; minor gaps (exact venue, budget) produce assumptions
    the run states out loud. Per the checkpoint: assumptions for minor gaps,
    follow-up questions for major ones.
    """
    rep = IntakeReport(raw=raw)
    rep.city = extract_city(raw)
    rep.event_date = extract_event_date(raw, today)
    rep.venue = _extract_venue(raw)
    rep.mentioned_names = _extract_names(raw, rep.city, rep.venue)

    # --- WHO ---------------------------------------------------------------
    others = [n for n in rep.mentioned_names
              if n.lower() != profile.name.lower()]
    if others:
        rep.questions.append(
            f"The tasking mentions {', '.join(repr(n) for n in others)}, but the "
            f"active protection profile is **{profile.name}** ({profile.role}). "
            f"Should I assess {profile.name}, or do you need to switch/create a "
            f"profile first (`threat-detector profile`)? I never guess who the "
            f"protectee is."
        )
    elif profile.name.lower() not in raw.lower():
        role_hit = next((w for w in _ROLE_WORDS if re.search(rf"\b{w}\b", raw.lower())), None)
        if role_hit and role_hit in profile.role.lower():
            rep.assumptions.append(
                f"Interpreting '{role_hit.upper()}' as the active profile: "
                f"{profile.name} ({profile.role})."
            )
        else:
            rep.questions.append(
                f"Who is the protectee? The active profile is **{profile.name}** "
                f"({profile.role}) — reply 'the {profile.role.split(',')[0]}' or a "
                f"name to confirm, or switch profiles."
            )

    # --- WHERE -------------------------------------------------------------
    if rep.city is None:
        rep.questions.append(
            "Where is this taking place? (a city at minimum — naming the exact "
            "venue enables real proximity math)"
        )
    elif rep.venue is None:
        rep.assumptions.append(
            f"No venue named — using the centre of {rep.city} for proximity "
            f"math. Name the venue for sharper distances."
        )

    # --- WHEN --------------------------------------------------------------
    if rep.event_date is None:
        rep.questions.append(
            "When is this happening? (a date, or e.g. 'in 3 weeks' — timing "
            "drives which permits/incidents matter and the protection plan)"
        )

    # --- budget (planner) is config-supplied: an assumption, not a question -
    if rep.city is not None:
        rep.assumptions.append(
            f"Protection-plan budget: using the configured "
            f"{cfg.plan_budget:.0f} for {cfg.trip_days} day(s) — adjust in the "
            f"ToT planner tab."
        )
    return rep



# ---------------------------------------------------------------------------
# INTAKE, SLOT FORM — the same Module 6 rule over accumulated state
# ---------------------------------------------------------------------------

def intake_gaps(slots, cfg: Config) -> IntakeReport:
    """The intake guardrail over ``TaskingSlots`` instead of a raw string.

    Identical rule to ``intake_check`` — major gaps (who/where/when) become
    questions, minor gaps become stated assumptions — but asked of state that
    ACCUMULATES. That difference is the whole point: ``intake_check`` re-derives
    everything from a string on every call, so when the console carried an
    unfinished tasking forward by concatenation, a question the analyst had
    already answered was re-asked forever and the run could never start.

    Here a filled slot is simply not a gap, so each question is asked at most
    once and the report is monotonic: more turns, never more questions.
    """
    rep = IntakeReport(raw=" ".join(slots.turns))
    rep.city = slots.city
    rep.event_date = slots.event_date
    rep.venue = slots.venue
    rep.mentioned_names = [slots.protectee_name] if slots.protectee_name else []

    missing = slots.missing()
    if "who" in missing:
        rep.questions.append(
            "Who is the protectee? Give me a name and their public role — I "
            "never guess who I am protecting."
        )
    if "where" in missing:
        rep.questions.append(
            "Where is this taking place? (a city at minimum — naming the exact "
            "venue enables real proximity math)"
        )
    if "when" in missing:
        rep.questions.append(
            "When is this happening? (a date, or e.g. 'in 3 weeks' — timing "
            "drives which permits/incidents matter and the protection plan)"
        )

    # --- minor gaps: state them, do not ask ---------------------------------
    if slots.city and not slots.venue:
        rep.assumptions.append(
            f"No venue named — using the centre of {slots.city} for proximity "
            f"math. Name the venue for sharper distances."
        )
    if slots.protectee_name and not slots.protectee_role:
        rep.assumptions.append(
            f"No public role given for {slots.protectee_name} — entity "
            f"resolution will be weaker, so unrelated namesakes are more likely "
            f"to survive into collection."
        )
    if slots.city:
        rep.assumptions.append(
            f"Protection-plan budget: using the configured "
            f"{cfg.plan_budget:.0f} for {cfg.trip_days} day(s) — adjust in the "
            f"ToT planner tab."
        )
    return rep

# ---------------------------------------------------------------------------
# STALENESS CHECK (post-collection)
# ---------------------------------------------------------------------------

def staleness_check(findings: list[Finding], cfg: Config,
                    today: date | None = None) -> list[str]:
    """Flag LIVE findings whose event date is months old but that would be
    scored as current signal. Case-memory retrievals are exempt: they are
    *labeled* historical (that's recall, not staleness). Down-weights severity
    by half and returns trace notes."""
    today = today or date.today()
    max_age = int(cfg.raw.get("guardrails", {}).get("stale_after_days", 180))
    notes = []
    for f in findings:
        if f.detail.get("retrieved"):        # deliberately-historical material
            continue
        if f.event_date and f.event_date < today - timedelta(days=max_age) \
                and not f.detail.get("stale"):
            old = f.severity
            f.severity = round(f.severity * 0.5, 3)
            f.detail["stale"] = f"event_date {f.event_date} > {max_age}d old"
            notes.append(
                f"staleness: '{f.summary[:60]}' is dated {f.event_date} — "
                f"severity {old:.2f} -> {f.severity:.2f} (a re-post of old news "
                f"must not read as a new development)."
            )
    return notes


# ---------------------------------------------------------------------------
# GEO SANITY CHECK
# ---------------------------------------------------------------------------

def geo_sanity_check(findings: list[Finding], cfg: Config) -> list[str]:
    """Cheap plausibility checks on coordinates: on the globe, not null island,
    and protest sites actually near the venue (false pins get added to online
    maps; an 800 km 'proximity' hit is a data error, not a threat)."""
    max_km = float(cfg.raw.get("guardrails", {}).get("geo_max_km", 100))
    notes = []
    for f in findings:
        if f.source_type != SourceType.GEOSPATIAL:
            continue
        for key in ("venue_coords", "protest_coords"):
            c = f.detail.get(key)
            if c is None:
                continue
            lat, lon = float(c[0]), float(c[1])
            if (abs(lat) < 0.5 and abs(lon) < 0.5) or not (-90 <= lat <= 90) \
                    or not (-180 <= lon <= 180):
                f.severity = 0.0
                f.detail["geo_invalid"] = f"{key}={c}"
                notes.append(f"geo sanity: '{f.summary[:60]}' has implausible "
                             f"{key} {c} — zeroed, flagged for review.")
        dist = f.detail.get("distance_km")
        if dist is not None and dist > max_km and not f.detail.get("geo_far"):
            f.detail["geo_far"] = f"{dist} km from venue"
            notes.append(
                f"geo sanity: '{f.summary[:60]}' is {dist:.0f} km out — beyond "
                f"the {max_km:.0f} km plausibility radius; kept informational, "
                f"not proximate."
            )
    return notes


# ---------------------------------------------------------------------------
# SOURCE VETTING
# ---------------------------------------------------------------------------

def is_vetted(source_id: str, reliability: ReliabilityMemory) -> bool:
    """A source is vetted once it exists in the reliability memory — i.e. an
    analyst (or the shipped trust list) has scored it. Everything else is
    provisional: usable, but not allowed to independently drive an escalation."""
    return source_id in reliability._scores


__all__ = ["IntakeReport", "intake_check", "intake_gaps", "extract_city", "extract_event_date",
           "staleness_check", "geo_sanity_check", "is_vetted"]
