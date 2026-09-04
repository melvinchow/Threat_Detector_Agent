"""Module 6 guardrails, metrics, and rate limits — all offline/deterministic."""

from datetime import date, timedelta

from threat_detector import evals, ratelimit
from threat_detector.config import Config
from threat_detector.guardrails import (extract_city, extract_event_date,
                                        geo_sanity_check, intake_check,
                                        staleness_check)
from threat_detector.orchestrator import Orchestrator
from threat_detector.schemas import Finding, SourceType, ThreatChannel


# --- intake: major gaps ask, minor gaps assume -------------------------------

def test_missing_date_asks_when(synthetic_cfg, example_profile):
    rep = intake_check("CEO has a conference in Berlin", example_profile, synthetic_cfg)
    assert not rep.ready
    assert any("When" in q for q in rep.questions)
    # Role word matched the profile -> WHO is an assumption, not a question.
    # Read the name off the fixture rather than hardcoding it: the protectee is
    # demo data and gets renamed, and a test that pins the name fails for a
    # reason that has nothing to do with the guardrail it is checking.
    assert any(example_profile.name in a for a in rep.assumptions)


def test_missing_city_asks_where(synthetic_cfg, example_profile):
    rep = intake_check("CEO speaking at a conference in 2 weeks",
                       example_profile, synthetic_cfg)
    assert any("Where" in q for q in rep.questions)


def test_unknown_name_asks_who_never_guesses(synthetic_cfg, example_profile):
    rep = intake_check("Maria Chen visiting a summit in Prague on 2026-09-20",
                       example_profile, synthetic_cfg)
    assert any("Maria Chen" in q for q in rep.questions)


def test_complete_tasking_is_ready_with_stated_assumptions(synthetic_cfg, example_profile):
    rep = intake_check("CEO keynoting a conference in Berlin in 3 weeks",
                       example_profile, synthetic_cfg)
    assert rep.ready
    assert rep.city == "Berlin" and rep.event_date is not None
    assert rep.assumptions      # venue + budget assumptions are stated, not hidden


def test_city_extraction_is_generalized():
    # Any typed city is honored — never silently replaced by the demo scenario.
    assert extract_city("Board offsite in Lisbon next week") == "Lisbon"
    assert extract_city("summit in Kuala Lumpur") == "Kuala Lumpur"


def test_relative_and_absolute_dates_parse():
    today = date(2026, 8, 31)
    assert extract_event_date("keynote in 3 weeks", today) == today + timedelta(days=21)
    assert extract_event_date("gala on 2026-09-20", today) == date(2026, 9, 20)
    assert extract_event_date("event on October 3", today) == date(2026, 10, 3)


# --- staleness: old live signal down-weighted; case memory exempt ------------

def _finding(**kw):
    base = dict(channel=ThreatChannel.REPUTATION, source_type=SourceType.SOCIAL,
                source_id="s", summary="x", severity=0.8)
    base.update(kw)
    return Finding(**base)


def test_stale_live_finding_is_downweighted(synthetic_cfg):
    old = _finding(event_date=date.today() - timedelta(days=400))
    notes = staleness_check([old], synthetic_cfg)
    assert notes and old.severity == 0.4 and old.detail["stale"]


def test_case_memory_retrieval_is_exempt_from_staleness(synthetic_cfg):
    hist = _finding(event_date=date.today() - timedelta(days=400),
                    detail={"retrieved": True, "index": "case"})
    assert staleness_check([hist], synthetic_cfg) == []
    assert hist.severity == 0.8      # labeled-historical recall is not staleness


# --- geo sanity --------------------------------------------------------------

def test_null_island_coordinates_are_zeroed(synthetic_cfg):
    f = _finding(source_type=SourceType.GEOSPATIAL,
                 detail={"venue_coords": (0.0, 0.0)})
    notes = geo_sanity_check([f], synthetic_cfg)
    assert notes and f.severity == 0.0 and f.detail["geo_invalid"]


def test_implausibly_far_proximity_hit_is_flagged(synthetic_cfg):
    f = _finding(source_type=SourceType.GEOSPATIAL,
                 detail={"distance_km": 800.0})
    notes = geo_sanity_check([f], synthetic_cfg)
    assert notes and f.detail["geo_far"]


# --- honesty: fixtures are labeled, other cities aren't faked ----------------

def test_synthetic_findings_carry_the_fixture_tag(synthetic_cfg, example_profile):
    b = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks")
    tagged = [f for f in b.findings if f.detail.get("synthetic_fixture")]
    assert tagged, "all-synthetic config must label its findings"


def test_other_cities_get_generated_records_never_the_berlin_ones(
        synthetic_cfg, example_profile):
    """A city outside the curated scenario is served, but never by relabeling.

    The curated fixtures describe Berlin. Returning them for Prague would be a
    lie; returning nothing at all left the console looking broken. So Prague
    gets records generated FOR Prague, and the tag proves which is which — the
    honesty rule is about provenance being legible, not about staying empty.
    """
    b = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Prague on 2026-09-25")
    permits = [f for f in b.findings
               if f.source_type == SourceType.PROTEST_PERMIT]
    assert permits, "a tasked city should not come back with an empty panel"
    for f in permits:
        assert f.detail.get("generated_for_city") == "Prague"
        assert f.detail.get("synthetic_fixture") is True
        assert "berlin" not in f.source_id.lower()
        assert "Berlin" not in f.summary


def test_curated_fixtures_do_not_bleed_into_another_scenario(synthetic_cfg):
    """`_matches` ORs any query token over two characters, so the word "CEO"
    alone pulled the curated Berlin posts into a Dublin tasking — five findings
    about a different protectee in a different country, scored as if they were
    about this one. Fixtures now declare the scenario they describe."""
    from threat_detector.tools import open_source

    berlin = open_source.search_social("Jordan Vale CEO", synthetic_cfg,
                                       entity_id="jordan_vale", city="Berlin")
    assert berlin, "the curated scenario must still get its curated posts"

    dublin = open_source.search_social("Dana Reyes CEO", synthetic_cfg,
                                       entity_id="dana_reyes", city="Dublin")
    for f in dublin:
        assert "Jordan Vale" not in f.summary
        assert "Berlin" not in f.summary
        assert f.detail.get("generated_for_city") == "Dublin"


def test_unscoped_callers_keep_the_curated_baseline(synthetic_cfg):
    """The deterministic Orchestrator and the existing suite pass no scenario
    and must see exactly what they saw before."""
    from threat_detector.tools import open_source

    assert open_source.search_news("Jordan Vale", synthetic_cfg)


# --- rate limiter ------------------------------------------------------------

def test_rate_limiter_enforces_daily_budget(tmp_path):
    cfg = Config(raw={"limits": {"exa": 2}}, root=tmp_path)
    assert ratelimit.spend(cfg, "exa")
    assert ratelimit.spend(cfg, "exa")
    assert not ratelimit.spend(cfg, "exa")           # budget exhausted
    assert ratelimit.used_today(cfg, "exa") == 2     # failed spend not recorded
    assert ratelimit.status(cfg)["exa"] == (2, 2)


# --- evaluation metrics ------------------------------------------------------

def test_metrics_computed_from_briefing(synthetic_cfg, example_profile):
    b = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks")
    m = evals.evaluate_briefing(b, synthetic_cfg)
    assert m["groundedness_rate"] >= 0.95      # below this is hallucination
    assert m["human_gate_hit"] is True
    assert 0.0 <= m["escalation_rate"] <= 1.0
    assert m["fallback_success"] is True


def test_seeded_recall_finds_all_planted_threats(synthetic_cfg):
    r = evals.seeded_recall(synthetic_cfg)
    assert r["recall"] == 1.0, f"missed planted threats: {r['missed']}"
