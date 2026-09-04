"""End-to-end orchestrator behavior: the wave loop and its pivots."""

from threat_detector.orchestrator import Orchestrator, parse_tasking


def test_parse_tasking_extracts_city_and_event():
    t = parse_tasking("CEO keynoting a conference in Berlin in 3 weeks")
    assert t.city == "Berlin"
    assert "conference" in t.event_terms


def test_full_assessment_escalates_and_scores(synthetic_cfg, example_profile):
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks"
    )

    trace = "\n".join(briefing.trace)
    # Wave 1 -> named group surfaces -> targeted wave 2 -> exposure escalation.
    assert "wave 1" in trace
    assert "Open Sky Collective" in trace
    assert "wave 2" in trace
    assert "exposure escalation" in trace

    # A real, multi-channel risk picture came out.
    assert briefing.score.overall > 0.3
    assert "physical_location" in briefing.score.by_channel
    assert "exposure" in briefing.score.by_channel

    # Every world-changing recommendation is gated.
    world_changing = [r for r in briefing.recommendations if r.requires_approval]
    assert world_changing, "expected at least one approval-gated action"


def test_focus_prunes_specialists(synthetic_cfg, example_profile):
    # Focusing on sentiment should skip the geospatial/public-records location sweep.
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "General sentiment check on the CEO", focus="sentiment"
    )
    trace = "\n".join(briefing.trace)
    assert "Geospatial" not in trace


def test_low_reliability_source_is_downweighted(synthetic_cfg, example_profile):
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks"
    )
    trace = "\n".join(briefing.trace)
    # r/rumors is scored 0.2 in source_reliability.json (< min_reliability 0.35).
    assert "down-weighting reddit:r/rumors" in trace


def test_retrieval_injects_case_memory_pattern(synthetic_cfg, example_profile):
    tasking = "CEO keynoting a conference in Berlin in 3 weeks"
    with_ = Orchestrator(example_profile, synthetic_cfg).assess(tasking, use_retrieval=True)
    without = Orchestrator(example_profile, synthetic_cfg).assess(tasking, use_retrieval=False)

    # Retrieval adds attributable findings the live-only run doesn't have...
    retrieved = [f for f in with_.findings if f.detail.get("retrieved")]
    assert retrieved, "retrieval wave should contribute findings"
    # ...specifically the 14-month-old case-memory surveillance post.
    assert any(f.source_id == "forum:redwing" and f.detail.get("index") == "case"
               for f in retrieved)
    assert len(with_.findings) > len(without.findings)


# --- Module 5: coordination rules ------------------------------------------

def test_collection_loop_reports_saturation(synthetic_cfg, example_profile):
    """The loop's satisfaction condition: a wave surfacing zero new entities
    closes collection — the cap is the safety net, not the exit."""
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks"
    )
    trace = "\n".join(briefing.trace)
    assert "saturation" in trace
    assert "collection loop closed" in trace


def test_single_sourced_high_severity_claim_is_capped_not_dropped(
        synthetic_cfg, example_profile):
    """The assessment<->orchestrator two-way edge: after the corroboration
    rounds are exhausted, a single-sourced claim is REPORTED with severity
    capped and flagged — never silently dropped."""
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks"
    )
    flagged = [f for f in briefing.findings if f.detail.get("uncorroborated")]
    assert flagged, "the dark-web itinerary claim should stay single-sourced"
    for f in flagged:
        assert f.severity <= synthetic_cfg.uncorroborated_cap + 1e-9
    trace = "\n".join(briefing.trace)
    assert "two-way edge" in trace
    assert "reported, not dropped" in trace


def test_corroborated_pattern_is_not_capped(synthetic_cfg, example_profile):
    """@drk_wolf appears across 3 independent sources (forum, social probe,
    incident log) — a documented pattern must NOT be confidence-capped."""
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks"
    )
    drk = [f for f in briefing.findings
           if "drk_wolf" in (f.summary + str(f.detail))]
    assert drk
    assert not any(f.detail.get("uncorroborated") for f in drk)


def test_live_trace_events_stream_to_observer(synthetic_cfg, example_profile):
    """The GUI hook: every trace line is pushed to on_event as it happens."""
    seen: list[str] = []
    briefing = Orchestrator(example_profile, synthetic_cfg,
                            on_event=seen.append).assess(
        "CEO keynoting a conference in Berlin in 3 weeks"
    )
    assert seen == briefing.trace


# --- Module 4: planner integration ------------------------------------------

def test_elevated_travel_tasking_attaches_plan_slate(synthetic_cfg, example_profile):
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks"
    )
    assert briefing.plans, "elevated risk + travel should trigger the ToT planner"
    assert all(p.complete for p in briefing.plans)
    # The slate rides in behind the analyst approval gate.
    plan_recs = [r for r in briefing.recommendations
                 if "protection plan" in r.action.lower()]
    assert plan_recs and all(r.requires_approval for r in plan_recs)


def test_no_travel_no_planner(synthetic_cfg, example_profile):
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "General sentiment check on the CEO", focus="sentiment"
    )
    assert briefing.plans == []


def test_retrieval_can_be_disabled(synthetic_cfg, example_profile):
    briefing = Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks", use_retrieval=False
    )
    assert not any(f.detail.get("retrieved") for f in briefing.findings)
