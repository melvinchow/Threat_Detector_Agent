"""ToT beam-search planner (Module 4): structure, pruning, and safety rules."""

from threat_detector.planner import REQUIREMENTS, load_vendors, plan_protection
from threat_detector.schemas import ProtectionPlan


def _all_items(plans):
    return [e for p in plans for e in p.line_items]


def test_beam_terminates_at_full_depth_with_complete_slate(synthetic_cfg):
    plans, trace = plan_protection(synthetic_cfg)
    # Terminates by construction: depth = trip_days x requirement types.
    depth = synthetic_cfg.trip_days * len(REQUIREMENTS)
    assert all(p.complete and len(p.line_items) == depth for p in plans)
    # A slate, not a single answer — and ranked.
    assert 1 < len(plans) <= synthetic_cfg.beam_width
    assert trace[-1]["event"] == "done"


def test_every_plan_respects_the_budget(synthetic_cfg):
    plans, _ = plan_protection(synthetic_cfg)
    for p in plans:
        assert p.total_cost <= p.budget
        assert abs(sum(e.cost for e in p.line_items) - p.total_cost) < 1e-6


def test_lookahead_pruning_rejects_the_budget_buster(synthetic_cfg):
    """The 'fighter jet escort' analog: the armored convoy is affordable on
    day 1 in isolation, but starves the floor cost of the remaining slots —
    constraint propagation must prune it before descending."""
    plans, trace = plan_protection(synthetic_cfg)
    assert all(e.vendor_id != "convoy-motorcade" for e in _all_items(plans))
    assert sum(t.get("pruned_by_lookahead", 0) for t in trace
               if t["event"] == "level") > 0


def test_hard_security_filter_beats_soft_scores(synthetic_cfg):
    """A hotel adjacent to a known protest site must never appear, no matter
    how cheap or well-reviewed: hard constraints are filters, never averaged
    into a weighted score where loyalty points could outvote them."""
    plans, _ = plan_protection(synthetic_cfg)
    assert all(e.vendor_id != "hotel-spreeblick" for e in _all_items(plans))


def test_availability_is_respected(synthetic_cfg):
    """Police escort fleet is booked on arrival day (available_days=[2]) —
    the checkpoint's worked example. No plan may use it on day 1."""
    plans, _ = plan_protection(synthetic_cfg)
    assert all(not (e.vendor_id == "police-escort" and e.day == 1)
               for e in _all_items(plans))


def test_diversity_enforcement_keeps_plans_distinct(synthetic_cfg):
    """The surviving beam must not be three variants of the same plan."""
    plans, _ = plan_protection(synthetic_cfg)
    signatures = {tuple(e.vendor_id for e in p.line_items) for p in plans}
    assert len(signatures) == len(plans)


def test_infeasible_budget_returns_labeled_incomplete_plan(synthetic_cfg):
    """The Module 5 give-up branch: when a cap/constraint is hit, the output is
    explicitly labeled incomplete, listing what went unresolved — a coverage
    gap must never silently read as coverage."""
    plans, trace = plan_protection(synthetic_cfg, budget=1000)
    assert plans and isinstance(plans[0], ProtectionPlan)
    assert not plans[0].complete
    assert plans[0].uncovered            # names the missing requirements
    assert any(t["event"] == "infeasible" for t in trace)


def test_near_identical_candidates_collapse_to_more_available_one(synthetic_cfg):
    """Kastanienhof and Lindenhof are same-block, same-price-band twins; the
    generator collapses them and keeps the one with wider availability."""
    plans, _ = plan_protection(synthetic_cfg)
    used = {e.vendor_id for e in _all_items(plans)}
    assert "hotel-kastanienhof" not in used     # day-1-only twin loses


def test_brand_preference_is_soft_not_decisive(synthetic_cfg):
    """Preferred brand nudges the score but a plan slate must not collapse to
    the preferred chain for every night (predictability is exposure)."""
    plans, _ = plan_protection(synthetic_cfg, preferred_brands=["Meridian"])
    stationary = [e.vendor_name for e in _all_items(plans)
                  if e.requirement == "stationary"]
    assert any("Meridian" not in name for name in stationary)


def test_vendors_fixture_loads(synthetic_cfg):
    vendors = load_vendors(synthetic_cfg)
    cats = {v.category for v in vendors}
    assert cats == set(REQUIREMENTS)
