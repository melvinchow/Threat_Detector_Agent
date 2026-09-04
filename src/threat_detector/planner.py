"""Tree-of-Thought protection planner (Module 4), nested inside Recommendation.

The one place in this system where reasoning is *exploratory*: allocating a
shared, limited security budget across every requirement of a multi-day trip.
Linear chain-of-thought fails here — spend the budget on a motorcade on day 1
and there is no way to "un-decide" it when day 2's lodging becomes unaffordable.

Structure (straight from the Module 4 checkpoint):

* A **node** is one expenditure (what is bought, from whom, $ spent).
* A **branch** is a chronological list of expenditures — like line items on a
  card statement. Depth = trip_days x requirement_types (2 days x
  [movement, stationary, monitoring] = 6 nodes per complete branch).
* **Beam search** (not DFS): level-by-level, keeping the best ``beam_width``
  partial plans alive. Chosen over DFS because the real problem is multi-day
  allocation against a *shared* budget — day 1's choice constrains day 5, and
  beam keeps "splurge on the high-exposure arrival, economize on quiet nights"
  and "spread evenly" alive to be compared as complete plans. It also ends with
  a *slate* of k complete plans, which is what an analyst wants: options with
  trade-offs, not a single answer from a black box. (Ties stop being a special
  case — ranking the final beam is just how the search terminates.)

The four ToT roles, mapped to code the same way the checkpoint maps them to
tools (generator / critic / controller / state):

* ``_generate_candidates``  — thought generator: proposes at most
  ``max_candidates`` options per slot, pre-filtered on HARD constraints.
  Capping at generation is cheaper than generating ten and pruning seven.
* ``_soft_score``           — critic: deterministic arithmetic (no LLM call can
  hallucinate a number here). Soft criteria only — hard constraints are
  filters, never averaged in, so loyalty points can't outvote a hotel sitting
  across from a permitted demonstration.
* ``plan_protection``       — controller: owns the level loop, the beam
  selection, diversity enforcement, and termination (by construction at depth).
* ``PlanNode``              — immutable state snapshots. Expanding a node makes
  a NEW node; abandoning a branch just drops it. Nothing is ever
  reverse-applied, which is what makes pruning/backtracking correct.

Safety rule carried over from the rest of the system: the search is READ-ONLY.
It queries price and availability but never books. Exploring 50 nodes must not
create 50 hotel reservations — only the analyst-approved plan is ever acted on.

Failure mode (per the checkpoint UPDATE): with beam, branch explosion is bounded
by construction (~k x b x d expansions). The honest risk becomes **premature
pruning under a weak partial-state score** — mitigated by (a) scoring partials
with the same floor-cost lookahead used for pruning, so cheap-early plans get no
free pass, and (b) diversity enforcement, so the surviving beam holds genuinely
different plans rather than three variants of the same hotel.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .schemas import Expenditure, ProtectionPlan

# Requirement types per trip day, in chronological order within the day —
# the checkpoint's "movement protection, stationary protection, online
# monitoring for leaks or new immediate dangers on-route".
REQUIREMENTS = ("movement", "stationary", "monitoring")


@dataclass(frozen=True)
class Vendor:
    id: str
    name: str
    category: str
    brand: str
    cost_per_day: float
    available_days: tuple[int, ...] | str      # tuple of day numbers, or "all"
    distance_km: float | None
    protest_adjacent: bool
    discretion: float
    review_stability: float
    recent_bad_press: bool

    def available(self, day: int) -> bool:
        return self.available_days == "all" or day in self.available_days


@dataclass(frozen=True)
class PlanNode:
    """Immutable snapshot of one partial plan: committed line items, remaining
    budget, and how far down the slot list we are. Frozen on purpose — beam
    abandonment is dropping a reference, never mutating a shared ledger."""

    items: tuple[Expenditure, ...]
    remaining: float
    slot: int              # index into the slots list = current depth
    soft_total: float      # accumulated soft score of committed items


def _real_lodging_vendors(city: str, cfg: Config) -> list[Vendor]:
    """REAL hotels near the city (names + distances from OpenStreetMap), with
    SYNTHETIC prices and security attributes — OSM has neither, so those are
    derived deterministically from the hotel name and honestly labeled. The
    checkpoint's hybrid: 'basic hotel venue data through open source maps,
    security services stay fake'."""
    from .tools.geospatial import hotels_near

    vendors = []
    for h in hotels_near(city, cfg):
        # Stable pseudo-attributes: same hotel -> same numbers on every run.
        seed = int(hashlib.sha1(h["name"].encode()).hexdigest()[:6], 16)
        r1, r2, r3 = (seed % 97) / 97, (seed // 97 % 89) / 89, (seed // 8633 % 83) / 83
        vendors.append(Vendor(
            id=f"osm-{seed:x}",
            name=f"{h['name']} (real hotel; price/security attrs synthetic)",
            category="stationary",
            brand=h["brand"],
            cost_per_day=round(900 + 1500 * r1, -1),
            available_days="all",
            distance_km=h["distance_km"],
            protest_adjacent=False,     # unknown from OSM; cross-check is the
                                        # proximity wave's job, not the catalog's
            discretion=round(0.5 + 0.4 * r2, 2),
            review_stability=round(0.6 + 0.35 * r3, 2),
            recent_bad_press=False,
        ))
    return vendors


def load_vendors(cfg: Config, city: str | None = None) -> list[Vendor]:
    raw = json.loads(Path(cfg.fixture("vendors.json")).read_text())
    out = []
    for v in raw["vendors"]:
        days = v["available_days"]
        out.append(Vendor(
            id=v["id"], name=v["name"], category=v["category"], brand=v["brand"],
            cost_per_day=float(v["cost_per_day"]),
            available_days="all" if days == "all" else tuple(days),
            distance_km=v["distance_km"], protest_adjacent=v["protest_adjacent"],
            discretion=v["discretion"], review_stability=v["review_stability"],
            recent_bad_press=v["recent_bad_press"],
        ))
    # Hybrid lodging: swap the synthetic hotels for real OSM ones when
    # configured and available. Movement and monitoring stay fully synthetic
    # (no legitimate open API exists for security escorts).
    if city and cfg.raw.get("planner", {}).get("lodging") == "osm":
        real = _real_lodging_vendors(city, cfg)
        if real:
            out = [v for v in out if v.category != "stationary"] + real
    return out


# --- critic: deterministic soft scoring -------------------------------------
# Weights are explicit and defensible. Brand preference is DELIBERATELY the
# smallest weight: a VIP who always stays at the same chain is a VIP whose
# lodging an adversary can predict, so comfort is scored low relative to
# discretion and proximity (the comfort-vs-predictability tension, resolved
# toward tradecraft).
_W_DISCRETION = 0.35
_W_PROXIMITY = 0.25
_W_REVIEWS = 0.20
_W_BAD_PRESS = -0.15
_W_BRAND_PREF = 0.05
# Pattern-variance penalty: booking the SAME brand for another night makes the
# VIP's lodging predictable, and predictability is exposure. Deliberately larger
# than the brand-preference bonus, so tradecraft beats comfort.
_W_REPEAT_BRAND = -0.15


def _soft_score(v: Vendor, preferred_brands: list[str]) -> float:
    s = _W_DISCRETION * v.discretion + _W_REVIEWS * v.review_stability
    if v.distance_km is not None:
        s += _W_PROXIMITY * max(0.0, 1.0 - v.distance_km / 10.0)
    else:
        s += _W_PROXIMITY * 0.5          # distance not applicable -> neutral
    if v.recent_bad_press:
        s += _W_BAD_PRESS
    if any(b.lower() in v.brand.lower() for b in preferred_brands):
        s += _W_BRAND_PREF
    return max(0.0, s)


# --- lookahead: constraint propagation --------------------------------------
def _floor_costs(vendors: list[Vendor], slots: list[tuple[int, str]]) -> list[float]:
    """floor[i] = cheapest possible way to cover slots i..end.

    The checkpoint's lookahead pruning: before descending from any node, if
    remaining budget < the sum of the floors of everything still uncovered, the
    branch is provably infeasible — prune now, no depth will fix it. The same
    number doubles as the budget-health term when scoring partial plans, so the
    score can't be gamed by spending nothing early.
    """
    floors = [0.0] * (len(slots) + 1)
    for i in range(len(slots) - 1, -1, -1):
        day, req = slots[i]
        options = [v.cost_per_day for v in vendors if v.category == req and v.available(day)]
        floors[i] = (min(options) if options else float("inf")) + floors[i + 1]
    return floors


# --- generator: candidates for one slot, hard-filtered ----------------------
def _generate_candidates(vendors: list[Vendor], day: int, req: str,
                         remaining: float, floor_after: float,
                         preferred_brands: list[str],
                         max_candidates: int) -> list[Vendor]:
    """HARD constraints only: available that day, no adjacency to a known
    protest site (minimum security adequacy), and affordable once the floor
    cost of every *remaining* requirement is set aside."""
    ok = [
        v for v in vendors
        if v.category == req and v.available(day) and not v.protest_adjacent
        and v.cost_per_day <= remaining - floor_after
    ]
    # Collapse near-identical candidates (same price band, similar distance):
    # two 4-star hotels on the same block shouldn't count as different plans.
    # Keep the one with wider availability — if the analyst approves late, the
    # more-available vendor is likelier to still have a room.
    collapsed: dict[tuple, Vendor] = {}
    for v in sorted(ok, key=lambda v: v.available_days == "all", reverse=True):
        band = int(v.cost_per_day // 200)
        dist = round(v.distance_km) if v.distance_km is not None else None
        collapsed.setdefault((band, dist), v)
    ranked = sorted(collapsed.values(),
                    key=lambda v: _soft_score(v, preferred_brands), reverse=True)
    return ranked[:max_candidates]


# --- controller: the beam ----------------------------------------------------
def plan_protection(cfg: Config, preferred_brands: list[str] | None = None,
                    trip_days: int | None = None, budget: float | None = None,
                    city: str | None = None,
                    ) -> tuple[list[ProtectionPlan], list[dict]]:
    """Run the beam search. Returns (slate of plans, beam trace).

    The beam trace is one dict per level — what was expanded, what lookahead
    pruned, what diversity dropped, and what survived — so the GUI's planner tab
    can show the tree actually being searched rather than only its answer.
    """
    preferred_brands = preferred_brands or []
    trip_days = trip_days or cfg.trip_days
    budget = budget if budget is not None else cfg.plan_budget
    width, max_cand = cfg.beam_width, cfg.max_candidates

    vendors = load_vendors(cfg, city=city)
    slots = [(day, req) for day in range(1, trip_days + 1) for req in REQUIREMENTS]
    floors = _floor_costs(vendors, slots)
    trace: list[dict] = [{
        "event": "start", "budget": budget, "depth": len(slots),
        "beam_width": width, "floor_total": floors[0],
        "note": f"depth = {trip_days} day(s) x {len(REQUIREMENTS)} requirement types; "
                f"cheapest conceivable full plan costs {floors[0]:.0f}",
    }]

    beam: list[PlanNode] = [PlanNode(items=(), remaining=budget, slot=0, soft_total=0.0)]

    for i, (day, req) in enumerate(slots):
        expansions: list[tuple[float, PlanNode, Vendor]] = []
        pruned = 0
        for node in beam:
            cands = _generate_candidates(vendors, day, req, node.remaining,
                                         floors[i + 1], preferred_brands, max_cand)
            # How many affordable-today options did lookahead veto?
            naive = [v for v in vendors
                     if v.category == req and v.available(day)
                     and not v.protest_adjacent and v.cost_per_day <= node.remaining]
            pruned += max(0, len(naive) - len(cands))
            by_id = {v.id: v for v in vendors}
            prior_brands = {by_id[e.vendor_id].brand for e in node.items
                            if e.requirement == "stationary"}
            for v in cands:
                item_soft = _soft_score(v, preferred_brands)
                if req == "stationary" and v.brand in prior_brands:
                    item_soft = max(0.0, item_soft + _W_REPEAT_BRAND)
                child = PlanNode(
                    items=node.items + (Expenditure(
                        day=day, requirement=req, vendor_id=v.id,
                        vendor_name=v.name, cost=v.cost_per_day,
                    ),),
                    remaining=node.remaining - v.cost_per_day,
                    slot=i + 1,
                    soft_total=node.soft_total + item_soft,
                )
                expansions.append((_partial_score(child, floors, budget), child, v))

        expansions.sort(key=lambda t: t[0], reverse=True)

        # Diversity enforcement: don't let the beam become three variants of the
        # same choice — a kept plan blocks other plans making the identical pick
        # at this slot, unless the beam can't be filled otherwise.
        kept: list[tuple[float, PlanNode, Vendor]] = []
        dropped_dupes = 0
        for entry in expansions:
            if len(kept) >= width:
                break
            if any(k[2].id == entry[2].id for k in kept):
                dropped_dupes += 1
                continue
            kept.append(entry)
        if len(kept) < width:                      # backfill if too strict
            for entry in expansions:
                if len(kept) >= width:
                    break
                if entry not in kept:
                    kept.append(entry)

        trace.append({
            "event": "level", "slot": i + 1, "day": day, "requirement": req,
            "expanded": len(expansions), "pruned_by_lookahead": pruned,
            "dropped_for_diversity": dropped_dupes,
            "beam": [{
                "picks": [e.vendor_name for e in n.items],
                "spent": round(budget - n.remaining, 0),
                "remaining": round(n.remaining, 0),
                "score": round(s, 3),
            } for s, n, _ in kept],
        })

        if not kept:
            # No feasible expansion at this slot: the Module 5 give-up branch.
            # Return best-so-far, explicitly labeled incomplete — never silently
            # pretend coverage exists ("no plan" must never read as "no risk").
            uncovered = [f"day {d} {r}" for d, r in slots[i:]]
            trace.append({"event": "infeasible", "at": f"day {day} {req}",
                          "uncovered": uncovered})
            slate = [_to_plan(n, budget, uncovered, rank, complete=False)
                     for rank, n in enumerate(beam, 1)]
            return slate, trace

        beam = [n for _, n, _ in kept]

    slate = [_to_plan(n, budget, [], rank, complete=True)
             for rank, n in enumerate(beam, 1)]
    trace.append({"event": "done", "complete_plans": len(slate),
                  "note": "terminated by construction at full depth — "
                          "a bounded search needs no retry cap"})
    return slate, trace


def _partial_score(node: PlanNode, floors: list[float], budget: float) -> float:
    """Rank a partial plan: quality of what's committed + budget health.

    Budget health = slack after reserving the floor cost of everything still
    uncovered, normalized. Using the lookahead floors here (not an even split)
    means a plan may legitimately front-load spending on a high-exposure day 1 —
    budget follows threat, and only genuine infeasibility scores badly.
    """
    committed = node.soft_total / max(1, len(node.items))
    slack = node.remaining - floors[node.slot]
    health = max(0.0, min(1.0, slack / max(budget * 0.25, 1.0)))
    return 0.7 * committed + 0.3 * health


def _to_plan(node: PlanNode, budget: float, uncovered: list[str],
             rank: int, complete: bool) -> ProtectionPlan:
    total = budget - node.remaining
    rationale = []
    if complete:
        rationale.append(
            f"Covers every requirement of the trip at {total:.0f} of {budget:.0f} budget."
        )
    else:
        rationale.append(
            "INCOMPLETE — no feasible option for: " + ", ".join(uncovered)
            + ". Reported as-is; a coverage gap is exactly what an adversary needs."
        )
    picks = {}
    for e in node.items:
        picks.setdefault(e.requirement, []).append(e.vendor_name)
    for req, names in picks.items():
        rationale.append(f"{req}: " + " / ".join(dict.fromkeys(names)))
    return ProtectionPlan(
        plan_id=f"option-{rank}",
        line_items=list(node.items),
        total_cost=round(total, 2),
        budget=budget,
        score=round(node.soft_total / max(1, len(node.items)), 3),
        complete=complete,
        uncovered=uncovered,
        rationale=rationale,
    )


__all__ = ["plan_protection", "load_vendors", "REQUIREMENTS", "PlanNode", "Vendor"]
