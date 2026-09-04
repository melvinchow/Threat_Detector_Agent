# Module 4 — Tree-of-Thought Integration

How the ToT design from the Module 4 checkpoint is implemented in this repo,
with pointers to the code that executes each decision. Runnable proof:
`python -m threat_detector.cli demo` (the plan slate appears in the briefing)
or the GUI's **Travel Planner** tab (`python app.py`), which shows the beam
searching level by level.

## Where ToT is used — and where it is forbidden

ToT lives in exactly one place: **inside the Recommendation agent**
(`planner.py`, dispatched from `orchestrator._dispatch_planner`), planning how
to spend a limited security budget across a multi-day trip. Linear
chain-of-thought fails this problem: commit the budget to a motorcade on day 1
and there is no way to "un-decide" it when day 2's lodging becomes
unaffordable — the checkpoint's fighter-jet-escort example, which is a real
fixture (`convoy-motorcade` in `data/fixtures/vendors.json`) that the planner
must reject.

Everywhere else, exploratory/divergent generation is **disallowed by design**:
brainstorming over *what threats might exist* is precisely how a system
hallucinates threats. Collection stays conservative; planning is exploratory.
(Enforced structurally: the Risk agent has no retrieval tools; the planner has
no collection tools.)

## Structure

| checkpoint term | implementation |
|---|---|
| Thought / node = one expenditure | `schemas.Expenditure` (day, requirement, vendor, cost) |
| Branch = chronological expenditure list | `PlanNode.items` (a tuple — immutable) |
| Depth limit = trip days × requirement types | `slots = days × (movement, stationary, monitoring)`; 2 days → depth 6 |
| Final output = full plan meeting budget + coverage | `ProtectionPlan(complete=True)`; termination is *by construction* at full depth |

**Immutable state snapshots** (`PlanNode`, a frozen dataclass): each node holds
committed line items + remaining budget. Expansion creates a new node;
abandoning a branch drops a reference. Nothing is ever reverse-applied, which
is what makes backtracking correct.

**Read-only search**: the tree queries price and availability but never
reserves. Only after the analyst approves a plan (the GUI's Approve button)
does anything get committed — exploring 50 nodes must not create 50 bookings.

## Search strategy: beam (per the checkpoint UPDATE)

Beam width 3 (`config.yaml planner.beam_width`), level by level, keeping the
best k partial plans alive. Chosen over DFS because the real problem is
multi-day allocation against a **shared** budget — day 1's choice constrains
day 2, and beam keeps "splurge on the high-exposure arrival, economize later"
and "spread evenly" alive to be compared as *complete* plans. It also ends
with a slate of k plans: an analyst wants options with trade-offs, not a
single answer from a black box — and ties stop being a special case, because
ranking the final beam is just how the search terminates. Cost is bounded by
construction at ~k×b×d expansions (3×3×6 ≈ 54 here; the demo run logs 44).

## Evaluation: hard filters vs. soft scores

Kept strictly separate (`_generate_candidates` vs. `_soft_score`), because a
hard constraint averaged into a weighted score can be outvoted by comfort:

- **Hard (binary, at generation)**: available on the date; affordable after
  reserving the *floor cost* of every remaining requirement; not adjacent to a
  known protest site. Test: `test_hard_security_filter_beats_soft_scores` —
  the cheap protest-adjacent `hotel-spreeblick` never survives, no matter its
  reviews.
- **Soft (weighted, at ranking)**: discretion capability 0.35, venue proximity
  0.25, review stability 0.20, recent bad press −0.15, VIP brand preference
  **+0.05 with a −0.15 repeat-brand penalty**. The preference weight is
  deliberately the smallest, and repeats are penalized harder than preference
  rewards: a VIP who always books the same chain is a VIP whose lodging an
  adversary can predict. Comfort-versus-predictability, resolved toward
  tradecraft (`test_brand_preference_is_soft_not_decisive`).

## Risk & mitigation (per the checkpoint UPDATE)

With beam, branch explosion is bounded by construction. The honest failure
mode becomes **premature pruning under a weak partial-state score**. Two
mitigations, both implemented:

1. **Floor-cost lookahead as the score, not just the prune**
   (`_floor_costs` / `_partial_score`): partial plans are ranked using the
   same constraint-propagation computation that prunes infeasible branches —
   budget health = slack after reserving the cheapest way to cover everything
   still open. Because it is floor-based rather than even-split, a plan may
   legitimately front-load spending on a high-exposure day 1 — budget follows
   threat, and only genuine infeasibility scores badly. This is why the
   `convoy-motorcade` (affordable today, starves tomorrow) is pruned before
   descent (`test_lookahead_pruning_rejects_the_budget_buster`).
2. **Diversity enforcement**: the surviving beam may not contain plans that
   made the identical pick at the current slot, so it can't collapse into
   three variants of the same hotel (`test_diversity_enforcement_keeps_plans_distinct`).
   Near-identical *candidates* (same price band, same block) are additionally
   collapsed at generation, keeping the one with wider availability — if the
   analyst approves late, that vendor is likelier to still have a room
   (`test_near_identical_candidates_collapse_to_more_available_one`).

## The give-up branch

If no feasible expansion exists (e.g. budget below the floor of a full plan),
the planner returns the best branch **explicitly labeled incomplete**, with
`uncovered` naming every unmet requirement
(`test_infeasible_budget_returns_labeled_incomplete_plan`). A coverage gap is
exactly what an adversary needs; it must never silently read as coverage.
