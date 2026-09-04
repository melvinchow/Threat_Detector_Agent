# Module 6 — Guardrails, Evaluation Metrics & Human Intervention

How the Module 6 checkpoint is implemented, with code and test pointers. Watch
it run: type an underspecified tasking into the GUI (`python app.py`) — e.g.
*"CEO has a conference in Berlin"* — and the system asks follow-up questions
instead of running; the **Evals** tab shows the metrics after each run.

## What already existed (the structural guardrails)

The strongest guardrails in this system were designed in Modules 2–5 and are
structural, not bolt-on:

| guardrail | mechanism | module |
|---|---|---|
| Anti-hallucination | Risk agent has **no retrieval tools** — it can only score what was actually collected | M2 |
| Wrong-person/wrong-place | `entity_id` **hard pre-filter** in retrieval | M3 |
| Honest absence | `no_supporting_evidence` ≠ "no threat identified" | M3 |
| Severity ≤ sourcing | corroboration loop caps single-sourced claims (reported, never dropped) | M5 |
| No side effects during search | ToT booking tools **read-only**; slate is propose-only | M4 |
| PII isolation | access-class metadata; only Exposure holds PII | M2/M3 |
| Bounded loops | every loop: satisfaction condition + hard cap + labeled-incomplete fallback | M5 |

## New in Module 6 (`guardrails.py`, `evals.py`, `ratelimit.py`)

**1. Intake check — pre-generation** (`intake_check`). Is everything needed
already in long-term memory (the profile), config, or the tasking? Per the
checkpoint: minor gaps become stated **assumptions** (no venue → city centre,
budget → configured value); major gaps (**who / where / when**) become
**follow-up questions to the analyst**, and the run does not start until they
are answered. The GUI folds the analyst's answers back into the tasking and
re-checks; the CLI exits with the questions unless `--assume`. City extraction
is generalized: any city the analyst types is honored — the demo scenario is
never silently substituted (`test_city_extraction_is_generalized`). A tasking
naming someone who is not the active profile triggers a who-question — the
system never guesses the protectee (`test_unknown_name_asks_who_never_guesses`).

**2. Staleness check** (`staleness_check`). A live finding whose event date is
months old (the karma-farming repost problem — and the checkpoint's new-hire
analyst who can't catch it) is down-weighted 50% and flagged. Case-memory
retrievals are exempt: they are *labeled* historical, which is the difference
between recall and staleness (`test_case_memory_retrieval_is_exempt_from_staleness`).

**3. Geo sanity check** (`geo_sanity_check`). False pins get added to online
maps, so coordinates must be plausible: on the globe, not null island (0,0),
and proximity hits within a configured radius of the target city — an 800 km
"proximity" hit is a data error, not a threat.

**4. Source vetting** (`is_vetted` + the corroboration loop). A source the
reliability memory has never seen **cannot corroborate an escalation** — new
sources need analyst promotion first. Vetting a new source by "checking online
sentiment" would be circular (using the untrusted internet to vet the
internet); membership in the analyst-maintained reliability memory is the
structural alternative.

**5. Tool-access limit** (the dead-query guard). A corroboration query that
returned nothing new is never re-issued verbatim — kill the search instead of
burning budget on zero information.

**6. Honesty labeling** (`tools/base.mark_synthetic`). Every finding that came
from a synthetic fixture carries `synthetic_fixture: true`, and the briefing
states "N from live collection, M from fixtures". A real-configured source that
silently fell back (missing key, spent budget, network error) can never pass
demo data off as live collection.

**7. Strict API budgets** (`ratelimit.py`, `config.yaml limits:`). Every real
API call spends from a per-day budget persisted across restarts; exhausted
budgets degrade to fixtures honestly. This is also a cost guardrail: the
analyst can test many times a day without exhausting free tiers.

## Evaluation metrics (`evals.py` — the GUI's Evals tab)

A guardrail is a per-item check; a **metric** is a number aggregated across
runs that says whether the checks are calibrated:

| metric | target / reading |
|---|---|
| Groundedness rate | 95–100%; less is hallucination. ~100% *by construction* here; the metric exists to catch regressions when an LLM harness is enabled |
| Escalation rate | too high → alert fatigue; too low → gates catch nothing |
| Corroboration rate | share of high-severity findings with 2+ independent sources |
| Retrieval recall (seeded) | plant known threats (`SEEDED_THREATS`), measure how many surface — the only honest measure of M3 false-negative risk (`test_seeded_recall_finds_all_planted_threats`) |
| Fallback success | when a cap is hit, a labeled-incomplete product still ships |

The feedback loop (checkpoint §5): guardrails constrain, metrics measure
whether the constraints are calibrated, human decisions feed back into both.
**If escalation rate climbs, tighten collection filters — don't loosen the
gate**: an analyst asked to approve forty things a day starts rubber-stamping,
and a rubber-stamped approval manufactures a false audit trail.

## Human-intervention conditions (checkpoint §4 — where each is enforced)

- Anything world-changing (takedowns, contacting law enforcement, bookings):
  `requires_approval=True` on every such Recommendation; the ToT search itself
  is read-only.
- **Changing the trust/blacklist** is analyst-only: the reliability memory is
  only written by `ReliabilityMemory.set_score` from the GUI's Source Reliability tab —
  no agent has a tool that writes it. (Privilege escalation insight: if the
  agent could edit its own whitelist, every source guardrail would be
  bypassable.)
- **Committing to one itinerary**: the planner returns a ranked slate; the
  choice among trade-offs is a value judgment the analyst owns (Approve /
  Reject / Re-plan buttons).
- Changing sensitive VIP details: the profile is only written by the
  interactive `cli profile` flow — no agent tool writes PII.
