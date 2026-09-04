# Module 5 — Multi-Agent Architecture & Coordination

How the coordination plan from the Module 5 checkpoint is implemented, with
pointers to the code and tests. Watch it run: `python -m threat_detector.cli
demo` (the trace prints every rule firing) or the GUI (`python app.py`), whose
Trace tab streams the loop live.

## Topology: hierarchical, 8 agents, shallow critical path

One orchestrator supervises a **parallel collection tier** (Open-Source,
Records, Exposure, Geospatial + the Retrieval agent), followed by a sequential
Assessment → Recommendation → Reporting spine, with the ToT beam nested inside
Recommendation. The count answer: coordination overhead scales with
*communicating pairs*, not agent count — the collection agents are 4 parallel
siblings with **zero peer edges**, so they are width, not depth; the critical
path is 3–5 agents deep despite 8 agents total.

The orchestrator states the topology and per-edge protocols in its trace at
the start of every run (`orchestrator.assess`, second `[think]` line), so the
coordination design is visible in the artifact, not just in this document.

## Protocol per edge (not per system)

| edge | protocol | why |
|---|---|---|
| orchestrator → collection tier | parallel fan-out | no dependencies between siblings |
| collection → orchestrator | **one-way** | a distance or a docket needs no debate; a round trip is an opportunity to revise a correct answer into a wrong one |
| assessment ↔ orchestrator | **two-way** | the corroboration loop (below) — the validation dialogue two-way exists for |
| assessment → reporting | one-way | the score is an input, not a negotiation |
| recommendation ↔ analyst | **conditional** | approve / revise / reject — real buttons in the GUI's ToT tab (`app.py on_decide` / `on_replan`) |
| inside Recommendation | **brainstorm** — and *nowhere else* | planning is exploratory; divergent generation during evidence collection is how a system hallucinates threats |

## Every loop: a satisfaction condition AND a hard cap

The condition is the intended exit; the cap is the safety net for when the
condition never fires. Both are config, not prompt text (`config.yaml
coordination:` block) — a rule that matters is enforced in code.

**1. Collection loop** (`orchestrator.assess`, the wave `while`):
- *Condition:* **saturation** — the latest wave surfaced zero new entities.
  A sweep is done when new queries stop returning anything you don't already
  have (`test_collection_loop_reports_saturation`).
- *Cap:* `max_collection_waves: 3`. Hitting it logs the unexplored entities as
  **unresolved** into short-term memory — never silently dropped.

**2. Assessment revision loop** (`orchestrator._corroboration_loop`):
- *Condition:* no claim serious enough to matter (severity ≥ 0.7) rests on a
  single source. Independence is by `source_id` — forty reposts of one wire
  story are still one source. Corroboration = same named actor from a
  different source; failing an actor, a different source of the *same kind*
  (a broker listing does not corroborate dark-web chatter just because both
  are "exposure"). Computed geometry (Geospatial) is exempt: arithmetic over
  already-validated inputs is not a claim needing a second source.
- *Cap:* `corroboration_rounds: 2` targeted requests back to collection. Then
  the claim is **reported with severity capped to 0.55 and flagged
  `uncorroborated`** — never dropped, because failure to confirm is normal,
  not exceptional (`test_single_sourced_high_severity_claim_is_capped_not_dropped`).
  The demo shows both outcomes honestly: the single dark-web itinerary post
  gets capped; the @drk_wolf pattern (3 independent sources: forum, social
  probe, incident log) does not (`test_corroborated_pattern_is_not_capped`).

**3. ToT beam:** terminates **by construction** at depth = trip days ×
requirement types — the contrast case: a well-bounded search needs no retry
cap because the structure itself is finite
(`test_beam_terminates_at_full_depth_with_complete_slate`).

## Role mapping (lab vocabulary)

Orchestrator = planner; the 4 collection agents = researchers; Risk
Assessment = **critic/evaluator** (it scores, it does not compose); the
Reporting surface (briefing renderer / GUI) = writer; Recommendation =
decision-maker with the ToT subgraph nested inside. The architecture is not
forced into the lab's three-role template — the lab is a teaching toy with one
task; this system has more shape, and the roles above say which agent carries
each function.

## The GUI as the coordination artifact

The TA's Local-Agent-Demo pattern (chat + live teaching panels) applied to
this system (`app.py`):

- **Short-term tab** — the per-run scratchpad (`memory.ShortTermMemory`):
  tasking, focus, which specialists answered, wave notes. Dies with the run.
- **Source Reliability tab** — what survives restarts: source reliability (analyst
  feedback persists via `ReliabilityMemory.set_score`, so a source
  down-weighted today stays down-weighted next session — the alert-fatigue
  fix), the curated case-memory index, and the profile's public view (the PII
  block is withheld from the GUI itself — the trust boundary includes the
  operator's screen).
- **Trace tab** — the Think → Act → Observe → Adapt loop streamed live over
  the orchestrator's `on_event` hook (`test_live_trace_events_stream_to_observer`).
- **Travel Planner tab** — the beam level by level (expansions, lookahead prunes,
  diversity drops, the surviving beam), the ranked slate, and the analyst's
  conditional edge: Approve ✅ / Reject ❌ / Re-plan 🔁 with a revised budget.
- **Agents tab** — the roster, edges, and loop limits.

Everything runs in deterministic mode: no API keys, reproducible, and the same
pipeline the pytest suite grades.
