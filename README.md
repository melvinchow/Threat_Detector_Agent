# ThreatDetector — Multi-Agent VIP Threat Assessment System

Capstone agentic system (Melvin Chow). An orchestrated crew of specialist agents
that assesses physical and reputational threats to a high-profile individual (a
"VIP" / protectee) based on their travel plans, internet activity, public-records
footprint, and exposure of their personal data.

This implements the design proposed in the **Module 1–5 capstone checkpoints**:
the agent roster and Think → Act → Observe → Adapt loop (M2), the RAG layer
(M3), the Tree-of-Thought protection planner (M4), the multi-agent coordination
rules (M5) — and an **analyst GUI** (`python app.py`) with live panels for
short-term memory, source-reliability memory, the reasoning trace, and the
planner's beam search.

---

## What it does

Given a **tasking** such as *"CEO is keynoting a conference in Berlin in 3 weeks"*,
the system:

1. Loads (or helps you build) a **VIP profile** — the protectee, their close
   relationships, and the sensitive PII only the security team should hold.
2. Runs the **orchestrator loop**: it reasons about *what it does not yet know
   that would most change the risk assessment*, dispatches specialist agents in
   waves, and re-plans based on what comes back.
3. Produces a **threat briefing**: a numeric risk score, the evidence behind it,
   and recommended next actions — with every action gated behind analyst approval.

## The architecture (from the Module 2 checkpoint)

The system is deliberately split into one **orchestrator** plus several
**specialist agents**, each owning a small, coherent set of tools. This is not
tidiness — it is a safety argument:

| Agent | Colour in the roster | Owns | Why it is isolated |
|---|---|---|---|
| **Orchestrator** | grey | delegation + memory; *no external tools* | Decides *what to ask*, never fetches. Can't invent a source mid-analysis. |
| **Open-Source Monitoring** | green | news, social (Reddit), sentiment, entity resolution | Read-only. Entity resolution stops "Jordan Vale the CEO" being confused with an unrelated namesake. |
| **Public Records** | green | court dockets, incident/CAD logs, protest permits | Forward-looking signal (permits filed for *future* dates). |
| **Exposure** | green | data-broker scan, breach lookup, dark-web, doxxing | **The only agent that holds the VIP's PII.** No other agent can read from it. |
| **Geospatial** | green | geocoding, distance/routing, proximity | Exists *because an LLM cannot do arithmetic on coordinates* — a real computation limitation. |
| **Case Memory / Retrieval** | teal | RAG pipeline (pre-filter → hybrid → rerank) + SQLite facts store | The RAG layer (Module 3): surfaces indexed historical/oblique signal a live scrape can't, scoped hard to the protectee. |
| **Risk Assessment** | purple | scoring rubric, calculator, source-reliability lookup | Has **no retrieval tools at all.** It can only score what was actually returned — this is what structurally prevents hallucinated threats. |
| **Recommendation** | red | playbooks, budget constraints, **the nested ToT beam planner** | Proposes actions; **never executes.** A human approves first. The system's one exploratory loop lives here — and nowhere else. |

The reasoning loop:

```
Think    → "What don't I know that would most change the risk assessment?"
Act      → dispatch a wave of collection specialists
Observe  → validate each finding against source-reliability memory
Adapt    → re-plan the next wave based on what came back (escalate if needed)
```

Collection runs in **waves**, not one big parallel fan-out: a broad sweep first,
then a targeted second wave triggered by what the first wave surfaced. That pivot
is the difference between "monitor everything, expensively" and "follow the
evidence."

## Data sources — real vs. synthetic

Every tool is defined by a **contract** (a function signature + a typed return
schema). Behind that contract are two interchangeable implementations, selected
per-source in `config.yaml`:

| Source | Default | Notes |
|---|---|---|
| Geocoding / mapping | **real** (OpenStreetMap Nominatim + Overpass) | No API key. Powers proximity math, the GUI map, and real hotel POIs for the planner. |
| Social media | **real** (Reddit via PRAW) | Needs `praw` installed *and* free Reddit app credentials in `.env`. Both are checked at call time; either missing falls back to synthetic. |
| News / current events | **real** (Exa, exa.ai) | Needs `EXA_API_KEY` in `.env`. Falls back to synthetic if unset. |
| Public records (court, CAD, permits) | synthetic | No public API exists for these — fake by design. |
| Exposure (brokers, breach, dark web) | synthetic | Nothing legitimate/free exists; always synthetic. |
| Protection vendors (escorts, monitoring) | synthetic | No open API for security services. Lodging is hybrid: real OSM hotel names/distances, synthetic prices/attributes. |
| VIP profile / PII | **always synthetic** | Never put a real person's PII in this repo. |

Every real API call passes a **strict per-day budget** (`config.yaml limits:`,
counters in `data/rate_limits.json`) so the analyst can test many times a day
without exhausting free tiers; exhausted budgets fall back to fixtures, and
every fixture-derived finding is tagged `synthetic_fixture` so demo data can
never masquerade as live collection. When a source configured `real` falls back,
the finding also carries a `fallback_reason` — missing key, missing client
library, spent budget, failed call — and the **Data Sources** tab prints it, so
"0 live" can never be misread as "nothing is out there".

### Two kinds of synthetic, and why the difference is labeled

The curated fixtures in `data/fixtures/` describe **one** scenario — a
conference in Berlin — and each declares the protectee and city it covers. For
any other tasking, records are **generated for that city** (`tools/synthetic.py`),
deterministically seeded on `(city, entity_id)` so a demo repeats exactly, with
coordinates from the real geocoder so the pins are somewhere that exists.

Curated always wins on a scenario match, so the graded Berlin baseline is
unchanged. Generated findings carry `generated_for_city` on top of
`synthetic_fixture`, and the Data Sources tab shows the three provenances
separately: **live**, curated fixture, generated fixture.

The alternative — serving the Berlin fixtures for whatever was asked — is not
neutral. The fixture matcher ORs any query token over two characters, so the
word "CEO" alone was enough to pull five findings about a different protectee in
a different country into a Dublin briefing, and score them as if they belonged
there. Scoping is what keeps "no curated data for this city" from silently
becoming "here is somebody else's threat picture".

Swapping a source from synthetic → real is: write one adapter, add the key to
`.env`, flip one line in `config.yaml`. The agents never know the difference —
the return type is identical either way. (This adapter pattern is the same way
production systems handle test doubles, and it is a good thing to write up.)

## Retrieval / RAG Layer

The system keeps its own indexed memory and retrieves from it — filtering the
daily ingest down to what matters, and catching threats phrased with no threat
words. The design is written up in [`docs/module3_rag_design.md`](docs/module3_rag_design.md);
in one paragraph:

- **Hybrid store.** Unstructured text goes in a vector index; structured facts
  (dates, coordinates, permits) go in **SQLite** (`store.py`) and are queried with
  a `WHERE` clause — because a similarity search answers "incidents within 500 m
  in the last 30 days" badly.
- **Pipeline:** metadata **pre-filter** (entity, date, geo, access class) →
  **hybrid search** (dense concept + lexical exact-match) → **reciprocal-rank
  fusion** → **rerank** to the top 6. Entity scoping is a hard exclusion, so a
  query about our CEO never returns threats aimed at a different same-named person.
- **Two indexes + tiering:** a *live* ingest that ages out (hot → warm → cold
  stub) and a *case* memory that never expires; analyst-confirmed items are
  promoted live → case.
- **Offline by default.** A local concept embedder and a pure-Python index (plus
  stdlib `sqlite3`) mean the RAG demo needs **no new dependencies and no keys**.
  Flip `retrieval.embeddings: voyage` / `retrieval.vector_store: chroma` (extras
  in `pyproject.toml`) to go real — Voyage AI for embeddings, since Claude has no
  embeddings endpoint.

See the contrast for yourself:

```bash
python -m threat_detector.cli rag-demo
```

It runs the same tasking with and without retrieval — directly, with no model
calls, so it costs nothing and repeats exactly. The without-retrieval answer
is fluent but sourceless; the with-retrieval answer cites dated, attributable
documents — including a 14-month-old case-memory post naming an individual with a
prior venue-surveillance pattern.

## ToT Protection Planner

Nested inside the Recommendation agent: a **beam search** (width 3) over
security expenditures for a multi-day trip — depth = trip days × requirement
types (movement / stationary / monitoring). Hard constraints (availability,
floor-cost lookahead, no adjacency to protest sites) are *filters*; soft
criteria (discretion, proximity, reviews, brand preference weighted low with a
repeat-brand predictability penalty) *rank* what survives. The search is
read-only — nothing is booked until the analyst approves — and it returns a
**slate** of complete plans, not a single answer. Design write-up:
[`docs/module4_tot_design.md`](docs/module4_tot_design.md).

## Coordination Rules

Topology: **hierarchical** — 8 agents, but the 4 collectors are parallel
siblings with zero peer edges, so the critical path stays 3–5 deep. Every loop
has a satisfaction condition AND a hard cap (`config.yaml coordination:`):
collection stops at **saturation** (a wave surfacing zero new entities, cap 3);
the assessment ↔ orchestrator **two-way edge** sends single-sourced
high-severity claims back for corroboration (cap 2 rounds), then reports them
with severity capped rather than dropping them; the ToT beam terminates by
construction. Protocols are chosen **per edge** (one-way returns, two-way
corroboration, a conditional analyst edge, brainstorm only inside the beam).
Write-up: [`docs/module5_architecture.md`](docs/module5_architecture.md).

## Guardrails & Evaluation

The structural guardrails from Modules 2–5 (a Risk agent that cannot retrieve,
the entity-ID hard filter, severity capped at what sourcing supports, read-only
search, PII isolation, bounded loops) are joined by the Module 6 layer
(`guardrails.py`, `evals.py`, `ratelimit.py`):

- **Intake check** — major gaps (who/where/when) become follow-up questions to
  the analyst *before* the run; minor gaps become stated assumptions. Type
  "CEO has a conference in Berlin" and the system asks *when*, then folds your
  answer into the tasking. The demo scenario is never silently substituted for
  what you typed.
- **Staleness + geo sanity** — months-old reposts down-weighted (case-memory
  history exempt: that's recall, not staleness); implausible coordinates
  zeroed; far "proximity" hits flagged as data errors.
- **Source vetting** — an unvetted source (absent from reliability memory)
  cannot corroborate an escalation; only the analyst writes the trust list.
- **Metrics** (GUI → Debugging tools → Evals tab): groundedness, escalation rate, corroboration
  rate, seeded retrieval recall, fallback success. Guardrails constrain;
  metrics say whether they're calibrated.

Write-up: [`docs/module6_guardrails.md`](docs/module6_guardrails.md).

## The analyst GUI

```bash
pip install -e ".[gui]"        # gradio
python app.py                  # opens a browser tab
```

Chat on the left; live teaching panels on the right, in the style of the
class's Local-Agent-Demo. **Every panel updates on every message**, not only
after an assessment — that is enforced structurally (each branch of `on_send`
returns the complete output tuple), because a panel that only some code paths
fill is a panel that spends most of the session stale.

Four **analyst** tabs are always on. Four **developer** tabs sit to their right,
hidden until you press **Debugging tools** — one button, not a deletion, so the
internals stay one click away without an analyst having to walk past a token
window readout to reach the map. Hidden panels keep rendering, so nothing is
stale when you open them.

| Analyst tab | Shows |
|---|---|
| **Source Reliability** | what survives restarts: the trust score on each source (edit it — feedback persists), the case-memory index, the active protectee (PII withheld even from the GUI). |
| **Travel Planner** | your decision controls first (plan, **Approve / Reject**, budget, trip days, **Re-plan**) — the analyst conditional edge — then the plan slate, then the beam search that produced it: expansions, lookahead prunes, diversity drops. |
| **Map** | a real GIS view (Leaflet + OpenStreetMap, no key). It geocodes and re-centres the moment you name a city — you do not wait for a run. Pins are scoped to *this* protectee and to the geo-sanity radius of *this* city, so another scenario's data cannot bleed in. |
| **Data Sources** | what each specialist actually returned this run — live, curated fixture, generated fixture, or not run and why. |

| Developer Tab | Shows |
|---|---|
| **Short-term** | the bounded window actually sent to the model this turn, and the tasking slots as they fill. Dies with the session. |
| **Trace** | the whole session in order: every routing decision, slot update, guardrail gap, specialist dispatch and model call, with elapsed time. |
| **Evals** | this run's metrics + the seeded retrieval-recall self-test. |
| **Agents** | the roster, per-edge protocols, and loop limits. |

Underspecified taskings don't run — the intake guardrail asks for what is
missing and the model phrases the question.

### Naming the protectee

Say who you are protecting in the chat: *"assess our CEO Dana Reyes, keynoting
in Dublin on 14 October"*. The profile on disk is a **default, not a lock**.

A protectee named in conversation gets an ad-hoc profile with an **empty PII
block** — telling the system a name is not the same as handing it someone's
home address, and inventing one to make Exposure look productive would be the
exact harm this system exists to prevent. Exposure then reports that it could
not run and why, rather than returning an empty result that reads like "nothing
exposed". Persist a full profile (PII included, synthetic only) with
`threat-detector profile`.

Deterministic mode by default: no API keys, reproducible, same pipeline the
tests grade.

## LLM mode — a real agent loop, on any of four backends

Set `llm.enabled: true` and the assessment runs as a genuine agent loop
(`agent_loop.py`, in the spirit of the class's Local-Agent-Demo): the model is
shown the specialist roster as tools, decides each dispatch, reacts to what
comes back, gets per-specialist persona commentary from the cheap model, and
writes the final assessment narrative — while the tools, guardrails,
corroboration, and the risk rubric stay deterministic (a model can direct
collection and interpret findings; it cannot invent them or re-score them).

Pick where tokens come from with `llm.backend` (verify any choice with
`python -m threat_detector.cli llm-check --deep`):

| backend | what it is | cost | needs |
|---|---|---|---|
| `ollama` | local open-source model (qwen3:8b, llama3.2, gemma3…) | free | `brew install ollama`, `ollama pull <model>`, `ollama serve` |
| `huggingface` | local model via transformers, no server | free | `pip install -e ".[huggingface]"` (big download) |
| `claude_cli` | your ordinary **Claude subscription** through the `claude` CLI, like ai-job-search | plan usage | Claude Code installed + signed in |
| `anthropic` | Anthropic API | pay-per-token | `ANTHROPIC_API_KEY` in `.env` (no quotes) |

The shipped default is `claude_cli` — `claude-sonnet-5` drives the loop and
`claude-haiku-4-5` voices the specialists (`config.yaml`, `llm:`). It needs no
API key: calls go through the `claude` CLI on your ordinary subscription. The
other three backends are fully supported; switch by editing `llm.backend` and
re-running `llm-check --deep`.

Config and `.env` are re-read **and re-verified** on every GUI message — switch
backend or fix a key and just send again. Verification makes one real model
call, because a shallow config check is exactly how this system once fooled
itself: a well-formed but rejected `ANTHROPIC_API_KEY` passed the check, every
real call 401'd, the errors were swallowed, and the GUI printed a Python
template that looked like a very boring LLM. Nothing degrades silently now — an
unusable backend refuses with the reason and the fix. (Note that a key exported
in your **shell** shadows `.env`; `llm-check` warns when that is the case.)

The GUI chat requires a model and will not fake a conversation: a failed call
refuses, with the reason and the fix, rather than printing a template.

### The LangChain harness (`llm.harness: langchain`)

The framework adapter the rubric asks for, and a working one:

```bash
python3 -m pip install -e ".[langchain]"
python3 -m threat_detector.cli demo --harness langchain
```

LangChain + LangGraph choose the dispatches; the specialists, the guardrails and
the scoring are the same objects the native loop uses. Three things make it a
real integration rather than a diagram:

- **One model path.** `ThreatDetectorChatModel` is a genuine `BaseChatModel`
  that delegates to `llm_backends`, so LangChain runs on `claude_cli`, `ollama`,
  `huggingface` or `anthropic` like everything else here — and every call still
  passes the `llm_calls` budget. A stock `ChatAnthropic` would have opened a
  second, metered, unrate-limited path to a different model: two systems
  claiming to be one.
- **Tool calling on a text backend.** `claude_cli` speaks text, not the
  tool-call protocol an agent needs, so `bind_tools` asks for a JSON decision
  and lifts it into real `AIMessage.tool_calls`.
- **A bounded blast radius.** LangChain replaces exactly one thing: choosing
  which specialist to dispatch next. Scoring, corroboration, staleness, the PII
  trust boundary and the approval gate stay in Python, where they are auditable.
  `exposure_scan` deliberately takes **no arguments** — the PII is closed over
  inside the tool, so no framework can route it through a prompt.

`claude_harness.py` (`llm.harness: claude`) remains as the Claude Agent SDK
adapter discussed in the module writeups.

## The chat (`conversation.py`)

The chat pane is not a form. Each message is routed by a model, in the context
of the conversation so far, into one of four intents:

| intent | what happens |
|---|---|
| `new_tasking` | fills/overwrites the tasking slots, then the intake guardrail, then the full agent loop |
| `answer_clarification` | fills the *blank* slots only, then re-checks |
| `followup` | answered from the briefing on screen, with `[source ids]`, no re-collection |
| `chitchat` | scope, method and limits — explicitly forbidden from citing anything |

The **guardrail decides what is missing**
(deterministic), the **model decides how to ask for it**. Intents are a closed
set, so the conversation can steer the system but never talk it out of its own
rules. Every grounded answer is redacted for PII at the prompt boundary, and a
failed model call raises — it is never replaced with a template.

### Tasking slots, and why intake now terminates

What the conversation establishes is stored as **slots** (`tasking_state.py`),
not as accumulated text. That distinction is load-bearing. The console used to
carry an unfinished tasking forward as a concatenated string and re-extract from
the whole blob every turn — so the moment you named a protectee, that name was
re-extracted on every later turn, the guardrail re-raised the same question
forever, and the assessment could never start. Answering re-triggered the
question; not answering left it open. There was no third move.

Slots make the loop monotonic: a filled slot is never re-asked, so more turns
can only mean fewer open questions. The model fills them from your own words
(`ConversationAgent.extract_slots`), with the deterministic extractors in
`guardrails.py` as a cross-check and a floor — a bad model turn degrades to
pattern matching rather than losing the tasking. Only an explicit re-tasking
("actually, make it Munich") may overwrite a filled slot.


## Quick start

Run these in order, in one terminal, on one interpreter. Each step is a
precondition for the ones under it.

```bash
cd ThreatDetector_SecurityAgentSystem
python3 -m pip install -e ".[gui,dev]"   # the package, the GUI, and pytest
cp .env.example .env                     # optional: EXA + Reddit keys

# 1. Prove the model backend answers, before trusting anything downstream:
python3 -m threat_detector.cli llm-check --deep

# 2. The analyst console — this is the demo:
python3 app.py

# 3. One assessment on the command line, against the fixture protectee:
python3 -m threat_detector.cli demo

# 4. Retrieval shown by contrast — same tasking, with and without RAG:
python3 -m threat_detector.cli rag-demo

# 5. Your own protectee: build the profile first, then task against it.
python3 -m threat_detector.cli profile
python3 -m threat_detector.cli assess "CEO keynoting a conference in Berlin in 3 weeks"
```


## Layout

```
ThreatDetector_SecurityAgentSystem/
├── agents/                     # one markdown spec per specialist (role + tools + protocol)
├── src/threat_detector/
│   ├── schemas.py              # typed Pydantic Findings/Chunks — the contract every tool returns
│   ├── config.py               # loads config.yaml + .env; source-selection logic
│   ├── memory.py               # short-term (this run) + long-term (source reliability, profile)
│   ├── profile.py              # VIP profile intake + persistence
│   ├── llm.py                  # model selection for both harnesses (cheap specialists, strong orchestrator)
│   ├── embeddings.py           # local concept embedder (+ Voyage AI seam)      ─┐
│   ├── chunking.py             # per-source-type document segmentation           │ Module 3
│   ├── retrieval.py            # pre-filter → hybrid → rerank; live + case index  │ RAG layer
│   ├── store.py                # SQLite structured-facts store (the pre-filter)  ─┘
│   ├── planner.py              # ToT beam-search protection planner (Module 4)
│   ├── guardrails.py           # intake/staleness/geo-sanity/source-vetting (Module 6)
│   ├── evals.py                # evaluation metrics + seeded-recall test (Module 6)
│   ├── ratelimit.py            # strict per-day budgets for every real API
│   ├── tools/                  # tools grouped by agent; each has synthetic + real impls
│   ├── llm_backends.py         # ONE chat interface: ollama / huggingface / claude_cli / anthropic
│   ├── agent_loop.py           # LLM mode: the model decides dispatches + writes the narrative
│   │                           #   begin() / finish() = the spine both harnesses share
│   ├── conversation.py         # the chat itself: routing, clarifying, grounded follow-ups
│   ├── langchain_roster.py     # LangChain/LangGraph harness (`.[langchain]` extra)
│   ├── claude_harness.py       # Claude Agent SDK adapter (course framework, legacy)
│   ├── orchestrator.py         # the Think→Act→Observe→Adapt wave loop (no-model path)
│   └── cli.py                  # command-line entry point
├── app.py                      # the analyst GUI (gradio): chat + live teaching panels
├── data/fixtures/              # internally-consistent fake data about a fictional VIP
│                               #   incl. vendors.json — the planner's synthetic catalog
├── data/corpus/                # RAG corpus (documents.json) + facts seed (facts.json)
├── docs/rag_design.md  # RAG design mapped to the Module 3 rubric
├── docs/tot_design.md  # ToT beam planner mapped to the Module 4 rubric
├── docs/architecture.md# coordination rules mapped to the Module 5 rubric
├── docs/guardrails.md  # guardrails + metrics mapped to the Module 6 rubric
└── tests/                      # pytest suite
```
