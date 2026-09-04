# Module 3 — RAG & Retrieval Design Integration

How the retrieval layer in this repo answers the five checkpoint questions, with
pointers to the code that implements each. The design mirrors the written
checkpoint; this document ties each decision to a runnable file so a grader can
see the architecture actually execute (`python -m threat_detector.cli rag-demo`).

Everything runs **offline by default**: a local concept embedder stands in for a
hosted embedding model and a pure-Python index stands in for a vector database,
so the pipeline is reproducible with no keys. Each offline backend swaps for a
real one via a single `config.yaml` flip (`retrieval.embeddings: voyage`,
`retrieval.vector_store: chroma`) without touching pipeline code — the same
real/synthetic seam the collection tools already use.

---

## #1 — Is retrieval required? Yes.

Three arguments, in strength order (`agents/retrieval.md`, `src/threat_detector/retrieval.py`):

1. **Relevance filtering is the product.** The corpus is the daily ingest —
   thousands of articles, posts, dockets per day, of which a handful matter. That
   volume cannot sit in a context window at any price.
2. **Semantic matching does real work.** Threat language usually contains no
   threat words ("I know which garage he parks in"). Embedding-based retrieval
   surfaces it because it is semantically near *surveillance of a routine*; a
   keyword search for "threat" never would.
3. **Temporal recall.** A 14-month-old post becomes relevant the moment its author
   resurfaces near a venue. No context window spans that; you need your own index.

**Judgment call:** not everything is embedded. Structured facts — coordinates,
dates, permit filings — live in a **SQLite** store (`store.py`) and are queried
with a `WHERE` clause, because asking a similarity search "how many incidents
within 500 m in the last 30 days" answers badly. Hybrid design: vector index for
unstructured text, structured store for facts.

## #2 — The retrieval mechanism

Pipeline stages (`retrieval.py`), one job each:

| stage | job | code |
|---|---|---|
| **pre-filter** | enforce *what's permissible* — `entity_id`, date, geo, access class | `RetrievalFilter.admits` |
| **hybrid** | find *what's plausible* — dense concept + lexical exact-match | `_dense_ranks`, `_lexical_ranks` |
| **fuse** | merge the two ranked lists | `_rrf` (reciprocal rank fusion) |
| **rerank** | pick *what's actually responsive* — joint query+chunk scoring | `_rerank_score` |

- **Pre-filter runs first** for correctness (scoping is a rule, not a ranking
  preference) and cost (similarity math over the admitted set, not the corpus).
  `entity_id` is load-bearing: it hard-excludes threats aimed at a different
  person of the same name — proven by `test_entity_prefilter_excludes_same_named_person`.
- **Two indexes**, differing only in retention: **live** ingest (ages out after
  `live_ttl_days`) and **case** memory (curated, never expires). Analyst-confirmed
  material is *promoted* live → case (`promote_to_case`), which is how Module 2's
  long-term memory and this retrieval layer become one mechanism.
- **Access class** keeps PII chunks off the public retrieval path — the Module 2
  trust boundary, now enforced at retrieval time (`test_access_class_isolates_pii`).

Real backends: `embeddings.py` has a Voyage AI adapter (Anthropic's embedding
partner; Claude itself has no embeddings endpoint), gated on `VOYAGE_API_KEY` with
graceful fallback, exactly like the Reddit adapter.

## #3 — Retrieval meaningfully changes the output

`cli rag-demo` runs the same tasking twice. **Without retrieval**, the baseline is
fluent, sourceless safety boilerplate. **With retrieval**, the pre-filter scopes
to the protectee and the hybrid search returns (a) the structured permit fact
400 m from the venue and (b) a 14-month-old **case-memory** post by `@drk_wolf`
describing loitering outside a *previous* venue, plus a current post showing the
same handle now probing the Berlin venue.

The assessment gains a **named individual** and a **documented prior pattern of
venue surveillance** — every retrieved item carries a source and a date the
analyst can verify. That is grounding: the output cites documents, not opinion
(`test_retrieval_injects_case_memory_pattern`).

## #4 — Key retrieval design choices

**Chunking by document type** (`chunking.py`), not uniformly:

- **Social post** → 1 post = 1 chunk (already atomic; splitting destroys it).
- **News** → accumulate paragraphs to a ~400-token target, snapping on paragraph
  boundaries, light (~50-token) overlap; inverted-pyramid prose loses little.
- **Narrative** (incident reports, court filings) → sliding window, 50% overlap,
  so a threat whose setup and payoff span paragraphs stays retrievable together.
- **CAD / docket** → 1 record = 1 chunk; structured fields go to metadata + SQL,
  not the embedded text. **VIP history** → 1 event = 1 chunk.

**Tiering** (`age_and_tier`): hot (full text + embeddings) → warm (keep text, drop
embeddings — cheap to re-embed) → cold (stub only: source, timestamp, author,
entity IDs, hash — ~200 bytes). The stub preserves *that content existed and who
was in it* so a pattern of interest is discoverable later without storing the body
(`test_tiering_demotes_and_stubs_old_live_chunks`).

**k:** retrieve `candidates` = 20, rerank to `top_k` = 6. Recall-first — a missed
indicator costs more than a wasted rerank pass, and reranking is cheaper than
enlarging the context.

**Source reliability** rides in metadata and doubles as a retrieval-time weight —
the same long-term reliability memory from Module 2.

## #5 — Retrieval failure mode: false negatives

A relevant document that exists in the index but the search misses is a security
failure; a false positive is only wasted analyst time. Three mitigations, each
already in the design:

- **Hybrid dense + lexical** covers both failure shapes — dense catches oblique
  threats (`test_dense_search_finds_oblique_threat`); lexical catches exact
  strings a vector blurs, like the handle `@drk_wolf`
  (`test_lexical_search_catches_exact_handle`).
- **Query expansion** — the reasoning loop issues several paraphrased queries per
  investigation (`_expand_query`), widening recall.
- **Honest reporting** — when nothing passes, the system returns
  `status = "no_supporting_evidence"`, never "no threat identified". Absence of
  evidence is not evidence of absence
  (`test_empty_scope_reports_no_supporting_evidence`).

Note the distinction kept clean from the checkpoint: *retrieval* false positives
(irrelevant chunks reaching the agent) are handled by the pre-filter and reranker;
*assessment* over-scoring is a separate, Risk-agent concern.

---

## What changed in the codebase for Module 3

New modules: `embeddings.py`, `chunking.py`, `retrieval.py`, `store.py`,
`claude_harness.py`. New fixtures: `data/corpus/documents.json`,
`data/corpus/facts.json`. New agent spec: `agents/retrieval.md`. The orchestrator
gained a retrieval-consult wave (`_wave_retrieval`) that is deduped against the
live waves so retrieval contributes genuinely new (historical / oblique) signal.
Nothing pre-existing was removed — the RAG layer is additive and behind
`retrieval.enabled`.
