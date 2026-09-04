"""The retrieval pipeline: pre-filter -> hybrid search -> fuse -> rerank.

This is the Module 3 architecture, made runnable. One sentence per stage, the way
the checkpoint frames it:

    pre-filter   enforces *what's permissible*   (entity, date, geo, access class)
    hybrid       finds *what's plausible*        (dense meaning + lexical exact)
    rerank       picks *what's actually responsive* (joint query+chunk scoring)

Two indexes share this pipeline, differing only in retention:

* **live**  — the Open-Source/Records ingest. Ages out after ``live_ttl_days``
  unless an analyst promotes it. Tiered hot -> warm -> cold to control cost.
* **case**  — curated memory: confirmed adversary profiles, prior briefings, VIP
  appearance history. Never expires. This is where a 14-month-old post about a
  named actor still lives when that actor resurfaces near a venue.

The offline backends (local concept embedder, pure-Python cosine) are swappable
for Voyage + Chroma via config; the pipeline code doesn't change.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import embeddings
from .config import Config, load_config
from .schemas import (
    AccessClass,
    Chunk,
    Document,
    RetrievalResult,
    RetrievedChunk,
    SourceType,
    Tier,
)

_WORD = re.compile(r"[a-z0-9'@_]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# --- the pre-filter's scope ------------------------------------------------
@dataclass
class RetrievalFilter:
    """Structured scope applied *before* any similarity math (Module 3 #2).

    ``access_class`` defaults to PUBLIC-only: PII-bearing chunks stay out of the
    general assessment path unless a PII-cleared caller (the Exposure agent) asks
    for them — the trust boundary, enforced at retrieval time.
    """

    entity_id: str | None = None
    since: date | None = None
    until: date | None = None
    near: tuple[float, float, float] | None = None      # (lat, lon, radius_km)
    index_name: str | None = None                       # None = both live + case
    access: AccessClass = AccessClass.PUBLIC
    min_reliability: float = 0.0

    def admits(self, c: Chunk) -> bool:
        if self.index_name and c.index_name != self.index_name:
            return False
        # Access class: a PUBLIC-scoped query never sees PII chunks; a PII-scoped
        # query may see both (it is the more privileged caller).
        if self.access == AccessClass.PUBLIC and c.access_class == AccessClass.PII:
            return False
        if self.entity_id and self.entity_id not in c.entity_ids:
            return False
        if self.since and c.authored_at and c.authored_at < self.since:
            return False
        if self.until and c.authored_at and c.authored_at > self.until:
            return False
        if c.reliability < self.min_reliability:
            return False
        if self.near and c.lat is not None and c.lon is not None:
            lat, lon, radius = self.near
            if _haversine_km(lat, lon, c.lat, c.lon) > radius:
                return False
        return True


class RetrievalIndex:
    """In-process vector + lexical index over chunked documents."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.chunks: list[Chunk] = []
        self._df: dict[str, int] = {}       # document frequency, for lexical idf

    # --- ingest ------------------------------------------------------------
    def add_documents(self, docs: list[Document]) -> "RetrievalIndex":
        from .chunking import chunk_document

        new_chunks: list[Chunk] = []
        for doc in docs:
            new_chunks.extend(chunk_document(doc))
        vectors = embeddings.embed([c.text for c in new_chunks], self.cfg)
        for c, v in zip(new_chunks, vectors):
            c.embedding = v
        self.chunks.extend(new_chunks)
        self._reindex_lexical()
        return self

    def _reindex_lexical(self) -> None:
        self._df = {}
        for c in self.chunks:
            for tok in set(_tokens(c.text)):
                self._df[tok] = self._df.get(tok, 0) + 1

    # --- the pipeline ------------------------------------------------------
    def retrieve(self, query: str, flt: RetrievalFilter | None = None) -> RetrievalResult:
        flt = flt or RetrievalFilter()
        result = RetrievalResult(query=query)

        # (1) PRE-FILTER — hard scope. Correctness first, cost second (we run
        #     similarity over the admitted set, not the whole corpus).
        pool = [c for c in self.chunks if flt.admits(c)]
        result.notes.append(f"pre-filter: {len(pool)}/{len(self.chunks)} chunks in scope")
        if not pool:
            result.status = "no_supporting_evidence"
            result.notes.append("nothing in scope — reporting 'no supporting evidence', not 'no threat'")
            return result

        # Query expansion (#5 mitigation): a few paraphrases widen recall so an
        # obliquely worded threat still surfaces.
        queries = _expand_query(query)

        # (2) HYBRID SEARCH over the pool, unioned across expanded queries.
        dense_rank = self._dense_ranks(queries, pool)
        lexical_rank = self._lexical_ranks(queries, pool)

        # (3) FUSE with reciprocal rank fusion.
        candidates = self.cfg.candidates
        fused = _rrf(dense_rank, lexical_rank, self.cfg.rrf_k)
        top_candidates = sorted(fused.items(), key=lambda kv: -kv[1])[:candidates]
        result.candidates_considered = len(top_candidates)

        by_id = {c.chunk_id: c for c in pool}
        retrieved = [
            RetrievedChunk(
                chunk=by_id[cid],
                dense_rank=dense_rank.get(cid),
                lexical_rank=lexical_rank.get(cid),
                fused_score=score,
            )
            for cid, score in top_candidates
        ]

        # (4) RERANK — expensive joint read of query + chunk, over just the
        #     candidates, then keep top_k.
        for rc in retrieved:
            rc.rerank_score = _rerank_score(query, rc.chunk)
        retrieved.sort(key=lambda rc: -rc.rerank_score)
        kept = [rc for rc in retrieved if rc.rerank_score >= self.cfg.rerank_threshold][: self.cfg.top_k]

        if not kept:
            result.status = "no_supporting_evidence"
            result.notes.append(
                f"{len(retrieved)} candidate(s) all below rerank threshold "
                f"{self.cfg.rerank_threshold}; reporting 'no supporting evidence'"
            )
            return result

        result.results = kept
        result.notes.append(
            f"hybrid retrieved {len(retrieved)} candidate(s); reranked to {len(kept)}"
        )
        return result

    def _dense_ranks(self, queries: list[str], pool: list[Chunk]) -> dict[str, int]:
        qvecs = embeddings.embed(queries, self.cfg)
        best: dict[str, float] = {}
        for c in pool:
            sim = max(embeddings.cosine(qv, c.embedding or []) for qv in qvecs)
            best[c.chunk_id] = sim
        return _to_ranks(best)

    def _lexical_ranks(self, queries: list[str], pool: list[Chunk]) -> dict[str, int]:
        n = max(len(self.chunks), 1)
        q_tokens = set()
        for q in queries:
            q_tokens |= set(_tokens(q))
        scores: dict[str, float] = {}
        for c in pool:
            ctoks = _tokens(c.text)
            ctset = set(ctoks)
            # idf-weighted overlap: rare terms (a handle, an address) dominate,
            # which is exactly how lexical search catches what dense search blurs.
            s = 0.0
            for tok in q_tokens & ctset:
                df = self._df.get(tok, 1)
                idf = math.log((n + 1) / df)
                s += idf
            scores[c.chunk_id] = s
        return _to_ranks(scores)

    # --- tiering (Module 3 hot/warm/cold) ----------------------------------
    def age_and_tier(self, now: date, relevance: dict[str, float] | None = None,
                     floor: float = 0.2) -> None:
        """Demote live-ingest chunks by age, discarding progressively more.

        * Past TTL, above the relevance floor -> WARM: keep text, drop embedding.
        * Past TTL, below the floor           -> COLD: keep only a stub (drop
          text and embedding); ``entity_ids``/``content_hash``/``meta`` survive so
          a pattern of interest is still discoverable later.
        Case memory never expires and is left untouched.
        """
        relevance = relevance or {}
        ttl = timedelta(days=self.cfg.live_ttl_days)
        for c in self.chunks:
            if c.index_name != "live" or c.authored_at is None:
                continue
            if now - c.authored_at <= ttl:
                continue
            if relevance.get(c.chunk_id, 0.0) >= floor:
                c.tier = Tier.WARM
                c.embedding = None
            else:
                c.tier = Tier.COLD
                c.embedding = None
                c.meta = {**c.meta, "stub": True, "author": c.meta.get("author")}
                c.text = ""

    def promote_to_case(self, chunk_id: str) -> None:
        """Analyst confirms a chunk matters -> move it to non-expiring case memory.

        This is where Module 2's long-term memory and the retrieval layer become
        the *same* mechanism rather than two bolted-on features.
        """
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                c.index_name = "case"
                c.tier = Tier.HOT
                if c.embedding is None and c.text:
                    c.embedding = embeddings.embed_one(c.text, self.cfg)


# --- query expansion -------------------------------------------------------
_EXPANSIONS = {
    "surveillance": "watching waiting outside route schedule routine",
    "itinerary": "travel flight schedule leaked",
    "protest": "demonstration rally activist",
    "venue": "conference keynote appearance entrance",
}


def _expand_query(query: str) -> list[str]:
    """Generate a few paraphrased queries to widen recall (Module 3 #5).

    The Module 2 reasoning loop 'generates several paraphrased queries per
    investigation rather than one'; here we expand deterministically by adding
    concept synonyms whenever a concept's cue appears in the query.
    """
    low = query.lower()
    variants = [query]
    for cue, extra in _EXPANSIONS.items():
        if cue in low:
            variants.append(f"{query} {extra}")
    return variants


# --- reranker (cross-encoder stand-in) -------------------------------------
_FIRST_PERSON = ("i know", "i'll", "i will", "i've", "i saw", "i waited", "we know", "i'm going")


def _rerank_score(query: str, chunk: Chunk) -> float:
    """Deterministic joint scoring of query + chunk together.

    A real reranker is a cross-encoder that reads both at once, catching
    distinctions a vector average misses — e.g. a first-person statement of
    intent versus a news article *about* a threat. This stand-in scores the same
    signals transparently: concept overlap, exact query-term presence, a
    first-person-intent bonus, and a recency/dated-event bonus.
    """
    qv = embeddings.embed_one(query)
    cv = chunk.embedding if chunk.embedding is not None else embeddings.embed_one(chunk.text)
    base = max(embeddings.cosine(qv, cv), 0.0)

    text = chunk.text.lower()
    qtoks = [t for t in _tokens(query) if len(t) > 2]
    overlap = sum(1 for t in set(qtoks) if t in text) / max(len(set(qtoks)), 1)

    intent = 0.15 if any(p in text for p in _FIRST_PERSON) else 0.0
    dated = 0.05 if chunk.authored_at is not None else 0.0

    return 0.6 * base + 0.25 * overlap + intent + dated


# --- rank / fusion helpers -------------------------------------------------
def _to_ranks(scores: dict[str, float]) -> dict[str, int]:
    """Turn raw scores into 1-based ranks (rank 1 = best). Zero scores dropped."""
    ordered = sorted((cid for cid, s in scores.items() if s > 0.0),
                     key=lambda cid: -scores[cid])
    return {cid: i + 1 for i, cid in enumerate(ordered)}


def _rrf(dense: dict[str, int], lexical: dict[str, int], k: int) -> dict[str, float]:
    """Reciprocal rank fusion: score = sum over lists of 1/(k + rank).

    Standard, simple, and parameter-light — it merges two ranked lists without
    needing their scores to be on the same scale.
    """
    fused: dict[str, float] = {}
    for ranks in (dense, lexical):
        for cid, r in ranks.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r)
    return fused


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --- construction from fixtures --------------------------------------------
def load_corpus(cfg: Config | None = None) -> list[Document]:
    """Load the corpus fixture into Document objects."""
    import json

    cfg = cfg or load_config()
    raw = json.loads(cfg.retrieval_path("corpus").read_text())
    docs: list[Document] = []
    for d in raw.get("documents", []):
        docs.append(Document(
            doc_id=d["doc_id"],
            text=d["text"],
            source_type=SourceType(d["source_type"]),
            source_id=d["source_id"],
            index_name=d.get("index", "live"),
            entity_ids=d.get("entity_ids", []),
            authored_at=date.fromisoformat(d["authored_at"]) if d.get("authored_at") else None,
            lat=d.get("lat"),
            lon=d.get("lon"),
            access_class=AccessClass(d.get("access_class", "public")),
            reliability=float(d.get("reliability", 0.6)),
            title=d.get("title", ""),
            meta=d.get("meta", {}),
        ))
    return docs


def build_index(cfg: Config | None = None) -> RetrievalIndex:
    """Build and populate the retrieval index from the corpus fixture."""
    cfg = cfg or load_config()
    return RetrievalIndex(cfg).add_documents(load_corpus(cfg))


__all__ = ["RetrievalIndex", "RetrievalFilter", "build_index", "load_corpus"]
