"""Document segmentation — chunk by document type, not uniformly.

This is the Module 3 #4 design choice made concrete. A news article, a social
post, and a court filing carry meaning at different granularities, so each gets
its own rule:

* **Social post** (SOCIAL): 1 post = 1 chunk, never split. A post is already
  atomic; splitting destroys it.
* **News article** (NEWS): accumulate paragraphs up to a ~400-token target,
  snapping on paragraph boundaries (never mid-paragraph), with light (~50-token)
  overlap. News is inverted-pyramid — largely self-contained paragraphs — so
  contiguous chunking with light overlap loses little.
* **Narrative** (INCIDENT_LOG, COURT_DOCKET when free-text): meaning accumulates
  across paragraphs, so use a **sliding window** (50% overlap) — every adjacent
  pair of paragraphs is retrievable as a unit.
* **Structured record** (PROTEST_PERMIT, DATA_BROKER, ...): 1 record = 1 chunk;
  the non-text fields (dates, coordinates) belong in the SQLite facts store and
  in metadata, not in the embedded text.

Token counts are approximated as whitespace words (≈ 0.75 tokens/word); precise
tokenization isn't needed to demonstrate the segmentation policy.
"""

from __future__ import annotations

import hashlib

from .schemas import Chunk, Document, SourceType

# ~400 tokens ≈ 300 words at this rough ratio; the target we snap paragraphs to.
_TARGET_WORDS = 300
_OVERLAP_WORDS = 40

# Source types whose free text is narrative (meaning spans paragraphs).
_NARRATIVE = {SourceType.INCIDENT_LOG, SourceType.COURT_DOCKET}
# Source types stored one-record-per-chunk (structured facts live in metadata).
_ATOMIC_RECORD = {SourceType.SOCIAL, SourceType.PROTEST_PERMIT, SourceType.DATA_BROKER,
                  SourceType.BREACH_CORPUS, SourceType.DARK_WEB, SourceType.GEOSPATIAL}


def chunk_document(doc: Document) -> list[Chunk]:
    """Segment one document into retrievable chunks per its source type."""
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [doc.text.strip()]

    if doc.source_type in _ATOMIC_RECORD:
        texts = [doc.text.strip()]                         # 1 record/post = 1 chunk
    elif doc.source_type in _NARRATIVE:
        texts = _sliding_window(paragraphs)                # 50% overlap
    else:  # NEWS and anything else: paragraph-snapped with light overlap
        texts = _paragraph_pack(paragraphs)

    return [_make_chunk(doc, i, t) for i, t in enumerate(texts)]


def _paragraph_pack(paragraphs: list[str]) -> list[str]:
    """Accumulate paragraphs up to the word target, snapping on boundaries.

    A paragraph larger than the target becomes its own oversized chunk rather
    than being split mid-paragraph. Light overlap carries the tail of the
    previous chunk so a statement straddling a boundary still retrieves.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for para in paragraphs:
        w = len(para.split())
        if cur and cur_words + w > _TARGET_WORDS:
            chunks.append("\n\n".join(cur))
            tail = _tail_words(cur[-1], _OVERLAP_WORDS)
            cur = [tail, para] if tail else [para]
            cur_words = len(tail.split()) + w
        else:
            cur.append(para)
            cur_words += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _sliding_window(paragraphs: list[str]) -> list[str]:
    """50%-overlap window over paragraphs: (1+2), (2+3), (3+4), ...

    Guarantees every adjacent pair is retrievable together — the right choice for
    narrative text where a threat's setup and payoff sit in different paragraphs.
    """
    if len(paragraphs) == 1:
        return paragraphs
    return ["\n\n".join(paragraphs[i:i + 2]) for i in range(len(paragraphs) - 1)]


def _tail_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[-n:]) if len(words) > n else text


def _make_chunk(doc: Document, i: int, text: str) -> Chunk:
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return Chunk(
        chunk_id=f"{doc.doc_id}#{i}",
        doc_id=doc.doc_id,
        text=text,
        source_type=doc.source_type,
        source_id=doc.source_id,
        index_name=doc.index_name,
        entity_ids=list(doc.entity_ids),
        authored_at=doc.authored_at,
        lat=doc.lat,
        lon=doc.lon,
        access_class=doc.access_class,
        reliability=doc.reliability,
        content_hash=content_hash,
        meta=dict(doc.meta),
    )


__all__ = ["chunk_document"]
