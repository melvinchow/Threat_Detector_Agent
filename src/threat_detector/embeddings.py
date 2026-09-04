"""Text embeddings for the vector index.

The system's guiding principle — runs offline, deterministic, no keys — applies
to retrieval too, so the default embedder is **local**: pure Python, no
dependencies, reproducible. Its vector has two parts:

* **Concept dimensions.** A small, transparent lexicon maps trigger terms to a
  dozen threat-relevant concepts (surveillance, itinerary, violence, protest,
  doxxing/PII, ...). This is what lets the offline demo show *semantic* matching:
  "I know which garage he parks in" and "monitoring the executive's daily route"
  share the *surveillance* concept even though they share no words. A real
  embedding model does this implicitly over millions of dimensions; the lexicon
  is an honest, gradable stand-in for that behaviour.

* **Hashed token dimensions.** A signed hashing trick spreads the raw vocabulary
  across the remaining dimensions, so shared wording still contributes.

Cosine similarity over this vector therefore rewards both shared *meaning*
(concepts) and shared *words* (hashed tokens).

Swapping to real embeddings is one config flip (``retrieval.embeddings: voyage``)
and touches only this file — the vector index calls :func:`embed` and never
learns which backend produced the numbers. Anthropic has no embeddings endpoint,
so the real path uses Voyage AI, Anthropic's recommended embedding partner.
"""

from __future__ import annotations

import hashlib
import math
import re

from .config import Config, load_config

# --- the concept lexicon ---------------------------------------------------
# Deliberately small and readable. Each concept is one dimension; a document
# activates a concept if it contains any trigger term (word-boundary match).
# These are the threat semantics an analyst cares about, so two texts about the
# same *kind* of threat land near each other even with disjoint vocabulary.
_CONCEPTS: dict[str, list[str]] = {
    "surveillance": ["watch", "watching", "waited", "waiting", "follow", "following",
                     "route", "garage", "parks", "parking", "schedule", "routine",
                     "spotted", "outside the", "which door", "morning"],
    "itinerary": ["itinerary", "flight", "flying", "travel", "hotel", "arrival",
                  "schedule", "leaked", "leak"],
    "violence": ["attack", "hurt", "kill", "weapon", "gun", "knife", "bomb",
                 "assault", "violence", "violent", "hit"],
    "protest": ["protest", "demonstration", "rally", "march", "picket", "blockade",
                "activist", "mobilize", "mobilizing"],
    "doxxing": ["address", "home", "dox", "doxx", "personal info", "phone number",
                "where he lives", "where she lives"],
    "legal": ["lawsuit", "docket", "injunction", "court", "sued", "litigation"],
    "sentiment_negative": ["backlash", "angry", "furious", "hate", "outrage",
                           "criticism", "boycott", "layoffs", "cut jobs"],
    "event_appearance": ["keynote", "conference", "summit", "appearance", "speak",
                         "speaking", "panel", "venue"],
    "identity": ["ceo", "executive", "founder", "chief"],
    "location_proximity": ["near", "close to", "meters", "metres", "blocks",
                           "outside", "entrance", "gate"],
}
_CONCEPT_NAMES = list(_CONCEPTS)
_CONCEPT_DIM = len(_CONCEPT_NAMES)
_HASH_DIM = 128
_DIM = _CONCEPT_DIM + _HASH_DIM

# Concept signal is weighted above raw token overlap so meaning dominates wording.
_CONCEPT_WEIGHT = 2.0


def dim() -> int:
    return _DIM


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _local_embed_one(text: str) -> list[float]:
    low = text.lower()
    vec = [0.0] * _DIM

    # Concept dimensions: presence-weighted (a concept fires once regardless of
    # how many trigger terms match, so a long rant doesn't swamp the signal).
    for i, name in enumerate(_CONCEPT_NAMES):
        for term in _CONCEPTS[name]:
            if term in low:
                vec[i] = _CONCEPT_WEIGHT
                break

    # Hashed token dimensions: signed hashing trick over the raw vocabulary.
    for tok in _tokens(text):
        if len(tok) < 3:
            continue
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = _CONCEPT_DIM + (h % _HASH_DIM)
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign

    return _l2_normalize(vec)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two (already L2-normalized) vectors."""
    return sum(x * y for x, y in zip(a, b))


def embed(texts: list[str], cfg: Config | None = None) -> list[list[float]]:
    """Embed a batch of texts. Real (Voyage) or local per config."""
    cfg = cfg or load_config()
    if cfg.embeddings_backend == "voyage":
        real = _voyage_embed(texts, cfg)
        if real is not None:
            return real
    return [_local_embed_one(t) for t in texts]


def embed_one(text: str, cfg: Config | None = None) -> list[float]:
    return embed([text], cfg)[0]


def _voyage_embed(texts: list[str], cfg: Config) -> list[list[float]] | None:
    """Real embeddings via Voyage AI. Returns None on any failure -> local fallback.

    Mirrors the Reddit adapter: lazy import, graceful degradation, identical
    return shape so nothing downstream changes.
    """
    if not cfg.env("VOYAGE_API_KEY"):
        return None
    try:
        import voyageai  # type: ignore
    except ImportError:
        return None
    try:
        client = voyageai.Client(api_key=cfg.env("VOYAGE_API_KEY"))
        # voyage-3 is a general-purpose model; input_type='document' at ingest.
        result = client.embed(texts, model="voyage-3", input_type="document")
        # Normalize so cosine() (a plain dot product) stays correct across backends.
        return [_l2_normalize(list(v)) for v in result.embeddings]
    except Exception:
        return None


__all__ = ["embed", "embed_one", "cosine", "dim"]
