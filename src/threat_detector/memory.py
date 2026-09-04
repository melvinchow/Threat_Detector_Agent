"""Short-term and long-term memory.

The Module 2/3 checkpoints called for two distinct memories:

* **Short-term** — what matters *right now* for this one tasking: the analyst's
  immediate focus, and a scratchpad the orchestrator uses to avoid re-dispatching
  a specialist it already has an answer from. Lives only for the run.

* **Long-term** — persists across sessions. Two things live here:
    1. **Source reliability** — which sources the analyst has learned to trust or
       distrust. This is the strongest argument for long-term memory: a source
       the analyst previously dismissed gets down-weighted on every future run
       rather than resurfacing and causing alert fatigue.
    2. The **VIP profile** (handled in profile.py, which uses this store).

For the capstone, long-term memory is a JSON file. In a production build this is
exactly the seam where a vector database / RAG would slot in — when "reliability"
grows from a lookup table into "retrieve the analyst's past notes about this
actor," you swap the JSON reads for embedding search behind the same methods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ShortTermMemory:
    """Per-run scratchpad. Reset for every tasking."""

    tasking: str = ""
    # The analyst's stated immediate concern narrows which specialists run.
    # e.g. "internet sentiment" -> skip the geospatial sweep entirely.
    focus: str = "all"
    # Specialists already answered this run, so the orchestrator doesn't repeat them.
    answered: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def remember(self, note: str) -> None:
        self.notes.append(note)


class ReliabilityMemory:
    """Long-term source-reliability store, backed by JSON.

    A reliability of 1.0 = fully trusted; lower values down-weight a source's
    findings during the orchestrator's Observe step. Unknown sources default to
    a neutral-but-cautious value so a brand-new source neither dominates nor is
    ignored.
    """

    DEFAULT = 0.6

    def __init__(self, path: Path):
        self.path = path
        self._scores: dict[str, float] = {}
        if path.exists():
            self._scores = json.loads(path.read_text())

    def score(self, source_id: str) -> float:
        return self._scores.get(source_id, self.DEFAULT)

    def set_score(self, source_id: str, value: float) -> None:
        """Record analyst feedback about a source, and persist it.

        This is the 'feedback guides behavior across runs' requirement: the
        analyst marks a source unreliable once, and it stays down-weighted.
        """
        self._scores[source_id] = max(0.0, min(1.0, value))
        self.path.write_text(json.dumps(self._scores, indent=2, sort_keys=True))
