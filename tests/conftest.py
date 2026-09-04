"""Shared test fixtures.

Every test runs against an all-synthetic config so the suite is hermetic — no
network, no API keys, fully reproducible. This mirrors what a grader gets when
they clone the repo and run `pytest`.
"""

from pathlib import Path

import pytest

from threat_detector import ratelimit
from threat_detector.config import Config
from threat_detector.profile import Profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _sandboxed_budgets(tmp_path, monkeypatch):
    """Point the per-day budget counters at a throwaway file.

    `ratelimit._path` resolves to `<cfg.root>/data/rate_limits.json`, and the
    test config's root is the real repo — so every run of the suite was
    spending the operator's real daily quota. That is a hermeticity bug on its
    own, but it also fails the suite for an unrelated reason: once `llm_calls`
    hit its cap, twenty tests that have nothing to do with budgets started
    failing with empty findings and missing trace lines.

    Autouse, because a test that forgets this opt-in is exactly the test that
    silently drains the budget again.
    """
    monkeypatch.setattr(ratelimit, "_path",
                        lambda cfg: tmp_path / "rate_limits.json")


@pytest.fixture
def synthetic_cfg() -> Config:
    raw = {
        "source": {
            "news": "synthetic",
            "social": "synthetic",
            "public_records": "synthetic",
            "exposure": "synthetic",
            "geospatial": "synthetic",
        },
        "llm": {"enabled": False},
        "orchestrator": {"max_waves": 4, "min_reliability": 0.35},
        "paths": {
            "fixtures": "data/fixtures",
            "reliability": "data/fixtures/source_reliability.json",
        },
        # RAG layer: local backends only, so the suite stays offline/hermetic.
        "retrieval": {
            "enabled": True,
            "embeddings": "local",
            "vector_store": "local",
            "candidates": 20,
            "top_k": 6,
            "rrf_k": 60,
            "rerank_threshold": 0.12,
            "live_ttl_days": 90,
            "paths": {
                "corpus": "data/corpus/documents.json",
                "facts": "data/corpus/facts.json",
                # In-memory in tests; on-disk path unused because tests pass in_memory=True.
                "facts_db": "data/rag/facts.db",
            },
        },
    }
    return Config(raw=raw, root=ROOT)


@pytest.fixture
def example_profile(synthetic_cfg) -> Profile:
    return Profile.load(synthetic_cfg.fixture("vip_profile.example.json"))


@pytest.fixture
def example_briefing(synthetic_cfg, example_profile):
    """A completed briefing, produced by the real deterministic pipeline.

    Built rather than hand-written so the conversational layer is always tested
    against the same shape the orchestrator actually emits.
    """
    from threat_detector.orchestrator import Orchestrator

    return Orchestrator(example_profile, synthetic_cfg).assess(
        "CEO keynoting a conference in Berlin in 3 weeks")
