"""Configuration + source selection.

Loads ``config.yaml`` and ``.env`` once, then answers the single question every
tool asks: "for my data source, am I real or synthetic right now?"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is a convenience; env vars still work without it
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


# Repo root = the directory that contains config.yaml (two levels up from here:
# src/threat_detector/config.py -> src/threat_detector -> src -> ROOT).
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict
    root: Path

    # --- source selection --------------------------------------------------
    def source_is_real(self, source: str) -> bool:
        """True if the named source should hit a real API rather than fixtures."""
        return self.raw.get("source", {}).get(source) == "real"

    # --- llm ---------------------------------------------------------------
    @property
    def llm_enabled(self) -> bool:
        return bool(self.raw.get("llm", {}).get("enabled", False))

    @property
    def llm_backend(self) -> str:
        """Where LLM-mode tokens come from: ollama | huggingface | claude_cli
        | anthropic (see llm_backends.py)."""
        return self.raw.get("llm", {}).get("backend", "anthropic")

    @property
    def llm_max_steps(self) -> int:
        """Hard cap on agent-loop tool steps (the loop's safety net)."""
        return int(self.raw.get("llm", {}).get("max_steps", 8))

    @property
    def llm_specialist_commentary(self) -> bool:
        """Whether specialists add persona commentary (extra cheap-model calls)."""
        return bool(self.raw.get("llm", {}).get("specialist_commentary", True))

    @property
    def harness(self) -> str:
        """Which framework harness drives collection.

        ``native``    -> agent_loop.py, the backend-based loop (default).
        ``langchain`` -> langchain_roster.py, the same tools and the same
                         deterministic spine, with LangChain choosing dispatches.
        ``claude``    -> the Claude Agent SDK adapter (claude_harness.py).
        """
        return self.raw.get("llm", {}).get("harness", "native")

    @property
    def orchestrator_model(self) -> str:
        return self.raw.get("llm", {}).get("orchestrator_model", "anthropic/claude-sonnet-5")

    @property
    def specialist_model(self) -> str:
        return self.raw.get("llm", {}).get("specialist_model", "anthropic/claude-haiku-4-5")

    @property
    def claude_orchestrator_model(self) -> str:
        return self.raw.get("llm", {}).get("claude_orchestrator_model", "claude-sonnet-5")

    @property
    def claude_specialist_model(self) -> str:
        return self.raw.get("llm", {}).get("claude_specialist_model", "claude-haiku-4-5")

    # --- retrieval (RAG) ---------------------------------------------------
    @property
    def retrieval_enabled(self) -> bool:
        return bool(self.raw.get("retrieval", {}).get("enabled", True))

    @property
    def embeddings_backend(self) -> str:
        return self.raw.get("retrieval", {}).get("embeddings", "local")

    @property
    def vector_store_backend(self) -> str:
        return self.raw.get("retrieval", {}).get("vector_store", "local")

    @property
    def candidates(self) -> int:
        return int(self.raw.get("retrieval", {}).get("candidates", 20))

    @property
    def top_k(self) -> int:
        return int(self.raw.get("retrieval", {}).get("top_k", 6))

    @property
    def rrf_k(self) -> int:
        return int(self.raw.get("retrieval", {}).get("rrf_k", 60))

    @property
    def rerank_threshold(self) -> float:
        return float(self.raw.get("retrieval", {}).get("rerank_threshold", 0.12))

    @property
    def live_ttl_days(self) -> int:
        return int(self.raw.get("retrieval", {}).get("live_ttl_days", 90))

    def retrieval_path(self, key: str) -> Path:
        rel = self.raw.get("retrieval", {}).get("paths", {}).get(key)
        if rel is None:
            raise KeyError(f"No retrieval path configured for '{key}'")
        return self.root / rel

    # --- orchestrator knobs ------------------------------------------------
    @property
    def max_waves(self) -> int:
        return int(self.raw.get("orchestrator", {}).get("max_waves", 4))

    @property
    def min_reliability(self) -> float:
        return float(self.raw.get("orchestrator", {}).get("min_reliability", 0.35))

    # --- ToT planner (Module 4) --------------------------------------------
    @property
    def beam_width(self) -> int:
        return int(self.raw.get("planner", {}).get("beam_width", 3))

    @property
    def max_candidates(self) -> int:
        return int(self.raw.get("planner", {}).get("max_candidates", 3))

    @property
    def trip_days(self) -> int:
        return int(self.raw.get("planner", {}).get("trip_days", 2))

    @property
    def plan_budget(self) -> float:
        return float(self.raw.get("planner", {}).get("budget", 12000))

    @property
    def planner_trigger(self) -> float:
        """Overall risk score at/above which the ToT planner runs."""
        return float(self.raw.get("planner", {}).get("trigger_score", 0.45))

    # --- coordination limits (Module 5) ------------------------------------
    @property
    def max_collection_waves(self) -> int:
        return int(self.raw.get("coordination", {}).get("max_collection_waves", 3))

    @property
    def corroboration_rounds(self) -> int:
        return int(self.raw.get("coordination", {}).get("corroboration_rounds", 2))

    @property
    def corroboration_severity(self) -> float:
        """A finding at/above this severity must not rest on a single source."""
        return float(self.raw.get("coordination", {}).get("corroboration_severity", 0.7))

    @property
    def uncorroborated_cap(self) -> float:
        """Severity ceiling for claims that stay single-sourced after the loop."""
        return float(self.raw.get("coordination", {}).get("uncorroborated_cap", 0.55))

    # --- paths -------------------------------------------------------------
    def path(self, key: str) -> Path:
        rel = self.raw.get("paths", {}).get(key)
        if rel is None:
            raise KeyError(f"No path configured for '{key}'")
        return self.root / rel

    @property
    def fixtures_dir(self) -> Path:
        return self.path("fixtures")

    def fixture(self, name: str) -> Path:
        return self.fixtures_dir / name

    # --- env passthrough ---------------------------------------------------
    @staticmethod
    def env(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> Config:
    """Load config once and cache it. Pass an explicit path in tests.

    ``override=True`` so an edited .env value (a fixed API key, say) actually
    replaces the stale one after ``reload_config()`` — without it, dotenv
    refuses to touch env vars that are already set in the process.
    """
    load_dotenv(ROOT / ".env", override=True)
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    return Config(raw=raw, root=ROOT)


def reload_config() -> Config:
    """Drop the cache and re-read config.yaml + .env. The GUI calls this on
    every message, so config edits apply WITHOUT restarting the app."""
    load_config.cache_clear()
    return load_config()
