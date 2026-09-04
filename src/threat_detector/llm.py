"""Model selection for the framework harnesses.

Two harnesses, two model-ID conventions:

* **LangChain** (``langchain_roster.py``) runs on this project's own backend
  layer, so it takes whatever ID ``llm.orchestrator_model`` holds — the
  ``anthropic/`` and ``ollama/`` prefixes are stripped by
  ``llm_backends.normalize_model``.
* The **Claude Agent SDK** uses bare first-party IDs (``claude-haiku-4-5``).

The design choice worth stating in the writeup is the same for both: the
**orchestrator** does the hard reasoning (planning which specialist to wake,
deciding when it has enough) and runs on the stronger, pricier model; the
**specialists** do high-volume, lower-stakes retrieval and interpretation and run
on the cheap, fast model. A tasking fans out to many specialist calls, so putting
them on the cheap model is where most of the cost is saved.
"""

from __future__ import annotations

from .config import Config, load_config


def specialist_llm(cfg: Config | None = None):
    """A LangChain chat model for specialist work (cheap/fast)."""
    cfg = cfg or load_config()
    from .langchain_roster import build_chat_model
    return build_chat_model(cfg, model=cfg.specialist_model)


def orchestrator_llm(cfg: Config | None = None):
    """A LangChain chat model for the orchestrator (stronger)."""
    cfg = cfg or load_config()
    from .langchain_roster import build_chat_model
    return build_chat_model(cfg, model=cfg.orchestrator_model)


def claude_models(cfg: Config | None = None) -> dict[str, str]:
    """Bare first-party model IDs for the Claude Agent SDK harness."""
    cfg = cfg or load_config()
    return {
        "orchestrator": cfg.claude_orchestrator_model,
        "specialist": cfg.claude_specialist_model,
    }
