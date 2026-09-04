"""Claude-native harness (Claude Agent SDK).

This is the alternative to ``langchain_roster.py``. Because the whole system is
committed to
Claude models, driving it with Anthropic's own agentic harness keeps the stack
consistent: native tool use and prompt caching instead of LangChain's
translation layer, and **subagents** that map one-to-one onto the specialist
roster (a lead agent that delegates to six focused subagents).

The split is identical to the LangChain adapter, and deliberately so — it is the
point of making the harness pluggable: the reasoning **algorithm** (the wave
loop and the step-4 pivot) is deterministic Python in ``orchestrator.py``; the
harness only supplies the **personas** and the model runtime. Swapping LangChain for
the Claude Agent SDK therefore touches this file and ``llm.py``, nothing in the
core.

Everything imports the SDK lazily and is only reached when
``config.yaml -> llm.enabled: true`` and ``llm.harness: claude``. The
deterministic orchestrator stays the reference implementation, so the project
runs and grades with no SDK installed and no API key.
"""

from __future__ import annotations

from .config import Config, load_config
from .llm import claude_models
from .profile import Profile
from .schemas import Briefing

# Subagent persona definitions — the same roster as langchain_roster.py, as the
# system prompts the Claude Agent SDK gives each subagent. Kept as data so they
# read like the submitted architecture and are easy to diff against agents/*.md.
_ROSTER = {
    "orchestrator": (
        "Protective Intelligence Orchestrator. Reason about what is not yet known "
        "that would most change the protectee's risk, and delegate one focused "
        "sub-investigation at a time. Never fetch data directly; never set the "
        "final threat level."
    ),
    "open_source": (
        "Open-Source Monitoring Specialist. Find and de-duplicate public mentions "
        "(news, social) and never conflate the protectee with a same-named person."
    ),
    "public_records": (
        "Public Records Specialist. Surface forward-looking signal: filed permits, "
        "recent incidents, court activity near the protectee's planned locations."
    ),
    "exposure": (
        "Exposure Specialist and sole holder of the protectee's PII. Report whether "
        "personal data or itinerary is discoverable; never echo the raw PII."
    ),
    "geospatial": (
        "Geospatial Specialist. Resolve places to coordinates and compute real "
        "distances — arithmetic the model must not do by hand — and catch "
        "same-named-city confusion by checking the resolved country."
    ),
    "retrieval": (
        "Case Memory / Retrieval Specialist. Query the indexed live and case "
        "memory, scoped to this protectee, for historical or obliquely-worded "
        "signal. Report 'no supporting evidence retrieved', never 'no threat'."
    ),
    "risk": (
        "Risk Assessment Specialist with NO retrieval tools. Score only the "
        "findings actually collected; you cannot fetch, so you cannot invent."
    ),
    "recommendation": (
        "Recommendation Specialist. Propose next actions behind an approval gate; "
        "propose only, never execute."
    ),
}


def build_subagents(cfg: Config | None = None) -> dict:
    """Construct the Claude Agent SDK subagent definitions. Returns {name: def}.

    Raises a clear, actionable error if the SDK is not installed, mirroring
    ``langchain_roster.build_tools`` so a misconfiguration fails informatively than
    with an obscure ImportError mid-run.
    """
    cfg = cfg or load_config()
    try:
        from claude_agent_sdk import AgentDefinition  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised only in LLM mode
        raise RuntimeError(
            "The Claude Agent SDK is not installed. Run `pip install claude-agent-sdk`, "
            "set llm.harness: langchain to use the LangChain harness instead, or "
            "set llm.enabled: false for deterministic mode."
        ) from e

    models = claude_models(cfg)
    subagents = {}
    for name, prompt in _ROSTER.items():
        model = models["orchestrator"] if name == "orchestrator" else models["specialist"]
        subagents[name] = AgentDefinition(
            description=prompt.split(".")[0],
            prompt=prompt,
            model=model,
        )
    return subagents


def assess_with_claude(profile: Profile, tasking: str, cfg: Config | None = None) -> Briefing:
    """LLM-driven assessment on the Claude-native harness.

    As with the LangChain adapter, this first build constructs the subagent roster
    (proving the SDK is present and the personas/models resolve) but still drives
    the graded wave-based control flow through the deterministic orchestrator. The
    natural next iteration turns each orchestrator ``_act`` dispatch into a real
    subagent turn; the seam is intentionally explicit here.
    """
    cfg = cfg or load_config()
    build_subagents(cfg)  # validates the SDK + models are available
    from .orchestrator import Orchestrator

    return Orchestrator(profile, cfg).assess(tasking)


__all__ = ["build_subagents", "assess_with_claude"]
