"""LangChain expression of the agent roster (Module 2 + 5).

This is the framework adapter the course rubric asks for, and it is a *real*
one: the agent below actually runs, on the same backend the rest of the system
uses, calling the same specialist tools. It is not a diagram in code.

Three design commitments, each of which is the interesting part:

1. **The model seam, not a second model.** ``ThreatDetectorChatModel`` is a
   genuine ``BaseChatModel`` that delegates to ``llm_backends`` — so LangChain
   runs on ``claude_cli`` (your subscription, no API key), ``ollama``,
   ``huggingface`` or ``anthropic``, exactly like every other model call here,
   and every call still passes the per-day ``llm_calls`` budget. A stock
   ``ChatAnthropic`` would have quietly opened a second, metered path to a
   different model with no rate limiting — two systems claiming to be one.

2. **Tool calling on a text backend.** ``claude_cli`` speaks text, not the
   tool-call protocol LangChain agents need. ``bind_tools`` therefore asks the
   model for a JSON decision and lifts it into real ``AIMessage.tool_calls``.
   This is the same technique ``agent_loop._decide`` uses, expressed so that
   LangGraph's agent loop can consume it.

3. **The graded spine is untouched.** LangChain replaces exactly one thing —
   deciding which specialist to dispatch next. Scoring, corroboration,
   staleness, the PII trust boundary and the approval gate stay in Python
   where they can be audited. A framework that could also re-score findings
   would have dissolved the safety argument the architecture is built on.
"""

from __future__ import annotations

import json
from typing import Any

from .config import Config, load_config
from .llm_backends import LLMUnavailable, extract_json
from .profile import Profile
from .schemas import Briefing


def _require_langchain():
    try:
        from langchain.agents import create_agent  # noqa: F401
        from langchain_core.language_models.chat_models import BaseChatModel  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised by hand, not in CI
        raise RuntimeError(
            "LangChain is not installed. Run `pip install -e \".[langchain]\"`."
        ) from e


# --- the model seam --------------------------------------------------------

def build_chat_model(cfg: Config | None = None, *, model: str = "",
                     backend=None, **kw):
    """A LangChain chat model backed by this project's own backend layer.

    ``backend`` is injectable for the same reason ``LLMOrchestrator`` takes one:
    the harness has to be testable offline, or its only proof of life is a bill.
    """
    _require_langchain()
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from . import llm_backends, ratelimit

    cfg = cfg or load_config()

    class ThreatDetectorChatModel(BaseChatModel):
        """LangChain over ``llm_backends`` — one model path, one budget."""

        cfg: Any
        llm_model: str
        backend: Any = None
        bound_tools: list = []
        max_tokens: int = 900

        model_config = {"arbitrary_types_allowed": True}

        @property
        def _llm_type(self) -> str:
            return f"threat-detector-{self.cfg.llm_backend}"

        def bind_tools(self, tools, **kwargs):
            """Give a text-only backend a tool-calling interface.

            LangGraph's agent expects ``AIMessage.tool_calls``; the backend
            returns prose. The bridge is a JSON decision schema, parsed back
            into the structure LangChain expects.
            """
            from langchain_core.utils.function_calling import convert_to_openai_tool
            specs = [convert_to_openai_tool(t) for t in tools]
            return self.__class__(cfg=self.cfg, llm_model=self.llm_model,
                                  backend=self.backend, bound_tools=specs,
                                  max_tokens=self.max_tokens)

        # -- the single place a model call happens ------------------------
        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> Any:
            # The same cost guardrail the data sources live under. A framework
            # is not a reason to stop counting.
            if not ratelimit.spend(self.cfg, "llm_calls"):
                raise LLMUnavailable("daily llm_calls budget exhausted "
                                     "(config.yaml limits: llm_calls)")
            system, transcript = _split_messages(messages)
            backend = self.backend or llm_backends.get_backend(self.cfg)

            if not self.bound_tools:
                text = llm_backends.chat(backend, system, transcript,
                                         model=self.llm_model,
                                         max_tokens=self.max_tokens)
                msg = AIMessage(content=llm_backends.strip_reasoning(text))
                return ChatResult(generations=[ChatGeneration(message=msg)])

            tools_txt = "\n".join(
                f"- {t['function']['name']}: {t['function'].get('description', '')}"
                f"\n    args: {json.dumps(t['function'].get('parameters', {}).get('properties', {}))}"
                for t in self.bound_tools)
            user = (
                f"{transcript}\n\n"
                f"AVAILABLE SPECIALISTS:\n{tools_txt}\n\n"
                "Choose the single next action. Broad sweep first, then target "
                "what surfaced. Never repeat a call you already made. Answer "
                "with JSON only: {\"tool\": \"<name>|done\", \"reasoning\": "
                "\"one short sentence\", \"args\": {…}}. Use \"done\" when a "
                "further call would not change the assessment."
            )
            raw = llm_backends.chat(backend, system, user, model=self.llm_model,
                                    json_schema=_DECISION_SCHEMA,
                                    max_tokens=self.max_tokens)
            decision = extract_json(raw)
            tool = decision.get("tool", "done")
            why = str(decision.get("reasoning", ""))[:200]
            known = {t["function"]["name"] for t in self.bound_tools}

            if tool not in known:
                # 'done', or an unparseable answer. Either way the agent stops
                # here rather than dispatching something that does not exist.
                msg = AIMessage(content=why or "Collection complete.")
            else:
                msg = AIMessage(
                    content=why,
                    tool_calls=[{"name": tool,
                                 "args": decision.get("args") or {},
                                 "id": f"call_{abs(hash(raw)) % 10**8}"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return ThreatDetectorChatModel(cfg=cfg, backend=backend,
                                   llm_model=model or cfg.orchestrator_model, **kw)


_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "reasoning": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["tool"],
}


def _split_messages(messages) -> tuple[str, str]:
    """LangChain messages -> (system, flattened transcript).

    The backends take a system string and a user string; tool results arrive as
    ``ToolMessage``s and must reach the model as observations, or the agent
    re-dispatches the specialist it just ran.
    """
    system_parts, lines = [], []
    for m in messages:
        kind = m.__class__.__name__
        text = m.content if isinstance(m.content, str) else json.dumps(m.content)
        if kind == "SystemMessage":
            system_parts.append(text)
        elif kind == "ToolMessage":
            lines.append(f"OBSERVATION from {getattr(m, 'name', 'tool')}: {text}")
        elif kind == "AIMessage":
            for call in getattr(m, "tool_calls", []) or []:
                lines.append(f"YOU CALLED: {call['name']}({json.dumps(call['args'])})")
            if text:
                lines.append(f"YOUR NOTE: {text}")
        else:
            lines.append(text)
    return "\n".join(system_parts), "\n".join(lines)


# --- the roster ------------------------------------------------------------

# Each entry is (tool name, owning specialist, what it is for). The tools
# themselves are the SAME functions the native loop dispatches — the adapter
# supplies personas and a graph, never a second implementation.
ROSTER = [
    ("search_news", "Open-Source Monitoring",
     "Search news for public mentions of the protectee. args: query, city"),
    ("search_social", "Open-Source Monitoring",
     "Search social media (Reddit) for chatter about the protectee. args: query, city"),
    ("public_records_sweep", "Public Records",
     "Protest permits filed for future dates plus recent incidents near a city. args: city"),
    ("court_dockets", "Public Records",
     "Court activity naming a person. args: name"),
    ("venue_proximity", "Geospatial",
     "Real distances between the venue, threat locations and safe havens. args: city"),
    ("case_memory_search", "Case Memory / Retrieval",
     "RAG over indexed case history, scoped to this protectee. args: query"),
    ("exposure_scan", "Exposure",
     "Broker, breach and dark-web exposure of the protectee. Takes NO arguments: "
     "the PII stays inside the tool and never enters a prompt."),
]


def build_tools(orch, tasking, findings: list):
    """The specialist roster as LangChain tools.

    Every tool delegates to ``LLMOrchestrator._run_tool``, so the guardrails
    that live there — the PII trust boundary, the "no city, so not run" skip,
    entity resolution — apply identically under LangChain. Note that
    ``exposure_scan`` takes no arguments: the PII is closed over, never passed
    through the model, which is the one boundary a framework must not be able
    to widen.
    """
    _require_langchain()
    from langchain_core.tools import StructuredTool

    def make(tool_name: str, owner: str, doc: str):
        def run(query: str = "", city: str = "", name: str = "") -> str:
            args = {k: v for k, v in
                    (("query", query), ("city", city), ("name", name)) if v}
            new = orch._run_tool(tool_name, args, tasking, findings)
            if new is None:
                return f"{tool_name} is not a known specialist tool."
            new = orch._observe(new, f"langchain ({tool_name})")
            findings.extend(new)
            if not new:
                return (f"{owner}: 0 findings. That is an absence of collected "
                        f"signal, not an absence of threat.")
            return f"{owner}: {len(new)} finding(s).\n" + "\n".join(
                f"- [{f.source_id}] {f.summary[:120]}" for f in new[:8])

        return StructuredTool.from_function(run, name=tool_name, description=doc)

    return [make(n, owner, doc) for n, owner, doc in ROSTER]


def build_roster(cfg: Config | None = None) -> dict[str, str]:
    """{tool: owning specialist} — the Module 2 roster, as wired here."""
    return {name: owner for name, owner, _ in ROSTER}


# --- the harness -----------------------------------------------------------

_PERSONA = (
    "You are the Protective Intelligence Orchestrator for a VIP protection "
    "team. You reason about what you do not yet know that would most change "
    "the risk assessment, and you dispatch specialists to find it. You never "
    "collect anything yourself and you never score risk — a deterministic "
    "rubric does that after you finish. Work in waves: a broad sweep first, "
    "then target whatever the sweep surfaced."
)


def assess_with_langchain(profile: Profile, tasking: str,
                          cfg: Config | None = None, *,
                          focus: str = "all", backend=None) -> Briefing:
    """Run one assessment with LangChain driving collection.

    Collection is LangChain's; everything downstream of it is the same code
    the native loop runs, which is what keeps the two harnesses comparable.
    """
    _require_langchain()
    from langchain.agents import create_agent

    from .agent_loop import LLMOrchestrator

    cfg = cfg or load_config()
    orch = LLMOrchestrator(profile, cfg, backend=backend)
    # `begin` runs the backend check and the intake guardrail. A framework does
    # not get its own, laxer entry point.
    parsed = orch.begin(tasking, focus=focus)
    findings: list = []

    orch._log(f"[langchain] roster wired: {len(ROSTER)} specialist tools on "
              f"backend={cfg.llm_backend}")
    tools = build_tools(orch, parsed, findings)
    model = build_chat_model(cfg, backend=orch.backend)
    agent = create_agent(model, tools, system_prompt=_PERSONA)

    try:
        agent.invoke(
            {"messages": [("user", f"TASKING: {tasking}\n"
                                   f"PROTECTEE (public view only): "
                                   f"{json.dumps(profile.public_view())}")]},
            {"recursion_limit": max(4, cfg.llm_max_steps * 2)},
        )
    except LLMUnavailable:
        raise
    except Exception as e:
        # A framework failure must not read as "nothing was out there".
        orch._log(f"[langchain] agent stopped early: {e}")

    orch._log(f"[langchain] collection complete: {len(findings)} finding(s).")
    return orch.finish(parsed, tasking, findings)


__all__ = ["assess_with_langchain", "build_chat_model", "build_roster",
           "build_tools", "ROSTER"]
