"""The LLM-driven mode: a REAL agent loop, in the style of the class demos.

An agent is a while-loop that (1) shows a model the tools, (2) asks "what
next?", (3) runs the chosen tool and feeds the result back, (4) repeats until
the model says it has enough, then (5) has the model write the assessment.
Here that loop is spelled out — no framework magic — and it runs on ANY
backend from `llm_backends.py` (Ollama, HuggingFace, claude CLI, Anthropic API).

What the model genuinely controls:
  * which specialist to dispatch next, with what query, and when to stop —
    the collection strategy is the model's, not a script's;
  * interpretation: each specialist's findings get a short commentary in that
    specialist's persona (the cheap `specialist_model`);
  * the final written threat assessment (`Briefing.narrative`).

What stays deterministic ON PURPOSE (these are the Module 6 guardrails, and
they hold no matter which model runs):
  * tools return typed Findings — the model reads them, it can't invent them;
  * reliability stamping, staleness, geo-sanity, corroboration, and the risk
    RUBRIC — severity math is not a vibe;
  * the ToT beam planner (search is control flow; a model narrating DFS loses
    the tree) and the analyst approval gate.

That split is exactly the TA demo's lesson ("the model returns words, tools
return artifacts") applied to protective intelligence.

``LLMOrchestrator`` subclasses the deterministic ``Orchestrator`` so Observe,
corroboration, planning, and tracing are shared code — the only thing replaced
is who decides the next move.
"""

from __future__ import annotations

import json

from .llm_backends import (LLMUnavailable, extract_json, get_backend,
                           normalize_model)
from .orchestrator import Orchestrator, parse_tasking
from .schemas import Briefing, Finding
from .tools import exposure, geospatial, open_source, public_records, recommendation, risk

# ---------------------------------------------------------------------------
# The tool surface shown to the model. Names map 1:1 onto the specialist
# roster; every callable is read-only and returns typed Findings.
# ---------------------------------------------------------------------------
_TOOL_DOCS = {
    "search_social": "Open-Source Monitoring: search social media (Reddit or fixtures) for a query. args: {query}",
    "search_news": "Open-Source Monitoring: search news/current events for a query. args: {query}",
    "public_records_sweep": "Public Records: protest permits + recent incidents for a city. args: {city}",
    "court_dockets": "Public Records: court dockets naming a person. args: {name}",
    "venue_proximity": "Geospatial: geocode the venue/city and measure distance to every protest site found so far. args: {city}",
    "case_memory_search": "Case Memory/Retrieval: RAG query over the indexed archive, scoped to the protectee. args: {query}",
    "exposure_scan": "Exposure: broker/breach/dark-web scan. Takes no args — the PII stays inside the tool; you never see it.",
    "done": "Stop collecting. Use when another call would not change the assessment.",
}

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "tool": {"type": "string", "enum": list(_TOOL_DOCS)},
        "args": {"type": "object",
                 "properties": {"query": {"type": "string"},
                                "city": {"type": "string"},
                                "name": {"type": "string"}},
                 "additionalProperties": False},
    },
    "required": ["reasoning", "tool", "args"],
}

_PERSONA = """You are the orchestrator of a multi-agent VIP threat-assessment
system for a corporate security department. You never fetch anything yourself:
you dispatch specialist agents (the tools) and reason over the typed findings
they return. Collection strategy: broad sweep first, then TARGETED follow-ups
on whatever surfaced (named groups, handles, venues) — follow the evidence.
Never re-issue a call that already ran. Findings carry source ids, dates, and
severities; you cannot invent, alter, or re-score them. When new calls would
no longer change the assessment, choose "done"."""


class LLMOrchestrator(Orchestrator):
    """The deterministic Orchestrator with its brain swapped for a model."""

    def __init__(self, profile, cfg=None, on_event=None, backend=None):
        super().__init__(profile, cfg, on_event=on_event)
        self.backend = backend or get_backend(self.cfg)

    def _chat(self, system: str, user: str, **kw) -> str:
        """Every model call passes the per-day LLM budget first (limits:
        llm_calls) — the same cost guardrail the data sources live under."""
        from . import ratelimit
        if not ratelimit.spend(self.cfg, "llm_calls"):
            raise RuntimeError("daily llm_calls budget exhausted "
                               "(config.yaml limits: llm_calls)")
        return self.backend.chat(system, user, **kw)

    # -- public entry point --------------------------------------------------
    def begin(self, raw_tasking: str, focus: str = "all"):
        """Set up a run and return the parsed tasking.

        Split out of ``assess`` so an alternative harness (``langchain_roster``)
        enters the run through exactly this code — the backend check, the intake
        guardrail and the short-term memory are not things a framework gets to
        skip.
        """
        from .memory import ShortTermMemory
        from . import guardrails

        stm = ShortTermMemory(tasking=raw_tasking, focus=focus)
        self.stm = stm
        tasking = parse_tasking(raw_tasking)

        ok, msg = self.backend.check()
        if not ok:
            raise RuntimeError(f"LLM backend '{self.backend.name}' unavailable: {msg}")
        omodel = normalize_model(self.cfg.orchestrator_model)
        self._log(f"[llm] agent loop on backend={self.backend.name} "
                  f"orchestrator={omodel} "
                  f"specialists={normalize_model(self.cfg.specialist_model)}")

        self.intake = guardrails.intake_check(raw_tasking, self.profile, self.cfg)
        for a in self.intake.assumptions:
            self._log(f"[guardrail] assumption: {a}")
        for q in self.intake.questions:
            self._log(f"[guardrail] OPEN QUESTION (running anyway): {q}")
        return tasking

    def assess(self, raw_tasking: str, focus: str = "all",
               use_retrieval: bool | None = None) -> Briefing:
        tasking = self.begin(raw_tasking, focus=focus)
        findings: list[Finding] = []
        called: set[str] = set()

        # ==== THE LOOP =====================================================
        for step in range(1, self.cfg.llm_max_steps + 1):
            decision = self._decide(tasking, findings, called, step)
            tool = decision.get("tool", "done")
            reasoning = str(decision.get("reasoning", ""))[:200]
            self._log(f"[llm-think] step {step}: {reasoning}")

            if tool == "done":
                self._log(f"[llm-think] step {step}: collection complete "
                          f"({len(findings)} findings).")
                break

            args = decision.get("args") or {}
            signature = tool + json.dumps(args, sort_keys=True)
            if signature in called:      # enforced in code, not in the prompt
                self._log(f"[guardrail] '{tool}' already ran with these args — "
                          f"treating as done.")
                break
            called.add(signature)

            new = self._run_tool(tool, args, tasking, findings)
            if new is None:              # unknown tool name survived parsing
                self._log(f"[guardrail] model chose unknown tool {tool!r}; ignored.")
                continue
            new = self._observe(new, f"step {step} ({tool})")
            findings += new
            self._specialist_commentary(tool, new)
        else:
            self._log(f"[adapt] hard cap: {self.cfg.llm_max_steps} steps reached.")

        return self.finish(tasking, raw_tasking, findings)

    def finish(self, tasking, raw_tasking: str,
               findings: list[Finding]) -> Briefing:
        """Score, corroborate, recommend, plan, narrate — the graded spine.

        Whoever collected the findings, this is what turns them into a briefing.
        Keeping it in one method is the reason the LangChain harness cannot
        drift into scoring its own evidence.
        """
        # ==== SHARED DETERMINISTIC SPINE (guardrails hold in LLM mode) =====
        self._think("Hand findings to Risk Assessment (rubric is deterministic — "
                    "severity math is not a vibe).")
        score = risk.score_findings(findings)
        self._log(f"[score] overall={score.overall:.3f} ({risk.band(score.overall)}) "
                  f"by_channel={score.by_channel}")

        findings, rescored = self._corroboration_loop(findings, tasking)
        if rescored:
            score = risk.score_findings(findings)
            self._log(f"[score] re-scored after corroboration: "
                      f"overall={score.overall:.3f} ({risk.band(score.overall)})")

        recs = recommendation.recommend(score, findings)
        plans = self._dispatch_planner(tasking, score, recs)

        narrative = self._write_narrative(tasking, findings, score)
        self._log("[llm] narrative written by the orchestrator model "
                  f"({len(narrative)} chars).")

        return Briefing(
            tasking=raw_tasking, protectee=self.profile.name, score=score,
            findings=findings, recommendations=recs, plans=plans,
            trace=list(self.trace), narrative=narrative,
        )

    # -- DECIDE --------------------------------------------------------------
    def _decide(self, tasking, findings, called, step) -> dict:
        tools_txt = "\n".join(f"- {k}: {v}" for k, v in _TOOL_DOCS.items())
        state = self._findings_digest(findings)
        already = "\n".join(sorted(called)) or "(none yet)"
        user = (
            f"TASKING: {tasking.raw}\n"
            f"PROTECTEE (public view only): {json.dumps(self.profile.public_view())}\n\n"
            f"TOOLS:\n{tools_txt}\n\n"
            f"FINDINGS SO FAR ({len(findings)}):\n{state}\n\n"
            f"ALREADY CALLED (never repeat):\n{already}\n\n"
            f"Step {step} of {self.cfg.llm_max_steps}. Decide the single next "
            f"action. Broad sweep first; then target what surfaced; 'done' when "
            f"further calls won't change the assessment. Keep `reasoning` to ONE "
            f"short sentence."
        )
        try:
            raw = self._chat(_PERSONA, user,
                                    model=self.cfg.orchestrator_model,
                                    json_schema=_DECISION_SCHEMA, max_tokens=900)
        except LLMUnavailable:
            # The model is DOWN. Do not quietly return "done": that produced a
            # zero-finding briefing rendered from a Python template, which is
            # indistinguishable from a hardcoded app. Fail where it happened.
            raise
        except Exception as e:
            self._log(f"[llm] backend error during decide: {e} — stopping collection.")
            return {"tool": "done", "reasoning": f"backend error: {e}", "args": {}}
        decision = extract_json(raw)
        if decision.get("tool") not in _TOOL_DOCS:
            return {"tool": "done",
                    "reasoning": f"unparseable decision: {raw[:80]!r}", "args": {}}
        return decision

    # -- ACT: dispatch the chosen specialist ----------------------------------
    def _run_tool(self, tool: str, args: dict, tasking,
                  findings: list[Finding]) -> list[Finding] | None:
        q = str(args.get("query") or args.get("name") or tasking.raw)
        city = str(args.get("city") or tasking.city or "")

        scope = {"entity_id": self.profile.entity_id, "city": city}
        if tool == "search_social":
            self._act("Open-Source Monitoring", f"social search: '{q}'")
            hits = open_source.search_social(q, self.cfg, **scope)
            return open_source.resolve_entity(hits, self.profile.name,
                                              _role_terms(self.profile))
        if tool == "search_news":
            self._act("Open-Source Monitoring", f"news search: '{q}'")
            return open_source.search_news(q, self.cfg, **scope)
        if tool == "public_records_sweep":
            if not city:
                self._log("[guardrail] public_records_sweep needs a city and the "
                          "tasking has none — not run. That is a gap in "
                          "collection, not an absence of records.")
                return []
            self._act("Public Records", f"permits + incidents near {city}")
            eid = self.profile.entity_id
            when = self.intake.event_date if self.intake else None
            return (public_records.protest_permits_near(city, self.cfg, when, eid)
                    + public_records.recent_incidents(city, self.cfg, eid))
        if tool == "court_dockets":
            self._act("Public Records", f"dockets naming {q}")
            return public_records.court_dockets(q, self.cfg, city)
        if tool == "venue_proximity":
            if not city:
                return []
            self._act("Geospatial", f"venue proximity for {city}")
            protest_locs = [f.detail.get("location") for f in findings
                            if f.source_type.value == "protest_permit"
                            and f.detail.get("location")]
            out = geospatial.assess_venue_proximity(city, protest_locs, self.cfg) \
                if protest_locs else []
            return out + geospatial.nearest_safe_havens(city, self.cfg)
        if tool == "case_memory_search":
            self._act("Case Memory / Retrieval",
                      f"RAG query scoped to '{self.profile.entity_id}': '{q}'")
            return self._retrieval_query(q, findings)
        if tool == "exposure_scan":
            # Trust boundary: the PII block goes into the tool, never the prompt.
            pii = self.profile.pii
            if not (pii.home_addresses or pii.phone_numbers or pii.emails
                    or pii.known_usernames):
                # An ad-hoc protectee named in chat has no PII on file. Returning
                # a silent [] here would read downstream as "scanned, nothing
                # exposed" — the exact false negative this system exists to
                # avoid. Say which one it is.
                self._log(f"[guardrail] Exposure NOT run: no PII on file for "
                          f"{self.profile.name}. Absence of a scan is not "
                          f"absence of exposure — add identifiers with "
                          f"`threat-detector profile` to cover this channel.")
                return []
            self._act("Exposure", "broker + breach + dark-web scan (PII stays in-tool)")
            return (exposure.scan_data_brokers(pii, self.cfg)
                    + exposure.breach_lookup(pii, self.cfg)
                    + exposure.darkweb_itinerary_scan(tasking.event_terms, self.cfg))
        return None

    def _retrieval_query(self, query: str, findings: list[Finding]) -> list[Finding]:
        from .orchestrator import _chunk_to_finding
        from .retrieval import RetrievalFilter, build_index

        if self._index is None:
            self._index = build_index(self.cfg)
        result = self._index.retrieve(query, RetrievalFilter(entity_id=self.profile.entity_id))
        if result.status == "no_supporting_evidence":
            self._log(f"[observe] retrieval: no supporting evidence retrieved for "
                      f"entity '{self.profile.entity_id}' — this archive holds no "
                      f"case history for this protectee. NOT 'no threat'.")
            return []
        seen = {f.source_id for f in findings}
        return [_chunk_to_finding(rc) for rc in result.results
                if rc.chunk.source_id not in seen]

    # -- specialist personas --------------------------------------------------
    _SPECIALIST_NAMES = {
        "search_social": "Open-Source Monitoring", "search_news": "Open-Source Monitoring",
        "public_records_sweep": "Public Records", "court_dockets": "Public Records",
        "venue_proximity": "Geospatial", "case_memory_search": "Case Memory/Retrieval",
        "exposure_scan": "Exposure",
    }

    def _specialist_commentary(self, tool: str, new: list[Finding]) -> None:
        """Each wave of findings gets 1–2 sentences in the owning specialist's
        voice (the cheap model) — the specialists are LLM personas too."""
        if not new or not self.cfg.llm_specialist_commentary:
            return
        name = self._SPECIALIST_NAMES.get(tool, tool)
        digest = self._findings_digest(new)
        try:
            comment = self._chat(
                f"You are the {name} specialist in a VIP protection team. In 1-2 "
                f"sentences of plain prose (no markdown, no headers), state what "
                f"your findings mean for the protectee's risk. Only reference "
                f"the findings given; never invent sources.",
                digest, model=self.cfg.specialist_model, max_tokens=150)
            self._log(f"[specialist:{name}] {comment.strip()[:300]}")
        except Exception as e:
            self._log(f"[llm] specialist commentary skipped ({e}).")

    # -- ANSWER ---------------------------------------------------------------
    def _write_narrative(self, tasking, findings: list[Finding], score) -> str:
        digest = self._findings_digest(findings, limit=25)
        user = (
            f"TASKING: {tasking.raw}\n"
            f"DETERMINISTIC RISK SCORE (rubric; do not alter): "
            f"{score.overall:.2f} [{risk.band(score.overall)}], "
            f"by channel {score.by_channel}\n\n"
            f"FINDINGS:\n{digest}\n\n"
            "Write the analyst-facing threat assessment (150-250 words): the "
            "picture, the named actors/patterns, what drives the score, and "
            "what remains uncertain. CITE source ids in [brackets] for every "
            "claim. If a claim is flagged uncorroborated or stale, say so. "
            "State facts only from the findings above — no invention."
        )
        last = ""
        for attempt in (1, 2):        # a transient empty reply gets one retry
            try:
                out = self._chat(_PERSONA, user,
                                        model=self.cfg.orchestrator_model,
                                        max_tokens=1200).strip()
            except LLMUnavailable as e:
                last = str(e)
                self._log(f"[llm] narrative attempt {attempt} failed: {e}")
                continue
            except Exception as e:
                self._log(f"[llm] narrative failed ({e}); briefing ships without it.")
                return ""
            if out:
                return out
            self._log(f"[llm] narrative attempt {attempt} came back empty"
                      + ("; retrying." if attempt == 1 else "; giving up."))
        # Never return "" here: the caller would render a template and the
        # analyst would have no way to tell the model never spoke.
        raise LLMUnavailable(
            f"The orchestrator model could not write the assessment. {last}")

    # -- helpers --------------------------------------------------------------
    def _redact(self, text: str) -> str:
        """Strip the protectee's PII values out of anything shown to a model.

        The PII block never enters a prompt directly, but Exposure findings can
        ECHO it ("<email> appears in breach X"). The trust boundary is 'no
        model sees PII', not 'no model sees the pii field' — so redact at the
        prompt boundary. The analyst-facing briefing keeps the real values.
        """
        pii = self.profile.pii
        for kind, values in (("address", pii.home_addresses),
                             ("phone", pii.phone_numbers),
                             ("email", pii.emails),
                             ("username", pii.known_usernames)):
            for v in values:
                if v:
                    text = text.replace(v, f"‹PII:{kind}›")
        return text

    def _findings_digest(self, findings: list[Finding], limit: int = 40) -> str:
        if not findings:
            return "(none)"
        rows = []
        for f in findings[:limit]:
            flags = "".join(
                f" [{k}]" for k in ("uncorroborated", "stale", "synthetic_fixture")
                if f.detail.get(k))
            rows.append(f"- [{f.source_id}] ({f.channel.value}, sev {f.severity:.2f}, "
                        f"rel {f.reliability:.2f}, {f.event_date or 'undated'}) "
                        f"{self._redact(f.summary[:140])}{flags}")
        return "\n".join(rows)


def _role_terms(profile) -> list[str]:
    import re
    return [t for t in re.split(r"[,\s]+", profile.role) if len(t) > 2]


__all__ = ["LLMOrchestrator"]
