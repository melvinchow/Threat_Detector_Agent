"""The orchestrator: reasons about questions, dispatches specialists in waves.

This is the outer loop from the Module 2 checkpoint, written as deterministic
Python so the whole pipeline runs with no LLM and no cost. (The LangChain
harness in langchain_roster.py drives these same tools; the loop logic is
identical either way — a framework gives you the personas, not the algorithm.)

The loop:

    Think    -> which risk classes matter for this tasking? what don't I know?
    Act      -> dispatch a WAVE of collection specialists (not one big fan-out)
    Observe  -> stamp each finding with source reliability; check if it answers
                the question that motivated the dispatch
    Adapt    -> re-plan the next wave based on what came back — including
                escalating to specialists that weren't in the original plan

Waves matter: a broad sweep first, then a targeted second wave triggered by what
the first surfaced. That is the step-4 pivot from the worked trace — a named group
appearing in public records narrows the open-source query from a broad sentiment
sweep to that group's channels specifically.

The orchestrator itself calls **no external tool**. Its only moves are delegating
to specialists and reading/writing memory. It never decides a threat level (that
is the Risk agent) and never proposes an action (that is the Recommendation
agent).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from datetime import date

from typing import Callable

from . import guardrails
from .config import Config, load_config
from .memory import ReliabilityMemory, ShortTermMemory
from .profile import Profile
from .schemas import Briefing, Finding, Recommendation, SourceType, ThreatChannel
from .tools import exposure, geospatial, open_source, public_records, recommendation, risk


@dataclass
class Tasking:
    """Structured view of a free-text tasking like
    'CEO keynoting a conference in Berlin in 3 weeks'."""

    raw: str
    city: str | None
    event_terms: list[str]


# A dependency-free tasking parser. City extraction is generalized (Module 6):
# any city the analyst types is honored — never silently replaced by the demo
# scenario's. In LLM mode the orchestrator agent does this instead.
def parse_tasking(raw: str) -> Tasking:
    city = guardrails.extract_city(raw)
    # Event terms: notable nouns for the dark-web/itinerary scan.
    terms = [w for w in ("conference", "keynote", "summit", "hearing", "earnings", "rally")
             if w in raw.lower()]
    return Tasking(raw=raw, city=city, event_terms=terms + ([city] if city else []))


class Orchestrator:
    def __init__(self, profile: Profile, cfg: Config | None = None,
                 on_event: Callable[[str], None] | None = None):
        self.profile = profile
        self.cfg = cfg or load_config()
        self.reliability = ReliabilityMemory(self.cfg.path("reliability"))
        self.trace: list[str] = []
        self._index = None            # RetrievalIndex, built lazily on first use
        # Live observers (the GUI's Trace tab). Every trace line is also pushed
        # here as it happens, so a caller can watch the loop run instead of only
        # reading the log afterwards.
        self.on_event = on_event
        # Exposed after a run for inspection (the GUI's memory/planner tabs).
        self.stm: ShortTermMemory | None = None
        self.plan_trace: list[dict] = []
        self.intake: guardrails.IntakeReport | None = None

    # --- public entry point ------------------------------------------------
    def assess(self, raw_tasking: str, focus: str = "all",
               use_retrieval: bool | None = None) -> Briefing:
        """Run one assessment.

        ``use_retrieval`` overrides config, so the paired with/without-retrieval
        demonstration (Module 3 #3) can force each mode from the same tasking.
        """
        if use_retrieval is None:
            use_retrieval = self.cfg.retrieval_enabled
        stm = ShortTermMemory(tasking=raw_tasking, focus=focus)
        self.stm = stm                      # exposed for the GUI's context tab
        tasking = parse_tasking(raw_tasking)
        findings: list[Finding] = []

        # ---- INTAKE GUARDRAIL (Module 6, pre-generation) -----------------
        # The GUI/CLI normally ask the analyst BEFORE calling assess; if we're
        # running anyway (tests, --assume), every gap is stated in the trace,
        # never silently papered over.
        self.intake = guardrails.intake_check(raw_tasking, self.profile, self.cfg)
        for a in self.intake.assumptions:
            self._log(f"[guardrail] assumption: {a}")
            stm.remember(f"assumption: {a}")
        for q in self.intake.questions:
            self._log(f"[guardrail] OPEN QUESTION (running anyway): {q}")

        self._think(
            f"Tasking: {raw_tasking!r}. Two risk classes to consider: "
            f"location-specific and actor-specific. Focus = {focus}. "
            f"retrieval={'on' if use_retrieval else 'off'}."
        )
        # Module 5: name the topology and the per-edge protocols up front so the
        # trace shows coordination is designed, not emergent.
        self._think(
            "Topology: hierarchical — orchestrator over a parallel collection tier "
            "(4 siblings, zero peer edges), then Assessment -> Recommendation -> "
            "Reporting sequential. Protocols per edge: collection->orchestrator "
            "one-way; assessment<->orchestrator two-way (corroboration); "
            "recommendation<->analyst conditional (approve/revise/reject); "
            "brainstorm allowed only inside the ToT beam."
        )

        # ---- COLLECTION LOOP (Module 5): saturation condition + hard cap --
        # Wave 1 is the broad parallel fan-out; each further wave is targeted at
        # entities the previous wave surfaced. The loop's satisfaction condition
        # is SATURATION — the latest wave surfaced zero new entities — with a
        # hard cap as the safety net for when the condition never fires.
        wave1 = self._wave_one(tasking, stm)
        findings += self._observe(wave1, "wave 1 (broad sweep)")
        stm.remember(f"wave 1: {len(wave1)} finding(s)")

        targeted_done: set[str] = set()
        wave_num = 1
        while True:
            new_entities = [g for g in self._named_groups(findings) if g not in targeted_done]
            if not new_entities:
                self._log(f"[adapt] saturation: wave {wave_num} surfaced 0 new entities "
                          f"— collection loop closed (condition met, cap unused).")
                break
            if wave_num >= self.cfg.max_collection_waves:
                self._log(f"[adapt] hard cap: {self.cfg.max_collection_waves} collection "
                          f"waves reached with entities still unexplored: {new_entities}. "
                          f"Reported as unresolved, not silently dropped.")
                stm.remember(f"UNRESOLVED at cap: {new_entities}")
                break
            wave_num += 1
            self._think(
                f"Adapt: named group(s) {new_entities} surfaced. Narrow the "
                f"open-source query to those channels instead of a broad sweep."
            )
            wave_n = self._wave_two_targeted(new_entities, stm)
            findings += self._observe(wave_n, f"wave {wave_num} (targeted on named group)")
            stm.remember(f"wave {wave_num}: targeted {new_entities}, {len(wave_n)} finding(s)")
            targeted_done.update(new_entities)
        named_groups = self._named_groups(findings)

        # ---- RETRIEVE: consult indexed memory (RAG) ----------------------
        # Historical/oblique signal a live scrape can't produce: a prior pattern
        # of venue surveillance by the named actor, sitting in case memory.
        # Deduped against what the live waves already collected, so retrieval
        # contributes the genuinely new (historical/obliquely-worded) material.
        if use_retrieval:
            seen = {f.source_id for f in findings}
            retr = self._wave_retrieval(tasking, named_groups, seen)
            findings += self._observe(retr, "retrieval (case + live memory)")

        # ---- ESCALATE: named actor + specific event -> check exposure ----
        if named_groups and tasking.event_terms:
            self._think(
                "Escalate: a named actor plus a specific event is the trigger to "
                "check whether the protectee's itinerary/PII is discoverable."
            )
            wave3 = self._wave_three_exposure(tasking, stm)
            findings += self._observe(wave3, "exposure escalation")

        # ---- SCORE (Risk agent — no retrieval) ---------------------------
        self._think("Hand findings to Risk Assessment agent (no retrieval; scores only what was collected).")
        score = risk.score_findings(findings)
        self._log(f"[score] overall={score.overall:.3f} ({risk.band(score.overall)}) by_channel={score.by_channel}")

        # ---- CORROBORATION LOOP (Module 5): assessment <-> orchestrator --
        # The two-way edge: assessment sends single-sourced high-severity claims
        # back for corroboration (up to corroboration_rounds), then anything
        # still single-sourced is kept but confidence-capped — never dropped.
        findings, rescored = self._corroboration_loop(findings, tasking)
        if rescored:
            score = risk.score_findings(findings)
            self._log(f"[score] re-scored after corroboration loop: "
                      f"overall={score.overall:.3f} ({risk.band(score.overall)})")

        # ---- RECOMMEND (propose-only, gated) -----------------------------
        recs = recommendation.recommend(score, findings)
        self._log(f"[recommend] {len(recs)} action(s) proposed; "
                  f"{sum(r.requires_approval for r in recs)} require analyst approval.")

        # ---- PLAN (Module 4): ToT beam nested inside Recommendation ------
        plans = self._dispatch_planner(tasking, score, recs)

        return Briefing(
            tasking=raw_tasking,
            protectee=self.profile.name,
            score=score,
            findings=findings,
            recommendations=recs,
            plans=plans,
            trace=list(self.trace),
        )

    # --- waves -------------------------------------------------------------
    def _wave_one(self, tasking: Tasking, stm: ShortTermMemory) -> list[Finding]:
        """Broad first sweep, dispatched in parallel across specialists.

        Short-term focus prunes irrelevant specialists: if the analyst is focused
        on internet sentiment, we don't touch geospatial or public records.
        """
        out: list[Finding] = []
        pub = self.profile.public_view()
        name = self.profile.name
        role_terms = _role_terms(self.profile)

        if stm.focus in ("all", "reputation", "sentiment"):
            self._act("Open-Source Monitoring", "news + social sweep")
            osint = open_source.search_news(name, self.cfg) + open_source.search_social(name, self.cfg)
            osint = open_source.resolve_entity(osint, name, role_terms)  # entity resolution step
            out += osint

        if stm.focus in ("all", "location") and tasking.city:
            self._act("Public Records", f"permits + incidents near {tasking.city}")
            out += public_records.protest_permits_near(tasking.city, self.cfg)
            out += public_records.recent_incidents(tasking.city, self.cfg)

            self._act("Geospatial", f"venue proximity for {tasking.city}")
            protest_locs = [f.detail.get("location") for f in out
                            if f.source_type.value == "protest_permit" and f.detail.get("location")]
            if protest_locs:
                out += geospatial.assess_venue_proximity(tasking.city, protest_locs, self.cfg)
            out += geospatial.nearest_safe_havens(tasking.city, self.cfg)

        if stm.focus in ("all", "reputation"):
            self._act("Public Records", f"court dockets naming {name}")
            out += public_records.court_dockets(name, self.cfg)

        return out

    def _wave_two_targeted(self, groups: list[str], stm: ShortTermMemory) -> list[Finding]:
        """Narrowed open-source query on the specific group(s) that surfaced."""
        out: list[Finding] = []
        for g in groups:
            self._act("Open-Source Monitoring", f"targeted query on '{g}'")
            hits = open_source.search_social(g, self.cfg) + open_source.search_news(g, self.cfg)
            out += hits
        return out

    def _wave_three_exposure(self, tasking: Tasking, stm: ShortTermMemory) -> list[Finding]:
        """Only the Exposure agent touches PII. Handed the real PII block here."""
        self._act("Exposure", "broker + breach + dark-web itinerary scan (PII-holding)")
        pii = self.profile.pii
        out: list[Finding] = []
        out += exposure.scan_data_brokers(pii, self.cfg)
        out += exposure.breach_lookup(pii, self.cfg)
        out += exposure.darkweb_itinerary_scan(tasking.event_terms, self.cfg)
        return out

    def _wave_retrieval(self, tasking: Tasking, groups: list[str],
                        seen_source_ids: set[str] | None = None) -> list[Finding]:
        """Consult the RAG layer for indexed historical / obliquely-worded signal.

        Scoped hard to THIS protectee via ``entity_id`` (the pre-filter), then a
        semantic query for surveillance-of-routine language plus the named
        actors. This is where the 14-month-old case-memory post about a prior
        venue-surveillance pattern re-enters the assessment — retrieval changing
        the output, not a fresh scrape.
        """
        from .retrieval import RetrievalFilter, build_index

        if self._index is None:
            self._index = build_index(self.cfg)
        self._act("Case Memory / Retrieval", f"RAG query scoped to entity '{self.profile.entity_id}'")

        query = (
            f"surveillance of the executive's route and schedule at the {tasking.city or 'event'} "
            f"venue {' '.join(groups)}"
        )
        # PUBLIC-only scope (default): PII-class chunks stay on the Exposure path.
        flt = RetrievalFilter(entity_id=self.profile.entity_id)
        result = self._index.retrieve(query, flt)

        if result.status == "no_supporting_evidence":
            # Module 3 #5: say exactly this, never "no threat identified".
            self._log("[observe] retrieval: no supporting evidence retrieved (NOT 'no threat').")
            return []

        self._log(
            f"[observe] retrieval: {' | '.join(result.notes)}"
        )
        # Dedupe: skip what the live waves already surfaced, and keep only the
        # highest-reranked chunk per source, so retrieval adds NEW signal.
        seen = set(seen_source_ids or set())
        out: list[Finding] = []
        for rc in result.results:
            sid = rc.chunk.source_id
            if sid in seen:
                continue
            seen.add(sid)
            out.append(_chunk_to_finding(rc))
        if not out:
            self._log("[observe] retrieval: all hits already collected by live waves; no new signal.")
        return out

    # --- Observe -----------------------------------------------------------
    def _observe(self, findings: list[Finding], label: str) -> list[Finding]:
        """Stamp reliability from long-term memory; down-weight/keep as needed."""
        kept: list[Finding] = []
        for f in findings:
            f.reliability = self.reliability.score(f.source_id)
            if f.reliability < self.cfg.min_reliability:
                f.detail["downweighted"] = f"reliability {f.reliability:.2f} < {self.cfg.min_reliability}"
                self._log(
                    f"[observe] {label}: down-weighting {f.source_id} "
                    f"(reliability {f.reliability:.2f})."
                )
            kept.append(f)
        # Module 6 post-collection guardrails: stale re-posts must not read as
        # new developments; implausible coordinates must not read as proximity.
        for note in guardrails.staleness_check(kept, self.cfg):
            self._log(f"[guardrail] {note}")
        for note in guardrails.geo_sanity_check(kept, self.cfg):
            self._log(f"[guardrail] {note}")
        self._log(f"[observe] {label}: {len(kept)} finding(s) validated against reliability memory.")
        return kept

    # --- Adapt helpers -----------------------------------------------------
    def _named_groups(self, findings: list[Finding]) -> list[str]:
        """Extract named adversary groups mentioned across findings.

        Draws on the profile's known adversaries plus any group named in a permit
        or incident. This is what turns the broad sweep into a targeted wave.
        """
        groups: set[str] = set()
        for f in findings:
            g = f.detail.get("group")
            if g:
                groups.add(g)
        for adv in self.profile.known_adversaries:
            for f in findings:
                if adv.lower() in (f.summary + str(f.detail)).lower():
                    groups.add(adv)
        return sorted(groups)

    # --- corroboration (Module 5) -------------------------------------------
    def _actor_of(self, f: Finding) -> str | None:
        """The named actor a finding is about, if any — the unit of corroboration."""
        actor = f.detail.get("group") or f.detail.get("author")
        if actor:
            return str(actor)
        blob = (f.summary + str(f.detail)).lower()
        for adv in self.profile.known_adversaries:
            if adv.lower() in blob:
                return adv
        return None

    def _support_count(self, f: Finding, findings: list[Finding]) -> int:
        """How many INDEPENDENT sources back this claim (distinct source_ids).

        Independence is by source, not by document — forty reposts of one wire
        story would still be one source_id here. Corroboration = same named
        actor from a different source; failing an actor, same channel from a
        different source.
        """
        actor = self._actor_of(f)
        sources = {f.source_id}
        for other in findings:
            if other.source_id == f.source_id:
                continue
            # Module 6 source vetting: a source the reliability memory has
            # never seen cannot corroborate an escalation — new sources need an
            # analyst promotion first. (Vetting a new source by "checking
            # online sentiment" would be circular.)
            if not guardrails.is_vetted(other.source_id, self.reliability):
                continue
            if actor is not None:
                other_actor = self._actor_of(other)
                if other_actor and other_actor.lower() == actor.lower():
                    sources.add(other.source_id)
            elif other.channel == f.channel and other.source_type == f.source_type:
                # No named actor to match on: only a different source of the
                # SAME kind counts. A broker listing does not corroborate
                # dark-web chatter just because both are 'exposure'.
                sources.add(other.source_id)
        return len(sources)

    def _corroboration_loop(self, findings: list[Finding],
                            tasking: Tasking) -> tuple[list[Finding], bool]:
        """Assessment <-> orchestrator two-way edge.

        Condition: no claim serious enough to matter (severity >= threshold)
        remains single-sourced. Cap: ``corroboration_rounds`` requests back to
        collection. If corroboration never turns up, the claim is REPORTED with
        its confidence marked down (severity capped, flagged uncorroborated) —
        never dropped: failure to confirm is normal, not exceptional.
        """
        threshold = self.cfg.corroboration_severity
        changed = False

        def _single_sourced() -> list[Finding]:
            # Corroboration applies to CLAIMS (human-sourced assertions).
            # Geospatial findings are computed geometry over already-validated
            # inputs — arithmetic doesn't need a second source.
            return [f for f in findings
                    if f.severity >= threshold and not f.detail.get("uncorroborated")
                    and f.source_type != SourceType.GEOSPATIAL
                    and self._support_count(f, findings) < 2]

        # Module 6 tool-access limit: a query that came back empty once is dead —
        # re-issuing it verbatim burns budget for zero information. Kill it.
        dead_queries: set[str] = set()

        for rnd in range(1, self.cfg.corroboration_rounds + 1):
            weak = _single_sourced()
            if not weak:
                self._log("[adapt] assessment<->orchestrator (two-way edge): every "
                          "high-severity claim has >1 independent source — "
                          "corroboration condition met.")
                return findings, changed
            for f in weak[:2]:
                actor = self._actor_of(f) or f.summary[:40]
                if str(actor) in dead_queries:
                    self._log(f"[guardrail] tool-access limit: corroboration query "
                              f"'{actor}' already returned nothing new — not "
                              f"re-issuing it.")
                    continue
                self._log(f"[adapt] assessment<->orchestrator (two-way edge, round "
                          f"{rnd}/{self.cfg.corroboration_rounds}): claim "
                          f"'{f.summary[:60]}' is single-sourced — requesting "
                          f"targeted corroboration on '{actor}'.")
                self._act("Open-Source Monitoring", f"corroboration query on '{actor}'")
                new = open_source.search_social(str(actor), self.cfg) \
                    + open_source.search_news(str(actor), self.cfg)
                seen = {x.source_id for x in findings}
                new = [n for n in new if n.source_id not in seen]
                if new:
                    findings += self._observe(new, f"corroboration round {rnd}")
                    changed = True
                else:
                    dead_queries.add(str(actor))

        for f in _single_sourced():
            self._log(f"[observe] '{f.summary[:60]}' stayed single-sourced after "
                      f"{self.cfg.corroboration_rounds} round(s): severity capped "
                      f"{f.severity:.2f} -> {self.cfg.uncorroborated_cap:.2f} and "
                      f"flagged uncorroborated — reported, not dropped.")
            f.severity = min(f.severity, self.cfg.uncorroborated_cap)
            f.detail["uncorroborated"] = True
            changed = True
        return findings, changed

    # --- ToT planner dispatch (Module 4) ------------------------------------
    def _dispatch_planner(self, tasking: Tasking, score, recs: list[Recommendation]):
        """Run the nested beam-search planner when travel is involved and risk
        is elevated. The search is read-only; the winning slate is attached as a
        propose-only recommendation behind the analyst approval gate."""
        if not tasking.city or score.overall < self.cfg.planner_trigger:
            return []
        from .planner import REQUIREMENTS, plan_protection

        self._think(
            f"Risk {score.overall:.2f} >= {self.cfg.planner_trigger} and travel "
            f"involved -> Recommendation agent opens its nested ToT beam "
            f"(brainstorming is allowed HERE and nowhere else — planning is "
            f"exploratory; evidence collection stays conservative)."
        )
        self._act("Recommendation / ToT planner",
                  f"beam width {self.cfg.beam_width}, depth "
                  f"{self.cfg.trip_days}x{len(REQUIREMENTS)}, budget "
                  f"{self.cfg.plan_budget:.0f} (search is read-only; nothing is booked)")
        plans, beam_trace = plan_protection(
            self.cfg,
            preferred_brands=list(self.profile.preferences.get("hotel_brands", [])),
            city=tasking.city,
        )
        self.plan_trace = beam_trace
        levels = [t for t in beam_trace if t.get("event") == "level"]
        pruned = sum(t["pruned_by_lookahead"] for t in levels)
        self._log(f"[observe] ToT beam: {len(levels)} level(s), "
                  f"{sum(t['expanded'] for t in levels)} expansion(s), "
                  f"{pruned} pruned by floor-cost lookahead, "
                  f"{len(plans)} plan(s) in the final slate.")
        if plans:
            best = plans[0]
            status = "complete" if best.complete else "INCOMPLETE (labeled)"
            recs.append(Recommendation(
                action=f"Adopt protection plan '{best.plan_id}' "
                       f"({best.total_cost:.0f} of {best.budget:.0f} budget, {status}); "
                       f"{len(plans)} option(s) on the slate for comparison.",
                channel=ThreatChannel.PHYSICAL_LOCATION,
                rationale="ToT beam search over vendors: hard security filters, "
                          "floor-cost lookahead, diversity-enforced slate.",
                requires_approval=True,
                estimated_cost=f"{best.total_cost:.0f}",
            ))
        return plans

    # --- trace helpers -----------------------------------------------------
    def _log(self, msg: str) -> None:
        """Append to the audit trace AND push to any live observer (the GUI)."""
        self.trace.append(msg)
        if self.on_event is not None:
            self.on_event(msg)

    def _think(self, msg: str) -> None:
        self._log(f"[think] {msg}")

    def _act(self, agent: str, what: str) -> None:
        if self.stm is not None:
            self.stm.answered.add(agent)
        self._log(f"[act] dispatch {agent}: {what}")


def _role_terms(profile: Profile) -> list[str]:
    """Context terms used by entity resolution to disambiguate the protectee."""
    terms = re.split(r"[,\s]+", profile.role)
    return [t for t in terms if len(t) > 2]


# Surveillance-of-routine language is the strongest oblique indicator, so a
# retrieved chunk carrying it maps to a physical-location threat.
_SURVEILLANCE_CUES = ("route", "routine", "garage", "waited", "outside", "which door",
                      "schedule", "floor plan", "entrance", "movements")


def _chunk_to_finding(rc) -> Finding:
    """Turn a retrieved chunk into a scoreable Finding, carrying its provenance.

    Severity is built transparently from the retrieval signal: a base level, a
    bump for surveillance-of-routine language, and a bump for corroboration from
    non-expiring case memory (an established prior pattern, not a one-off).
    """
    chunk = rc.chunk
    text = chunk.text.lower()
    surveillance = any(cue in text for cue in _SURVEILLANCE_CUES)
    channel = ThreatChannel.PHYSICAL_LOCATION if surveillance else ThreatChannel.HOSTILE_ACTOR

    severity = 0.4
    if surveillance:
        severity += 0.3
    if chunk.index_name == "case":
        severity += 0.1
    severity = min(severity, 1.0)

    return Finding(
        channel=channel,
        source_type=chunk.source_type,
        source_id=chunk.source_id,
        summary=(chunk.meta.get("author", "") + ": " if chunk.meta.get("author") else "")
        + chunk.text[:120],
        event_date=chunk.authored_at if isinstance(chunk.authored_at, date) else None,
        severity=severity,
        detail={
            "retrieved": True,
            "index": chunk.index_name,
            "rerank_score": round(rc.rerank_score, 3),
            "snippet": chunk.text[:200],
            "author": chunk.meta.get("author"),
        },
    )
