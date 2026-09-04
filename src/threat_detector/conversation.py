"""The conversational layer — the analyst actually talks to a model.

Why this module exists
----------------------
The console used to be a *form* wearing a chat's clothes. Every message went
straight into the assessment pipeline, and everything the analyst read back was
a Python f-string: the clarifying questions came from ``guardrails.intake_check``
as fixed sentences, and the briefing summary came from an ``_summary_md()``
template. Change a word of the tasking and you got the same sentences back.
That is a chatbot in the 1966 sense, not an agent.

So the chat pane now runs through a model that does four things no template can:

1. **Routes.** It reads the message *in the context of the conversation so far*
   and decides what the analyst wants: a new assessment, an answer to the
   questions it just asked, a follow-up about the briefing already on screen,
   or ordinary conversation about the system. "Actually make it Munich" after a
   Berlin run is a re-tasking; "why is that scored so high?" is a follow-up.
2. **Asks in its own words.** The intake guardrail still decides *what is
   missing* deterministically (that is a Module 6 rule, and rules stay in code).
   The model decides how to *say* it, in the register of the conversation.
3. **Answers follow-ups, grounded.** Questions about a completed briefing are
   answered from that briefing's findings, score, and plans — with source ids —
   and never by re-running collection.
4. **Talks.** Scope, method, limitations, "what can you do" — real answers, in
   whatever language the analyst is writing in.

The safety split from Module 6 is unchanged: the model chooses words and
strategy; tools, the risk rubric, the guardrails, and the ToT beam stay
deterministic. A conversational front end must not become a way to talk the
system out of its own rules — so ``route`` can only pick from a fixed set of
intents, and every grounded answer is handed findings it cannot alter.
"""

from __future__ import annotations

import json
import re
from datetime import date

from .llm_backends import LLMUnavailable, chat, extract_json
from .schemas import Briefing
from .tasking_state import TaskingSlots

# ---------------------------------------------------------------------------
# INTENTS — the router's entire vocabulary. A closed set on purpose: the model
# steers the conversation, it cannot invent a new mode of operation.
# ---------------------------------------------------------------------------
INTENTS = {
    "new_tasking": "The analyst is asking for a threat assessment of a person/event/trip, "
                   "or changing the target of one (different city, date, protectee).",
    "answer_clarification": "The analyst is supplying details the system just asked for "
                            "(a city, a date, who the protectee is). Only valid when "
                            "questions are pending.",
    "followup": "A question ABOUT the assessment already on screen — why a score is what "
                "it is, what a finding means, which plan to pick, what to do next.",
    "chitchat": "Anything else: greetings, what the system can do, how it works, its "
                "limits, or general protective-intelligence questions.",
}

_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "reasoning"],
}

_SLOT_SCHEMA = {
    "type": "object",
    "properties": {
        "protectee_name": {"type": ["string", "null"],
                           "description": "Full name of the person being protected."},
        "protectee_role": {"type": ["string", "null"],
                           "description": "Their public role/affiliation, e.g. 'CEO, Acme'."},
        "city": {"type": ["string", "null"],
                 "description": "City the trip/event is in."},
        "venue": {"type": ["string", "null"],
                  "description": "Named venue, if one was given."},
        "event_date": {"type": ["string", "null"],
                       "description": "Event date as YYYY-MM-DD, resolved against TODAY."},
    },
    "required": ["protectee_name", "protectee_role", "city", "venue", "event_date"],
}

_SYSTEM = """You are the analyst-facing interface of a multi-agent VIP threat
assessment system used by a corporate protective-intelligence team.

The system you speak for:
- Eight agents under one orchestrator. Collection specialists (Open-Source
  Monitoring, Public Records, Exposure, Geospatial, Case Memory/RAG) gather
  typed findings; Risk Assessment scores them against a fixed rubric;
  Recommendation runs a Tree-of-Thought beam search over protection plans.
- Every finding carries a source id, a date, a severity and a reliability.
- The risk score, the guardrails and the plan search are DETERMINISTIC code.
  You explain them. You never recompute, re-score or overrule them.

How you talk: like an experienced protective-intelligence analyst briefing a
colleague. Direct, concrete, no filler, no marketing. Plain prose with short
markdown where it genuinely helps. Never open with "Certainly" or "Great
question". Answer in the language the analyst writes in.

Hard rules:
- State facts ONLY from the findings and data you are given. If something was
  not collected, say it was not collected. "No evidence retrieved" is never
  "no threat" — say which one you mean.
- Cite source ids in [brackets] for factual claims about the protectee.
- The protectee's PII never appears in your context and must never appear in
  your answer.
- You cannot book, contact, purchase or surveil. You propose; a human approves.
- You never collect anything yourself and you have NO background process. When
  collection runs it runs synchronously, and its results are in the SAME reply
  that reports them. So never write that you are running, tasking, dispatching,
  gathering, checking, or that results are "coming" / "still processing" /
  "will follow" — none of that can be true. If a run has not happened, say so
  plainly and say what you need in order to start one."""


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


class ConversationAgent:
    """Holds the thread and turns each analyst message into a real reply.

    One instance per GUI session. It keeps its own transcript (separate from
    Gradio's display list) so the model sees the conversation, not just the
    latest line — which is what makes "make it Munich instead" resolvable.
    """

    MAX_TURNS = 12          # trailing turns shown to the model

    def __init__(self, backend, cfg, profile):
        self.backend = backend
        self.cfg = cfg
        self.profile = profile
        self.history: list[dict] = []

    # -- transcript ---------------------------------------------------------
    def remember(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": _clip(content, 2000)})
        del self.history[:-self.MAX_TURNS]

    def _chat(self, system: str, user: str, *, model=None, cheap=False,
              with_history=True, **kw) -> str:
        from . import ratelimit
        if not ratelimit.spend(self.cfg, "llm_calls"):
            raise LLMUnavailable("Daily llm_calls budget exhausted "
                                 "(raise it in config.yaml → limits: llm_calls).")
        return chat(self.backend, system, user,
                    model=model or (self.cfg.specialist_model if cheap
                                    else self.cfg.orchestrator_model),
                    history=self.history if with_history else None, **kw)

    # -- 1. ROUTE -----------------------------------------------------------
    def route(self, user_text: str, *, has_briefing: bool,
              pending_questions: bool) -> tuple[str, str]:
        """Classify the message against the live conversation. -> (intent, why)"""
        options = {k: v for k, v in INTENTS.items()
                   if not (k == "answer_clarification" and not pending_questions)
                   and not (k == "followup" and not has_briefing)}
        menu = "\n".join(f"- {k}: {v}" for k, v in options.items())
        state = (f"State: a completed assessment is "
                 f"{'ON SCREEN' if has_briefing else 'NOT on screen'}; "
                 f"clarifying questions are "
                 f"{'PENDING an answer' if pending_questions else 'not pending'}.")
        schema = dict(_ROUTE_SCHEMA)
        schema["properties"] = dict(_ROUTE_SCHEMA["properties"],
                                    intent={"type": "string",
                                            "enum": list(options)})
        try:
            raw = self._chat(
                "You classify an analyst's message into exactly one intent. "
                "Consider the conversation so far, not just the last line. "
                "Reply with JSON only.",
                f"{state}\n\nINTENTS:\n{menu}\n\nMESSAGE: {user_text}",
                cheap=True, json_schema=schema, max_tokens=400)
            intent = extract_json(raw).get("intent")
            why = _clip(extract_json(raw).get("reasoning", ""), 160)
            if intent in options:
                return intent, why
        except LLMUnavailable:
            raise
        except Exception:
            pass
        return self._route_fallback(user_text, options), "heuristic fallback"

    @staticmethod
    def _route_fallback(text: str, options: dict) -> str:
        """Used only when the router's JSON is unusable — never as the norm."""
        low = text.lower()
        if "answer_clarification" in options and len(text.split()) <= 12:
            return "answer_clarification"
        if "followup" in options and re.search(
                r"\b(why|how|what|which|explain|elaborate|that|this|it)\b", low):
            return "followup"
        if re.search(r"\b(assess|threat|travel|trip|keynote|visit|event|"
                     r"conference|summit|speaking)\b", low):
            return "new_tasking"
        return "followup" if "followup" in options else "chitchat"

    # -- 1b. FILL THE TASKING SLOTS -----------------------------------------
    def extract_slots(self, user_text: str, slots: TaskingSlots,
                      *, override: bool = False,
                      today: date | None = None) -> list[str]:
        """Read who/where/when/venue out of the message into ``slots``.

        This is the half of intake a regex cannot do: "he's speaking in Dublin,
        Ireland the week after next" carries a city and a date that no pattern
        list will ever cover, and "make it Munich instead" is a correction
        rather than an addition. The model proposes; the deterministic
        extractors in ``guardrails`` cross-check and fill anything it missed, so
        a bad model turn degrades to the old behaviour instead of losing the
        tasking.

        Returns the names of the slots that changed, for the trace.
        """
        from . import guardrails

        today = today or date.today()
        known = {k: v for k, v in {
            "protectee_name": slots.protectee_name,
            "protectee_role": slots.protectee_role,
            "city": slots.city,
            "venue": slots.venue,
            "event_date": slots.event_date.isoformat() if slots.event_date else None,
        }.items()}

        proposed: dict = {}
        try:
            raw = self._chat(
                "You extract tasking details for a VIP protection assessment. "
                "Return JSON only. Use null for anything the analyst has not "
                "actually said — never guess a name, a city or a date, and "
                "never copy the example values from a question you were asked. "
                "Resolve relative dates ('in 3 weeks', 'next Tuesday') against "
                "TODAY into YYYY-MM-DD.",
                f"TODAY: {today.isoformat()}\n"
                f"ALREADY KNOWN: {json.dumps(known)}\n\n"
                f"ANALYST MESSAGE: {user_text}",
                cheap=True, json_schema=_SLOT_SCHEMA, max_tokens=400)
            proposed = extract_json(raw) or {}
        except LLMUnavailable:
            raise
        except Exception:
            proposed = {}

        values: dict = {}
        for key in ("protectee_name", "protectee_role", "city", "venue"):
            v = proposed.get(key)
            if isinstance(v, str) and v.strip():
                values[key] = v.strip()
        raw_date = proposed.get("event_date")
        if isinstance(raw_date, str) and raw_date.strip():
            try:
                values["event_date"] = date.fromisoformat(raw_date.strip()[:10])
            except ValueError:
                pass

        # Deterministic cross-check: fills whatever the model left out. Runs on
        # THIS message only — never on the accumulated transcript, which is the
        # re-parsing that deadlocked the old console.
        values.setdefault("city", guardrails.extract_city(user_text))
        values.setdefault("event_date",
                          guardrails.extract_event_date(user_text, today))

        changed = (slots.override(**values) if override else slots.update(**values))
        for key in changed:
            label = {"protectee_name": "protectee", "protectee_role": "role",
                     "city": "city", "venue": "venue",
                     "event_date": "date"}[key]
            slots.origin[label] = "model" if key in proposed and proposed.get(key) \
                else "pattern"
        return changed

    # -- the truthful picture of what has and has not run --------------------
    @staticmethod
    def run_state(briefing: Briefing | None, slots: TaskingSlots | None = None) -> str:
        """What is actually true right now, handed to every ungrounded prompt.

        The console once answered "what did you find?" with "nothing yet —
        collection is still running." Nothing was running; the model had simply
        been told nothing about the system's state and filled the hole with
        something plausible. A model with no facts about itself will invent
        them, so state the facts.
        """
        if briefing is None:
            line = ("RUN STATE: no assessment has been run in this session. "
                    "Zero findings exist. Nothing is in progress.")
        else:
            line = (f"RUN STATE: one assessment has completed — "
                    f"{len(briefing.findings)} finding(s), overall score "
                    f"{briefing.score.overall:.2f}. Nothing is in progress.")
        if slots is not None:
            known = ", ".join(f"{k}={v.split('  ')[0]}" for k, v in slots.filled().items())
            line += f"\nTASKING SO FAR: {known or '(nothing yet)'}"
            if slots.missing():
                line += f"\nSTILL MISSING: {', '.join(slots.missing())}"
        return line

    # -- 2. ASK FOR WHAT'S MISSING ------------------------------------------
    def ask_clarification(self, tasking: str, intake, *,
                          run_state: str = "") -> str:
        """The guardrail decided WHAT is missing; the model decides how to ask.

        The gaps below are computed in ``guardrails.intake_check`` and are not
        negotiable — the model is rephrasing a fixed checklist, not choosing
        whether to enforce it.
        """
        gaps = "\n".join(f"- {q}" for q in intake.questions)
        assumed = "\n".join(f"- {a}" for a in intake.assumptions) or "(none)"
        return self._chat(
            _SYSTEM + "\n\nRight now you are at INTAKE. NOTHING HAS RUN. A "
            "deterministic input guardrail found gaps it will not guess past. "
            "Ask the analyst for exactly those, conversationally — in your own "
            "words, as a short lead-in plus a numbered list. State the safe "
            "assumptions you are already making.\n"
            "- Your numbered list must contain EXACTLY the missing items below, "
            "one each, no more. Do not add a question of your own, however "
            "reasonable it sounds — the checklist is the guardrail, and padding "
            "it makes the system look like it is stalling.\n"
            "- Do not answer the tasking, and do not say you are starting, "
            "preparing or dispatching anything. You are waiting on the analyst.\n"
            "Keep it under 120 words.",
            f"{run_state}\n\nTHE ANALYST ASKED FOR: {tasking}\n\n"
            f"MISSING (ask about exactly these, all of them):\n{gaps}\n\n"
            f"ALREADY ASSUMED (state, don't ask):\n{assumed}",
            max_tokens=700)

    # -- 3. FOLLOW-UPS, GROUNDED IN THE BRIEFING ----------------------------
    def answer_followup(self, question: str, briefing: Briefing,
                        trace: list[str] | None = None) -> str:
        """Answer from the briefing on screen. No new collection, no invention."""
        return self._chat(
            _SYSTEM + "\n\nThe analyst is asking about the assessment below. "
            "Answer from it and only it.\n"
            "- Cite [source ids] — but ONLY ids that appear verbatim above. "
            "Never attach a citation to something that was NOT found: an "
            "invented id on a negative claim ('no medical records [case:health]') "
            "reads as evidence of absence and is the most dangerous thing you "
            "can write here. State negatives with no citation at all.\n"
            "- If the answer is not in this material, say plainly what would "
            "have to be collected to get it, and offer to re-task.\n"
            "60-180 words unless asked for more.",
            f"{self._briefing_context(briefing, trace)}\n\nANALYST ASKS: {question}",
            max_tokens=900)

    # -- 4. ORDINARY CONVERSATION -------------------------------------------
    def chitchat(self, user_text: str, *, has_briefing: bool,
                 run_state: str = "") -> str:
        """Talk about the system, its method, its limits — never about results.

        Deliberately runs on the ORCHESTRATOR model, not the cheap one. A small
        model asked "what can you do?" with a security-analyst persona will
        happily answer by inventing a threat report, complete with plausible
        source ids. In this domain a fabricated finding is the worst possible
        output, so the capable model gets this turn and the prompt forbids
        results talk outright.
        """
        return self._chat(
            _SYSTEM + "\n\nYou are BETWEEN assessments. NO collection has run "
            "for this question and you hold NO findings.\n"
            "ABSOLUTE RULES FOR THIS REPLY:\n"
            "- Do NOT state, summarise or allude to any finding, risk score, "
            "actor, location or protection plan.\n"
            "- Do NOT write any source id or citation. There is nothing to cite.\n"
            "- Do NOT report progress or status. Nothing is running. If asked "
            "what you found, the honest answer is that no collection has run "
            "and why — never 'still collecting' or 'results shortly'.\n"
            "- Describe only what the system DOES and what you need to start.\n"
            "If they want an assessment, say you need three things: who the "
            "protectee is, where they are going, and when. Under 120 words.",
            f"{run_state}\n\n"
            f"{'A previous assessment is on screen, but it is NOT the subject of '
               'this question unless the analyst says so. ' if has_briefing else ''}"
            f"ANALYST: {user_text}",
            max_tokens=600)

    # -- 5. THE HEADLINE OVER A COMPLETED RUN -------------------------------
    def summarize_briefing(self, briefing: Briefing, tasking: str) -> str:
        """Two or three sentences of prose to open the briefing.

        The full written assessment is produced inside the agent loop
        (``agent_loop._write_narrative``). This is the conversational turn that
        hands it over — what changed, what matters most, what the analyst
        should do next — so the chat reads as a reply, not a report dump.
        """
        return self._chat(
            _SYSTEM + "\n\nThe assessment just finished. Open the reply with "
            "2-3 sentences in your own voice: the bottom line, the single "
            "biggest driver, and the one thing you want the analyst to decide "
            "or look at next. No headers, no bullet list, no preamble.",
            f"THE ANALYST ASKED FOR: {tasking}\n\n"
            f"{self._briefing_context(briefing)}",
            max_tokens=500)

    # -- shared grounding context -------------------------------------------
    def _briefing_context(self, b: Briefing, trace: list[str] | None = None) -> str:
        """Everything the model may use to answer, redacted at the boundary.

        Redaction happens ONCE over the finished block rather than per-field:
        Exposure findings echo PII values, and those values then propagate into
        the rubric rationale, the written narrative and the trace. Scrubbing
        only the obvious field is how a boundary springs a leak.
        """
        from .tools import risk

        out = [f"TASKING: {b.tasking}",
               f"PROTECTEE: {b.protectee} ({self.profile.role})",
               f"RISK SCORE (deterministic rubric, do not recompute): "
               f"{b.score.overall:.2f} [{risk.band(b.score.overall)}]",
               f"BY CHANNEL: {json.dumps(b.score.by_channel)}",
               "RUBRIC RATIONALE:"]
        out += [f"  - {r}" for r in b.score.rationale[:6]]

        out.append(f"\nFINDINGS ({len(b.findings)}):")
        for f in b.findings[:30]:
            flags = "".join(f" [{k}]" for k in
                            ("uncorroborated", "stale", "synthetic_fixture")
                            if f.detail.get(k))
            out.append(f"  - [{f.source_id}] ({f.channel.value}, sev "
                       f"{f.severity:.2f}, rel {f.reliability:.2f}, "
                       f"{f.event_date or 'undated'}) "
                       f"{_clip(f.summary, 160)}{flags}")

        out.append(f"\nRECOMMENDATIONS ({len(b.recommendations)}):")
        for r in b.recommendations:
            out.append(f"  - {'[needs analyst approval] ' if r.requires_approval else ''}"
                       f"{r.action}")

        if b.plans:
            out.append(f"\nPROTECTION PLAN SLATE ({len(b.plans)}, from the ToT "
                       f"beam search; nothing is booked):")
            for p in b.plans:
                out.append(f"  - {p.plan_id}: cost {p.total_cost:.0f}/{p.budget:.0f}, "
                           f"score {p.score:.3f}, "
                           f"{'complete' if p.complete else 'INCOMPLETE'} — "
                           + "; ".join(f"day {e.day} {e.requirement}: {e.vendor_name}"
                                       for e in p.line_items[:6]))
        else:
            out.append("\nPROTECTION PLAN SLATE: none (the planner runs only when "
                       "the tasking involves travel and risk is ELEVATED or above).")

        if b.narrative:
            out.append(f"\nWRITTEN ASSESSMENT:\n{b.narrative}")
        if trace:
            out.append("\nORCHESTRATOR TRACE (what actually ran):")
            out += [f"  {t}" for t in trace[-25:]]
        return self._redact("\n".join(out))

    def _redact(self, text: str) -> str:
        """The trust boundary from the agent loop, enforced here too: Exposure
        findings can echo PII values, and this context goes into a prompt."""
        pii = self.profile.pii
        for kind, values in (("address", pii.home_addresses),
                             ("phone", pii.phone_numbers),
                             ("email", pii.emails),
                             ("username", pii.known_usernames)):
            for v in values:
                if v:
                    text = text.replace(v, f"‹PII:{kind}›")
        return text


__all__ = ["ConversationAgent", "INTENTS"]
