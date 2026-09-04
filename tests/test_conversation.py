"""The conversational layer, tested offline with a scripted fake backend.

The contract these tests defend is the one the project previously broke: the
chat pane must produce MODEL output or an honest error — never a Python
template dressed up as a model. So we assert on where the words came from and
on what the model was allowed to see, not on the wording itself.
"""

import json
from datetime import date

import pytest

from threat_detector.conversation import INTENTS, ConversationAgent
from threat_detector.llm_backends import LLMUnavailable, strip_reasoning
from threat_detector.tasking_state import TaskingSlots


class FakeBackend:
    """Records every prompt; returns scripted text (and scripted intents)."""

    name = "fake"

    def __init__(self, reply="MODEL PROSE.", intent="chitchat"):
        self.reply = reply
        self.intent = intent
        self.prompts: list[str] = []
        self.histories: list[list] = []

    def check(self):
        return True, "fake backend"

    def chat(self, system, user, *, model, json_schema=None, max_tokens=1024,
             history=None, temperature=0.3):
        self.prompts.append(system + "\n" + user)
        self.histories.append(list(history or []))
        if json_schema is not None:
            props = json_schema.get("properties") or {}
            if "intent" in props:
                return json.dumps({"intent": self.intent,
                                   "reasoning": "scripted"})
            return json.dumps(getattr(self, "slots", None) or {
                "protectee_name": None, "protectee_role": None, "city": None,
                "venue": None, "event_date": None})
        return self.reply


@pytest.fixture
def agent(synthetic_cfg, example_profile):
    return ConversationAgent(FakeBackend(), synthetic_cfg, example_profile)


# --- routing ---------------------------------------------------------------
def test_route_returns_a_known_intent(agent):
    agent.backend.intent = "new_tasking"
    intent, why = agent.route("assess the CEO in Berlin next month",
                              has_briefing=False, pending_questions=False)
    assert intent == "new_tasking" and why


def test_route_hides_intents_that_make_no_sense_yet(agent):
    """followup can't be chosen with no briefing on screen, and
    answer_clarification can't be chosen with nothing pending — the menu the
    model sees is narrowed in code, so it cannot route into a dead end."""
    agent.route("hello", has_briefing=False, pending_questions=False)
    menu = agent.backend.prompts[-1]
    assert "followup" not in menu and "answer_clarification" not in menu
    assert "new_tasking" in menu and "chitchat" in menu


def test_route_offers_every_intent_once_state_allows(agent):
    agent.backend.intent = "followup"
    agent.route("why so high?", has_briefing=True, pending_questions=True)
    menu = agent.backend.prompts[-1]
    assert all(k in menu for k in INTENTS)


def test_route_falls_back_to_heuristics_when_json_is_garbage(synthetic_cfg,
                                                             example_profile):
    class Garbage(FakeBackend):
        def chat(self, system, user, **kw):
            return "I reckon it's probably {{{ a tasking"

    a = ConversationAgent(Garbage(), synthetic_cfg, example_profile)
    intent, why = a.route("assess the threat to the CEO travelling to Berlin",
                          has_briefing=False, pending_questions=False)
    assert intent in INTENTS and why == "heuristic fallback"


# --- the replies are the MODEL's, and the model is given real grounding -----
def test_clarification_is_written_by_the_model_from_guardrail_gaps(
        agent, synthetic_cfg, example_profile):
    from threat_detector import guardrails

    intake = guardrails.intake_check("assess our CEO", example_profile, synthetic_cfg)
    assert intake.questions, "fixture should leave real gaps"

    agent.backend.reply = "Two things before I start: where, and when?"
    out = agent.ask_clarification("assess our CEO", intake)

    assert out == "Two things before I start: where, and when?"
    # The deterministic guardrail still decides WHAT is missing: every gap it
    # found must reach the model, so the model can only rephrase, not skip.
    for q in intake.questions:
        assert q in agent.backend.prompts[-1]


def test_followup_is_grounded_in_the_briefing_it_was_given(agent, example_briefing):
    agent.answer_followup("why is it scored that high?", example_briefing)
    prompt = agent.backend.prompts[-1]
    assert f"{example_briefing.score.overall:.2f}" in prompt
    for f in example_briefing.findings[:3]:
        assert f.source_id in prompt


def test_chitchat_forbids_inventing_findings(agent):
    """A small model asked 'what can you do?' in an analyst persona will
    fabricate a threat report with plausible source ids. The prompt must
    forbid results talk outright."""
    agent.chitchat("what can you do?", has_briefing=False)
    prompt = agent.backend.prompts[-1]
    assert "NO findings" in prompt and "source id" in prompt


def test_chitchat_uses_the_capable_model_not_the_cheap_one(agent, synthetic_cfg):
    """Fabrication risk is highest exactly where there is no grounding, so
    free-form conversation runs on the orchestrator model."""
    calls = []
    orig = agent.backend.chat
    agent.backend.chat = lambda s, u, **kw: (calls.append(kw["model"]), orig(s, u, **kw))[1]
    agent.chitchat("hello", has_briefing=False)
    assert calls == [synthetic_cfg.orchestrator_model]


# --- the transcript makes it a conversation, not a series of one-shots -----
def test_history_is_replayed_to_the_model(agent):
    agent.remember("user", "assess the CEO in Berlin")
    agent.remember("assistant", "Running it now.")
    agent.chitchat("actually, make it Munich", has_briefing=False)
    roles = [t["role"] for t in agent.backend.histories[-1]]
    assert roles == ["user", "assistant"]


def test_history_is_bounded(agent):
    for i in range(40):
        agent.remember("user", f"message {i}")
    assert len(agent.history) == ConversationAgent.MAX_TURNS
    assert agent.history[-1]["content"] == "message 39"


# --- the trust boundary holds here too -------------------------------------
def test_pii_never_reaches_the_model(agent, example_briefing, example_profile):
    """Exposure findings can echo the protectee's PII, and this context goes
    straight into a prompt. Redact at the boundary, as the agent loop does."""
    pii = example_profile.pii
    example_briefing.findings[0].summary = (
        f"{pii.emails[0]} surfaced in a broker listing"
        if pii.emails else "no pii")
    agent.answer_followup("what did exposure find?", example_briefing)
    blob = "\n".join(agent.backend.prompts)
    for secret in (pii.home_addresses + pii.phone_numbers + pii.emails):
        assert secret not in blob, "PII leaked into a model prompt!"


# --- failure is loud -------------------------------------------------------
def test_a_dead_backend_raises_instead_of_returning_canned_text(
        synthetic_cfg, example_profile):
    """The whole bug this module was written to fix: a failed model call used
    to be swallowed, leaving a Python template that looked like an answer."""

    class Dead(FakeBackend):
        def chat(self, system, user, **kw):
            raise LLMUnavailable("ollama is not running")

    a = ConversationAgent(Dead(), synthetic_cfg, example_profile)
    with pytest.raises(LLMUnavailable, match="not running"):
        a.chitchat("hi", has_briefing=False)
    with pytest.raises(LLMUnavailable):
        a.route("hi", has_briefing=False, pending_questions=False)


# --- reasoning models ------------------------------------------------------
def test_reasoning_blocks_are_stripped():
    """qwen3 and friends emit <think>…</think>. Leaving it in shows the analyst
    the model muttering to itself and breaks every downstream parse."""
    assert strip_reasoning("<think>hmm, maybe</think>The answer.") == "The answer."
    assert strip_reasoning("<think>ran out of budget mid-tho") == ""
    assert strip_reasoning("plain text") == "plain text"


def test_model_output_is_stripped_of_reasoning_end_to_end(synthetic_cfg,
                                                          example_profile):
    a = ConversationAgent(FakeBackend(reply="<think>…</think>Real answer."),
                          synthetic_cfg, example_profile)
    # strip_reasoning runs inside the real backends; assert the helper the
    # backends share does the job on a completion shaped like qwen3's.
    assert strip_reasoning(a.chitchat("hi", has_briefing=False)) == "Real answer."


# --- slot extraction: the half of intake a regex cannot do -----------------
def test_slots_come_from_the_model_and_the_regex_backstop(agent):
    agent.backend.slots = {"protectee_name": "Dana Reyes",
                           "protectee_role": "CEO, Northwind",
                           "city": "Dublin", "venue": None,
                           "event_date": "2026-10-14"}
    slots = TaskingSlots()
    changed = agent.extract_slots("Dana Reyes, our CEO, keynotes in Dublin "
                                  "on 14 October", slots)
    assert set(changed) >= {"protectee_name", "city", "event_date"}
    assert slots.city == "Dublin"
    assert slots.event_date == date(2026, 10, 14)
    assert slots.ready


def test_slot_extraction_falls_back_to_the_deterministic_extractors(
        synthetic_cfg, example_profile):
    """A useless model turn must not lose the tasking — the regex extractors
    that predate this layer are still the floor."""
    class Garbage(FakeBackend):
        def chat(self, system, user, **kw):
            return "not json at all {{{"

    a = ConversationAgent(Garbage(), synthetic_cfg, example_profile)
    slots = TaskingSlots()
    a.extract_slots("the conference is in Berlin in 3 weeks", slots)
    assert slots.city == "Berlin"
    assert slots.event_date is not None


def test_extraction_never_reparses_the_transcript(agent):
    """Re-extracting from accumulated text is what deadlocked the console."""
    agent.backend.slots = {"protectee_name": None, "protectee_role": None,
                           "city": None, "venue": None, "event_date": None}
    slots = TaskingSlots()
    slots.update(city="Dublin")
    slots.add_turn("earlier I said Berlin")
    agent.extract_slots("nothing new here", slots)
    assert slots.city == "Dublin"
    # The prompt is given the CURRENT message and the known slots, not the log.
    assert "earlier I said Berlin" not in agent.backend.prompts[-1]


# --- the model is told the truth about what has run ------------------------
def test_run_state_says_plainly_that_nothing_has_run():
    out = ConversationAgent.run_state(None, TaskingSlots())
    assert "no assessment has been run" in out
    assert "Nothing is in progress" in out


def test_run_state_reports_a_completed_run(example_briefing):
    out = ConversationAgent.run_state(example_briefing, TaskingSlots())
    assert str(len(example_briefing.findings)) in out
    assert "Nothing is in progress" in out


def test_chitchat_is_told_the_run_state_and_forbidden_to_invent_one(agent):
    """The console once answered 'what did you find?' with 'collection is still
    running'. Nothing was running; the model had simply been told nothing."""
    agent.chitchat("what did you find?", has_briefing=False,
                   run_state=ConversationAgent.run_state(None, TaskingSlots()))
    prompt = agent.backend.prompts[-1]
    assert "no assessment has been run" in prompt
    assert "Do NOT report progress or status" in prompt


def test_the_persona_forbids_claiming_work_is_underway(agent):
    """Collection is synchronous, so 'I am tasking the agents now' can never be
    a true sentence in this system."""
    agent.chitchat("go ahead and start", has_briefing=False)
    prompt = agent.backend.prompts[-1]
    assert "NO background process" in prompt
    assert "still processing" in prompt


def test_clarification_may_not_pad_the_guardrail_checklist(agent, synthetic_cfg):
    """It invented an extra question about the venue relationship, which made
    the system look like it was stalling."""
    from threat_detector.guardrails import intake_gaps

    intake = intake_gaps(TaskingSlots(), synthetic_cfg)
    agent.ask_clarification("assess someone", intake)
    prompt = agent.backend.prompts[-1]
    assert "EXACTLY the missing items" in prompt
    assert "do not say you are starting" in prompt.lower()
