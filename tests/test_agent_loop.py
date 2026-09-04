"""The LLM agent loop, tested offline with a scripted fake backend.

The loop's contract must hold for ANY model: decisions drive real tools,
findings stay typed, guardrails and the deterministic rubric still apply, and
model failures degrade gracefully instead of crashing the assessment.
"""

import json

import pytest

from threat_detector.agent_loop import LLMOrchestrator
from threat_detector.llm_backends import extract_json, normalize_model


class FakeBackend:
    """Plays back scripted decisions; records every prompt it was shown."""

    name = "fake"

    def __init__(self, decisions, narrative="Assessment: pattern [forum:redwing]."):
        self.decisions = list(decisions)
        self.narrative = narrative
        self.prompts: list[str] = []

    def check(self):
        return True, "fake backend"

    def chat(self, system, user, *, model, json_schema=None, max_tokens=1024):
        self.prompts.append(system + "\n" + user)
        if json_schema is not None:                      # a DECIDE call
            d = self.decisions.pop(0) if self.decisions else {"tool": "done"}
            return json.dumps({"reasoning": "scripted", "args": {}, **d})
        return self.narrative                            # commentary / ANSWER


def _loop(profile, cfg, decisions, **kw):
    return LLMOrchestrator(profile, cfg,
                           backend=FakeBackend(decisions, **kw))


def test_decisions_drive_real_tools_and_produce_a_briefing(synthetic_cfg, example_profile):
    b = _loop(example_profile, synthetic_cfg, [
        {"tool": "search_social", "args": {"query": "Jordan Vale"}},
        {"tool": "public_records_sweep", "args": {"city": "Berlin"}},
        {"tool": "case_memory_search", "args": {"query": "venue surveillance"}},
        {"tool": "done"},
    ]).assess("CEO keynoting a conference in Berlin in 3 weeks")

    assert b.findings, "tools must have produced typed findings"
    assert b.narrative.startswith("Assessment")
    trace = "\n".join(b.trace)
    assert "[llm-think]" in trace and "[specialist:" in trace
    # The deterministic spine still ran: rubric score, recommendations, planner.
    assert b.score.overall > 0 and b.recommendations and b.plans


def test_repeated_call_is_blocked_in_code(synthetic_cfg, example_profile):
    b = _loop(example_profile, synthetic_cfg, [
        {"tool": "search_social", "args": {"query": "Jordan Vale"}},
        {"tool": "search_social", "args": {"query": "Jordan Vale"}},  # dupe
        {"tool": "search_news", "args": {"query": "never reached"}},
    ]).assess("CEO keynoting a conference in Berlin in 3 weeks")
    assert "already ran with these args" in "\n".join(b.trace)


def test_unparseable_decision_degrades_to_done(synthetic_cfg, example_profile):
    class Garbage(FakeBackend):
        def chat(self, system, user, *, model, json_schema=None, max_tokens=1024):
            if json_schema is not None:
                return "I think we should probably {{{ look at"
            return "n/a"

    orch = LLMOrchestrator(example_profile, synthetic_cfg, backend=Garbage([]))
    b = orch.assess("CEO keynoting a conference in Berlin in 3 weeks")
    assert b.score is not None            # never crashes; ships what it has


def test_pii_never_reaches_the_model(synthetic_cfg, example_profile):
    fake = FakeBackend([
        {"tool": "exposure_scan", "args": {}},
        {"tool": "done"},
    ])
    LLMOrchestrator(example_profile, synthetic_cfg, backend=fake).assess(
        "CEO keynoting a conference in Berlin in 3 weeks")
    blob = "\n".join(fake.prompts)
    pii = example_profile.pii
    for secret in (pii.home_addresses + pii.phone_numbers + pii.emails):
        assert secret not in blob, "PII leaked into a model prompt!"


def test_unusable_backend_raises_instead_of_silent_fallback(synthetic_cfg, example_profile):
    class Down(FakeBackend):
        def check(self):
            return False, "server offline"

    with pytest.raises(RuntimeError, match="server offline"):
        LLMOrchestrator(example_profile, synthetic_cfg,
                        backend=Down([])).assess("CEO event in Berlin in 3 weeks")


def test_model_prefixes_are_normalized():
    assert normalize_model("anthropic/claude-sonnet-5") == "claude-sonnet-5"
    assert normalize_model("ollama/qwen3:8b") == "qwen3:8b"
    assert normalize_model("claude-haiku-4-5") == "claude-haiku-4-5"


def test_extract_json_survives_fences_and_prose():
    assert extract_json('Sure! ```json\n{"tool": "done", "args": {}}\n```')["tool"] == "done"
    assert extract_json("no json here at all") == {}
    assert extract_json('{"a": {"b": 1}} trailing')["a"]["b"] == 1
