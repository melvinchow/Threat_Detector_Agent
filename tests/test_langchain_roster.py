"""The LangChain harness, driven offline by the same scripted fake backend.

The point of these tests is that the framework is *load-bearing and bounded*:
LangChain really chooses the dispatches, and it really cannot reach past
collection into scoring, the PII boundary, or the daily budget.
"""

import json

import pytest

pytest.importorskip("langchain", reason="the LangChain harness is an extra")

from threat_detector import langchain_roster as lc
from threat_detector import ratelimit
from tests.test_agent_loop import FakeBackend


def _assess(profile, cfg, decisions, **kw):
    return lc.assess_with_langchain(
        profile, "CEO keynoting a conference in Berlin in 3 weeks", cfg,
        backend=FakeBackend(decisions, **kw))


def test_langchain_drives_real_tools_and_produces_a_briefing(synthetic_cfg,
                                                             example_profile):
    b = _assess(example_profile, synthetic_cfg, [
        {"tool": "search_social", "args": {"query": "Jordan Vale"}},
        {"tool": "public_records_sweep", "args": {"city": "Berlin"}},
        {"tool": "done"},
    ])
    assert b.findings, "the agent's tool calls produced no typed findings"
    trace = "\n".join(b.trace)
    assert "[langchain] roster wired" in trace
    # The deterministic spine ran on LangChain's output, unchanged.
    assert b.score.overall > 0 and b.recommendations


def test_the_roster_covers_every_specialist_the_native_loop_dispatches():
    from threat_detector.agent_loop import LLMOrchestrator

    assert set(lc.build_roster()) == set(LLMOrchestrator._SPECIALIST_NAMES)


def test_exposure_takes_no_arguments_so_pii_cannot_enter_a_prompt():
    """The one boundary a framework must not be able to widen.

    Every other tool takes model-authored arguments. `exposure_scan` must not,
    because an argument is something the model writes — and the model must
    never be in a position to write, or read back, the protectee's PII.
    """
    tool = next(t for t in lc.ROSTER if t[0] == "exposure_scan")
    assert "no arguments" in tool[2].lower()


def test_every_model_call_still_passes_the_daily_budget(synthetic_cfg,
                                                        example_profile):
    before = ratelimit.used_today(synthetic_cfg, "llm_calls")
    _assess(example_profile, synthetic_cfg, [{"tool": "done"}])
    assert ratelimit.used_today(synthetic_cfg, "llm_calls") > before, (
        "LangChain calls bypassed the llm_calls budget")


def test_an_exhausted_budget_refuses_rather_than_returning_an_empty_briefing(
        synthetic_cfg, example_profile, monkeypatch):
    """A dead model must not read as a clean assessment.

    The native loop raises here rather than returning a zero-finding briefing,
    because a briefing rendered from a Python template is indistinguishable
    from a hardcoded app. The framework harness owes the same guarantee.
    """
    from threat_detector.llm_backends import LLMUnavailable

    monkeypatch.setattr(ratelimit, "spend",
                        lambda cfg, key, n=1: key != "llm_calls")
    with pytest.raises(LLMUnavailable, match="budget exhausted"):
        _assess(example_profile, synthetic_cfg, [{"tool": "done"}])
