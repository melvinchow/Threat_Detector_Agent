"""The GUI panels, and the structural rule that keeps them alive.

The console's tabs looked frozen for a whole session. The cause was structural rather than
cosmetic: every early-return path in `on_send` yielded `gr.skip()` for outputs
4-11, and only the assessment branch ever filled them. A turn that asked a
clarifying question therefore left every tab showing its import-time render.

So the rule these tests defend is not "the panels are correct" but "**no code
path can leave a panel stale**" — every yield from `on_send` is a complete
output tuple, on every branch, including the error branches.

Runs entirely offline against a scripted fake backend.
"""

import json
from datetime import date

import pytest

gr = pytest.importorskip("gradio")

import app as gui                                        # noqa: E402
from threat_detector.tasking_state import TaskingSlots   # noqa: E402


class FakeBackend:
    """Routes to a scripted intent; returns scripted prose and scripted slots."""

    name = "fake"

    def __init__(self, intent="chitchat", reply="MODEL PROSE.", slots=None):
        self.intent, self.reply, self.slots = intent, reply, slots or {}

    def check(self):
        return True, "fake"

    def chat(self, system, user, *, model, json_schema=None, max_tokens=1024,
             history=None, temperature=0.3):
        if json_schema is None:
            return self.reply
        if "intent" in (json_schema.get("properties") or {}):
            return json.dumps({"intent": self.intent, "reasoning": "scripted"})
        return json.dumps({"protectee_name": None, "protectee_role": None,
                           "city": None, "venue": None, "event_date": None,
                           **self.slots})


@pytest.fixture
def wired(monkeypatch):
    """A GUI whose backend is fake, geocoder offline, and budget untouched."""
    backend = FakeBackend()
    monkeypatch.setattr(gui.llm_backends, "verify",
                        lambda cfg, **kw: (True, "ok", backend))
    # These tests drive the real handler, which spends the real per-day
    # `llm_calls` budget from data/rate_limits.json. Running the suite must not
    # eat the operator's quota — and an exhausted budget made these tests fail
    # for a reason that had nothing to do with the panels.
    from threat_detector import ratelimit
    monkeypatch.setattr(ratelimit, "spend", lambda cfg, key, n=1: True)
    coords = {"Dublin": (53.3498, -6.2603), "Berlin": (52.5200, 13.4050)}
    monkeypatch.setattr(gui.geospatial, "geocode",
                        lambda place, cfg=None: next(
                            (c for k, c in coords.items() if k in place), None))
    return backend


def _drive(user_text, state=None, focus="all"):
    return list(gui.on_send(user_text, [], focus, state or gui._new_state()))


# --- the structural invariant ----------------------------------------------
def test_render_all_matches_the_wired_output_count():
    """If a panel is added to the UI without being added to `_render_all`,
    Gradio silently drops updates. Pin the two together."""
    assert len(gui._render_all(gui._new_state(), [], gui.CFG)) == len(gui.outputs)


def test_every_yield_on_every_branch_is_a_full_output_tuple(wired):
    """The bug itself: branches that returned `gr.skip()` for most panels."""
    wired.intent = "chitchat"
    chitchat = _drive("what can you do?")

    wired.intent = "new_tasking"
    clarify = _drive("assess someone somewhere")

    for label, frames in (("chitchat", chitchat), ("clarification", clarify)):
        assert frames, f"{label} produced no output"
        for i, frame in enumerate(frames):
            assert len(frame) == len(gui.outputs), f"{label} frame {i} is short"
            for j, cell in enumerate(frame):
                # gr.skip() is a sentinel dict, so compare by value — the
                # session state is also a dict and would trip an isinstance.
                assert cell != gr.skip(), \
                    f"{label} frame {i} left output {j} stale"


def test_llm_disabled_branch_still_renders_every_panel(monkeypatch):
    """Even the refusal paths must not freeze the tabs."""
    cfg = gui.load_config()
    monkeypatch.setattr(type(cfg), "llm_enabled", property(lambda self: False))
    monkeypatch.setattr(gui, "_refresh_cfg", lambda: cfg)
    for frame in _drive("hello"):
        assert len(frame) == len(gui.outputs)


# --- the panels are session-aware, not run-aware ---------------------------
def test_trace_grows_on_a_turn_that_runs_no_assessment(wired):
    wired.intent = "chitchat"
    frames = _drive("hello there")
    trace = frames[-1][gui.outputs.index(gui.trace_md)]
    assert "[route]" in trace and "intent=chitchat" in trace
    assert "nothing yet" not in trace


def test_short_term_shows_the_slots_and_the_model_window(wired):
    wired.intent = "new_tasking"
    wired.slots = {"city": "Dublin", "protectee_name": "Dana Reyes"}
    frames = _drive("assess Dana Reyes in Dublin")
    stm = frames[-1][gui.outputs.index(gui.stm_md)]
    assert "Dublin" in stm and "Dana Reyes" in stm
    assert "Conversation window" in stm
    assert "when" in stm            # the still-missing slot is named


def test_sources_says_not_exercised_before_any_run():
    out = gui.render_sources(gui._new_state())
    assert "No assessment has run" in out
    assert "Case Memory / Retrieval" in out


# --- the map follows the conversation --------------------------------------
def test_map_centres_on_the_named_city_before_any_assessment(wired):
    state = gui._new_state()
    state["slots"].update(city="Dublin")
    html = gui.render_map(state)
    assert "53.3498" in html, "the map must move when a city is named"


def test_map_excludes_another_scenarios_pins(wired):
    """Berlin facts must not appear during a conversation about Dublin —
    plotting the whole facts store unconditionally is what put them there."""
    state = gui._new_state()
    state["slots"].update(city="Dublin")
    pts, center = gui._map_points(state)
    assert center == (53.3498, -6.2603)
    for p in pts:
        assert abs(p["lat"] - 52.52) > 0.5 or abs(p["lon"] - 13.4) > 0.5, \
            f"a Berlin pin leaked into a Dublin map: {p['label']}"


def test_map_is_honest_when_nothing_is_located():
    assert "No mappable location yet" in gui.render_map(gui._new_state())


# --- the ad-hoc protectee ---------------------------------------------------
def test_a_protectee_named_in_chat_gets_an_empty_pii_block():
    """An analyst naming someone has not handed us their home address, and the
    system must not invent one to make Exposure look productive."""
    slots = TaskingSlots()
    slots.update(protectee_name="Dana Reyes", protectee_role="CEO",
                 city="Dublin", event_date=date(2026, 10, 14))
    prof = gui._session_profile(slots)
    assert prof.name == "Dana Reyes"
    assert prof.entity_id == "dana_reyes"
    assert not (prof.pii.emails or prof.pii.home_addresses
                or prof.pii.phone_numbers or prof.pii.known_usernames)


def test_the_configured_protectee_keeps_its_pii():
    slots = TaskingSlots()
    slots.update(protectee_name=gui.DEFAULT_PROFILE.name)
    assert gui._session_profile(slots) is gui.DEFAULT_PROFILE


# --- console polish: what an analyst sees first ----------------------------
# These are cosmetic rules, but they are the ones that quietly rot: a tab gets
# renamed in the UI and the prose that points at it keeps the old name, or a
# developer panel gets un-hidden by an unrelated edit and a demo opens on a
# token-window readout.
def _tabs():
    return [(b.label, b.visible) for b in gui.demo.blocks.values()
            if isinstance(b, gr.Tab)]


def test_the_analyst_tabs_come_first_and_the_developer_tabs_are_hidden():
    labels = [lbl for lbl, _ in _tabs()]
    assert labels == ["Source Reliability", "Travel Planner", "Map",
                      "Data Sources", "Short-term", "Trace", "Evals", "Agents"]
    assert all(vis for _, vis in _tabs()[:4]), "an analyst tab is hidden"
    assert not any(vis for _, vis in _tabs()[4:]), "a developer tab is exposed"


def test_the_debug_button_toggles_exactly_the_developer_tabs():
    shown, label, *tabs = gui.on_toggle_debug(False)
    assert shown is True
    assert label["value"] == "Hide debugging tools"
    assert len(tabs) == 4 and all(t["visible"] is True for t in tabs)

    hidden, label, *tabs = gui.on_toggle_debug(True)
    assert hidden is False
    assert label["value"] == "Debugging tools"
    assert all(t["visible"] is False for t in tabs)


def test_no_panel_mentions_the_course_module_numbers():
    """Course scaffolding is for the report, not for the analyst's screen."""
    panels = {
        "ltm": gui.render_ltm(None), "beam": gui.render_beam(None, None),
        "sources": gui.render_sources(None), "stm": gui.render_stm(None),
        "trace": gui.render_trace(None), "evals": gui.render_evals(None),
        "agents": gui.render_agents(),
        "stats": gui._stats(gui._new_state(), gui.CFG),
    }
    offenders = {k: v for k, v in panels.items() if "Module" in v}
    assert not offenders, f"module numbers leaked into: {sorted(offenders)}"

    static = [b.value for b in gui.demo.blocks.values()
              if isinstance(b, gr.Markdown) and isinstance(b.value, str)]
    assert not [t for t in static if "Module" in t]


def test_the_planner_puts_the_options_above_the_search_that_found_them():
    """The analyst decides between plans; the beam search is the justification
    underneath, not several screens to scroll past first."""
    class Orch:
        plan_trace = [{"event": "done", "complete_plans": 1, "note": "closed"}]

    class Item:
        day, requirement, vendor_name, cost = 1, "close protection", "V", 900.0

    class Plan:
        plan_id, complete = "option-1", True
        total_cost, budget, score = 900.0, 12000.0, 0.8
        line_items = [Item()]

    class B:
        plans = [Plan()]

    body = gui.render_beam(Orch(), B(), "approve option-1")
    assert body.index("Your options") < body.index("Analyst decision") \
        < body.index("How the planner reached")


def test_the_chat_points_at_tabs_that_actually_exist():
    labels = {lbl for lbl, _ in _tabs()}
    prose = " ".join([gui.render_ltm(None), gui.render_evals(None),
                      gui.render_sources(None), gui.render_stm(None)])
    for stale in ("Long-term tab", "ToT planner tab", "Sources tab"):
        assert stale not in prose, f"prose still points at the old {stale!r}"
    assert "Source Reliability" in labels and "Data Sources" in labels


def test_news_and_social_are_reported_as_separate_rows():
    """One row per source, not per specialist.

    Open-Source Monitoring owns Exa (news) and Reddit (social). Merged into one
    row, a healthy Exa count read as "the specialist is live" while Reddit had
    silently served fixtures all session — which is exactly how a broken
    collection channel went unnoticed in a live demo.
    """
    assert gui._SPECIALIST_OF["news"] != gui._SPECIALIST_OF["social"]
    assert gui._SPECIALIST_OF["news"] in gui._SPECIALISTS
    assert gui._SPECIALIST_OF["social"] in gui._SPECIALISTS


def test_a_fallback_reason_reaches_the_data_sources_tab():
    """A `real`-configured source that served fixtures must say why, on screen."""
    from threat_detector.schemas import Finding, SourceType, ThreatChannel

    f = Finding(channel=ThreatChannel.REPUTATION, source_type=SourceType.SOCIAL,
                source_id="reddit:r/test", summary="post", severity=0.3,
                detail={"synthetic_fixture": True,
                        "fallback_reason": "praw is not installed"})
    b = gui.Briefing(protectee="Test", tasking="t", findings=[f],
                     score=gui.risk.score_findings([]), recommendations=[])
    md = gui.render_sources({"briefing": b})
    assert "praw is not installed" in md, "the analyst is not told why it fell back"
