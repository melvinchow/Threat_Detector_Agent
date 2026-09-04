"""The intake deadlock, and the invariant that prevents it coming back.

The console used to carry an unfinished tasking forward as a concatenated
string and re-extract from the whole blob every turn. Once the analyst named a
protectee, that name was re-extracted forever, the WHO question re-fired
forever, and `on_send` returned before ever reaching the assessment. The system
looked frozen because it *was* frozen — there was no sequence of analyst
messages that could clear intake.

The invariant that fixes it, and that these tests defend, is monotonicity:
**more turns can only mean fewer open questions, never more.**
"""

from datetime import date

import pytest

from threat_detector.guardrails import intake_gaps
from threat_detector.tasking_state import TaskingSlots


@pytest.fixture
def slots():
    return TaskingSlots()


# --- the regression --------------------------------------------------------
def test_intake_terminates_on_the_conversation_that_used_to_deadlock(
        slots, synthetic_cfg):
    """The exact three turns from the bug report."""
    turns = [
        dict(city="Dublin"),
        dict(protectee_name="Daniel Okonkwo", protectee_role="CEO"),
        dict(venue="Convention Centre Dublin", event_date=date(2026, 10, 14)),
    ]
    open_counts = []
    for upd in turns:
        slots.update(**upd)
        open_counts.append(len(intake_gaps(slots, synthetic_cfg).questions))

    assert open_counts == sorted(open_counts, reverse=True), \
        f"questions must never increase, got {open_counts}"
    assert open_counts[-1] == 0, "intake must be satisfiable"
    assert slots.ready


def test_a_filled_slot_is_never_asked_about_again(slots, synthetic_cfg):
    """The deadlock's mechanism: answering re-triggered the same question."""
    slots.update(protectee_name="Daniel Okonkwo")
    asked_once = intake_gaps(slots, synthetic_cfg).questions
    assert not any("protectee" in q.lower() for q in asked_once)

    # Ten more turns that mention the name again, as a real analyst would.
    for _ in range(10):
        slots.add_turn("Daniel Okonkwo is speaking there")
        slots.update(protectee_name="Daniel Okonkwo")
    assert not any("protectee" in q.lower()
                   for q in intake_gaps(slots, synthetic_cfg).questions)


def test_turns_are_recorded_but_never_reparsed(slots):
    """The audit trail must not feed back into extraction — that coupling is
    what made the old console's state depend on its own history."""
    slots.add_turn("assess Maria Chen in Berlin")
    slots.update(city="Dublin")
    assert slots.city == "Dublin"          # not overwritten by the older turn
    assert slots.protectee_name is None    # the name in `turns` was NOT extracted
    assert len(slots.turns) == 1


# --- update vs. override ---------------------------------------------------
def test_update_only_fills_blanks(slots):
    slots.update(city="Dublin")
    assert slots.update(city="Munich") == []
    assert slots.city == "Dublin"


def test_override_retargets(slots):
    """'actually, make it Munich' is a change of target, not a re-parse."""
    slots.update(city="Dublin", protectee_name="Daniel Okonkwo")
    changed = slots.override(city="Munich")
    assert changed == ["city"]
    assert slots.city == "Munich"
    assert slots.protectee_name == "Daniel Okonkwo"   # unrelated slots survive


def test_retask_keeps_the_transcript_but_drops_the_target(slots):
    slots.add_turn("first tasking")
    slots.update(city="Dublin", protectee_name="Daniel Okonkwo",
                 event_date=date(2026, 10, 14))
    fresh = slots.retask()
    assert fresh.turns == ["first tasking"]
    assert fresh.missing() == ["who", "where", "when"]


# --- what gets handed to the orchestrator ----------------------------------
def test_as_tasking_is_one_clean_sentence(slots):
    """The orchestrator re-parses its tasking, so it must not receive the
    semicolon-joined pile of half-sentences the old console built."""
    slots.update(protectee_name="Daniel Okonkwo", protectee_role="CEO",
                 city="Dublin", venue="Convention Centre",
                 event_date=date(2026, 10, 14))
    out = slots.as_tasking()
    assert ";" not in out
    for token in ("Daniel Okonkwo", "CEO", "Dublin", "Convention Centre",
                  "2026-10-14"):
        assert token in out


def test_as_tasking_survives_a_partly_empty_slot_set(slots):
    slots.update(city="Dublin")
    assert "Dublin" in slots.as_tasking()
    assert "None" not in slots.as_tasking()
