"""Accumulated tasking state — the slots the system must fill before it runs.

Why this module exists
----------------------
The console used to carry an unfinished tasking forward as a *string*::

    tasking_text = f"{pending}; {user_text}"       # app.py, before this module

and then re-ran extraction over the whole blob on every turn. That is a
deadlock, not a conversation. Extraction is not idempotent over an accumulating
string: the moment the analyst named a protectee, that name was re-extracted on
every subsequent turn, the intake guardrail re-raised the same question, and the
assessment could never start. Answering the question re-triggered it (the answer
contains the name); not answering it left it open. There was no third move.

The fix is to keep the *answers*, not the transcript. A slot, once filled, stays
filled — so a question can be asked at most once, and ``ready`` is monotonic:
turns only ever add information.

``turns`` keeps the raw messages for the audit trail and for the orchestrator's
narrative context. Nothing is ever re-parsed out of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

# The three MAJOR gaps from the Module 6 intake rule. Anything else (venue,
# budget) is a minor gap the run states as an assumption and proceeds under.
MAJOR_SLOTS = ("who", "where", "when")

_SLOT_OF = {"who": "protectee_name", "where": "city", "when": "event_date"}


@dataclass
class TaskingSlots:
    """What the analyst has told us so far, one field per question we may ask."""

    protectee_name: str | None = None
    protectee_role: str | None = None
    city: str | None = None
    venue: str | None = None
    event_date: date | None = None
    # Raw analyst messages, oldest first. Audit trail only — NEVER re-parsed.
    turns: list[str] = field(default_factory=list)
    # Which slot each value came from ('analyst' | 'profile' | 'regex' | 'model'),
    # so the GUI can show the analyst what was inferred rather than told.
    origin: dict[str, str] = field(default_factory=dict)

    # -- state ---------------------------------------------------------------
    def missing(self) -> list[str]:
        """The major gaps still open, in the order they should be asked."""
        return [k for k in MAJOR_SLOTS if getattr(self, _SLOT_OF[k]) in (None, "")]

    @property
    def ready(self) -> bool:
        return not self.missing()

    def filled(self) -> dict[str, str]:
        """Human-readable view of what is known, for the short-term panel."""
        out: dict[str, str] = {}
        for label, value in (("protectee", self.protectee_name),
                             ("role", self.protectee_role),
                             ("city", self.city),
                             ("venue", self.venue),
                             ("date", self.event_date.isoformat()
                              if self.event_date else None)):
            if value:
                src = self.origin.get(label, "analyst")
                out[label] = f"{value}  _({src})_"
        return out

    # -- updates -------------------------------------------------------------
    def add_turn(self, text: str) -> None:
        if text and text.strip():
            self.turns.append(text.strip())

    def update(self, **values) -> list[str]:
        """Fill EMPTY slots only, and report which ones changed.

        Filling only empties is what makes the loop terminate: a later turn can
        add information but cannot un-answer a question. Genuine corrections
        (a different city, a new protectee) come through ``retask`` instead,
        which is reached via the router's ``new_tasking`` intent — an explicit
        change of target, not an accident of re-parsing.
        """
        changed = []
        for key, value in values.items():
            if value in (None, "") or not hasattr(self, key):
                continue
            if getattr(self, key) in (None, ""):
                setattr(self, key, value)
                changed.append(key)
        return changed

    def override(self, **values) -> list[str]:
        """Replace slots outright — the analyst explicitly changed the target."""
        changed = []
        for key, value in values.items():
            if value in (None, "") or not hasattr(self, key):
                continue
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed.append(key)
        return changed

    def retask(self) -> "TaskingSlots":
        """A fresh tasking that keeps only the conversation's audit trail."""
        return TaskingSlots(turns=list(self.turns))

    def copy(self) -> "TaskingSlots":
        return replace(self, turns=list(self.turns), origin=dict(self.origin))

    # -- output --------------------------------------------------------------
    def as_tasking(self) -> str:
        """One clean sentence for the orchestrator.

        The orchestrator re-parses its tasking (``parse_tasking``) and the
        intake guardrail runs again inside ``assess``, so what we hand over must
        be unambiguous — not a semicolon-joined pile of half-sentences.
        """
        who = self.protectee_name or "the protectee"
        if self.protectee_role:
            who += f" ({self.protectee_role})"
        parts = [f"Threat assessment for {who}"]
        if self.venue:
            parts.append(f"at {self.venue}")
        if self.city:
            parts.append(f"in {self.city}")
        if self.event_date:
            parts.append(f"on {self.event_date.isoformat()}")
        return " ".join(parts) + "."


__all__ = ["TaskingSlots", "MAJOR_SLOTS"]
