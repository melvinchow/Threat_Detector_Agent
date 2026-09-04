"""VIP profile: the protectee, their close relationships, and their sensitive PII.

Just as the ai-job-search agent builds a profile of *you* before it looks for
jobs, this system builds a profile of the person (or small group) it is
protecting before it assesses anything. The profile is the input the whole crew
is oriented around.

**Trust boundary.** Only the Exposure agent is ever handed the ``pii`` block.
Every other agent receives a *redacted* view (name and public affiliation only).
This is enforced in code here — ``Profile.public_view()`` — not left to
convention, because the PII (home address, family, credentials) is exactly what
the system is trying to keep from leaking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Relationship:
    name: str
    relation: str          # e.g. "spouse", "child", "chief of staff"
    protect: bool = True    # is this person also under our protection?


@dataclass
class PII:
    """Sensitive fields. NEVER logged, NEVER handed to any agent but Exposure."""

    home_addresses: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    # Identifiers the Exposure agent scans breach/broker corpora *for* — we hold
    # them so we can detect when they appear somewhere they shouldn't.
    known_usernames: list[str] = field(default_factory=list)


@dataclass
class Profile:
    name: str
    role: str                              # public affiliation, e.g. "CEO, Vantage Robotics"
    is_public_figure: bool = True
    relationships: list[Relationship] = field(default_factory=list)
    known_adversaries: list[str] = field(default_factory=list)  # named groups/individuals
    pii: PII = field(default_factory=PII)
    # Stable identifier used by the retrieval pre-filter to scope every query to
    # THIS protectee — the hard exclusion that keeps a query about our CEO from
    # returning threats aimed at a different person of the same name. Defaults to
    # a slug of the name if not supplied.
    entity_id: str = ""
    # Soft preferences consumed by the ToT planner (Module 4), e.g.
    # {"hotel_brands": ["Meridian"]}. Deliberately weighted LOW there: a VIP who
    # always books the same chain is predictable, and predictability is exposure.
    preferences: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = _slug(self.name)

    # --- trust boundary ----------------------------------------------------
    def public_view(self) -> dict:
        """Redacted profile safe to hand to non-Exposure agents (no PII)."""
        return {
            "name": self.name,
            "role": self.role,
            "is_public_figure": self.is_public_figure,
            "relationships": [
                {"name": r.name, "relation": r.relation} for r in self.relationships
            ],
            "known_adversaries": list(self.known_adversaries),
        }

    # --- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Profile":
        raw = json.loads(path.read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Profile":
        return cls(
            name=raw["name"],
            role=raw["role"],
            is_public_figure=raw.get("is_public_figure", True),
            relationships=[Relationship(**r) for r in raw.get("relationships", [])],
            known_adversaries=list(raw.get("known_adversaries", [])),
            pii=PII(**raw.get("pii", {})),
            entity_id=raw.get("entity_id", ""),
            preferences=dict(raw.get("preferences", {})),
        )


def build_profile_interactively(input_fn=input, print_fn=print) -> Profile:
    """Walk the analyst through creating a profile (the `cli profile` flow).

    ``input_fn`` / ``print_fn`` are injected so this is testable without a TTY.
    """
    print_fn("\n=== Build a VIP protection profile ===")
    print_fn("(Use synthetic/test data only — never a real person's PII.)\n")

    name = input_fn("Protectee name: ").strip()
    role = input_fn("Public role/affiliation: ").strip()

    rels: list[Relationship] = []
    print_fn("\nClose relationships also worth watching (blank name to finish):")
    while True:
        rn = input_fn("  relation name: ").strip()
        if not rn:
            break
        rr = input_fn("  relation type (spouse/child/aide/...): ").strip() or "associate"
        rels.append(Relationship(name=rn, relation=rr))

    adversaries_raw = input_fn(
        "\nKnown adversaries / opposition groups (comma-separated, optional): "
    ).strip()
    adversaries = [a.strip() for a in adversaries_raw.split(",") if a.strip()]

    print_fn("\nSensitive PII (held ONLY by the Exposure agent). Comma-separated, all optional.")
    addrs = _csv(input_fn("  home address(es): "))
    phones = _csv(input_fn("  phone number(s): "))
    emails = _csv(input_fn("  email(s): "))
    usernames = _csv(input_fn("  known usernames/handles to monitor: "))

    return Profile(
        name=name,
        role=role,
        relationships=rels,
        known_adversaries=adversaries,
        pii=PII(
            home_addresses=addrs,
            phone_numbers=phones,
            emails=emails,
            known_usernames=usernames,
        ),
    )


def _csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _slug(name: str) -> str:
    return "_".join(part for part in "".join(
        ch.lower() if ch.isalnum() else " " for ch in name
    ).split())
