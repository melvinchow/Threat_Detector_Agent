"""Tools, grouped by the agent that owns them.

Grouping mirrors the Module 2 agent roster. The three grouping rules from that
checkpoint:
  1. Trust boundary  — PII-touching tools live only in `exposure`.
  2. Shared schema   — tools whose findings are reasoned over the same way sit
                       together (news + social; geocode + distance).
  3. Read vs. write  — collection tools are read-only; `recommendation` is the
                       only place actions originate, and even there they're gated.
"""

from . import (  # noqa: F401
    exposure,
    geospatial,
    open_source,
    public_records,
    recommendation,
    risk,
)
