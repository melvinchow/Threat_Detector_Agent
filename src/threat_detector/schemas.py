"""Typed data contracts shared across the whole system.

Every collection tool returns a ``Finding`` (or a list of them). The Risk
Assessment agent consumes only ``Finding`` objects — it never sees raw web
pages or free text. Because the schema is fixed, a synthetic fixture and a real
API adapter are interchangeable: whichever produces the ``Finding``, the rest of
the pipeline is identical.

Keeping these as Pydantic models (rather than loose dicts) is what lets the
system catch a malformed specialist result at the boundary instead of deep
inside the scoring logic.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ThreatChannel(str, Enum):
    """The kind of risk a finding speaks to.

    Mirrors the two risk classes named in the Module 2 worked trace
    (location-specific vs. actor-specific), plus the reputational/exposure
    channels the profile is meant to protect.
    """

    PHYSICAL_LOCATION = "physical_location"   # something about where the VIP will be
    HOSTILE_ACTOR = "hostile_actor"           # a person/group expressing intent
    REPUTATION = "reputation"                 # sentiment / narrative shift
    EXPOSURE = "exposure"                     # PII / itinerary leaked or discoverable


class SourceType(str, Enum):
    NEWS = "news"
    SOCIAL = "social"
    COURT_DOCKET = "court_docket"
    INCIDENT_LOG = "incident_log"
    PROTEST_PERMIT = "protest_permit"
    DATA_BROKER = "data_broker"
    BREACH_CORPUS = "breach_corpus"
    DARK_WEB = "dark_web"
    GEOSPATIAL = "geospatial"


class Finding(BaseModel):
    """One atomic, attributable observation.

    The point of forcing everything through this shape is *anti-hallucination*:
    the Risk Assessment agent can only score claims that carry a real source and
    date. It has no tool to fetch a new source mid-analysis, so it cannot invent
    one.
    """

    channel: ThreatChannel
    source_type: SourceType
    source_id: str = Field(..., description="Which source produced this, for reliability lookup.")
    summary: str = Field(..., description="One-line, human-readable claim.")
    observed_at: datetime = Field(default_factory=_utcnow)
    event_date: Optional[date] = Field(
        None, description="Forward-looking findings (e.g. a filed protest permit) carry the future date."
    )
    # 0-1 how strongly this finding, if true, indicates elevated risk.
    severity: float = Field(0.0, ge=0.0, le=1.0)
    # Free-form structured payload (coordinates, distances, post URLs, etc.).
    detail: dict = Field(default_factory=dict)

    # Filled in by the orchestrator during Observe, from source-reliability memory.
    # 1.0 = fully trusted; low values down-weight the finding in scoring.
    reliability: float = Field(1.0, ge=0.0, le=1.0)


class Tier(str, Enum):
    """Storage tier for an ingested document (Module 3 hot/warm/cold design).

    * ``HOT``  — full text **and** embeddings; everything ingested; ~90-day window.
    * ``WARM`` — scored above a low relevance floor: keep the full text, drop the
                 embeddings (cheap to re-embed on demand).
    * ``COLD`` — below the floor: keep only a stub (source, timestamp, author,
                 entity IDs, content hash). ~200 bytes. Enough to establish a
                 pattern of interest later without storing the content.
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class AccessClass(str, Enum):
    """Trust-boundary tag carried on every chunk's metadata.

    ``PII`` chunks are isolated so the Exposure agent's material never enters the
    general-assessment retrieval path — the same trust boundary the profile
    enforces, now enforced at retrieval time by the metadata pre-filter.
    """

    PUBLIC = "public"
    PII = "pii"


class Chunk(BaseModel):
    """One retrievable unit of text plus the structured metadata the pre-filter
    scopes on *before* any similarity math runs.

    The metadata is load-bearing: ``entity_ids`` is what keeps a query about the
    protectee from retrieving threatening language aimed at a different executive
    of the same name. Structured facts (coordinates, dates) live in metadata, not
    in the embedded text — embedding them would answer "incidents within 500m"
    badly; the SQLite facts store answers that instead.
    """

    chunk_id: str
    doc_id: str
    text: str
    source_type: SourceType
    source_id: str
    index_name: str = Field("live", description="'live' (ingest) or 'case' (curated memory).")
    entity_ids: list[str] = Field(default_factory=list)
    authored_at: Optional[date] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    access_class: AccessClass = AccessClass.PUBLIC
    reliability: float = Field(0.6, ge=0.0, le=1.0)
    tier: Tier = Tier.HOT
    # Present only while HOT/WARM; None once demoted to COLD (stub only).
    embedding: Optional[list[float]] = None
    content_hash: str = ""
    # Free-form (thread parent, url, author handle, etc.).
    meta: dict = Field(default_factory=dict)


class Document(BaseModel):
    """A source document before chunking. Chunking is dispatched by
    ``source_type`` (see chunking.py) — a news article, a social post, and a
    court filing are segmented differently on purpose."""

    doc_id: str
    text: str
    source_type: SourceType
    source_id: str
    index_name: str = "live"
    entity_ids: list[str] = Field(default_factory=list)
    authored_at: Optional[date] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    access_class: AccessClass = AccessClass.PUBLIC
    reliability: float = Field(0.6, ge=0.0, le=1.0)
    title: str = ""
    meta: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk that survived the retrieval pipeline, with its provenance scores
    so a grader can see *why* it was retrieved (pre-filter → hybrid → rerank)."""

    chunk: Chunk
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    fused_score: float = 0.0
    rerank_score: float = 0.0


class RetrievalResult(BaseModel):
    """Output of one retrieval query.

    ``status`` implements the Module 3 #5 rule: the agent reports
    'no_supporting_evidence' rather than 'no threat' when the search comes back
    empty. Absence of evidence is not evidence of absence.
    """

    query: str
    status: str = "ok"                     # "ok" | "no_supporting_evidence"
    candidates_considered: int = 0
    results: list[RetrievedChunk] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RiskScore(BaseModel):
    """Output of the Risk Assessment agent. Computed only from Findings."""

    overall: float = Field(..., ge=0.0, le=1.0, description="Blended 0-1 risk level.")
    by_channel: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    top_findings: list[Finding] = Field(default_factory=list)


class Recommendation(BaseModel):
    """A proposed action. `requires_approval` is always True for anything that
    changes the world — the Recommendation agent proposes, a human executes."""

    action: str
    channel: ThreatChannel
    rationale: str
    requires_approval: bool = True
    estimated_cost: Optional[str] = None


class Expenditure(BaseModel):
    """One node of the ToT protection-plan tree (Module 4): a single security
    expenditure — what is bought, from whom, for which day, at what cost. A
    branch is a chronological list of these, like line items on a card statement."""

    day: int = Field(..., ge=1, description="Trip day this expenditure covers (1-based).")
    requirement: str = Field(..., description="movement | stationary | monitoring")
    vendor_id: str
    vendor_name: str
    cost: float = Field(..., ge=0.0)
    note: str = ""


class ProtectionPlan(BaseModel):
    """One complete branch that survived the beam — a full multi-day protection
    plan. The planner returns a *slate* of these (the surviving beam), because an
    analyst wants options with trade-offs, not a single answer from a black box.

    ``complete=False`` plans are the Module 5 'give-up' output: explicitly
    labeled incomplete with ``uncovered`` listing what went unresolved — never
    silently dropped, never passed off as a full plan."""

    plan_id: str
    line_items: list[Expenditure] = Field(default_factory=list)
    total_cost: float = 0.0
    budget: float = 0.0
    score: float = Field(0.0, description="Beam ranking score (soft criteria + budget health).")
    complete: bool = True
    uncovered: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class Briefing(BaseModel):
    """The final deliverable handed back to the analyst."""

    tasking: str
    protectee: str
    generated_at: datetime = Field(default_factory=_utcnow)
    score: RiskScore
    findings: list[Finding]
    recommendations: list[Recommendation]
    # Slate of complete protection plans from the ToT beam planner (Module 4),
    # populated when the tasking involves travel and risk is elevated. Every
    # plan is propose-only — nothing is booked until the analyst approves.
    plans: list[ProtectionPlan] = Field(default_factory=list)
    trace: list[str] = Field(
        default_factory=list,
        description="The orchestrator's Think/Act/Observe/Adapt log, for auditability.",
    )
    # LLM mode only: the model's written threat assessment, grounded in (and
    # only in) the collected findings. Empty in deterministic mode.
    narrative: str = ""
