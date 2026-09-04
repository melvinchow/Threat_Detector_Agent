"""app.py — The analyst GUI. Run this file:

    python app.py            (after: pip install -e ".[gui]")

Layout (same teaching pattern as the TA's Local-Agent-Demo):

    +---------------------------+----------------------------------------------+
    |                           | [Source Reliability] [Travel Planner] [Map]  |
    |   chat with the system    |                                              |
    |   (type a tasking)        |   live panels showing what the agentic       |
    |                           |   system is actually DOING                   |
    |  [ type here ]   [send]   |                                              |
    +---------------------------+----------------------------------------------+

The chat is a REAL conversation with a model (`conversation.py`): it routes
what you type, fills the tasking slots from your own words, phrases its own
clarifying questions, and answers follow-ups grounded in the findings it
actually collected. Nothing in the chat pane is a canned string — if the model
is down you get an error saying so, never a template pretending to be an answer.
It also cannot claim to be doing work: collection is synchronous, so a reply
that mentions results is a reply that has them.

**The panels update on EVERY turn**, not only after an assessment. That is the
TA demo's lesson — the panels are the point, the chat is how you poke it — and
it is enforced structurally: every branch of `on_send` returns `_render_all`,
so no code path can leave a stale tab behind.

The four analyst tabs are always on; the four developer tabs sit to their
right and stay hidden until you press **Debugging tools**. An analyst should
not have to walk past a token-window readout to reach the map.

Analyst tabs:

* **Source Reliability** — what survives restarts: the trust score you have
  given each source (persisted to disk), the case-memory index, and the active
  protectee (public view only — the PII block is withheld from every agent
  except Exposure).
* **Travel Planner** — the plan slate, and under it the beam search that
  produced it. Approve / reject / re-plan at the top: that is the
  recommendation<->analyst conditional edge, as buttons.
* **Map** — follows the conversation: the moment you name a city it geocodes
  and centres there. Facts-store pins are scoped to THIS protectee and to the
  radius of the tasked city, so another scenario's pins cannot bleed in.
* **Data Sources** — what each specialist actually returned this run: live,
  curated fixture, generated fixture, or not run and why.

Developer tabs (behind **Debugging tools**):

* **Short-term** — the bounded window the model actually sees this turn, plus
  the tasking slots as they fill. Dies with the session.
* **Trace** — every routing decision, slot update, guardrail gap and tool
  dispatch, in order, with elapsed time. The whole session, not one run.
* **Evals** — groundedness, escalation and corroboration rates across the run.
* **Agents** — the roster and the loop limits.

Model backend is `config.yaml → llm.backend`, re-read on every message:
`ollama` (local, free, offline, no key), `claude_cli` (your Claude
subscription), `anthropic` (API key), or `huggingface`. Tools, guardrails, the
risk rubric and the ToT beam stay deterministic on every backend — that split
is the Module 6 safety layer, and it is why the tests can still grade this.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import html as html_mod

import gradio as gr

from threat_detector import evals, guardrails, llm_backends, ratelimit
from threat_detector.agent_loop import LLMOrchestrator
from threat_detector.config import load_config, reload_config
from threat_detector.conversation import ConversationAgent
from threat_detector.llm_backends import LLMUnavailable
from threat_detector.memory import ReliabilityMemory
from threat_detector.orchestrator import Orchestrator
from threat_detector.planner import plan_protection
from threat_detector.profile import PII, Profile
from threat_detector.schemas import Briefing
from threat_detector.tasking_state import TaskingSlots
from threat_detector.tools import geospatial, risk

CFG = load_config()


def _refresh_cfg():
    """Re-read config.yaml and .env before every run, so edits apply WITHOUT
    restarting the app (switch backend, fix a key, change models — just send
    the next message)."""
    global CFG
    CFG = reload_config()
    return CFG


def _mode_line(cfg) -> str:
    if not cfg.llm_enabled:
        return "mode: deterministic (no LLM — flip `llm.enabled: true` in config.yaml)"
    from threat_detector.llm_backends import normalize_model
    return (f"mode: **LLM agent loop** · backend `{cfg.llm_backend}` · "
            f"orchestrator `{normalize_model(cfg.orchestrator_model)}` · "
            f"specialists `{normalize_model(cfg.specialist_model)}`")


PROFILE_PATH = CFG.path("profile")
DEFAULT_PROFILE = Profile.load(PROFILE_PATH if PROFILE_PATH.exists()
                               else CFG.fixture("vip_profile.example.json"))


# ---------------------------------------------------------------------------
# SESSION STATE — everything the panels read, in one place
# ---------------------------------------------------------------------------
def _new_state() -> dict:
    return {"slots": TaskingSlots(), "session_trace": [], "t0": time.time(),
            "agent": None, "orch": None, "briefing": None, "metrics": None,
            "recall": None, "decision": "", "awaiting": False,
            "last_intent": None, "profile": DEFAULT_PROFILE}


def _trace(state: dict, msg: str) -> None:
    """One session-wide event log. The Trace tab is this list, rendered."""
    state["session_trace"].append(f"+{time.time() - state['t0']:6.1f}s  {msg}")


def _profile_of(state: dict) -> Profile:
    return state.get("profile") or DEFAULT_PROFILE


def _session_profile(slots: TaskingSlots) -> Profile:
    """The protectee for THIS tasking.

    Named in chat, not locked to a file. When the name matches the profile on
    disk we use that one — PII block and all, so Exposure can run. When it does
    not, we build an ad-hoc profile with an EMPTY PII block: an analyst naming
    someone in conversation has not given us their home address, and inventing
    one would be the exact harm this system exists to prevent. Exposure then
    reports that it could not run, which is the honest answer.
    """
    name = (slots.protectee_name or "").strip()
    if not name or name.lower() == DEFAULT_PROFILE.name.lower():
        return DEFAULT_PROFILE
    return Profile(name=name,
                   role=slots.protectee_role or "role not stated",
                   is_public_figure=True, pii=PII())


# ---------------------------------------------------------------------------
# PANEL RENDERERS — turn session state into readable markdown
# ---------------------------------------------------------------------------
def render_stm(state: dict | None) -> str:
    """SHORT-TERM MEMORY: the window the model sees, and the tasking slots.

    Two things live here, and the split is the teaching point. The *transcript*
    is what the model is shown this turn — bounded, so watch it drop the oldest
    turn once it fills. The *slots* are what the conversation has actually
    established; unlike the transcript they are structured, and once a slot is
    filled it stays filled. That is what stops the intake loop repeating itself.
    """
    header = ("### Short-term memory (this session only)\n"
              "_Dies when you close the app. Compare with the Source Reliability tab, "
              "which survives restarts._\n\n")
    if not state:
        return header + "_empty — say something to start_"

    slots: TaskingSlots = state.get("slots") or TaskingSlots()
    out = [header, "**Tasking slots** — filled by the model from your own words; "
                   "a filled slot is never re-asked:\n"]
    out.append("| slot | value |\n|---|---|")
    filled = slots.filled()
    for label in ("protectee", "role", "city", "venue", "date"):
        out.append(f"| {label} | {filled.get(label, '_— not yet given_')} |")
    missing = slots.missing()
    out.append(f"\n**Still required:** "
               + (", ".join(f"`{m}`" for m in missing) if missing
                  else "nothing — ready to assess ✅"))

    agent = state.get("agent")
    limit = ConversationAgent.MAX_TURNS
    hist = list(agent.history) if agent else []
    out.append(f"\n---\n**Conversation window** — `{len(hist)} / {limit}` turns "
               f"sent to the model (`MAX_TURNS` in `conversation.py`).")
    if len(hist) >= limit:
        out.append("> **FULL.** Every new turn now pushes an old one out "
                   "permanently — the model stops being able to see it.")
    if not hist:
        out.append("\n_nothing in the window yet_")
    else:
        out.append("\n| role | content |\n|---|---|")
        for m in hist:
            text = m["content"].replace("\n", " ").replace("|", "\\|")
            out.append(f"| {m['role']} | {text[:90]}{'…' if len(text) > 90 else ''} |")

    orch = state.get("orch")
    if orch is not None and orch.stm is not None:
        out.append(f"\n---\n**Last run's scratchpad** (rebuilt per tasking)\n")
        out.append(f"- **Tasking:** {orch.stm.tasking}")
        out.append(f"- **Focus:** `{orch.stm.focus}` (prunes which specialists run)")
        out.append(f"- **Specialists answered:** "
                   f"{', '.join(sorted(orch.stm.answered)) or '—'}")
        for n in orch.stm.notes:
            out.append(f"  - {n}")
    return "\n".join(out)


def render_ltm(state: dict | None = None) -> str:
    """SOURCE RELIABILITY: reliability scores + case memory + profile.

    This is the long-term memory tier. It is named for the part an analyst
    actually operates — the trust score on each source — because "long-term
    memory" told a first-time user nothing about what the tab was for.
    """
    out = ["### Source Reliability — the memory that survives restarts"]

    rel = ReliabilityMemory(CFG.path("reliability"))
    out.append("\n**1. Source reliability** — the analyst's accumulated trust, "
               f"persisted in `{CFG.path('reliability').name}`. A source you "
               "down-weight here stays down-weighted on every future run "
               "(the fix for alert fatigue):\n")
    out.append("| source | reliability |\n|---|---|")
    for sid, score in sorted(rel._scores.items()):
        if not isinstance(score, (int, float)):    # the file's _comment key
            continue
        flag = " ⚠️ below min" if score < CFG.min_reliability else ""
        out.append(f"| `{sid}` | {score:.2f}{flag} |")

    corpus = json.loads(CFG.retrieval_path("corpus").read_text())["documents"]
    case = [d for d in corpus if d.get("index") == "case"]
    live = [d for d in corpus if d.get("index") != "case"]
    out.append(f"\n**2. Case memory (RAG)** — curated, never expires: "
               f"`{len(case)}` case document(s) + `{len(live)}` live-ingest "
               f"document(s) (live ages out after {CFG.live_ttl_days} days). "
               "Analyst-confirmed material is promoted live → case:")
    for d in case:
        out.append(f"  - `{d['doc_id']}` ({d.get('authored_at', '?')}): "
                   f"{d['text'][:90]}...")

    prof = _profile_of(state or {})
    pub = prof.public_view()
    out.append(f"\n**3. Active protectee** — {pub['name']} ({pub['role']}), "
               f"entity id `{prof.entity_id}`. "
               f"Known adversaries: {', '.join(pub['known_adversaries']) or '—'}.")
    if prof is DEFAULT_PROFILE:
        out.append("_Loaded from the profile on disk._")
    else:
        out.append("_Defined in this conversation, so it has **no PII block** and "
                   "**no case history** — Exposure and Case Memory will say so "
                   "rather than return a misleading empty result. Persist it with "
                   "`threat-detector profile`._")
    out.append("_The PII block exists but is not shown here — only the Exposure "
               "agent is ever handed it. That trust boundary includes this GUI._")
    return "\n".join(out)


def render_trace(state: dict | None) -> str:
    """TRACE: the whole session — routing, slots, guardrails, tool dispatch."""
    header = ("### Trace — everything the system did, in order\n"
              "_Router decisions, slot updates, guardrail gaps, each specialist "
              "dispatch and each model call. Timestamps are seconds since the "
              "session started._\n\n")
    lines = (state or {}).get("session_trace") or []
    if not lines:
        return header + "_nothing yet — send a message._"
    return header + "\n".join(f"`{l}`" for l in lines)


def render_beam(orch: Orchestrator | None, briefing: Briefing | None,
                decision: str = "") -> str:
    """TRAVEL PLANNER: the slate first, then the search that produced it.

    Deliberate order. The controls above this text are what the analyst came
    here to use, so the options they choose between come next, and the beam
    search reads as the justification *underneath* them. The old order made you
    scroll past several screens of search output to find the three plans.

    The title and blurb live in the UI block, not here, so that the decision
    controls can sit between them and this body.
    """
    if orch is None or not orch.plan_trace:
        return ("_No plan yet. The planner runs when a tasking involves travel "
                "and risk scores ELEVATED or above._")

    out = []
    if briefing and briefing.plans:
        out.append("### Your options (ranked)")
        for p in briefing.plans:
            status = "✅ complete" if p.complete else "⚠️ INCOMPLETE"
            out.append(f"\n**{p.plan_id}** — {status}, cost "
                       f"`{p.total_cost:.0f}` / `{p.budget:.0f}`, score `{p.score:.3f}`")
            for e in p.line_items:
                out.append(f"  - day {e.day} · {e.requirement}: {e.vendor_name} "
                           f"(`{e.cost:.0f}`)")
    if decision:
        out.append(f"\n**Analyst decision:** {decision}")

    out.append("\n---\n### How the planner reached those options\n"
               "_The search, level by level. Each level keeps only the strongest "
               "branches and prunes the rest, which is why three plans come back "
               "and not three hundred._\n")
    for t in orch.plan_trace:
        if t["event"] == "start":
            out.append(f"**Search opened** — budget `{t['budget']:.0f}`, depth "
                       f"`{t['depth']}`, beam width `{t['beam_width']}`. {t['note']}.\n")
        elif t["event"] == "level":
            out.append(f"**Level {t['slot']}** — day {t['day']}, "
                       f"*{t['requirement']}*: expanded {t['expanded']}, "
                       f"pruned by floor-cost lookahead {t['pruned_by_lookahead']}, "
                       f"dropped for diversity {t['dropped_for_diversity']}")
            for i, b in enumerate(t["beam"], 1):
                picks = " → ".join(b["picks"])
                out.append(f"  {i}. `{b['score']:.3f}` spent {b['spent']:.0f} "
                           f"(remaining {b['remaining']:.0f}): {picks}")
            out.append("")
        elif t["event"] == "infeasible":
            out.append(f"**⚠️ INFEASIBLE at {t['at']}** — uncovered: "
                       f"{', '.join(t['uncovered'])}. Returned best-so-far, "
                       "explicitly labeled incomplete.")
        elif t["event"] == "done":
            out.append(f"**Search closed** — {t['complete_plans']} complete "
                       f"plan(s) on the slate. {t['note']}.")
    return "\n".join(out)


# --- Sources ---------------------------------------------------------------
# One row per SOURCE, not per specialist. Open-Source Monitoring owns two
# unrelated pipes — Exa for news, Reddit for social — and merging them into one
# row is a real reporting failure: live news made the cell read "N LIVE" while
# the Reddit half was silently serving fixtures, so the row vouched for a
# collection channel that had never run.
_SPECIALIST_OF = {
    "news": "Open-Source Monitoring · news", "social": "Open-Source Monitoring · social",
    "court_docket": "Public Records", "incident_log": "Public Records",
    "protest_permit": "Public Records", "data_broker": "Exposure",
    "breach_corpus": "Exposure", "dark_web": "Exposure",
    "geospatial": "Geospatial",
}
_SPECIALISTS = ["Open-Source Monitoring · news", "Open-Source Monitoring · social",
                "Public Records", "Exposure", "Geospatial",
                "Case Memory / Retrieval"]


def render_sources(state: dict | None = None) -> str:
    """What each specialist ACTUALLY returned — live, curated, generated, or
    not run. Config says what a source is *allowed* to be; this says what it
    was, which is the only thing an analyst can act on."""
    state = state or {}
    b: Briefing | None = state.get("briefing")
    out = ["### Data Sources — what each specialist actually returned\n"]

    if b is None:
        out.append("_No assessment has run this session, so nothing below has "
                   "been exercised yet. The table shows configuration only._\n")
    counts: dict[str, dict[str, int]] = {s: {"live": 0, "curated": 0, "generated": 0}
                                         for s in _SPECIALISTS}
    fallbacks: dict[str, str] = {}
    if b is not None:
        for f in b.findings:
            if f.detail.get("retrieved"):
                spec = "Case Memory / Retrieval"
            else:
                spec = _SPECIALIST_OF.get(f.source_type.value, "Geospatial")
            kind = ("generated" if f.detail.get("generated_for_city")
                    else "curated" if f.detail.get("synthetic_fixture") else "live")
            counts[spec][kind] += 1
            # A source configured `real` that returned fixtures owes the analyst
            # a reason. Without it, "0 live" reads as "nothing out there".
            if f.detail.get("fallback_reason"):
                fallbacks[spec] = f.detail["fallback_reason"]

    skips = {}
    for line in (b.trace if b else []):
        if "[guardrail]" in line and "NOT run" in line:
            for s in _SPECIALISTS:
                if s.split(" /")[0].split()[0].lower() in line.lower():
                    skips[s] = line.split("]", 1)[1].strip()

    out.append("| specialist | configured | this run |\n|---|---|---|")
    cfg_src = CFG.raw.get("source", {})
    cfg_of = {"Open-Source Monitoring · news": f"news={cfg_src.get('news')} (Exa)",
              "Open-Source Monitoring · social": f"social={cfg_src.get('social')} (Reddit)",
              "Public Records": str(cfg_src.get("public_records")),
              "Exposure": str(cfg_src.get("exposure")),
              "Geospatial": str(cfg_src.get("geospatial")),
              "Case Memory / Retrieval": "local RAG index"}
    for s in _SPECIALISTS:
        c = counts[s]
        if s in skips:
            cell = f"⚠️ **NOT RUN** — {skips[s][:110]}"
        elif b is None:
            cell = "_not exercised yet_"
        elif sum(c.values()) == 0:
            cell = "0 findings — nothing matched (not the same as 'no threat')"
        else:
            bits = []
            if c["live"]:
                bits.append(f"**{c['live']} LIVE**")
            if c["curated"]:
                bits.append(f"{c['curated']} curated fixture")
            if c["generated"]:
                bits.append(f"{c['generated']} generated for this city")
            if s in fallbacks:
                bits.append(f"⚠️ fell back: {fallbacks[s]}")
            cell = " · ".join(bits)
        out.append(f"| {s} | {cfg_of[s]} | {cell} |")

    out.append("\n**Are the real APIs actually wired?**\n")
    out.append("| API | key present | effect |\n|---|---|---|")
    for label, key, note in (
            ("Exa (news)", "EXA_API_KEY", "real news search"),
            ("Reddit (social)", "REDDIT_CLIENT_ID", "real social search"),
            ("OpenStreetMap", None, "geocoding + hotels; no key needed")):
        have = True if key is None else bool(CFG.env(key))
        out.append(f"| {label} | {'✅' if have else '❌ falls back to fixtures'} "
                   f"| {note} |")
    out.append("| VIP profile / PII | — | **always synthetic** (never a real person) |")

    out.append("\n### Today's API budgets (strict on purpose)\n")
    out.append("| API | used today | daily limit |\n|---|---|---|")
    for key, (used, limit) in ratelimit.status(CFG).items():
        warn = " ⚠️" if used >= limit else ""
        out.append(f"| {key} | {used}{warn} | {limit} |")
    out.append("\n_When a budget runs out, the tool falls back to synthetic "
               "fixtures and the run degrades honestly. Edit limits in "
               "`config.yaml → limits:`._")
    return "\n".join(out)


def render_evals(metrics: dict | None, recall: dict | None = None) -> str:
    """Metrics: numbers across runs, not per-item checks."""
    out = ["### Evaluation Metrics\n",
           "_A guardrail passes or fails on one item; a metric tells you whether "
           "the guardrails are calibrated. If escalation climbs, tighten "
           "collection filters — don't loosen the gate._\n"]
    if metrics is None:
        out.append("_Run an assessment to see this run's metrics._")
    else:
        out.append("| metric | value | reading |\n|---|---|---|")
        out.append(f"| groundedness rate | {metrics['groundedness_rate']:.0%} | "
                   "target 95–100%; less = hallucination |")
        out.append(f"| escalation rate | {metrics['escalation_rate']:.0%} | "
                   "too high = alert fatigue; too low = gates catch nothing |")
        out.append(f"| corroboration rate | {metrics['corroboration_rate']:.0%} | "
                   f"{metrics['confidence_capped']} of "
                   f"{metrics['high_severity_claims']} high-severity claim(s) "
                   "confidence-capped |")
        out.append(f"| stale / geo flags | {metrics['stale_flagged']} / "
                   f"{metrics['geo_flagged']} | guardrail catches this run |")
        out.append(f"| cap events | {metrics['cap_events_in_trace']} | "
                   f"fallback success: {metrics['fallback_success']} |")
        out.append(f"| volume | {metrics['findings']} findings, "
                   f"{metrics['trace_steps']} trace steps | |")
    if recall:
        out.append("\n### Retrieval recall (seeded-threat test)")
        out.append(f"**Recall: {recall['recall']:.0%}** — surfaced "
                   f"{len(recall['surfaced'])}/{len(recall['expected'])} planted "
                   "threats.")
        for s in recall["surfaced"]:
            out.append(f"  - ✅ `{s}`")
        for s in recall["missed"]:
            out.append(f"  - ❌ MISSED `{s}` — false-negative risk!")
    else:
        out.append("\n_Click **Run seeded-recall test** to measure retrieval's "
                   "false-negative risk against planted threats._")
    return "\n".join(out)


# --- Map -------------------------------------------------------------------
def _map_points(state: dict) -> tuple[list[dict], tuple[float, float] | None]:
    """Plottable coordinates for the CURRENT conversation.

    Scoped twice, because an unscoped map is actively misleading: the facts
    store holds rows for every scenario in the corpus, and plotting all of them
    put Berlin pins on screen during a conversation about Dublin. A row is drawn
    only if it belongs to this protectee AND falls within the geo-sanity radius
    of the city actually being tasked.
    """
    state = state or {}
    slots: TaskingSlots = state.get("slots") or TaskingSlots()
    b: Briefing | None = state.get("briefing")
    prof = _profile_of(state)
    pts: list[dict] = []
    center: tuple[float, float] | None = None

    # 1. The tasked location itself — drawn as soon as the analyst names it.
    place = None
    if slots.venue and slots.city:
        place = f"{slots.venue}, {slots.city}"
    elif slots.city:
        place = slots.city
    if place:
        try:
            center = geospatial.geocode(place, CFG)
        except Exception:
            center = None
        if center:
            pts.append({"lat": center[0], "lon": center[1], "color": "blue",
                        "label": f"TASKING LOCATION: {place}"})

    # 2. Findings from the run.
    if b:
        for f in b.findings:
            d = f.detail
            if d.get("venue_coords"):
                pts.append({"lat": d["venue_coords"][0], "lon": d["venue_coords"][1],
                            "label": f"VENUE: {d.get('venue', place or '')}",
                            "color": "blue"})
            if d.get("protest_coords"):
                pts.append({"lat": d["protest_coords"][0], "lon": d["protest_coords"][1],
                            "label": f"PROTEST SITE: {d.get('protest_location', '')} "
                                     f"({d.get('distance_km', '?')} km)", "color": "red"})

    # 3. Facts store — scoped to this protectee and this city's radius.
    max_km = float(CFG.raw.get("guardrails", {}).get("geo_max_km", 100))
    try:
        facts = json.loads(CFG.retrieval_path("facts").read_text())["facts"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        facts = []
    for row in facts:
        if not (row.get("lat") and row.get("lon")):
            continue
        if row.get("entity_id") and row["entity_id"] != prof.entity_id:
            continue
        if center is None:
            # No located tasking yet, so there is no radius to judge against.
            # Drawing the facts store anyway is what put Berlin pins on screen
            # during a conversation about Dublin — and on a blank session that
            # had said nothing at all.
            continue
        if geospatial.haversine_km(center, (row["lat"], row["lon"])) > max_km:
            continue
        pts.append({"lat": row["lat"], "lon": row["lon"], "color": "orange",
                    "label": f"{row.get('fact_type', 'fact')}: "
                             f"{row.get('summary', '')[:80]}"})

    seen, unique = set(), []
    for p in pts:
        k = (round(p["lat"], 5), round(p["lon"], 5), p["label"])
        if k not in seen:
            seen.add(k)
            unique.append(p)
    if center is None and unique:
        center = (unique[0]["lat"], unique[0]["lon"])
    return unique, center


def render_map(state: dict | None) -> str:
    """A real GIS view: Leaflet + OpenStreetMap tiles (open source, no key)."""
    pts, center = _map_points(state or {})
    if not pts or center is None:
        return ("<p><em>No mappable location yet. Name a city in the chat and "
                "this map geocodes it immediately — you do not have to wait for "
                "an assessment. Venue, protest sites and facts-store entries for "
                "<b>this</b> protectee near <b>this</b> city are pinned as they "
                "are found (Leaflet + OpenStreetMap, no API key).</em></p>")
    markers_js = ";".join(
        f"L.circleMarker([{p['lat']},{p['lon']}],"
        f"{{radius:8,color:'{p['color']}',fillOpacity:0.7}})"
        f".addTo(m).bindPopup({json.dumps(p['label'])})"
        for p in pts
    )
    doc = f"""<!DOCTYPE html><html><head>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style></head>
<body><div id="map"></div><script>
var m=L.map('map').setView([{center[0]},{center[1]}],13);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'&copy; OpenStreetMap contributors'}}).addTo(m);
{markers_js};
</script></body></html>"""
    return (f'<iframe srcdoc="{html_mod.escape(doc)}" '
            f'style="width:100%;height:440px;border:1px solid #ccc;'
            f'border-radius:8px"></iframe>'
            "<p style='font-size:0.85em'>🔵 tasking location / venue · "
            "🔴 permitted protest sites · 🟠 facts-store entries for this "
            "protectee within the geo-sanity radius. Tiles: OpenStreetMap.</p>")


def render_agents() -> str:
    return """### Agent Roster

Topology: **hierarchical** — one orchestrator over a parallel collection tier
(4 siblings with **zero peer edges** — width, not depth; the critical path is
3–5 agents deep despite 8 agents total).

| Agent | Role | Owns | Edge to orchestrator |
|---|---|---|---|
| Orchestrator | planner | delegation + memory, **no external tools** | — |
| Open-Source Monitoring | researcher | news, social, sentiment, entity resolution | one-way return |
| Public Records | researcher | dockets, incidents, protest permits | one-way return |
| Exposure | researcher | brokers, breach, dark web — **only agent holding PII** | one-way return |
| Geospatial | researcher | geocoding, routing, proximity math | one-way return |
| Case Memory / Retrieval | researcher | RAG: pre-filter → hybrid → rerank + SQLite facts | one-way return |
| Risk Assessment | critic/evaluator | scoring rubric, **no retrieval tools** | **two-way** (corroboration) |
| Recommendation | decision-maker | playbooks + the **nested ToT beam** | conditional → analyst |

Loop limits (every loop = a satisfaction condition **and** a hard cap):
- **Collection**: stop at *saturation* (a wave surfaces zero new entities); cap 3 waves.
- **Corroboration**: stop when no high-severity claim is single-sourced; cap 2 rounds —
  then the claim is reported with severity capped, never dropped.
- **ToT beam**: terminates *by construction* at depth = trip days × requirement types.

Brainstorming is allowed **only inside the ToT beam** — divergent generation
during evidence collection is how a system hallucinates threats."""


def _summary_md(b: Briefing) -> str:
    lines = [f"**Overall risk: {b.score.overall:.2f} "
             f"[{risk.band(b.score.overall)}]** for {b.protectee}\n"]
    for ch, v in sorted(b.score.by_channel.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {ch}: {v:.2f}")
    lines.append("\n**Top evidence:**")
    for r in b.score.rationale[:4]:
        lines.append(f"- {r}")
    uncorro = [f for f in b.findings if f.detail.get("uncorroborated")]
    if uncorro:
        lines.append(f"\n_{len(uncorro)} claim(s) reported single-sourced with "
                     "confidence marked down — see Trace._")
    synth = sum(1 for f in b.findings if f.detail.get("synthetic_fixture"))
    lines.append(f"\n_Provenance: {len(b.findings) - synth} finding(s) from live "
                 f"collection, {synth} from synthetic fixtures (see the Data "
                 f"Sources tab)._")
    lines.append(f"\n**Recommendations ({len(b.recommendations)}):**")
    for r in b.recommendations:
        gate = "🔒 approval required" if r.requires_approval else "auto-ok"
        lines.append(f"- [{gate}] {r.action}")
    if b.plans:
        lines.append(f"\n**Protection plan slate:** {len(b.plans)} option(s) — "
                     "review and approve in the *Travel Planner* tab.")
    return "\n".join(lines)


def _stats(state: dict, cfg) -> str:
    b: Briefing | None = state.get("briefing")
    if b is None:
        slots: TaskingSlots = state.get("slots") or TaskingSlots()
        missing = ", ".join(slots.missing()) or "nothing"
        return (f"{_mode_line(cfg)}\n\n**No assessment yet.** Still needed: "
                f"`{missing}`.")
    return (f"{_mode_line(cfg)}\n\n**Findings:** {len(b.findings)} · "
            f"**Score:** {b.score.overall:.2f} "
            f"[{risk.band(b.score.overall)}] · **Recommendations:** "
            f"{len(b.recommendations)} · **Plans on slate:** {len(b.plans)} · "
            f"**Trace steps:** {len(b.trace)}")


# ---------------------------------------------------------------------------
# THE MAIN EVENT HANDLER
# ---------------------------------------------------------------------------
def _render_all(state: dict, chat, cfg, *, msg: str = "") -> tuple:
    """Every output, every turn.

    The old handler returned `gr.skip()` for most panels on most code paths, so
    a turn that did not run an assessment left every tab showing whatever was
    there at import time — which is exactly what made the console look frozen.
    Building the full tuple in one place makes a stale panel impossible.
    """
    return (msg, chat, state,
            render_trace(state),
            render_stm(state),
            render_ltm(state),
            render_beam(state.get("orch"), state.get("briefing"),
                        state.get("decision", "")),
            _stats(state, cfg),
            render_map(state),
            render_sources(state),
            render_evals(state.get("metrics"), state.get("recall")))


def _conversation(cfg, backend, state) -> ConversationAgent:
    """One conversation agent per session, rebuilt if the backend changed."""
    agent = state.get("agent")
    if agent is None or agent.backend is not backend:
        history = agent.history if agent else []
        agent = ConversationAgent(backend, cfg, _profile_of(state))
        agent.history = history          # a backend swap doesn't erase the thread
        state["agent"] = agent
    agent.cfg = cfg                      # config.yaml is re-read every message
    agent.profile = _profile_of(state)
    return agent


def on_send(user_text, chat, focus, state):
    if not user_text.strip():
        yield _render_all(dict(state or _new_state()), chat, CFG, msg=user_text)
        return

    state = {**_new_state(), **dict(state or {})}
    chat = chat + [{"role": "user", "content": user_text},
                   {"role": "assistant", "content": "_thinking…_"}]

    def emit(md, cfg=None):
        chat[-1]["content"] = md
        return _render_all(state, chat, cfg or CFG)

    # ---- CONFIG + BACKEND, re-read and re-VERIFIED on every message ---------
    cfg = _refresh_cfg()
    _trace(state, f"[analyst] {user_text[:120]}")
    if not cfg.llm_enabled:
        _trace(state, "[error] llm.enabled is false — refusing to fake a reply")
        yield emit(
            "⚠️ **LLM mode is off**, so there is no model to talk to — this "
            "console would only be able to hand you templated text.\n\n"
            "Set `llm.enabled: true` in `config.yaml` and send again "
            "(config is re-read every message; no restart needed).", cfg)
        return

    t = time.time()
    ok, msg, backend = llm_backends.verify(cfg, deep=True)
    _trace(state, f"[backend] {cfg.llm_backend} verify={'OK' if ok else 'FAILED'} "
                  f"({time.time() - t:.1f}s)")
    if not ok:
        # NEVER silently fall back to deterministic output. A canned reply that
        # looks like an LLM is worse than an error that says the model is down.
        yield emit(
            f"⚠️ **LLM backend `{cfg.llm_backend}` is not usable:**\n\n{msg}\n\n"
            "Nothing is being faked in its place. Fix it and send again, or "
            "switch `llm.backend` in `config.yaml` "
            "(`ollama` runs locally and offline). Diagnose with "
            "`python -m threat_detector.cli llm-check --deep`.", cfg)
        return

    agent = _conversation(cfg, backend, state)
    slots: TaskingSlots = state["slots"]
    briefing: Briefing | None = state.get("briefing")
    run_state = agent.run_state(briefing, slots)

    # ---- 1. ROUTE: what does this message actually want? -------------------
    try:
        t = time.time()
        intent, why = agent.route(user_text, has_briefing=briefing is not None,
                                  pending_questions=bool(state.get("awaiting")))
    except LLMUnavailable as e:
        _trace(state, f"[error] router: {e}")
        yield emit(f"⚠️ **The model went away mid-conversation:** {e}", cfg)
        return
    state["last_intent"] = intent
    _trace(state, f"[route] intent={intent} ({time.time() - t:.1f}s) — {why}")

    # ---- 2. THE TWO NON-TASKING INTENTS ------------------------------------
    try:
        if intent == "followup" and briefing is not None:
            yield emit("_reading the briefing…_", cfg)
            t = time.time()
            reply = agent.answer_followup(user_text, briefing, briefing.trace)
            _trace(state, f"[answer] grounded follow-up over "
                          f"{len(briefing.findings)} finding(s) "
                          f"({time.time() - t:.1f}s)")
            agent.remember("user", user_text)
            agent.remember("assistant", reply)
            yield emit(reply + _intent_footer(intent, why, cfg), cfg)
            return

        if intent == "chitchat":
            yield emit("_…_", cfg)
            t = time.time()
            reply = agent.chitchat(user_text, has_briefing=briefing is not None,
                                   run_state=run_state)
            _trace(state, f"[answer] conversation, no collection "
                          f"({time.time() - t:.1f}s)")
            agent.remember("user", user_text)
            agent.remember("assistant", reply)
            yield emit(reply + _intent_footer(intent, why, cfg), cfg)
            return
    except LLMUnavailable as e:
        _trace(state, f"[error] {e}")
        yield emit(f"⚠️ **Model call failed:** {e}\n\nNothing was substituted "
                   f"for it.", cfg)
        return

    # ---- 3. TASKING: fill slots, then the intake guardrail ------------------
    # 'new_tasking' may CHANGE an existing target ("make it Munich"), so it can
    # overwrite a filled slot. 'answer_clarification' only ever fills a blank —
    # which is what makes the intake loop terminate.
    if intent == "new_tasking" and briefing is not None:
        state["slots"] = slots = slots.retask()
        state["briefing"] = briefing = None
        state["orch"] = None
        state["metrics"] = None
        _trace(state, "[slots] new tasking — previous briefing cleared")
    slots.add_turn(user_text)
    try:
        changed = agent.extract_slots(user_text, slots,
                                      override=(intent == "new_tasking"))
    except LLMUnavailable as e:
        _trace(state, f"[error] slot extraction: {e}")
        yield emit(f"⚠️ **Model call failed:** {e}", cfg)
        return
    _trace(state, f"[slots] {'filled ' + ', '.join(changed) if changed else 'no new detail in this message'}"
                  f" — still missing: {', '.join(slots.missing()) or 'nothing'}")
    state["profile"] = _session_profile(slots)
    agent.profile = state["profile"]

    intake = guardrails.intake_gaps(slots, cfg)
    for a in intake.assumptions:
        _trace(state, f"[guardrail] assumption: {a[:110]}")
    if intake.questions:
        # The guardrail decides WHAT is missing (deterministic, Module 6);
        # the model decides how to ASK for it.
        state["awaiting"] = True
        for q in intake.questions:
            _trace(state, f"[guardrail] intake gap: {q[:110]}")
        try:
            reply = agent.ask_clarification(
                slots.as_tasking(), intake,
                run_state=agent.run_state(briefing, slots))
        except LLMUnavailable as e:
            yield emit(f"⚠️ **Model call failed:** {e}", cfg)
            return
        agent.remember("user", user_text)
        agent.remember("assistant", reply)
        yield emit(reply + _intent_footer(intent, why, cfg), cfg)
        return
    state["awaiting"] = False

    # ---- 4. RUN THE MULTI-AGENT ASSESSMENT ---------------------------------
    tasking_text = slots.as_tasking()
    profile = state["profile"]
    _trace(state, f"[dispatch] intake satisfied — assessing: {tasking_text}")
    _trace(state, f"[dispatch] protectee={profile.name} entity_id={profile.entity_id} "
                  f"pii_on_file={'yes' if profile.pii.emails or profile.pii.home_addresses else 'no'}")

    q: queue.Queue = queue.Queue()
    orch = LLMOrchestrator(profile, cfg, on_event=q.put, backend=backend)
    state["orch"] = orch
    result: dict = {}

    def work():
        try:
            result["briefing"] = orch.assess(tasking_text, focus=focus)
        except Exception as e:                        # surface, never swallow
            result["error"] = f"{type(e).__name__}: {e}"
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            break
        _trace(state, item)
        chat[-1]["content"] = (f"_assessing: {tasking_text}_\n\n{_mode_line(cfg)}\n\n"
                               f"`{item}`")
        yield _render_all(state, chat, cfg)

    if "error" in result:
        _trace(state, f"[error] assessment failed: {result['error']}")
        yield emit(f"⚠️ **Assessment failed:** {result['error']}\n\n"
                   f"The run is not being replaced with canned output — see the "
                   f"Trace tab for how far it got.", cfg)
        return

    b: Briefing = result["briefing"]
    state["briefing"] = b
    state["metrics"] = evals.evaluate_briefing(b, CFG)
    state["decision"] = ""

    # The chat reply is MODEL PROSE: a conversational hand-off written now,
    # then the full assessment the orchestrator wrote during the run. The
    # structured template is demoted to a collapsed appendix — it is a data
    # dump, not the answer.
    try:
        opener = agent.summarize_briefing(b, tasking_text)
    except LLMUnavailable as e:
        opener = f"_(couldn't write the hand-off: {e})_"
    reply = opener
    if b.narrative:
        reply += f"\n\n---\n\n{b.narrative}"
    reply += (f"\n\n<details><summary>Structured briefing (scores, findings, "
              f"recommendations)</summary>\n\n{_summary_md(b)}\n\n</details>")
    reply += _intent_footer(intent, why, cfg)

    agent.remember("user", user_text)
    agent.remember("assistant", opener + ("\n\n" + b.narrative if b.narrative else ""))
    _trace(state, f"[done] {len(b.findings)} finding(s), score {b.score.overall:.2f}")

    yield emit(reply, cfg)


def _intent_footer(intent: str, why: str, cfg) -> str:
    """Show the routing decision. The analyst should be able to see that a
    model classified the message — and disagree with it out loud."""
    from threat_detector.llm_backends import normalize_model
    return (f"\n\n<sub>routed as `{intent}` by `"
            f"{normalize_model(cfg.specialist_model)}` — {why}</sub>")


# ---------------------------------------------------------------------------
# ANALYST CONTROLS — the conditional edges, as buttons
# ---------------------------------------------------------------------------
def on_decide(state, plan_id, decision):
    """recommendation <-> analyst conditional edge: approve / reject."""
    if not state or not state.get("briefing") or not state["briefing"].plans:
        return gr.skip()
    b = state["briefing"]
    if decision == "approve":
        note = (f"✅ APPROVED `{plan_id}` — booking may now proceed (outside this "
                "system; the search itself never reserves anything).")
    else:
        note = (f"❌ REJECTED `{plan_id}` — revise constraints below and re-plan, "
                "or adjust the tasking.")
    state["decision"] = note
    return render_beam(state["orch"], b, decision=note)


def on_replan(state, budget, days):
    """The 'revise' branch of the conditional edge: re-run ONLY the planner."""
    if not state or not state.get("orch"):
        return gr.skip()
    orch: Orchestrator = state["orch"]
    city = orch.intake.city if orch.intake else None
    plans, trace = plan_protection(
        CFG, preferred_brands=list(_profile_of(state).preferences.get("hotel_brands", [])),
        trip_days=int(days), budget=float(budget), city=city)
    orch.plan_trace = trace
    b = state["briefing"]
    b.plans = plans
    state["decision"] = ""
    note = f"🔁 re-planned with budget {budget:.0f}, {int(days)} day(s)"
    return render_beam(orch, b, decision=note)


def on_feedback(state, source_id, score):
    """Analyst feedback -> long-term memory. Persists to disk: the next run
    (and the next session) down-weights this source. This is the alert-fatigue
    fix from Module 2, operated from the GUI."""
    if source_id:
        ReliabilityMemory(CFG.path("reliability")).set_score(source_id, float(score))
    return render_ltm(state)


_DEBUG_TABS = 4          # Short-term, Trace, Evals, Agents


def on_toggle_debug(shown: bool):
    """Show or hide the developer tabs.

    They are built with `visible=False` rather than omitted, so the debugging
    view is the same objects the analyst view already updated — flipping the
    button never shows a panel that stopped being rendered while it was hidden.
    """
    shown = not shown
    label = "Hide debugging tools" if shown else "Debugging tools"
    return (shown, gr.update(value=label),
            *(gr.update(visible=shown) for _ in range(_DEBUG_TABS)))


def on_recall_test(state):
    """Seeded retrieval-recall self-test (offline; local embedder)."""
    state = dict(state or _new_state())
    state["recall"] = evals.seeded_recall(CFG)
    return state, render_evals(state.get("metrics"), state["recall"])


def _known_sources() -> list[str]:
    rel = ReliabilityMemory(CFG.path("reliability"))
    return sorted(k for k, v in rel._scores.items()
                  if isinstance(v, (int, float)))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="ThreatDetector — Analyst Console") as demo:
    gr.Markdown(
        f"# ThreatDetector — Analyst Console\n"
        "Multi-agent VIP threat assessment (capstone). **Talk to it** — name a "
        "protectee, where they are going and when (*\"assess our CEO Dana Reyes, "
        "keynoting in Dublin on 14 October\"*), answer what it asks, then ask it "
        "*why* a score landed where it did. Every panel on the right updates on "
        f"every message. Currently: {_mode_line(CFG)}. Config and `.env` are "
        "**re-read and re-verified on every message** — edit them and just send "
        "again, no restart. All VIP data is synthetic."
    )

    state = gr.State(_new_state())

    with gr.Row():
        # ---------------- LEFT: chat ----------------
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=480, show_label=False)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Name the protectee, the city and the date…",
                    show_label=False, scale=8, autofocus=True)
                send_btn = gr.Button("Assess", variant="primary", scale=1)
            stats_md = gr.Markdown("")
            focus = gr.Radio(["all", "location", "reputation", "sentiment"],
                             value="all", label="Analyst focus (short-term memory)",
                             info="Prunes which specialists run this wave.")

        # ---------------- RIGHT: the panels that teach ----------------
        # Analyst tabs first, developer tabs last and hidden. A tab that only
        # a developer can read is noise in a demo, but deleting it would hide
        # how the system works — so it is one button away, not gone.
        with gr.Column(scale=5):
            with gr.Tab("Source Reliability"):
                ltm_md = gr.Markdown(render_ltm(None))
                gr.Markdown("---\n**Analyst feedback → long-term memory** "
                            "(persists across sessions):")
                with gr.Row():
                    src_dd = gr.Dropdown(choices=_known_sources(), label="source",
                                         allow_custom_value=True, scale=2)
                    rel_sl = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                       label="reliability", scale=2)
                    fb_btn = gr.Button("Save", size="sm", scale=1)

            with gr.Tab("Travel Planner"):
                # Order is the ask: title, what it does, YOUR decision, the
                # options you are deciding between, then the search underneath.
                gr.Markdown(
                    "### Travel Planner\n"
                    "_Beam search over protection expenditures. The search is "
                    "read-only — nothing is booked until you approve a plan._")
                gr.Markdown("---\n**Conditional edge — your decision:**")
                with gr.Row():
                    plan_dd = gr.Dropdown(choices=["option-1", "option-2", "option-3"],
                                          value="option-1", label="plan", scale=2)
                    approve_btn = gr.Button("Approve ✅", size="sm", scale=1)
                    reject_btn = gr.Button("Reject ❌", size="sm", scale=1)
                with gr.Row():
                    budget_sl = gr.Slider(3000, 25000, value=CFG.plan_budget,
                                          step=500, label="budget", scale=2)
                    days_sl = gr.Slider(1, 4, value=CFG.trip_days, step=1,
                                        label="trip days", scale=1)
                    replan_btn = gr.Button("Re-plan 🔁", size="sm", scale=1)
                gr.Markdown("---")
                beam_md = gr.Markdown(render_beam(None, None))

            with gr.Tab("Map"):
                map_html = gr.HTML(render_map(None))

            with gr.Tab("Data Sources"):
                sources_md = gr.Markdown(render_sources(None))

            # --- developer tabs: hidden until asked for ---------------------
            with gr.Tab("Short-term", visible=False) as stm_tab:
                stm_md = gr.Markdown(render_stm(None))
            with gr.Tab("Trace", visible=False) as trace_tab:
                trace_md = gr.Markdown(render_trace(None))
            with gr.Tab("Evals", visible=False) as evals_tab:
                evals_md = gr.Markdown(render_evals(None))
                recall_btn = gr.Button("Run seeded-recall test 🧪", size="sm")
            with gr.Tab("Agents", visible=False) as agents_tab:
                gr.Markdown(render_agents())

            debug_btn = gr.Button("Debugging tools", size="sm")
            debug_on = gr.State(False)

    # ---- wiring ----
    outputs = [msg_box, chatbot, state, trace_md, stm_md, ltm_md, beam_md,
               stats_md, map_html, sources_md, evals_md]
    send_btn.click(on_send, [msg_box, chatbot, focus, state], outputs)
    msg_box.submit(on_send, [msg_box, chatbot, focus, state], outputs)
    recall_btn.click(on_recall_test, state, [state, evals_md])
    debug_btn.click(on_toggle_debug, debug_on,
                    [debug_on, debug_btn, stm_tab, trace_tab, evals_tab,
                     agents_tab])

    approve_btn.click(lambda s, p: on_decide(s, p, "approve"),
                      [state, plan_dd], beam_md)
    reject_btn.click(lambda s, p: on_decide(s, p, "reject"),
                     [state, plan_dd], beam_md)
    replan_btn.click(on_replan, [state, budget_sl, days_sl], beam_md)
    fb_btn.click(on_feedback, [state, src_dd, rel_sl], ltm_md)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
