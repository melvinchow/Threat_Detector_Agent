"""Open-Source Monitoring agent tools: news + social + entity resolution.

Strictly read-only. Two collection tools (news, social) plus an entity-resolution
step that keeps the protectee from being confused with an unrelated namesake —
the "LiSA vs. LISA" / "Jordan Vale the CEO vs. some other Jordan Vale" failure
mode from the Module 2 writeup.

Real social source: Reddit via PRAW (falls back to synthetic if creds are unset).
Real news source: Exa (https://exa.ai) semantic search with category=news, gated
on EXA_API_KEY. Both real paths pass through the per-day rate limiter first and
fall back to synthetic fixtures when the budget is spent — the run degrades
honestly instead of failing.
"""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlparse

import requests

from .. import ratelimit
from ..config import Config, load_config
from ..schemas import Finding, SourceType, ThreatChannel
from . import synthetic
from .base import load_fixture, mark_fallback, mark_synthetic

_EXA_SEARCH = "https://api.exa.ai/search"


def search_news(query: str, cfg: Config | None = None, *,
                entity_id: str = "", city: str = "") -> list[Finding]:
    """Return news-derived findings. Real (Exa) or synthetic per config."""
    cfg = cfg or load_config()
    reason: str | None = None
    if cfg.source_is_real("news"):
        if not cfg.env("EXA_API_KEY"):
            reason = "no EXA_API_KEY in .env"
        elif not ratelimit.spend(cfg, "exa"):
            reason = "exa daily budget spent"
        else:
            real = _search_exa(query, cfg)
            if real is not None:
                return real
            reason = "the Exa call failed"
    fixture = load_fixture("news.json", cfg)
    if not synthetic.in_scope(fixture, entity_id, city):
        return mark_fallback(synthetic.generate_news(query, city, cfg, entity_id), reason)
    return mark_fallback(
        mark_synthetic([_article_to_finding(a) for a in fixture["articles"]
                        if _matches(query, a["text"] + " " + a["title"])]), reason)


def search_social(query: str, cfg: Config | None = None, *,
                  entity_id: str = "", city: str = "") -> list[Finding]:
    """Return social-media findings. Real (Reddit) or synthetic per config."""
    cfg = cfg or load_config()
    # Order matters: every precondition is checked BEFORE `spend`. The budget
    # used to be charged first, so a missing `praw` burned a Reddit token on a
    # call that never left the process — the daily quota drained to zero while
    # live social collection had in fact never run once.
    reason: str | None = None
    if cfg.source_is_real("social"):
        if not _praw_available():
            reason = ("praw is not installed — run "
                      '`python3 -m pip install -e "."` to restore live Reddit')
        elif not _reddit_configured(cfg):
            reason = "no REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in .env"
        elif not ratelimit.spend(cfg, "reddit"):
            reason = "reddit daily budget spent (config.yaml -> limits.reddit)"
        else:
            real = _search_reddit(query, cfg)
            if real is not None:
                return real
            reason = "the Reddit call failed"
    # Fallback / synthetic path. The curated posts describe one scenario, and
    # `_matches` is a bare token OR — "CEO" alone was enough to pull the Berlin
    # posts into a Dublin briefing and inflate its score. Scope them.
    fixture = load_fixture("reddit.json", cfg)
    if not synthetic.in_scope(fixture, entity_id, city):
        return mark_fallback(synthetic.generate_social(query, city, cfg, entity_id), reason)
    return mark_fallback(
        mark_synthetic([_post_to_finding(p) for p in fixture["posts"]
                        if _matches(query, p["title"] + " " + p["body"])]), reason)


# --- entity resolution -----------------------------------------------------

def resolve_entity(findings: list[Finding], protectee_name: str, role_terms: list[str]) -> list[Finding]:
    """Drop findings that mention the name but clearly refer to someone else.

    A finding survives if it mentions the name AND at least one role/context term
    (company, title). This is the explicit tool step that prevents identity
    conflation, rather than trusting the model to notice implicitly.
    """
    kept: list[Finding] = []
    name_l = protectee_name.lower()
    terms = [t.lower() for t in role_terms if t]
    for f in findings:
        text = (f.summary + " " + str(f.detail)).lower()
        if name_l not in text:
            kept.append(f)  # not name-dependent; leave as-is
            continue
        if not terms or any(t in text for t in terms):
            kept.append(f)
        else:
            f.detail["entity_resolution"] = "dropped: name matched but no role context"
    return kept


# --- helpers ---------------------------------------------------------------

def _matches(query: str, text: str) -> bool:
    q = query.lower()
    text = text.lower()
    return any(tok in text for tok in q.replace(",", " ").split() if len(tok) > 2)


def _article_to_finding(a: dict) -> Finding:
    return Finding(
        channel=ThreatChannel.REPUTATION if a.get("kind") == "sentiment" else ThreatChannel.HOSTILE_ACTOR,
        source_type=SourceType.NEWS,
        source_id=a["source_id"],
        summary=a["title"],
        severity=float(a.get("severity", 0.3)),
        detail={"text": a["text"], "sentiment": a.get("sentiment")},
    )


def _post_to_finding(p: dict) -> Finding:
    return Finding(
        channel=ThreatChannel.HOSTILE_ACTOR if p.get("names_target") else ThreatChannel.REPUTATION,
        source_type=SourceType.SOCIAL,
        source_id=p["source_id"],
        summary=p["title"],
        severity=float(p.get("severity", 0.3)),
        detail={
            "body": p["body"],
            "subreddit": p.get("subreddit"),
            "names_target": p.get("names_target", False),
            "mentions_event": p.get("mentions_event"),
        },
    )


def _search_exa(query: str, cfg: Config, limit: int = 5) -> list[Finding] | None:
    """Current-events search via Exa's news category. Returns None on any
    failure so the caller falls back to fixtures. Findings carry the article's
    publication date, so the staleness guardrail applies to real news too."""
    try:
        resp = requests.post(
            _EXA_SEARCH,
            headers={"x-api-key": cfg.env("EXA_API_KEY"), "Content-Type": "application/json"},
            json={
                "query": query,
                "category": "news",
                "numResults": limit,
                # Recent by default: threat level is a function of this month.
                "startPublishedDate": (date.today() - timedelta(days=45)).isoformat(),
                "contents": {"text": {"maxCharacters": 400}},
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError):
        return None

    findings: list[Finding] = []
    for r in results:
        domain = urlparse(r.get("url", "")).netloc.removeprefix("www.")
        published = None
        if r.get("publishedDate"):
            try:
                published = date.fromisoformat(r["publishedDate"][:10])
            except ValueError:
                pass
        findings.append(Finding(
            channel=ThreatChannel.REPUTATION,
            source_type=SourceType.NEWS,
            source_id=f"news:{domain or 'exa-unknown'}",
            summary=r.get("title") or "(untitled article)",
            event_date=published,
            severity=0.3,
            detail={"url": r.get("url"), "text": (r.get("text") or "")[:400],
                    "via": "exa"},
        ))
    return findings


def _praw_available() -> bool:
    """Is the Reddit client importable at all?

    `praw` is a declared dependency, but it is imported lazily inside the call,
    so an incomplete install degrades to synthetic rather than raising. Checking
    it here — before the rate limiter — is what keeps that degradation honest
    and free.
    """
    try:
        import praw  # noqa: F401
    except ImportError:
        return False
    return True


def _reddit_configured(cfg: Config) -> bool:
    return bool(cfg.env("REDDIT_CLIENT_ID") and cfg.env("REDDIT_CLIENT_SECRET"))


def _search_reddit(query: str, cfg: Config, limit: int = 15) -> list[Finding] | None:
    """Query Reddit for recent posts. Returns None on any failure (-> fallback)."""
    import praw  # availability is checked by `_praw_available` before we get here

    try:
        reddit = praw.Reddit(
            client_id=cfg.env("REDDIT_CLIENT_ID"),
            client_secret=cfg.env("REDDIT_CLIENT_SECRET"),
            user_agent=cfg.env("REDDIT_USER_AGENT", "threat-detector-capstone/0.1"),
            check_for_async=False,
        )
        findings: list[Finding] = []
        for post in reddit.subreddit("all").search(query, sort="new", limit=limit):
            findings.append(
                Finding(
                    channel=ThreatChannel.REPUTATION,
                    source_type=SourceType.SOCIAL,
                    source_id=f"reddit:r/{post.subreddit.display_name}",
                    summary=post.title,
                    severity=0.3,
                    detail={
                        "body": (post.selftext or "")[:500],
                        "subreddit": post.subreddit.display_name,
                        "url": f"https://reddit.com{post.permalink}",
                        "score": post.score,
                    },
                )
            )
        return findings
    except Exception:
        return None


__all__ = ["search_news", "search_social", "resolve_entity"]
