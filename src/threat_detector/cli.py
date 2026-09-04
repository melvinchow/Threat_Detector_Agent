"""Command-line interface.

    python -m threat_detector.cli demo
    python -m threat_detector.cli profile
    python -m threat_detector.cli assess "CEO keynoting a conference in Berlin in 3 weeks"
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .profile import Profile, build_profile_interactively
from .schemas import Briefing
from .tools import risk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="threat-detector", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run the full pipeline on the example VIP + fixtures.")
    p_demo.add_argument("--harness", choices=["native", "langchain"], default=None,
                        help="Which framework drives collection (default: config.yaml llm.harness).")
    sub.add_parser("rag-demo", help="Paired with/without-retrieval assessment showing how RAG grounds the output.")
    sub.add_parser("profile", help="Interactively build and save a VIP protection profile.")

    p_check = sub.add_parser("llm-check",
                             help="Diagnose the configured LLM backend: is it usable right now?")
    p_check.add_argument("--deep", action="store_true",
                         help="Also make one tiny real model call to prove it end to end.")

    p_assess = sub.add_parser("assess", help="Assess a tasking against the saved profile.")
    p_assess.add_argument("tasking", help="Free-text tasking, e.g. 'CEO keynoting in Berlin in 3 weeks'.")
    p_assess.add_argument("--focus", default="all",
                          choices=["all", "location", "reputation", "sentiment"],
                          help="Narrow which specialists run (short-term memory focus).")
    p_assess.add_argument("--harness", choices=["native", "langchain"], default=None,
                          help="Which framework drives collection (default: config.yaml llm.harness).")
    p_assess.add_argument("--assume", action="store_true",
                          help="Run even if the intake guardrail has open questions "
                               "(gaps are logged in the trace, never papered over).")

    args = parser.parse_args(argv)
    cfg = load_config()

    if args.command == "demo":
        profile = Profile.load(cfg.fixture("vip_profile.example.json"))
        briefing = _run(profile, "CEO keynoting a conference in Berlin in 3 weeks",
                        cfg, harness=args.harness)
        _print_briefing(briefing)
        return 0

    if args.command == "rag-demo":
        _rag_demo(cfg)
        return 0

    if args.command == "llm-check":
        return _llm_check(cfg, deep=args.deep)

    if args.command == "profile":
        profile = build_profile_interactively()
        out = cfg.path("profile")
        profile.save(out)
        print(f"\nSaved profile to {out}")
        return 0

    if args.command == "assess":
        profile_path = cfg.path("profile")
        if not profile_path.exists():
            print("No saved profile found. Run `threat-detector profile` first, "
                  "or `threat-detector demo` to use the example VIP.", file=sys.stderr)
            return 1
        profile = Profile.load(profile_path)

        # Module 6 intake guardrail: major gaps get follow-up questions, not guesses.
        from .guardrails import intake_check
        intake = intake_check(args.tasking, profile, cfg)
        if intake.questions and not args.assume:
            print("Before running, the intake guardrail needs answers:\n")
            for i, q in enumerate(intake.questions, 1):
                print(f"  {i}. {q}")
            if intake.assumptions:
                print("\nWhere a safe assumption exists, it will be stated, not asked:")
                for a in intake.assumptions:
                    print(f"  - {a}")
            print("\nAdd the missing details to the tasking, or re-run with --assume.")
            return 2

        briefing = _run(profile, args.tasking, cfg, focus=args.focus,
                        harness=args.harness)
        _print_briefing(briefing)
        return 0

    return 1


def _run(profile: Profile, tasking: str, cfg, focus: str = "all",
         harness: str | None = None) -> Briefing:
    if cfg.llm_enabled:
        if (harness or cfg.harness) == "langchain":
            # Same tools, same deterministic spine; LangChain + LangGraph
            # choose the dispatches instead of agent_loop's own loop.
            from .langchain_roster import assess_with_langchain
            return assess_with_langchain(profile, tasking, cfg, focus=focus)
        # The REAL agent loop: the model picks tools, reacts, and writes the
        # narrative — on whichever backend config selects (ollama /
        # huggingface / claude_cli / anthropic).
        from .agent_loop import LLMOrchestrator
        return LLMOrchestrator(profile, cfg).assess(tasking, focus=focus)
    from .orchestrator import Orchestrator
    return Orchestrator(profile, cfg).assess(tasking, focus=focus)


# What a prompt-only model (no tools, no retrieval) produces: fluent, sourceless
# safety platitudes — identical for almost any executive at almost any venue.
_PROMPT_ONLY_BASELINE = [
    "Large public event; expect elevated crowd exposure.",
    "Keep emergency contacts handy and note the nearest hospital and police station.",
    "Avoid sharing real-time location on social media.",
    "Recommend standard advance work and close protection coverage.",
]


def _rag_demo(cfg) -> None:
    """The Module 3 #3 paired demonstration: same tasking, with vs. without retrieval.

    The point isn't the overall number (live collection already reads HIGH here);
    it's *grounding*. Retrieval adds dated, attributable evidence — including a
    14-month-old case-memory post naming an individual with a prior pattern of
    venue surveillance — that a prompt-only answer cannot produce.
    """
    from .orchestrator import Orchestrator

    profile = Profile.load(cfg.fixture("vip_profile.example.json"))
    tasking = "CEO keynoting a conference in Berlin in 3 weeks"

    without = Orchestrator(profile, cfg).assess(tasking, use_retrieval=False)
    with_ = Orchestrator(profile, cfg).assess(tasking, use_retrieval=True)

    line = "=" * 72
    print(f"\n{line}\nRAG PAIRED DEMONSTRATION — {profile.name}\n{line}")
    print(f"Tasking: {tasking}\n")

    print("(a) PROMPT-ONLY baseline — fluent, but cites nothing verifiable:")
    for item in _PROMPT_ONLY_BASELINE:
        print(f"      • {item}")

    retrieved = [f for f in with_.findings if f.detail.get("retrieved")]
    print(f"\n(b) WITH RETRIEVAL — every item carries a source and a date the analyst can check:")
    for f in retrieved:
        d = f.event_date.isoformat() if f.event_date else "n/a"
        author = f.detail.get("author") or "—"
        print(f"      • [{f.channel.value:<18}] {f.source_id}  (authored {d}, "
              f"index={f.detail.get('index')}, author={author})")
        print(f"            {f.detail.get('snippet', '')[:104]}")

    print("\nWhat retrieval uniquely surfaced:")
    print("  - A NAMED individual (@drk_wolf) tied to a prior venue-surveillance pattern")
    print("    (Munich 2025, in non-expiring case memory) — now probing the Berlin venue.")
    print("  - This is grounding: the assessment cites documents, not opinion.")

    print(f"\nOverall score:  without-retrieval {without.score.overall:.2f} "
          f"[{risk.band(without.score.overall)}]   ->   "
          f"with-retrieval {with_.score.overall:.2f} [{risk.band(with_.score.overall)}]")
    print("The band was already elevated by live signal; retrieval changes it from")
    print("'elevated on current chatter' to 'elevated with a documented, attributed pattern'.")
    print(line)


def _llm_check(cfg, deep: bool = False) -> int:
    """Answer 'why is LLM mode not working?' in one command."""
    from .llm_backends import get_backend, normalize_model

    print(f"llm.enabled : {cfg.llm_enabled}")
    if not cfg.llm_enabled:
        print("  -> deterministic mode. Set llm.enabled: true in config.yaml "
              "to use a model.")
    print(f"llm.backend : {cfg.llm_backend}")
    print(f"orchestrator: {normalize_model(cfg.orchestrator_model)}")
    print(f"specialist  : {normalize_model(cfg.specialist_model)}")

    # A key exported in the SHELL silently overrides .env — that is how this
    # project once ran for weeks "in LLM mode" while every call 401'd.
    import os
    if cfg.llm_backend == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        from .config import ROOT
        dotenv = ROOT / ".env"
        in_file = dotenv.exists() and "ANTHROPIC_API_KEY" in dotenv.read_text()
        if not in_file:
            print("\n  ! ANTHROPIC_API_KEY comes from your SHELL environment, "
                  "not .env.\n    Editing .env will not change it. Check with: "
                  "env | grep ANTHROPIC")

    backend = get_backend(cfg)
    ok, msg = backend.check()
    print(f"\ncheck       : {'OK' if ok else 'NOT USABLE'}\n  {msg}")
    if not ok:
        return 1
    if not deep:
        # A shallow check only proves the config LOOKS right. Say so, loudly:
        # a well-formed but rejected key passes here and fails every real call.
        print("\ndeep check  : SKIPPED — this only checked configuration, not "
              "that the model\n              actually answers. Run with --deep "
              "before trusting it.")
        return 0
    if hasattr(backend, "ping"):
        pok, pmsg = backend.ping()
        print(f"deep check  : {'OK' if pok else 'FAILED'}\n  {pmsg}")
        ok = pok
    return 0 if ok else 1


def _print_briefing(b: Briefing) -> None:
    line = "=" * 72
    print(f"\n{line}\nTHREAT BRIEFING — {b.protectee}\n{line}")
    print(f"Tasking : {b.tasking}")
    print(f"Overall : {b.score.overall:.2f}  [{risk.band(b.score.overall)}]")
    if b.narrative:
        print(f"\nAssessment (written by the orchestrator model):\n{b.narrative}")
    print("By channel:")
    for ch, v in sorted(b.score.by_channel.items(), key=lambda kv: -kv[1]):
        print(f"    {ch:<20} {v:.2f}")

    print("\nTop evidence:")
    for r in b.score.rationale:
        print(f"  - {r}")

    print(f"\nRecommendations ({len(b.recommendations)}):")
    for rec in b.recommendations:
        gate = "APPROVAL REQUIRED" if rec.requires_approval else "auto-ok"
        print(f"  [{gate}] {rec.action}")
        print(f"            why: {rec.rationale}")

    if b.plans:
        print(f"\nProtection plan slate (ToT beam — {len(b.plans)} option(s), "
              "nothing booked until approved):")
        for p in b.plans:
            status = "complete" if p.complete else "INCOMPLETE: " + ", ".join(p.uncovered)
            print(f"  {p.plan_id}: {p.total_cost:.0f} of {p.budget:.0f} "
                  f"(score {p.score:.3f}, {status})")
            for e in p.line_items:
                print(f"      day {e.day} {e.requirement:<11} {e.vendor_name} ({e.cost:.0f})")

    print("\nReasoning trace (Think -> Act -> Observe -> Adapt):")
    for step in b.trace:
        print(f"  {step}")
    print(line)


if __name__ == "__main__":
    raise SystemExit(main())
