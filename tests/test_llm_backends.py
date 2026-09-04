"""Backend-level regressions: failures here are invisible upstream."""

from pathlib import Path


def test_claude_cli_passes_the_prompt_after_a_double_dash(monkeypatch):
    """Every findings digest starts "- [source_id] ...", and the CLI read that
    leading dash as a flag — so specialist commentary failed on every wave,
    silently, because that call site catches exceptions. Guard the separator."""
    import subprocess

    from threat_detector.config import Config
    from threat_detector.llm_backends import ClaudeCLIBackend

    seen = {}

    class Result:
        returncode, stdout, stderr = 0, "commentary", ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    digest = "- [reddit:r/x] (hostile_actor, sev 0.30) a finding"
    backend = ClaudeCLIBackend(Config(raw={}, root=Path(".")))
    assert backend.chat("sys", digest, model="claude-haiku-4-5") == "commentary"

    cmd = seen["cmd"]
    assert "--" in cmd, "the prompt must be separated from the flags"
    assert cmd.index("--") < cmd.index(digest), "prompt must come after --"
    assert cmd[-1] == digest
