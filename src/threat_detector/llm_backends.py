"""Model backends: ONE chat interface, four ways to run it.

The agent loop (`agent_loop.py`) doesn't care where tokens come from. Pick in
`config.yaml → llm.backend`:

* ``ollama``      — a local open-source model served by Ollama
                    (http://localhost:11434). Free, offline, no key.
                    Models: whatever you've `ollama pull`ed (e.g. qwen3:8b).
* ``huggingface`` — a local model via `transformers` (pip install -e ".[huggingface]").
                    Heavy download + RAM; for when you want no server at all.
* ``claude_cli``  — your ordinary Claude subscription through the `claude` CLI
                    in headless print mode (the same auth the ai-job-search
                    project uses). No API key; billed to your plan's usage.
* ``anthropic``   — the Anthropic API with ANTHROPIC_API_KEY from `.env`
                    (pay-per-token). Called over plain HTTPS — no SDK needed.

Every backend implements:

    chat(system, user, model=..., json_schema=None, max_tokens=...) -> str
    check() -> (ok: bool, message: str)      # is this backend usable RIGHT NOW?

``json_schema`` asks for constrained/structured output where the backend
supports it (Ollama enforces it at decode time, like the TA demo); elsewhere we
instruct + parse defensively (`extract_json`). ``check()`` powers
`cli llm-check` and the GUI's refusal-with-reasons — when LLM mode can't run,
you get told exactly why, never a silent fall back to deterministic mode.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import requests

from .config import Config

_OLLAMA_URL = "http://localhost:11434"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


def normalize_model(name: str) -> str:
    """Strip framework prefixes (litellm-style 'anthropic/x', 'ollama/x') so a
    config written for the old LiteLLM-style harness still works everywhere."""
    for prefix in ("anthropic/", "ollama/", "huggingface/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# Reasoning models (qwen3, deepseek-r1, ...) emit a chain-of-thought block. It is
# NOT the answer: leaving it in poisons every downstream parse and shows the
# analyst the model muttering to itself. Stripped centrally, for every backend.
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>…</think> blocks (and an unclosed trailing one)."""
    text = _THINK_RE.sub("", text or "")
    # An unclosed block means the model hit the token cap mid-thought — the
    # visible answer never arrived, so there is nothing to keep after it.
    text = re.sub(r"<(?:think|thinking|reasoning)>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class LLMUnavailable(RuntimeError):
    """The backend could not produce a completion. Raised, never swallowed —
    a silent empty string is what makes an LLM app look like a hardcoded one."""


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, defensively.

    Handles code fences and leading prose. Returns {} when nothing parses —
    the caller treats that as 'no tool, just answer', never a crash.
    """
    text = re.sub(r"```(?:json)?", "", strip_reasoning(text))
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return {}


class OllamaBackend:
    """Local open-source models. The default backend: free, offline, and it
    cannot 401 or rate-limit in the middle of a demo."""

    name = "ollama"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def chat(self, system: str, user: str, *, model: str,
             json_schema: dict | None = None, max_tokens: int = 1024,
             history: list[dict] | None = None, temperature: float = 0.3) -> str:
        messages = [{"role": "system", "content": system}]
        # Real multi-turn: prior turns go in as their own messages, not as text
        # pasted into the prompt. This is what lets the chat follow a thread.
        for turn in (history or []):
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"],
                                 "content": str(turn["content"])})
        messages.append({"role": "user", "content": user})

        body = {
            "model": normalize_model(model),
            "messages": messages,
            "stream": False,
            # Reasoning models (qwen3) otherwise spend the ENTIRE num_predict
            # budget inside <think> and return an EMPTY answer — the single
            # nastiest failure here, because it looks like "the LLM did nothing"
            # rather than an error. Ollama ignores this key on non-thinking models.
            "think": False,
            "keep_alive": "10m",           # don't pay model load on every call
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_schema is not None:
            body["format"] = json_schema      # constrained decoding, TA-demo style

        try:
            resp = requests.post(f"{_OLLAMA_URL}/api/chat", json=body, timeout=300)
            if resp.status_code == 400 and "think" in resp.text.lower():
                body.pop("think")             # older ollama build: retry without
                resp = requests.post(f"{_OLLAMA_URL}/api/chat", json=body, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise LLMUnavailable(
                f"Ollama call failed ({e}). Is `ollama serve` running, and is "
                f"`{normalize_model(model)}` pulled?") from e

        msg = resp.json().get("message", {})
        out = strip_reasoning(msg.get("content", ""))
        if not out:
            # The model produced only reasoning, or nothing at all. Say so —
            # never hand the caller an empty string to paper over.
            hint = ("the model spent its whole token budget thinking; raise "
                    "llm.max_tokens or use a non-reasoning model"
                    if msg.get("thinking") else "the model returned no content")
            raise LLMUnavailable(
                f"Ollama model {normalize_model(model)} returned an empty "
                f"completion ({hint}).")
        return out

    def check(self) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=3)
            resp.raise_for_status()
        except requests.RequestException:
            return False, ("Ollama server not reachable at localhost:11434. "
                           "Install: `brew install ollama`, then run `ollama serve`.")
        have = {m["name"] for m in resp.json().get("models", [])}
        want = {normalize_model(self.cfg.orchestrator_model),
                normalize_model(self.cfg.specialist_model)}
        missing = {w for w in want
                   if w not in have and f"{w}:latest" not in have
                   and not any(h.startswith(w) for h in have)}
        if missing:
            return False, (f"Ollama is running but model(s) not pulled: "
                           f"{sorted(missing)}. Run `ollama pull <model>`. "
                           f"Available: {sorted(have) or 'none'}.")
        return True, f"Ollama up; models available: {sorted(want)}"

    def ping(self) -> tuple[bool, str]:
        """One tiny real generation, to prove the model actually answers."""
        try:
            out = self.chat("Answer in one word.", "Say: ok",
                            model=self.cfg.orchestrator_model, max_tokens=200)
            return True, f"Live generation succeeded (reply: {out.strip()[:30]!r})."
        except Exception as e:
            return False, f"Live generation failed: {e}"


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def chat(self, system: str, user: str, *, model: str,
             json_schema: dict | None = None, max_tokens: int = 1024,
             history: list[dict] | None = None, temperature: float = 0.3) -> str:
        if json_schema is not None:
            user += ("\n\nRespond with ONLY a JSON object matching this schema, "
                     "no prose:\n" + json.dumps(json_schema))
        messages = [t for t in (history or [])
                    if t.get("role") in ("user", "assistant") and t.get("content")]
        messages = [{"role": t["role"], "content": str(t["content"])} for t in messages]
        messages.append({"role": "user", "content": user})
        key = (self.cfg.env("ANTHROPIC_API_KEY", "") or "").strip().strip('"').strip("'")
        resp = requests.post(
            _ANTHROPIC_URL,
            headers={"x-api-key": key,
                     "anthropic-version": _ANTHROPIC_VERSION,
                     "content-type": "application/json"},
            json={"model": normalize_model(model), "max_tokens": max_tokens,
                  "system": system, "temperature": temperature,
                  "messages": messages},
            timeout=120,
        )
        if resp.status_code == 401:
            raise LLMUnavailable(
                "Anthropic API rejected the key (401). Note that a stale "
                "ANTHROPIC_API_KEY exported in your SHELL overrides .env for "
                "this check — run `env | grep ANTHROPIC` if .env looks right.")
        if resp.status_code == 429:
            raise LLMUnavailable("Anthropic API rate limit (429). Retry shortly, "
                                 "or switch llm.backend to `ollama` in config.yaml.")
        resp.raise_for_status()
        out = strip_reasoning("".join(b.get("text", "")
                                      for b in resp.json().get("content", [])))
        if not out:
            raise LLMUnavailable(f"Anthropic model {normalize_model(model)} "
                                 f"returned an empty completion.")
        return out

    def check(self) -> tuple[bool, str]:
        key = self.cfg.env("ANTHROPIC_API_KEY", "") or ""
        key = key.strip().strip('"').strip("'")
        if not key:
            return False, ("ANTHROPIC_API_KEY is empty. Put it in .env as "
                           "ANTHROPIC_API_KEY=sk-ant-... (no quotes, no spaces).")
        if not key.startswith("sk-ant-"):
            return False, ("ANTHROPIC_API_KEY doesn't look like an Anthropic key "
                           "(should start with sk-ant-). Check for stray quotes "
                           "or truncation in .env.")
        return True, f"API key present ({key[:10]}…). Deep check: `cli llm-check --deep`."

    def ping(self) -> tuple[bool, str]:
        """One tiny real request (~a token) to prove the key works end to end."""
        try:
            out = self.chat("Reply with exactly: ok", "ping",
                            model=self.cfg.orchestrator_model, max_tokens=8)
            return True, f"Live API call succeeded (reply: {out.strip()[:20]!r})."
        except Exception as e:
            return False, f"Live API call failed: {e}"


class ClaudeCLIBackend:
    """Your Claude *subscription* via the `claude` CLI in print mode — the same
    way ai-job-search runs on your ordinary Claude account, no API key."""

    name = "claude_cli"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def chat(self, system: str, user: str, *, model: str,
             json_schema: dict | None = None, max_tokens: int = 1024,
             history: list[dict] | None = None, temperature: float = 0.3) -> str:
        # The CLI is one-shot per invocation, so prior turns are replayed as a
        # transcript inside the prompt rather than as API message roles.
        prompt = user
        if history:
            transcript = "\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in history
                if t.get("role") in ("user", "assistant") and t.get("content"))
            prompt = (f"Conversation so far:\n{transcript}\n\n"
                      f"Now respond to the latest message:\n{user}")
        if json_schema is not None:
            prompt += ("\n\nRespond with ONLY a JSON object matching this "
                       "schema, no prose:\n" + json.dumps(json_schema))
        # `--` before the prompt, because the prompt is arbitrary text and the
        # CLI parses a leading dash as a flag. Every findings digest begins
        # "- [source_id] ...", so without this the specialist-commentary calls
        # died with `unknown option` on every wave — silently, since that call
        # site swallows exceptions. The trace then showed collection happening
        # with no specialist ever speaking.
        cmd = ["claude",
               "--model", normalize_model(model),
               "--append-system-prompt", system,
               "--output-format", "text",
               "-p", "--", prompt]
        # Strip the API key from the child env: with it set, the CLI would bill
        # the API instead of the subscription — defeating this backend's point.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                             env=env)
        if out.returncode != 0:
            raise LLMUnavailable(f"claude CLI failed: {out.stderr.strip()[:300]}")
        text = strip_reasoning(out.stdout)
        if not text:
            raise LLMUnavailable("claude CLI returned an empty completion "
                                 "(is the session logged in? try `claude login`).")
        return text

    def check(self) -> tuple[bool, str]:
        if not shutil.which("claude"):
            return False, ("`claude` CLI not found on PATH. Install Claude Code "
                           "and sign in (`claude login`) to use your subscription.")
        return True, "claude CLI found; runs on your subscription's usage."

    def ping(self) -> tuple[bool, str]:
        try:
            out = self.chat("Answer in one word.", "Say: ok",
                            model=self.cfg.specialist_model, max_tokens=16)
            return True, f"Live CLI call succeeded (reply: {out.strip()[:30]!r})."
        except Exception as e:
            return False, f"Live CLI call failed: {e}"


class HuggingFaceBackend:
    """Local models via transformers. Heavy (downloads weights, needs RAM) —
    install with `pip install -e \".[huggingface]\"`."""

    name = "huggingface"
    _pipes: dict = {}

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _pipe(self, model: str):
        if model not in self._pipes:
            from transformers import pipeline  # lazy: torch is a big import
            self._pipes[model] = pipeline("text-generation", model=model)
        return self._pipes[model]

    def chat(self, system: str, user: str, *, model: str,
             json_schema: dict | None = None, max_tokens: int = 1024,
             history: list[dict] | None = None, temperature: float = 0.3) -> str:
        if json_schema is not None:
            user += ("\n\nRespond with ONLY a JSON object matching this schema, "
                     "no prose:\n" + json.dumps(json_schema))
        messages = [{"role": "system", "content": system}]
        messages += [{"role": t["role"], "content": str(t["content"])}
                     for t in (history or [])
                     if t.get("role") in ("user", "assistant") and t.get("content")]
        messages.append({"role": "user", "content": user})
        out = self._pipe(normalize_model(model))(
            messages, max_new_tokens=max_tokens, do_sample=False)
        reply = out[0]["generated_text"]
        text = strip_reasoning(
            reply[-1]["content"] if isinstance(reply, list) else str(reply))
        if not text:
            raise LLMUnavailable(f"HuggingFace model {normalize_model(model)} "
                                 f"returned an empty completion.")
        return text

    def check(self) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, ("`transformers` not installed. Run "
                           "`pip install -e \".[huggingface]\"` (large download; "
                           "a small instruct model like Qwen/Qwen2.5-3B-Instruct "
                           "is a realistic laptop choice).")
        return True, "transformers available; model downloads on first use."

    def ping(self) -> tuple[bool, str]:
        try:
            out = self.chat("Answer in one word.", "Say: ok",
                            model=self.cfg.specialist_model, max_tokens=16)
            return True, f"Live generation succeeded (reply: {out.strip()[:30]!r})."
        except Exception as e:
            return False, f"Live generation failed: {e}"


_BACKENDS = {b.name: b for b in
             (OllamaBackend, AnthropicBackend, ClaudeCLIBackend, HuggingFaceBackend)}


def get_backend(cfg: Config):
    name = cfg.llm_backend
    if name not in _BACKENDS:
        raise ValueError(f"Unknown llm.backend {name!r}. "
                         f"Options: {sorted(_BACKENDS)}")
    return _BACKENDS[name](cfg)


def chat(backend, system: str, user: str, **kw) -> str:
    """Call ``backend.chat`` with multi-turn ``history`` when the backend takes
    it, and degrade to a flattened transcript when it doesn't.

    Test doubles and any third-party backend implement only the original
    ``chat(system, user, *, model, json_schema, max_tokens)`` signature; this
    keeps them working without every call site branching.
    """
    try:
        return backend.chat(system, user, **kw)
    except TypeError as e:
        if "history" not in str(e) and "temperature" not in str(e):
            raise
        history = kw.pop("history", None) or []
        kw.pop("temperature", None)
        if history:
            transcript = "\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in history
                if t.get("role") in ("user", "assistant") and t.get("content"))
            user = (f"Conversation so far:\n{transcript}\n\n"
                    f"Now respond to the latest message:\n{user}")
        return backend.chat(system, user, **kw)


# A deep check costs a real (tiny) model call, so it is done once per process
# per backend+model, not on every message.
_VERIFIED: dict[tuple, tuple[bool, str]] = {}


def verify(cfg: Config, *, deep: bool = True, force: bool = False):
    """Is LLM mode actually usable RIGHT NOW? Returns (ok, message, backend).

    ``deep`` runs one real completion. This exists because a shallow check is
    exactly how this project used to lie to itself: an ``ANTHROPIC_API_KEY``
    that merely *looked* valid passed ``check()``, every real call then 401'd,
    the errors were swallowed, and the GUI printed a Python template that
    looked like a very boring LLM. Nothing silently degrades any more.
    """
    backend = get_backend(cfg)
    ok, msg = backend.check()
    if not ok or not deep:
        return ok, msg, backend

    key = (cfg.llm_backend, normalize_model(cfg.orchestrator_model),
           normalize_model(cfg.specialist_model))
    if force:
        _VERIFIED.pop(key, None)
    if key not in _VERIFIED:
        ping = getattr(backend, "ping", None)
        _VERIFIED[key] = ping() if ping else (True, msg)
    ok, deep_msg = _VERIFIED[key]
    return ok, (f"{msg} {deep_msg}" if ok else deep_msg), backend


__all__ = ["get_backend", "chat", "verify", "extract_json", "normalize_model",
           "strip_reasoning", "LLMUnavailable",
           "OllamaBackend", "AnthropicBackend", "ClaudeCLIBackend",
           "HuggingFaceBackend"]
