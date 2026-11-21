"""
rag_llm.py: the one place SHAKTI talks to a language model. Every other module calls

    from rag_llm import ask
    out = ask("Question ...", system="...", schema={...}, model="sonnet")   # -> dict (schema) or str

Choose the backend with SHAKTI_LLM_BACKEND (see README.md, "Connecting a language model"):

  claude   (default) Claude Code CLI signed in with your own account: `claude -p`, no API key needed.
  codex    OpenAI Codex CLI signed in with your ChatGPT account (or OPENAI_API_KEY): `codex exec`.
  openai   any OpenAI-compatible /v1/chat/completions server: Ollama, vLLM, LM Studio, llama.cpp
           server (offline models such as Qwen2.5-VL), or the OpenAI API itself.

The code uses three model aliases: "sonnet" (answers, grading), "opus" (judge in the evaluation) and
"haiku". For the claude backend they are passed through unchanged. For the other backends they are mapped
by SHAKTI_LLM_MODEL_<ALIAS> (e.g. SHAKTI_LLM_MODEL_OPUS), falling back to SHAKTI_LLM_MODEL.

Each call is stateless (our own system prompt, no tools). With a JSON schema the answer comes back as a
dict. Every call is cached on disk by (backend, model, system, prompt, schema), so an interrupted run
resumes without re-asking and a rerun of the scoring costs nothing.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import re
import json
import time
import glob
import hashlib
import shutil
import tempfile
import subprocess
import threading
import urllib.request
import urllib.error

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
CACHE = os.path.join(RESEARCH_DIR, "rag_eval", "llm_cache")
USAGE_LOG = os.path.join(RESEARCH_DIR, "rag_eval", "llm_usage.jsonl")
CWD = os.path.join(RESEARCH_DIR, "rag_eval", "llm_cwd")        # empty folder: no CLAUDE.md / AGENTS.md to discover
BACKEND = os.environ.get("SHAKTI_LLM_BACKEND", "claude").strip().lower()
OPENAI_BASE = os.environ.get("SHAKTI_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
OPENAI_KEY = os.environ.get("SHAKTI_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
HOME = os.path.expanduser("~")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_lock = threading.Lock()
_tl = threading.local()             # per-thread usage sink, see collect()


class collect:
    """with collect() as calls: ... -> every ask() in this thread appends its usage record to `calls`
    (cache hits too, flagged cached=True, with the usage of the original call kept for reference)."""

    def __enter__(self):
        self.prev, _tl.sink = getattr(_tl, "sink", None), []
        return _tl.sink

    def __exit__(self, *exc):
        _tl.sink = self.prev


def _sink(rec):
    s = getattr(_tl, "sink", None)
    if s is not None:
        s.append(rec)


class RateLimited(RuntimeError):
    """The account's usage limit was hit; the caller should pause and retry later."""


def resolve_model(alias):
    if BACKEND == "claude":
        return alias
    return (os.environ.get(f"SHAKTI_LLM_MODEL_{alias.upper()}") or os.environ.get("SHAKTI_LLM_MODEL")
            or ("gpt-5" if BACKEND == "codex" else alias))


def _key(model, system, prompt, schema):
    h = hashlib.sha256(json.dumps([BACKEND, model, system, prompt, schema], sort_keys=True).encode()).hexdigest()
    return os.path.join(CACHE, h[:2], h + ".json")


def _is_limit(msg):
    low = (msg or "").lower()
    return "limit" in low and ("usage" in low or "rate" in low or "reset" in low or "quota" in low)


def _parse_json(text):
    """Model text -> dict: accepts a bare object, a ```json fenced block, or an object inside prose."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    try:
        return json.loads(t)
    except Exception:
        pass
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except Exception:
            return None
    return None


def _schema_instruction(schema):
    return ("\n\nReply with ONE JSON object only (no prose, no code fence) that matches this JSON schema:\n"
            + json.dumps(schema))

# ---------------------------------------------------------------- backends: each returns (out, usage, cost)


def claude_cli():
    """claude on PATH, else the newest VS Code extension binary."""
    hits = glob.glob(os.path.join(HOME, ".vscode", "extensions", "anthropic.claude-code-*",
                                  "resources", "native-binary", "claude*"))
    return shutil.which("claude") or (max(hits, key=os.path.getmtime) if hits else None)


def _claude(prompt, system, schema, model, timeout):
    cli = claude_cli()
    if not cli:
        raise SystemExit("Claude Code CLI not found: install it and run `claude` once to sign in.")
    cmd = [cli, "-p", "--model", model, "--system-prompt", system, "--tools", "",
           "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                       cwd=CWD, timeout=timeout, creationflags=NO_WINDOW)
    try:
        res = json.loads(p.stdout)
    except Exception:
        res = {"is_error": True, "result": (p.stdout or "") + (p.stderr or "")}
    msg = str(res.get("result") or "")
    if res.get("is_error") or res.get("subtype") != "success":
        raise _Retry(msg[:500])
    out = res.get("structured_output") if schema else msg
    if schema and out is None:
        raise _Retry("no structured_output: " + msg[:300])
    return out, res.get("usage", {}), res.get("total_cost_usd")


def _codex(prompt, system, schema, model, timeout):
    cli = shutil.which("codex")
    if not cli:
        raise SystemExit("Codex CLI not found: `npm install -g @openai/codex`, then run `codex login`.")
    full = f"{system}\n\n---\n\n{prompt}" + (_schema_instruction(schema) if schema else "")
    with tempfile.TemporaryDirectory() as td:
        last = os.path.join(td, "last.txt")
        cmd = [cli, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
               "--output-last-message", last]
        if model:
            cmd += ["-m", model]
        cmd += ["-"]                                         # prompt from stdin
        p = subprocess.run(cmd, input=full, capture_output=True, text=True, encoding="utf-8",
                           cwd=CWD, timeout=timeout, creationflags=NO_WINDOW)
        text = open(last, encoding="utf-8").read() if os.path.exists(last) else ""
    if p.returncode != 0 or not text.strip():
        raise _Retry(((p.stderr or "") + (p.stdout or ""))[-500:] or f"codex exited {p.returncode}")
    if not schema:
        return text.strip(), {}, None
    out = _parse_json(text)
    if out is None:
        raise _Retry("codex reply was not JSON: " + text[:300])
    return out, {}, None


def _openai(prompt, system, schema, model, timeout):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": prompt + (_schema_instruction(schema) if schema else "")}]
    body = {"model": model, "messages": msgs, "temperature": float(os.environ.get("SHAKTI_LLM_TEMPERATURE", "0"))}
    if schema and os.environ.get("SHAKTI_OPENAI_JSON_SCHEMA", "1") != "0":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "shakti", "schema": schema}}
    headers = {"Content-Type": "application/json"}
    if OPENAI_KEY:
        headers["Authorization"] = "Bearer " + OPENAI_KEY

    def post(b):
        req = urllib.request.Request(OPENAI_BASE + "/chat/completions", data=json.dumps(b).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    try:
        res = post(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code == 429 or _is_limit(detail):
            raise RateLimited(detail)
        if e.code == 400 and "response_format" in body:      # server without structured output: prompt-only JSON
            body.pop("response_format")
            res = post(body)
        else:
            raise _Retry(f"HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise _Retry(f"cannot reach {OPENAI_BASE}: {e}")
    text = res["choices"][0]["message"].get("content") or ""
    u = res.get("usage") or {}
    usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
    if not schema:
        return text.strip(), usage, None
    out = _parse_json(text)
    if out is None:
        raise _Retry("reply was not JSON: " + text[:300])
    return out, usage, None


class _Retry(RuntimeError):
    pass


BACKENDS = {"claude": _claude, "codex": _codex, "openai": _openai}


def ask(prompt, system, schema=None, model="sonnet", timeout=600, retries=3):
    if BACKEND not in BACKENDS:
        raise SystemExit(f"SHAKTI_LLM_BACKEND={BACKEND!r}; choose one of {', '.join(BACKENDS)}")
    model = resolve_model(model)
    path = _key(model, system, prompt, schema)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        _sink({"model": model, "secs": 0.0, "cached": True, "usage": c.get("usage", {}), "cost_list_usd": 0.0,
               "original_cost_list_usd": c.get("cost_list_usd"), "original_secs": c.get("secs")})
        return c["out"]
    os.makedirs(CWD, exist_ok=True)
    last = None
    for attempt in range(retries):
        s = time.time()
        try:
            out, usage, cost = BACKENDS[BACKEND](prompt, system, schema, model, timeout)
        except subprocess.TimeoutExpired:            # a stalled call (latency spike): kill it and try again
            last = f"timed out after {timeout}s"
            continue
        except _Retry as e:
            last = str(e)
            if _is_limit(last):
                raise RateLimited(last)
            time.sleep(10 * (attempt + 1))
            continue
        rec = {"model": model, "backend": BACKEND, "secs": round(time.time() - s, 1), "usage": usage,
               "cost_list_usd": cost}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"out": out, **rec}, f, ensure_ascii=False)
        os.replace(path + ".tmp", path)
        with _lock, open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")
        _sink({**rec, "cached": False})
        return out
    raise RuntimeError(f"{BACKEND} call failed after {retries} tries: {last}")


if __name__ == "__main__":                            # smoke test: python rag_llm.py
    t = time.time()
    print(BACKEND, ask("Is the sky blue on a clear day?", system="You answer questions with yes, no or maybe.",
                       schema={"type": "object", "properties": {"answer": {"type": "string",
                                                                           "enum": ["yes", "no", "maybe"]}},
                               "required": ["answer"]}, model="sonnet"), f"{time.time() - t:.1f}s")
