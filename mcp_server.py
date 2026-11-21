"""
mcp_server.py: SHAKTI as an MCP tool server (stdio), for Claude Desktop, Claude Code, Codex CLI, Cursor or any
other MCP client.

It is a thin client of a running shakti_chat_server.py (start that first): every tool call goes through the
same six stages (understand, USDA food data, literature search + MedCPT rerank, link check, answer, verification)
and returns the cited answer with its sources and the verification report.

  pip install "mcp>=1.2"
  python mcp_server.py                      # normally started by the MCP client, see README.md

Environment: SHAKTI_URL (default http://127.0.0.1:8780), SHAKTI_MCP_TIMEOUT seconds (default 600).
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
import time
import uuid
import urllib.request

from mcp.server.fastmcp import FastMCP

URL = os.environ.get("SHAKTI_URL", "http://127.0.0.1:8780").rstrip("/")
TIMEOUT = int(os.environ.get("SHAKTI_MCP_TIMEOUT", "600"))
mcp = FastMCP("shakti")
_sid = None


def _call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")


def _session():
    global _sid
    if _sid is None:
        j = _call("/api/session", {"consent": True, "adult": True, "role": "mcp", "background": "mcp"})
        if "sid" not in j:
            raise RuntimeError(f"SHAKTI server at {URL} refused a session: {j}")
        _sid = j["sid"]
    return _sid


def _format(res):
    lines = [res.get("answer", "").strip(), ""]
    for p in res.get("papers", []):
        if p.get("cited"):
            link = p.get("link") or (("https://doi.org/" + p["doi"]) if p.get("doi") else "")
            lines.append(f"[{p['n']}] {p.get('title')} ({p.get('year')}). {p.get('venue') or ''} {link}".strip())
    for i, f in enumerate(res.get("foods", []), 1):
        lines.append(f"[F{i}] USDA FoodData Central: {f.get('desc') or f.get('query')}")
    v = res.get("verification", {})
    lines += ["", f"Verification: risk {v.get('risk')}"
              + (" - " + "; ".join(v.get("risk_reasons", [])) if v.get("risk_reasons") else "")]
    return "\n".join(lines).strip()


@mcp.tool()
def ask_shakti(question: str, conversation: str = "default") -> str:
    """Answer a nutrition, food or health question from ~230 million research abstracts and USDA food data.
    Returns a cited answer, the sources and an automatic hallucination check (citations, verbatim quotes,
    grounded numbers, semantic support). Use the same `conversation` id for follow-up questions.
    Not medical advice."""
    sid = _session()
    j = _call("/api/ask", {"sid": sid, "conv": conversation[:40] or "default", "question": question})
    if j.get("reset"):                                   # server restarted: new session, one retry
        global _sid
        _sid = None
        j = _call("/api/ask", {"sid": _session(), "conv": conversation[:40] or "default", "question": question})
    if "job" not in j:
        return "SHAKTI error: " + str(j.get("error", j))
    t0 = time.time()
    while time.time() - t0 < TIMEOUT:
        time.sleep(2)
        r = _call(f"/api/job?id={j['job']}&sid={_sid}")
        if r.get("status") == "done":
            return _format(r["result"])
        if r.get("status") in ("error", "unknown"):
            return "SHAKTI error: " + str(r.get("error") or r.get("status"))
    return f"SHAKTI did not answer within {TIMEOUT}s."


@mcp.tool()
def ask_shakti_json(question: str, conversation: str = "default") -> str:
    """Same as ask_shakti, but returns SHAKTI's full JSON result (answer, papers with abstracts, quotes,
    food rows, charts data, verification metrics)."""
    sid = _session()
    j = _call("/api/ask", {"sid": sid, "conv": conversation[:40] or "default", "question": question})
    if "job" not in j:
        return json.dumps({"error": j.get("error", j)})
    t0 = time.time()
    while time.time() - t0 < TIMEOUT:
        time.sleep(2)
        r = _call(f"/api/job?id={j['job']}&sid={sid}")
        if r.get("status") == "done":
            return json.dumps(r["result"], ensure_ascii=False)
        if r.get("status") in ("error", "unknown"):
            return json.dumps({"error": r.get("error") or r.get("status")})
    return json.dumps({"error": f"timeout after {TIMEOUT}s"})


if __name__ == "__main__":
    mcp.run()
