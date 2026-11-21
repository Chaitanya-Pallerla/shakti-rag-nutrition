"""
survey_server.py: the human-evaluation chat for the RAG study. Participants ask nutrition / health
questions, get an evidence-cited answer, and rate it. Each question is answered by one of the
compared methods, assigned at random and balanced per session, and blinded (participants never see
which). Ratings are stored anonymously for the paper's user study.

Back end: serve_index.py (PQ search over the ~230M papers with an abstract, GPU) -> MedCPT
cross-encoder rerank -> Claude (Claude Code CLI, rag_llm.py) with the evaluation's prompts.

  python survey_server.py [--port 8780]        (needs <SHAKTI_SERVE_INDEX>)

Binds to 127.0.0.1 only. Sharing it is a separate, deliberate step: START_SURVEY_PUBLIC.bat.
Data: <SHAKTI_RESEARCH_DIR>\\survey\\*.jsonl (no IP addresses, no names; a random session id only).
Admin summary: /admin?token=<token in survey\\admin_token.txt>.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import re
import sys
import json
import time
import uuid
import random
import secrets
import argparse
import threading
from collections import defaultdict, deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import rag_agents as RA
from rag_llm import ask, RateLimited

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
DATA = os.path.join(RESEARCH_DIR, "survey")
MEDCPT = os.path.join(os.environ.get("SHAKTI_MODELS_DIR", "models"), "medcpt-cross-encoder")
ARMS = os.environ.get("SURVEY_ARMS", "single,crag,deep").split(",")
MAX_Q_PER_SESSION = int(os.environ.get("SURVEY_MAX_Q", "8"))
DAILY_CAP = int(os.environ.get("SURVEY_DAILY_CAP", "300"))        # protects the Claude subscription
MAX_RUNNING = 2                                                    # questions processed at once
MAX_QUEUE = 12
MAX_CHARS = 600

SYS_SURVEY = RA.SYS_ANSWER + (
    " Write for a general audience in plain language, in 3 to 6 sentences, citing papers like [2]. "
    "State how strong the evidence is and its limits (for example animal or small studies). This is "
    "research information, not personal medical advice; if the question describes an emergency or a "
    "personal medical decision, say that a clinician should be consulted.")
SCHEMA = RA.answer_schema("bioasq")          # verdict: answered / insufficient_evidence


def log(s):
    os.makedirs(DATA, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    with open(os.path.join(DATA, "server.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


_wlock = threading.Lock()


def append(name, rec):
    with _wlock:
        with open(os.path.join(DATA, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- retrieval + methods
class Engine:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import serve_index
        t = time.time()
        self.S = serve_index.Searcher()
        self.enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        self.tok = AutoTokenizer.from_pretrained(MEDCPT)
        self.ce = AutoModelForSequenceClassification.from_pretrained(MEDCPT).to("cuda").half().eval()
        self.gpu = threading.Lock()
        self.textlock = threading.Lock()
        self.torch = torch
        log(f"engine ready: {self.S.n:,} papers on the GPU, {time.time() - t:.0f}s")

    def search(self, queries, k=150):
        """PQ top-k per query -> list (per query) of paper records."""
        with self.gpu:
            qv = self.enc.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
            _, rows = self.S.search(qv, k=k)
        out = []
        for rr in rows:
            with self.textlock:                      # the Searcher's open parquet files are shared
                recs = self.S.text(rr.tolist())
            out.append([r for r in recs if r.get("id")])
        return out

    def rerank(self, query, recs, top):
        if not recs:
            return []
        texts = [" ".join(x for x in (r.get("title"), r.get("abstract")) if x) or "(no text)" for r in recs]
        scores = []
        with self.gpu, self.torch.inference_mode():
            for a in range(0, len(texts), 64):
                enc = self.tok([query] * len(texts[a:a + 64]), texts[a:a + 64], truncation="only_second",
                               max_length=512, padding=True, return_tensors="pt").to("cuda")
                scores += self.ce(**enc).logits.squeeze(-1).float().cpu().tolist()
        order = sorted(range(len(recs)), key=lambda i: -scores[i])
        return [recs[i] for i in order[:top]]

    def find(self, query, top, exclude=(), k=300):
        """PQ top-k (k=300 holds 98.2% of the exact top-10, pq_check.json) -> MedCPT -> top."""
        recs = [r for r in self.search([query], k=k)[0] if r["id"] not in exclude]
        return self.rerank(query, recs, top)


def papers_text(recs):
    if not recs:
        return "(no papers were retrieved)"
    return "\n\n".join(f"[{n}] {r.get('title') or '(untitled)'} ({r.get('year') or 'n.d.'})\n"
                       f"{(r.get('abstract') or '(no abstract)')[:RA.ABSTRACT_CHARS]}" for n, r in enumerate(recs, 1))


def task(q):
    return f"Question: {q}\nTask: answer the question from the research evidence, for a general audience."


def run_single(E, q, trace):
    ev = E.find(q, 10)
    return ev, ask(task(q) + "\n\nPapers:\n" + papers_text(ev), system=SYS_SURVEY, schema=SCHEMA)


def run_crag(E, q, trace):
    docs = E.find(q, 10)
    g = ask(task(q) + "\n\nPapers:\n" + papers_text(docs), system=RA.SYS_GRADE, schema=RA.SCH_GRADE)
    by_n = {x["paper"]: x["grade"] for x in g["grades"]}
    graded = [(d, by_n.get(n, "incorrect")) for n, d in enumerate(docs, 1)]
    trace["grades"] = [gr for _, gr in graded]
    if not any(gr == "correct" for _, gr in graded):
        rq = g["rewritten_query"].strip() or q
        trace["rewrite"] = rq
        more = E.find(rq, 10, exclude={d["id"] for d in docs})
        g2 = ask(task(q) + "\n\nPapers:\n" + papers_text(more), system=RA.SYS_GRADE, schema=RA.SCH_GRADE)
        by2 = {x["paper"]: x["grade"] for x in g2["grades"]}
        graded += [(d, by2.get(n, "incorrect")) for n, d in enumerate(more, 1)]
    good = [d for d, gr in graded if gr == "correct"]
    if len(good) < 3:
        good += [d for d, gr in graded if gr == "ambiguous"][:3 - len(good)]
    ev = good[:10]
    return ev, ask(task(q) + "\n\nPapers:\n" + papers_text(ev), system=SYS_SURVEY, schema=SCHEMA)


def run_deep(E, q, trace):
    p = ask(task(q), system=RA.SYS_PLAN, schema=RA.SCH_PLAN)
    plan = [x.strip() for x in p["queries"] if x.strip()][:4] or [q]
    trace["plan"] = plan
    pool, seen = [], set()
    for d in E.find(q, 6) + [d for pq in plan for d in E.find(pq, 5, k=150)]:
        if d["id"] not in seen:
            seen.add(d["id"])
            pool.append(d)
    pool = pool[:25]
    r = ask(task(q) + "\n\nPapers:\n" + papers_text(pool), system=RA.SYS_READ, schema=RA.SCH_READ)
    kept = [pool[n - 1] for n in r["relevant"] if 1 <= n <= len(pool)]
    kept = list({d["id"]: d for d in kept}.values())
    fu = [t.strip() for t in r["followup_queries"] if t.strip()][:2]
    trace["followups"] = fu
    for t in fu:
        kept += E.find(t, 5, exclude=seen | {d["id"] for d in kept}, k=150)
    ev = kept[:15]
    return ev, ask(task(q) + "\n\nPapers:\n" + papers_text(ev), system=SYS_SURVEY, schema=SCHEMA)


RUN = {"single": run_single, "crag": run_crag, "deep": run_deep}


def proof_check(ev, a):
    """Same verbatim rule as the evaluation (rag_agents._norm): quote must occur in the cited paper."""
    out = []
    for qt in a.get("quotes", []):
        n = qt.get("paper", 0)
        if 1 <= n <= len(ev):
            d = ev[n - 1]
            nq = RA._norm(qt.get("quote"))
            out.append(bool(nq) and nq in RA._norm(f"{d.get('title')} {d.get('abstract')}"))
    return out


# ---------------------------------------------------------------- sessions + jobs
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}                 # sid -> {"arms": [...], "asked": n, "role": ...}
        self.jobs = {}                     # job id -> dict
        self.queue = deque()
        self.running = 0
        self.today = (time.strftime("%Y-%m-%d"), 0)
        self.load()

    def load(self):
        p = os.path.join(DATA, "sessions.jsonl")
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                r = json.loads(ln)
                self.sessions[r["sid"]] = {"arms": r["arms"], "asked": 0, "role": r.get("role")}
        p = os.path.join(DATA, "answers.jsonl")
        if os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                r = json.loads(ln)
                if r["sid"] in self.sessions:
                    self.sessions[r["sid"]]["asked"] += 1

    def count_today(self):
        d = time.strftime("%Y-%m-%d")
        if self.today[0] != d:
            self.today = (d, 0)
        return self.today[1]


ST = State()
ENGINE = None
WORK = threading.Semaphore(MAX_RUNNING)


def process(job):
    with WORK:
        with ST.lock:
            job["status"] = "running"
        t0 = time.time()
        trace = {}
        try:
            ev, a = RUN[job["arm"]](ENGINE, job["question"], trace)
            proofs = proof_check(ev, a)
            papers = [{"n": n, "id": d["id"], "title": d.get("title"), "year": d.get("year"),
                       "venue": d.get("venue"), "doi": d.get("doi")} for n, d in enumerate(ev, 1)]
            quotes = [{"paper": q.get("paper"), "quote": q.get("quote"), "verbatim": ok}
                      for q, ok in zip([q for q in a.get("quotes", []) if 1 <= q.get("paper", 0) <= len(ev)], proofs)]
            res = {"answer": a.get("answer", ""), "verdict": a.get("verdict"), "citations": a.get("citations", []),
                   "papers": papers, "quotes": quotes}
            append("answers.jsonl", {"job": job["id"], "sid": job["sid"], "arm": job["arm"], "qn": job["qn"],
                                     "question": job["question"], "time": time.time(), "secs": round(time.time() - t0, 1),
                                     "trace": trace, **res})
            with ST.lock:
                job.update(status="done", result=res)
        except RateLimited:
            with ST.lock:
                job.update(status="error", error="The assistant has reached its usage limit for now. "
                                                 "Please try again later; thank you for your patience.")
        except Exception as e:
            log(f"job {job['id']} ({job['arm']}) failed: {e!r}")
            with ST.lock:
                job.update(status="error", error="Sorry, something went wrong answering this question. "
                                                 "Please try a different question.")


# ---------------------------------------------------------------- HTTP
PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey_page.html"), encoding="utf-8").read()
ADMIN_TOKEN = None
LIKERT = ["relevance", "evidence", "clarity", "trust", "satisfaction"]


def clean(s, n):
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(s or ""))
    return s.strip()[:n]


class H(BaseHTTPRequestHandler):
    server_version = "survey"

    def log_message(self, *a):            # no access log: it would hold participants' IP addresses
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(b)

    def js(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False))

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 20000:
            raise ValueError("too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        if path == "/":
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/job":
            jid = dict(p.split("=", 1) for p in qs.split("&") if "=" in p).get("id", "")
            with ST.lock:
                j = ST.jobs.get(jid)
                if not j:
                    return self.js({"status": "unknown"}, 404)
                pos = list(ST.queue).index(jid) + 1 if jid in ST.queue else 0
                return self.js({"status": j["status"], "result": j.get("result"), "error": j.get("error"),
                                "queue": pos})
        if path == "/admin":
            tok = dict(p.split("=", 1) for p in qs.split("&") if "=" in p).get("token", "")
            if not ADMIN_TOKEN or not secrets.compare_digest(tok, ADMIN_TOKEN):
                return self.send(404, "not found", "text/plain")
            return self.send(200, admin_page(), "text/html; charset=utf-8")
        return self.send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            b = self.body()
        except Exception:
            return self.js({"error": "bad request"}, 400)
        if self.path == "/api/session":
            if not (b.get("consent") is True and b.get("adult") is True):
                return self.js({"error": "consent required"}, 400)
            sid = uuid.uuid4().hex
            arms = ARMS[:]
            random.shuffle(arms)                               # balanced order, random start
            rec = {"sid": sid, "time": time.time(), "arms": arms,
                   "role": clean(b.get("role"), 40), "nutrition_background": clean(b.get("background"), 40)}
            append("sessions.jsonl", rec)
            with ST.lock:
                ST.sessions[sid] = {"arms": arms, "asked": 0, "role": rec["role"]}
            return self.js({"sid": sid, "max_questions": MAX_Q_PER_SESSION})
        if self.path == "/api/ask":
            sid, q = b.get("sid", ""), clean(b.get("question"), MAX_CHARS)
            with ST.lock:
                s = ST.sessions.get(sid)
                if not s:
                    return self.js({"error": "Please start a new session (reload the page)."}, 400)
                if len(q) < 8:
                    return self.js({"error": "Please type a full question."}, 400)
                if s["asked"] >= MAX_Q_PER_SESSION:
                    return self.js({"error": f"Thank you! You have reached the limit of {MAX_Q_PER_SESSION} questions."}, 400)
                if ST.count_today() >= DAILY_CAP:
                    return self.js({"error": "The study has reached today's question limit. Please come back tomorrow."}, 503)
                if len(ST.queue) >= MAX_QUEUE:
                    return self.js({"error": "Many people are asking right now. Please try again in a minute."}, 503)
                qn = s["asked"]
                arm = s["arms"][qn % len(s["arms"])]
                s["asked"] += 1
                ST.today = (ST.today[0], ST.today[1] + 1)
                jid = uuid.uuid4().hex
                job = {"id": jid, "sid": sid, "qn": qn, "arm": arm, "question": q, "status": "queued"}
                ST.jobs[jid] = job
                ST.queue.append(jid)

            def go():
                try:
                    process(job)
                finally:
                    with ST.lock:
                        if jid in ST.queue:
                            ST.queue.remove(jid)
            threading.Thread(target=go, daemon=True).start()
            return self.js({"job": jid, "remaining": MAX_Q_PER_SESSION - qn - 1})
        if self.path == "/api/rate":
            sid, jid = b.get("sid", ""), b.get("job", "")
            with ST.lock:
                j = ST.jobs.get(jid)
                if not j or j["sid"] != sid or j.get("rated"):
                    return self.js({"error": "unknown answer"}, 400)
                j["rated"] = True
            r = {k: int(b[k]) for k in LIKERT if str(b.get(k, "")).isdigit() and 1 <= int(b[k]) <= 5}
            if len(r) != len(LIKERT):
                with ST.lock:
                    j["rated"] = False
                return self.js({"error": "Please answer all five rating questions."}, 400)
            append("ratings.jsonl", {"job": jid, "sid": sid, "arm": j["arm"], "qn": j["qn"], "time": time.time(),
                                     **r, "wrong_or_harmful": bool(b.get("harm")),
                                     "comment": clean(b.get("comment"), 1000)})
            return self.js({"ok": True})
        if self.path == "/api/final":
            sid = b.get("sid", "")
            if sid not in ST.sessions:
                return self.js({"error": "unknown session"}, 400)
            rec = {"sid": sid, "time": time.time(), "comment": clean(b.get("comment"), 2000)}
            for k in ("umux_capabilities", "umux_easy"):
                v = str(b.get(k, ""))
                if v.isdigit() and 1 <= int(v) <= 7:
                    rec[k] = int(v)
            append("final.jsonl", rec)
            return self.js({"ok": True})
        return self.js({"error": "not found"}, 404)


def admin_page():
    import html
    rows = defaultdict(lambda: defaultdict(list))
    p = os.path.join(DATA, "ratings.jsonl")
    n_r = 0
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            r = json.loads(ln)
            n_r += 1
            for k in LIKERT:
                rows[r["arm"]][k].append(r[k])
            rows[r["arm"]]["harm"].append(1 if r.get("wrong_or_harmful") else 0)
    n_s = len(ST.sessions)
    n_a = sum(1 for _ in open(os.path.join(DATA, "answers.jsonl"), encoding="utf-8")) if os.path.exists(os.path.join(DATA, "answers.jsonl")) else 0
    t = "".join(f"<tr><td>{html.escape(a)}</td><td>{len(v['relevance'])}</td>" +
                "".join(f"<td>{sum(v[k]) / len(v[k]):.2f}</td>" for k in LIKERT) +
                f"<td>{100 * sum(v['harm']) / len(v['harm']):.0f}%</td></tr>" for a, v in sorted(rows.items()))
    return (f"<!doctype html><meta charset=utf-8><title>Survey admin</title><body style='font-family:sans-serif'>"
            f"<h2>Survey status</h2><p>{n_s} sessions, {n_a} answers, {n_r} ratings; today {ST.count_today()}/{DAILY_CAP} "
            f"questions; queue {len(ST.queue)}</p><table border=1 cellpadding=4><tr><th>method</th><th>n</th>" +
            "".join(f"<th>{k}</th>" for k in LIKERT) + f"<th>flagged</th></tr>{t}</table>"
            "<p>Mean ratings 1-5. Full data: <SHAKTI_RESEARCH_DIR>\\survey\\*.jsonl; analysis: survey_analyze.py</p>")


def main():
    global ENGINE, ADMIN_TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    tp = os.path.join(DATA, "admin_token.txt")
    if not os.path.exists(tp):
        with open(tp, "w") as f:
            f.write(secrets.token_urlsafe(24))
    ADMIN_TOKEN = open(tp).read().strip()
    ENGINE = Engine()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    log(f"survey server on http://127.0.0.1:{a.port}/  (admin: /admin?token={ADMIN_TOKEN}); arms {ARMS}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
