"""
rag_agents.py: the query-time RAG methods of the comparison, with Claude as the LLM (Claude Code CLI
signed in with the user's own account, see rag_llm.py) over the exact hybrid retrieval of all 484M
papers (MiniLM + SPECTER2, reciprocal-rank fusion; rag_search.py).

Methods. Every method ends with the SAME answer prompt (numbered papers -> verdict, answer, cited
paper numbers, verbatim proof quotes), so only the retrieval strategy differs:
  vanilla   single-shot RAG: hybrid top-10 for the question -> answer                  (reference)
  multihop  IRCoT / Self-Ask: top-5; then up to 2 times Claude reasons about what is still missing
            and asks one follow-up search; +5 papers per hop -> answer
  crag      Self-RAG / Corrective RAG: top-10; Claude grades every paper correct / ambiguous /
            incorrect; if none is correct it rewrites the query and retrieves once more; answers
            only from papers graded correct (ambiguous ones fill up to 3)
  deep      deep-research agent: plans 3-4 searches -> reads ~25 papers, keeps the evidence, notes
            gaps -> up to 2 gap-filling searches -> answer from the kept papers

Follow-up searches are batched: the pending queries of all methods and all questions go into one
exact pass over all papers (~85 min), then every method continues. A run is a few passes, not one
pass per query. All Claude calls are cached on disk, so a stopped run resumes where it was.

Then a pooled relevance judgement (Claude, TREC-style 0/1/2) of every paper any method used, and for
BioASQ a judgement of the answer against the expert answer. Scores -> agents\\scores.json.

  python rag_agents.py run   [--per-dataset 250]
  python rag_agents.py score
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import re
import sys
import json
import time
import random
import bisect
import argparse
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import jobstatus
from rag_llm import ask, RateLimited

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
EVAL = os.path.join(RESEARCH_DIR, "rag_eval")
AG = os.environ.get("RAG_AGENTS_DIR", os.path.join(EVAL, "agents"))
SMOKE_SHARDS = os.environ.get("RAG_SMOKE_SHARDS")           # smoke tests: follow-up searches see only N shards
MASK = os.environ.get("RAG_MASK", "abstract")              # search only papers with an abstract ("" = all)
ROUND1 = os.path.join(EVAL, "round1abs" if MASK else "round1")   # first-round hybrid lists of the questions
META = os.path.join(RESEARCH_DIR, "full", "meta")
LOG = os.path.join(EVAL, "rag_eval.log")
PY_FULL = os.environ.get("SHAKTI_PY_MAIN", sys.executable)
PY_S2 = os.environ.get("SHAKTI_PY_SPECTER2", sys.executable)
MODEL = os.environ.get("RAG_MODEL", "sonnet")
JUDGE_MODEL = os.environ.get("RAG_JUDGE_MODEL", "sonnet")
WORKERS = int(os.environ.get("RAG_WORKERS", "4"))          # parallel claude processes
PASS_K = 20                                                 # papers kept per follow-up query
ABSTRACT_CHARS = 1500
METHODS = ["vanilla", "multihop", "crag", "deep"]
JOB = {"vanilla": ("m_vanilla", "Single-shot RAG (hybrid top-10) + Claude answer"),
       "multihop": ("m_multihop", "Multi-hop / iterative RAG (IRCoT, Self-Ask)"),
       "crag": ("m_selfrag", "Self-RAG / Corrective RAG (CRAG)"),
       "deep": ("m_deepresearch", "Deep-research agent")}
VERDICTS = {"pubmedqa": ["yes", "no", "maybe"], "scifact": ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"],
            "bioasq": ["answered", "insufficient_evidence"], "nfcorpus": ["answered", "insufficient_evidence"]}


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + "[agents] " + str(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------- data
def load_questions():
    with open(os.path.join(EVAL, "questions.jsonl"), encoding="utf-8") as f:
        return {q["qid"]: q for q in map(json.loads, f)}


def make_sample(Q, gold, per_dataset):
    """Stratified: per_dataset questions from each benchmark, among those with gold papers in the index."""
    path = os.path.join(AG, f"sample_{per_dataset}.json")
    if os.path.exists(path):
        return json.load(open(path))
    by = defaultdict(list)
    for qid in sorted(Q):
        if qid in gold:
            by[Q[qid]["dataset"]].append(qid)
    rng = random.Random(13)
    sample = [qid for ds in sorted(by) for qid in sorted(rng.sample(by[ds], min(per_dataset, len(by[ds]))))]
    json.dump(sample, open(path, "w"))
    return sample


class DocStore:
    """W id -> {title, abstract, year}; reads only the 50k-row groups that hold the wanted rows."""

    def __init__(self, path=None):
        from rag_idmap import IdMap
        self.idmap = IdMap()
        self.path = path or os.path.join(AG, "docs.jsonl")
        self.d = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for ln in f:
                    r = json.loads(ln)
                    self.d[r.pop("id")] = r

    def fetch(self, ids):
        import pyarrow.parquet as pq
        missing = sorted({i for i in ids if i and i not in self.d})
        if not missing:
            return
        s = time.time()
        by = defaultdict(list)
        for i, loc in zip(missing, self.idmap.lookup(missing)):
            if loc:
                by[loc[0]].append((loc[1], i))
            else:
                self.d[i] = {"title": None, "abstract": None, "year": None}
        new = []
        for shard, lst in by.items():
            pf = pq.ParquetFile(os.path.join(META, shard + ".parquet"))
            starts, acc = [], 0
            for g in range(pf.metadata.num_row_groups):
                starts.append(acc)
                acc += pf.metadata.row_group(g).num_rows
            groups = defaultdict(list)
            for row, i in lst:
                g = bisect.bisect_right(starts, row) - 1
                groups[g].append((row - starts[g], i))
            for g, items in groups.items():
                t = pf.read_row_group(g, columns=["title", "abstract", "year"]).take([r for r, _ in items])
                for (_, i), rec in zip(items, t.to_pylist()):
                    self.d[i] = rec
                    new.append({"id": i, **rec})
        with open(self.path, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"fetched {len(new):,} paper texts from {len(by):,} shards in {time.time() - s:.0f}s")

    def get(self, i):
        return self.d.get(i) or {}


# ---------------------------------------------------------------- prompts
def task_text(q):
    ds, text = q["dataset"], q["question"]
    if ds == "scifact":
        return (f"Claim: {text}\nTask: decide whether the papers SUPPORT or CONTRADICT this claim, or "
                f"NOT_ENOUGH_INFO if they do not settle it; explain briefly.")
    if ds == "pubmedqa":
        return f"Question: {text}\nTask: answer yes, no or maybe, then explain briefly."
    if ds == "nfcorpus":
        return f"Topic: {text}\nTask: summarise briefly what the research evidence says about this health / nutrition topic."
    return f"Question: {text}\nTask: answer the question briefly."


def papers_text(ids):
    if not ids:
        return "(no papers were retrieved)"
    out = []
    for n, i in enumerate(ids, 1):
        d = DOCS.get(i)
        ab = (d.get("abstract") or "(no abstract)")[:ABSTRACT_CHARS]
        out.append(f"[{n}] {d.get('title') or '(untitled)'} ({d.get('year') or 'n.d.'})\n{ab}")
    return "\n\n".join(out)


SYS_ANSWER = (
    "You answer biomedical, health and nutrition questions using ONLY the numbered papers provided, "
    "never outside knowledge. Cite papers by their number. As proof, copy for each key claim one exact "
    "sentence, verbatim, from the cited paper's title or abstract into quotes. If the papers do not "
    "contain enough evidence, say so and choose the insufficient-evidence verdict (maybe / "
    "NOT_ENOUGH_INFO / insufficient_evidence).")


def answer_schema(ds):
    return {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": VERDICTS[ds]},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "quotes": {"type": "array", "items": {"type": "object", "properties": {
            "paper": {"type": "integer"}, "quote": {"type": "string"}}, "required": ["paper", "quote"]}}},
        "required": ["verdict", "answer", "citations", "quotes"]}


SYS_MULTIHOP = (
    "You do multi-hop retrieval (IRCoT / Self-Ask) over a search engine of 484 million scholarly papers. "
    "Given a question and the papers retrieved so far, reason step by step about what the papers "
    "establish and what is still missing. If they already hold enough evidence to answer, set enough "
    "to true. Otherwise give ONE follow-up search query (short, keyword-style) for the missing piece.")
SCH_MULTIHOP = {"type": "object", "properties": {"reasoning": {"type": "string"}, "enough": {"type": "boolean"},
                                                 "followup_query": {"type": "string"}},
                "required": ["reasoning", "enough", "followup_query"]}

SYS_GRADE = (
    "You are the retrieval evaluator of Corrective RAG. Grade every numbered paper for the question: "
    "correct = it contains evidence that directly helps answer it; ambiguous = related but only "
    "indirect or partial; incorrect = not useful. Also write a better, short search query for the "
    "question, used if the papers are not good enough.")
SCH_GRADE = {"type": "object", "properties": {
    "grades": {"type": "array", "items": {"type": "object", "properties": {
        "paper": {"type": "integer"}, "grade": {"type": "string", "enum": ["correct", "ambiguous", "incorrect"]}},
        "required": ["paper", "grade"]}},
    "rewritten_query": {"type": "string"}}, "required": ["grades", "rewritten_query"]}

SYS_PLAN = (
    "You are a deep-research agent with a semantic search engine over 484 million scholarly papers "
    "(titles and abstracts). Plan 3 or 4 distinct, short search queries that together find the "
    "evidence needed: the direct question, human trials or meta-analyses, mechanisms, and evidence "
    "that could contradict it.")
SCH_PLAN = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"},
                                                         "minItems": 3, "maxItems": 4}}, "required": ["queries"]}

SYS_READ = (
    "You are a deep-research agent. Read the numbered papers, keep only those that give real evidence "
    "for the question (relevant), summarise the findings, and give up to 2 follow-up search queries "
    "for important gaps (an empty list if there are none).")
SCH_READ = {"type": "object", "properties": {
    "relevant": {"type": "array", "items": {"type": "integer"}}, "findings": {"type": "string"},
    "followup_queries": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
    "required": ["relevant", "findings", "followup_queries"]}

SYS_JUDGE = (
    "You are an expert biomedical relevance assessor (TREC style). For each numbered paper judge its "
    "relevance to the question or claim, from its title and abstract only: 2 = contains direct "
    "evidence that answers the question or supports / refutes the claim; 1 = on topic but only "
    "background or indirect evidence; 0 = not relevant.")
SCH_JUDGE = {"type": "object", "properties": {"judgments": {"type": "array", "items": {"type": "object", "properties": {
    "paper": {"type": "integer"}, "relevance": {"type": "integer", "enum": [0, 1, 2]}},
    "required": ["paper", "relevance"]}}}, "required": ["judgments"]}

SYS_GRADE_ANSWER = (
    "You grade an answer to a biomedical question against the expert reference answer. correct = it "
    "agrees with the reference on the key facts; partial = some key facts right, others missing or "
    "wrong; wrong = it contradicts or misses the reference, or declines to answer. Judge meaning, not wording.")
SCH_GRADE_ANSWER = {"type": "object", "properties": {"grade": {"type": "string", "enum": ["correct", "partial", "wrong"]}},
                    "required": ["grade"]}


def call(st, prompt, system, schema, model=None):
    st["calls"] = st.get("calls", 0) + 1
    return ask(prompt, system=system, schema=schema, model=model or MODEL)


def answer(q, st, ids):
    return call(st, task_text(q) + "\n\nPapers:\n" + papers_text(ids), SYS_ANSWER, answer_schema(q["dataset"]))


def dedupe(xs):
    return list(dict.fromkeys(x for x in xs if x))


# ---------------------------------------------------------------- the four methods (resumable state machines)
R = {}          # retrieval results: request id -> ranked W ids (the question's own qid = round-1 hybrid)
DOCS = None


def step_vanilla(q, st):
    docs = R[q["qid"]][:10]
    st.update(evidence=docs, answer=answer(q, st, docs), done=True)


MAX_HOPS = 2


def step_multihop(q, st):
    qid = q["qid"]
    if "docs" not in st:
        st.update(docs=R[qid][:5], hops=0, trace=[])
    if st.get("wait"):
        for rq in st["wait"]:
            st["docs"] = dedupe(st["docs"] + [d for d in R[rq] if d not in st["docs"]][:5])
        st["wait"] = None
        st["hops"] += 1
    if st["hops"] < MAX_HOPS:
        r = call(st, task_text(q) + "\n\nPapers so far:\n" + papers_text(st["docs"]), SYS_MULTIHOP, SCH_MULTIHOP)
        st["trace"].append(r)
        if not r["enough"] and r["followup_query"].strip():
            st["wait"] = {f"{qid}|multihop|{st['hops'] + 1}": r["followup_query"].strip()}
            return
    st.update(evidence=st["docs"], answer=answer(q, st, st["docs"]), done=True)


def _grade(q, st, docs):
    if not docs:
        return {}, ""
    g = call(st, task_text(q) + "\n\nPapers:\n" + papers_text(docs), SYS_GRADE, SCH_GRADE)
    by_n = {x["paper"]: x["grade"] for x in g["grades"]}
    return {d: by_n.get(n, "incorrect") for n, d in enumerate(docs, 1)}, g["rewritten_query"].strip()


def step_crag(q, st):
    qid = q["qid"]
    if "grades" not in st:
        st["grades"], st["rewrite"] = _grade(q, st, R[qid][:10])
        if "correct" not in st["grades"].values():
            st["wait"] = {f"{qid}|crag|1": st["rewrite"] or q["question"]}
            return
    elif st.get("wait"):
        rq = next(iter(st["wait"]))
        g, _ = _grade(q, st, [d for d in R[rq] if d not in st["grades"]][:10])
        st["grades"].update(g)
        st["wait"] = None
        st["retried"] = True
    order = list(st["grades"])
    good = [d for d in order if st["grades"][d] == "correct"]
    if len(good) < 3:
        good += [d for d in order if st["grades"][d] == "ambiguous"][:3 - len(good)]
    st["evidence"] = good[:10]
    st.update(answer=answer(q, st, st["evidence"]), done=True)


def step_deep(q, st):
    qid = q["qid"]
    if "plan" not in st:
        p = call(st, task_text(q), SYS_PLAN, SCH_PLAN)
        st["plan"] = [x.strip() for x in p["queries"] if x.strip()][:4] or [q["question"]]
        st["wait"] = {f"{qid}|deep|p{i}": t for i, t in enumerate(st["plan"])}
        return
    if "read" not in st:
        pool = dedupe(R[qid][:5] + [d for rq in st["wait"] for d in R[rq][:5]])[:25]
        r = call(st, task_text(q) + "\n\nPapers:\n" + papers_text(pool), SYS_READ, SCH_READ)
        st.update(pool=pool, wait=None, read=r)      # only after the call: a failed call must leave "wait" for the retry
        st["kept"] = dedupe(pool[n - 1] for n in r["relevant"] if 1 <= n <= len(pool))
        fu = [t.strip() for t in r["followup_queries"] if t.strip()][:2]
        if fu:
            st["wait"] = {f"{qid}|deep|f{i}": t for i, t in enumerate(fu)}
            return
    elif st.get("wait"):
        seen = set(st["pool"]) | set(st["kept"])
        st["kept"] = dedupe(st["kept"] + [d for rq in st["wait"] for d in [x for x in R[rq] if x not in seen][:5]])
        st["wait"] = None
    st["evidence"] = st["kept"][:15]
    st.update(answer=answer(q, st, st["evidence"]), done=True)


STEPS = {"vanilla": step_vanilla, "multihop": step_multihop, "crag": step_crag, "deep": step_deep}


# ---------------------------------------------------------------- engine
def state_path():
    return os.path.join(AG, "state.json")


def save_state(S):
    with open(state_path() + ".tmp", "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False)
    os.replace(state_path() + ".tmp", state_path())


def save_R():
    with open(os.path.join(AG, "retrieved.json.tmp"), "w", encoding="utf-8") as f:
        json.dump(R, f)
    os.replace(os.path.join(AG, "retrieved.json.tmp"), os.path.join(AG, "retrieved.json"))


def waiting(st):
    return bool(st.get("wait")) and any(rq not in R for rq in st["wait"])


def advance(Q, m, qid, st):
    while not st.get("done") and not waiting(st):
        STEPS[m](Q[qid], st)


def retrieval_pass(n, pend):
    """One exact MiniLM + SPECTER2 pass over all papers for every pending query, fused (RRF)."""
    import rag_score
    import pyarrow.parquet as pq
    base = os.path.join(AG, f"pass{n}")
    qf = base + ".jsonl"
    with open(qf, "w", encoding="utf-8") as f:
        for rq, text in sorted(pend.items()):
            f.write(json.dumps({"qid": rq, "text": text}, ensure_ascii=False) + "\n")
    log(f"retrieval pass {n}: {len(pend):,} follow-up queries")
    jobstatus.Job("rag_agents", "RAG eval: query-time methods (Claude)", unit="", order=55).update(
        "running", note=f"retrieval pass {n}: {len(pend):,} follow-up queries x all 484M papers (~85 min)")
    steps = [(PY_FULL, "encode", "minilm", base + ".minilm.q.npy"), (PY_S2, "encode", "specter2", base + ".specter2.q.npy"),
             (PY_S2, "search", "minilm", base + ".minilm.hits.parquet"), (PY_S2, "search", "specter2", base + ".specter2.hits.parquet")]
    for py, cmd, model, out in steps:
        if not os.path.exists(out):
            subprocess.run([py, "rag_search.py", cmd, "--model", model, "--queries", qf, "--out", base,
                            "--k", str(PASS_K)] + (["--max-shards", SMOKE_SHARDS] if SMOKE_SHARDS else [])
                           + (["--mask", MASK] if MASK and cmd == "search" else []),
                           cwd=HERE, check=True)
    rag_score.fuse(base + ".hybrid.hits.parquet", [base + ".minilm.hits.parquet", base + ".specter2.hits.parquet"], k=PASS_K)
    got = defaultdict(list)
    for r in sorted(pq.read_table(base + ".hybrid.hits.parquet").to_pylist(), key=lambda r: (r["qid"], r["rank"])):
        got[r["qid"]].append(r["id"])
    for rq in pend:
        R[rq] = got.get(rq, [])
    save_R()


def run_parallel(items, fn, label_update=None):
    """fn(item) in WORKERS threads. Stops cleanly on the subscription's usage limit (RateLimited)."""
    errors = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        last = 0
        try:
            for k, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except RateLimited:
                    raise
                except Exception as e:
                    errors += 1
                    log(f"error on {futs[f]}: {e!r}"[:300])
                if label_update and time.time() - last > 15:
                    label_update(k, len(items))
                    last = time.time()
        except RateLimited:
            for f in futs:
                f.cancel()
            raise
    return errors


def update_jobs(S, sample, note=""):
    for m in METHODS:
        key, label = JOB[m]
        done = sum(1 for qid in sample if S[m][qid].get("done"))
        failed = sum(1 for qid in sample if S[m][qid].get("failed"))
        jobstatus.Job(key, label, unit="questions", total=len(sample), order=60).update(
            "done" if done == len(sample) else "running", done=done,
            note=note or f"{done - failed:,}/{len(sample):,} answered" + (f", {failed} failed" if failed else "")
            + f"; Claude {MODEL} via your Claude Code sign-in")


def run(per_dataset):
    global DOCS
    import pyarrow.parquet as pq
    import rag_score
    os.makedirs(AG, exist_ok=True)
    Q = load_questions()
    gold = rag_score.load_gold()
    sample = make_sample(Q, gold, per_dataset)
    log(f"=== agents run: {len(sample):,} questions x {METHODS}, model {MODEL}, {WORKERS} workers ===")
    S = json.load(open(state_path(), encoding="utf-8")) if os.path.exists(state_path()) else {}
    for m in METHODS:
        S.setdefault(m, {})
        for qid in sample:
            S[m].setdefault(qid, {})
    S.setdefault("next_pass", 1)
    if os.path.exists(os.path.join(AG, "retrieved.json")):
        R.update(json.load(open(os.path.join(AG, "retrieved.json"), encoding="utf-8")))
    if not all(qid in R for qid in sample):                 # round-1 hybrid lists for the questions themselves
        want = set(sample)
        lists = defaultdict(list)
        for r in sorted((r for r in pq.read_table(ROUND1 + ".hybrid.hits.parquet").to_pylist()
                         if r["qid"] in want), key=lambda r: (r["qid"], r["rank"])):
            lists[r["qid"]].append(r["id"])
        for qid in sample:
            R[qid] = lists[qid][:PASS_K]
        save_R()
    DOCS = DocStore()
    overall = jobstatus.Job("rag_agents", "RAG eval: query-time methods (Claude)", unit="", order=55)

    while True:
        DOCS.fetch([d for ids in R.values() for d in ids])
        todo = [(m, qid) for m in METHODS for qid in sample if not S[m][qid].get("done") and not waiting(S[m][qid])]
        if todo:
            overall.update("running", note=f"Claude steps for {len(todo):,} method x question items")
            try:
                errs = run_parallel(todo, lambda it: advance(Q, it[0], it[1], S[it[0]][it[1]]),
                                    lambda k, n: update_jobs(S, sample))
            except RateLimited as e:
                save_state(S)
                update_jobs(S, sample, note="paused: Claude usage limit reached; retrying every 15 min")
                overall.update("paused", note=f"Claude usage limit: {str(e)[:120]}; retrying every 15 min")
                log(f"usage limit: {e}; sleeping 15 min")
                time.sleep(900)
                continue
            save_state(S)
            update_jobs(S, sample)
            if errs:
                log(f"{errs} items failed this round; they are retried next round")
                for m, qid in todo:
                    st = S[m][qid]
                    if not st.get("done") and not st.get("wait"):
                        st["errors"] = st.get("errors", 0) + 1
                        if st["errors"] >= 3:
                            st.update(done=True, failed=True, evidence=st.get("evidence", []))
                save_state(S)
                update_jobs(S, sample)                       # else the dashboard stays at n-1 "running"
        pend = {rq: t for m in METHODS for qid in sample if waiting(S[m][qid])
                for rq, t in S[m][qid]["wait"].items() if rq not in R}
        if pend:
            save_state(S)
            retrieval_pass(S["next_pass"], pend)
            S["next_pass"] += 1
            save_state(S)
            continue
        if all(S[m][qid].get("done") for m in METHODS for qid in sample):
            break
    log("all methods answered; judging")
    judge(Q, S, sample)
    save_state(S)
    overall.update("done", note="answered + judged; see agents\\scores.json")
    score(S, sample, Q, gold)


# ---------------------------------------------------------------- judging + scoring
def judge(Q, S, sample):
    S.setdefault("judge", {})
    S.setdefault("bioasq_grade", {m: {} for m in METHODS})
    items = []
    for qid in sample:
        pool = dedupe(d for m in METHODS for d in S[m][qid].get("evidence", []))
        for c in range(0, len(pool), 20):
            items.append(("rel", qid, pool[c:c + 20]))
        if Q[qid]["dataset"] == "bioasq":
            items += [("ans", qid, m) for m in METHODS if S[m][qid].get("answer")]
    DOCS.fetch([d for it in items if it[0] == "rel" for d in it[2]])
    lock = threading.Lock()
    job = jobstatus.Job("rag_judge", "RAG eval: Claude relevance judge (pooled, 0/1/2)", unit="calls",
                        total=len(items), order=56)

    def do(it):
        kind, qid, x = it
        q, st = Q[qid], {}
        if kind == "rel":
            r = call(st, task_text(q) + "\n\nPapers:\n" + papers_text(x), SYS_JUDGE, SCH_JUDGE, JUDGE_MODEL)
            by_n = {j["paper"]: j["relevance"] for j in r["judgments"]}
            with lock:
                S["judge"].setdefault(qid, {}).update({d: by_n.get(n, 0) for n, d in enumerate(x, 1)})
        else:
            a = S[x][qid]["answer"]["answer"]
            r = call(st, f"Question: {q['question']}\n\nReference answer: {q['answer']}\n\nAnswer to grade: {a}",
                     SYS_GRADE_ANSWER, SCH_GRADE_ANSWER, JUDGE_MODEL)
            with lock:
                S["bioasq_grade"][x][qid] = r["grade"]

    while True:
        try:
            run_parallel(items, do, lambda k, n: job.update("running", done=k, total=n))
            break
        except RateLimited as e:
            job.update("paused", note=f"Claude usage limit: {str(e)[:120]}; retrying every 15 min")
            time.sleep(900)
    job.update("done", done=len(items))


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def score(S, sample, Q, gold):
    import rag_score
    res, rows = {}, []                                   # rows: per question, for confidence intervals
    for m in METHODS:
        per = defaultdict(lambda: defaultdict(list))
        for qid in sample:
            q, st = Q[qid], S[m][qid]
            ds, ev, a = q["dataset"], st.get("evidence", []), st.get("answer") or {}
            rel = S.get("judge", {}).get(qid, {})
            g = gold.get(qid, {})
            gold_ids = {i for _, ids in g.values() for i in ids}
            vals = {"calls": st.get("calls", 0), "papers_used": len(ev)}
            if g:
                gs = rag_score.score_question(ev, g)
                vals.update(gold_recall=gs["recall@100"], gold_hit=gs["hit@10"])
            if ev:
                vals["judged_precision"] = sum(rel.get(d, 0) >= 1 for d in ev) / len(ev)
                vals["judged_precision_strict"] = sum(rel.get(d, 0) == 2 for d in ev) / len(ev)
            vals["has_direct_evidence"] = float(any(rel.get(d, 0) == 2 or d in gold_ids for d in ev))
            cited = [ev[n - 1] for n in a.get("citations", []) if 1 <= n <= len(ev)]
            if cited:
                vals["citation_precision"] = sum(rel.get(d, 0) == 2 or d in gold_ids for d in cited) / len(cited)
            proofs = []
            for qt in a.get("quotes", []):
                n = qt.get("paper", 0)
                if 1 <= n <= len(ev):
                    d = DOCS.get(ev[n - 1])
                    verbatim = _norm(qt.get("quote")) and _norm(qt.get("quote")) in _norm(f"{d.get('title')} {d.get('abstract')}")
                    proofs.append(bool(verbatim) and (rel.get(ev[n - 1], 0) >= 1 or ev[n - 1] in gold_ids))
            if proofs:
                vals["proof_valid"] = sum(proofs) / len(proofs)
            vals["answered_with_valid_proof"] = float(any(proofs))
            if ds == "pubmedqa":
                vals["verdict_acc"] = float(a.get("verdict") == q["label"])
            elif ds == "scifact" and q["label"] in VERDICTS["scifact"]:
                vals["verdict_acc"] = float(a.get("verdict") == q["label"])
            elif ds == "bioasq" and qid in S.get("bioasq_grade", {}).get(m, {}):
                gr = S["bioasq_grade"][m][qid]
                vals["answer_correct"] = float(gr == "correct")
                vals["answer_correct_or_partial"] = float(gr in ("correct", "partial"))
            vals["failed"] = float(bool(st.get("failed")))
            rows.append({"method": m, "qid": qid, "dataset": ds, **vals})
            for k, v in vals.items():
                per[ds][k].append(v)
                per["all"][k].append(v)
        res[m] = {ds: {"n": len(v["calls"]), **{k: round(sum(x) / len(x), 4) for k, x in v.items()}}
                  for ds, v in per.items()}
    json.dump(res, open(os.path.join(AG, "scores.json"), "w"), indent=1)
    with open(os.path.join(AG, "per_question.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    keys = ["verdict_acc", "answer_correct", "gold_hit", "judged_precision", "has_direct_evidence",
            "citation_precision", "proof_valid", "calls"]
    for ds in ["all", "pubmedqa", "scifact", "bioasq", "nfcorpus"]:
        print(f"== {ds}")
        print("  method    " + "  ".join(f"{k[:14]:>14s}" for k in keys))
        for m in METHODS:
            r = res[m].get(ds, {})
            print(f"  {m:9s} " + "  ".join(f"{r[k]:14.3f}" if k in r else f"{'-':>14s}" for k in keys))
    return res


def main():
    global DOCS
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--per-dataset", type=int, default=250)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.per_dataset)
    else:
        import rag_score
        Q = load_questions()
        gold = rag_score.load_gold()
        DOCS = DocStore()
        score(json.load(open(state_path(), encoding="utf-8")), make_sample(Q, gold, a.per_dataset), Q, gold)


if __name__ == "__main__":
    main()
