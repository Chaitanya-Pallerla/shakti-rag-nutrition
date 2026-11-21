"""
rag_benchmarks.py: build the ~10k health / nutrition question set for the RAG comparison and link
every gold paper to a row of the full index (full\\meta\\<shard>.parquet, row i = vector row i).

Sources (public, expert-labelled):
  * NFCorpus (BEIR)       nutrition questions -> relevant PubMed docs (graded 1 / 2)      gold by title
  * BioASQ (rag-mini)     biomedical questions -> relevant PMIDs + ideal answer           gold by PMID
  * SciFact               health claims -> evidence abstracts + SUPPORT / CONTRADICT        gold by title
  * PubMedQA (labelled)   question -> its own PMID + yes / no / maybe + long answer       gold by PMID
PMIDs -> DOI + title via NCBI esummary (free, no key); then one DuckDB pass over full\\meta matches
gold papers by DOI or by normalised title (lowercase, [a-z0-9] only). Every matching row counts as the
gold paper (OpenAlex keeps duplicates / preprint versions).

Output in <SHAKTI_RESEARCH_DIR>\\rag_eval:
  questions.jsonl   one question per line: qid, dataset, question, answer, label, gold[]
  gold.parquet      qid, gkey, rel, label, id, shard, row   (one line per matching index row)
  coverage.json     per dataset: questions, gold papers, gold papers found in the index
Run with the fullindex environment. Downloads are cached in rag_eval\\raw, so a rerun is cheap.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import re
import io
import json
import hashlib
import time
import tarfile
import urllib.request
import urllib.parse

import jobstatus

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
META = os.path.join(RESEARCH_DIR, "full", "meta")
OUT = os.path.join(RESEARCH_DIR, "rag_eval")
RAW = os.path.join(OUT, "raw")
LOG = os.path.join(OUT, "rag_eval.log")
HF = "https://huggingface.co/api/datasets/{}/parquet/{}/{}/0.parquet"
SOURCES = {
    "nfcorpus_queries": HF.format("BeIR/nfcorpus", "queries", "queries"),
    "nfcorpus_corpus": HF.format("BeIR/nfcorpus", "corpus", "corpus"),
    "nfcorpus_qrels_train": HF.format("BeIR/nfcorpus-qrels", "default", "train"),
    "nfcorpus_qrels_validation": HF.format("BeIR/nfcorpus-qrels", "default", "validation"),
    "nfcorpus_qrels_test": HF.format("BeIR/nfcorpus-qrels", "default", "test"),
    "bioasq_qa": HF.format("rag-datasets/rag-mini-bioasq", "question-answer-passages", "test"),
    "pubmedqa": HF.format("qiaojin/PubMedQA", "pqa_labeled", "train"),
    "scifact": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
}
MIN_TITLE = 20          # normalised titles shorter than this ("editorial", "introduction") are too generic
MAX_TITLE_ROWS = 5      # a title shared by more papers than this is ambiguous, not a gold match


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + "[data] " + str(s)
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def fetch(name):
    """Download a source once into raw\\<name>.<ext>; returns the local path."""
    url = SOURCES[name]
    path = os.path.join(RAW, name + (".tar.gz" if url.endswith(".tar.gz") else ".parquet"))
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "shakti-rag-eval"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
        log(f"downloaded {name}: {len(data):,} bytes")
    return path


def read_parquet(name):
    import pyarrow.parquet as pq
    return pq.read_table(fetch(name)).to_pylist()


def _pmids(v):
    if isinstance(v, (list, tuple)):
        return [str(int(x)) for x in v]
    return re.findall(r"\d+", str(v))


# ---------------------------------------------------------------- the four benchmarks
def load_nfcorpus():
    queries = {r["_id"]: r["text"] for r in read_parquet("nfcorpus_queries")}
    docs = {r["_id"]: r["title"] for r in read_parquet("nfcorpus_corpus")}
    qrels = {}
    for split in ("train", "validation", "test"):
        for r in read_parquet("nfcorpus_qrels_" + split):
            q = qrels.setdefault(r["query-id"], {"split": split, "docs": {}})
            q["docs"][r["corpus-id"]] = max(int(r["score"]), q["docs"].get(r["corpus-id"], 0))
    out = []
    for qid, q in qrels.items():
        if qid not in queries:
            continue
        gold = [{"gkey": "title:" + norm_title(docs.get(d)), "title": docs.get(d), "rel": s, "src": d}
                for d, s in q["docs"].items() if s > 0 and docs.get(d)]
        if gold:
            out.append({"qid": "nfcorpus:" + qid, "dataset": "nfcorpus", "split": q["split"],
                        "question": queries[qid], "answer": None, "label": None, "gold": gold})
    return out


def load_bioasq():
    out = []
    for r in read_parquet("bioasq_qa"):
        gold = [{"gkey": "pmid:" + p, "rel": 1, "src": p} for p in dict.fromkeys(_pmids(r["relevant_passage_ids"]))]
        if gold:
            out.append({"qid": f"bioasq:{r['id']}", "dataset": "bioasq", "split": "test",
                        "question": r["question"], "answer": r["answer"], "label": None, "gold": gold})
    return out


def load_pubmedqa():
    return [{"qid": f"pubmedqa:{r['pubid']}", "dataset": "pubmedqa", "split": "labeled",
             "question": r["question"], "answer": r["long_answer"], "label": r["final_decision"],
             "gold": [{"gkey": f"pmid:{r['pubid']}", "rel": 1, "src": str(r["pubid"])}]}
            for r in read_parquet("pubmedqa")]


def load_scifact():
    """Labelled claims (train + dev; test has no labels). Gold = evidence docs (with their label),
    or the cited docs for not-enough-info claims."""
    files = {}
    with tarfile.open(fetch("scifact"), "r:gz") as tar:   # read members in memory, never extract
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if m.isfile() and base in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"):
                files[base] = tar.extractfile(m).read().decode("utf-8")
    docs = {}
    for ln in files["corpus.jsonl"].splitlines():
        d = json.loads(ln)
        docs[str(d["doc_id"])] = d["title"]
    out = []
    for split in ("train", "dev"):
        for ln in files[f"claims_{split}.jsonl"].splitlines():
            c = json.loads(ln)
            ev = c.get("evidence") or {}
            if ev:
                gold = [{"gkey": "title:" + norm_title(docs.get(d)), "title": docs.get(d), "rel": 2,
                         "label": e[0]["label"], "src": d} for d, e in ev.items() if docs.get(d)]
                labels = {g["label"] for g in gold}
                label = labels.pop() if len(labels) == 1 else "MIXED"
            else:
                gold = [{"gkey": "title:" + norm_title(docs.get(str(d))), "title": docs.get(str(d)), "rel": 1,
                         "src": str(d)} for d in c.get("cited_doc_ids", []) if docs.get(str(d))]
                label = "NOT_ENOUGH_INFO"
            if gold:
                out.append({"qid": f"scifact:{split}:{c['id']}", "dataset": "scifact", "split": split,
                            "question": c["claim"], "answer": None, "label": label, "gold": gold})
    return out


# ---------------------------------------------------------------- PMID -> DOI + title (NCBI)
def pubmed_summaries(pmids):
    cache_path = os.path.join(OUT, "pubmed_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    todo = [p for p in pmids if p not in cache]
    log(f"PubMed esummary: {len(pmids):,} PMIDs, {len(todo):,} not cached")
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        body = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(chunk), "retmode": "json",
                                       "tool": "shakti-rag-eval"}).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                                             data=body, headers={"User-Agent": "shakti-rag-eval"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    res = json.load(r).get("result", {})
                break
            except Exception as e:
                log(f"esummary retry {attempt + 1}: {e!r}"[:200])
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("NCBI esummary keeps failing")
        for p in chunk:
            d = res.get(p) or {}
            doi = next((a["value"] for a in d.get("articleids", []) if a.get("idtype") == "doi"), None)
            cache[p] = {"title": d.get("title"), "doi": doi.lower() if doi else None}
        if (i // 200) % 10 == 9 or i + 200 >= len(todo):
            with open(cache_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(cache_path + ".tmp", cache_path)
            log(f"  esummary {min(i + 200, len(todo)):,}/{len(todo):,}")
        time.sleep(0.4)                                # NCBI: max 3 requests/s without a key
    return cache


# ---------------------------------------------------------------- match gold papers to index rows
def match_index(dois, titles):
    """One DuckDB pass over full\\meta: rows whose DOI or normalised title is a gold one."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    cache = os.path.join(OUT, "match_rows.parquet")
    sig = hashlib.sha256(json.dumps([sorted(dois), sorted(titles)]).encode()).hexdigest()
    if os.path.exists(cache) and (pq.read_schema(cache).metadata or {}).get(b"sig") == sig.encode():
        log("reusing match_rows.parquet (same gold DOIs / titles)")
        return pq.read_table(cache).to_pylist()
    con = duckdb.connect()
    con.register("gd", pa.table({"doi": sorted(dois)}))
    con.register("gt", pa.table({"nt": sorted(titles)}))
    glob_ = os.path.join(META, "*.parquet").replace("\\", "/")
    log(f"DuckDB scan of full\\meta for {len(dois):,} DOIs and {len(titles):,} titles ...")
    s = time.time()
    rows = con.execute(f"""
        WITH m AS (
            SELECT id, lower(doi) AS doi, regexp_replace(lower(title), '[^a-z0-9]', '', 'g') AS nt,
                   filename, file_row_number
            FROM read_parquet('{glob_}', filename=true, file_row_number=true))
        SELECT id, doi, nt, filename, file_row_number FROM m
        WHERE doi IN (SELECT doi FROM gd) OR nt IN (SELECT nt FROM gt)
    """).fetchall()
    log(f"  {len(rows):,} matching rows in {time.time() - s:.0f}s")
    rows = [{"id": r[0], "doi": r[1], "nt": r[2],
             "shard": os.path.basename(r[3])[:-8], "row": int(r[4])} for r in rows]
    t = pa.Table.from_pylist(rows)
    pq.write_table(t.replace_schema_metadata({"sig": sig}), cache)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    job = jobstatus.Job("rag_data", "RAG eval: benchmark questions + gold papers", unit="", order=50)
    job.update("running", note="downloading benchmarks")
    qs = []
    for name, loader in (("nfcorpus", load_nfcorpus), ("bioasq", load_bioasq),
                         ("scifact", load_scifact), ("pubmedqa", load_pubmedqa)):
        part = loader()
        log(f"{name}: {len(part):,} questions")
        qs += part

    job.update("running", note="PubMed IDs -> DOI + title")
    pm = pubmed_summaries(sorted({g["gkey"][5:] for q in qs for g in q["gold"] if g["gkey"].startswith("pmid:")}))
    # a gold paper is matched by DOI or by title; keep both handles per gold key
    handles = {}                                       # gkey -> (doi or None, normalised title or None)
    for q in qs:
        for g in q["gold"]:
            k = g["gkey"]
            if k.startswith("pmid:"):
                d = pm.get(k[5:]) or {}
                nt = norm_title(d.get("title"))
                handles[k] = (d.get("doi"), nt if len(nt) >= MIN_TITLE else None)
                g["title"] = d.get("title")
            else:
                handles[k] = (None, k[6:] if len(k) - 6 >= MIN_TITLE else None)

    job.update("running", note="matching gold papers to the 484M-row index (DuckDB scan)")
    rows = match_index({h[0] for h in handles.values() if h[0]}, {h[1] for h in handles.values() if h[1]})
    by_doi, by_nt = {}, {}
    for r in rows:
        if r["doi"]:
            by_doi.setdefault(r["doi"], []).append(r)
        by_nt.setdefault(r["nt"], []).append(r)
    hits, ambiguous = {}, 0
    for k, (doi, nt) in handles.items():
        # DOI is exact; a title is only a fallback, and only when it is (nearly) unique: titles like
        # "Editorial comment" match hundreds of unrelated papers
        cand = by_doi.get(doi, []) if doi else []
        if not cand and nt:
            cand = by_nt.get(nt, [])
            if len({r["id"] for r in cand}) > MAX_TITLE_ROWS:
                cand, ambiguous = [], ambiguous + 1
        hits[k] = list({(r["shard"], r["row"]): r for r in cand}.values())
    log(f"gold papers dropped because their title matches > {MAX_TITLE_ROWS} papers: {ambiguous:,}")

    import pyarrow as pa
    import pyarrow.parquet as pq
    recs, cov = [], {}
    for q in qs:
        c = cov.setdefault(q["dataset"], {"questions": 0, "questions_with_gold_in_index": 0,
                                          "gold_papers": 0, "gold_papers_in_index": 0})
        c["questions"] += 1
        found_any = False
        for g in q["gold"]:
            c["gold_papers"] += 1
            g["in_index"] = bool(hits[g["gkey"]])
            c["gold_papers_in_index"] += g["in_index"]
            found_any |= g["in_index"]
            for r in hits[g["gkey"]]:
                recs.append({"qid": q["qid"], "gkey": g["gkey"], "rel": g["rel"], "label": g.get("label"),
                             "id": r["id"], "shard": r["shard"], "row": r["row"]})
        c["questions_with_gold_in_index"] += found_any
    with open(os.path.join(OUT, "questions.jsonl.tmp"), "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    os.replace(os.path.join(OUT, "questions.jsonl.tmp"), os.path.join(OUT, "questions.jsonl"))
    pq.write_table(pa.Table.from_pylist(recs), os.path.join(OUT, "gold.parquet"))
    cov["total"] = {k: sum(c[k] for c in cov.values()) for k in next(iter(cov.values()))}
    json.dump(cov, open(os.path.join(OUT, "coverage.json"), "w"), indent=1)
    log("coverage: " + json.dumps(cov))
    t = cov["total"]
    job.update("done", note=f"{t['questions']:,} questions; {t['gold_papers_in_index']:,}/{t['gold_papers']:,} "
                            f"gold papers found in the index")


if __name__ == "__main__":
    main()
