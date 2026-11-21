"""
rag_rerank.py: re-score each question's top-100 papers with a medical cross-encoder (GPU), on top of
the existing exact retrieval (rag_search.py). Nothing in the index changes.

Model: MedCPT Cross-Encoder (NCBI; PubMedBERT trained on PubMed search logs). Input per pair:
[question] [SEP] [title + abstract], up to 512 tokens; the logit is the score.

  python rag_rerank.py --hits RUN.hybrid.hits.parquet [--depth 100]
     -> RUN.hybrid_medcpt.hits.parquet  (qid, rank, score, id), scored by rag_score.py like any run

Resumable: scores are appended per question to <out>.scores.jsonl. Paper texts are cached in
rag_eval\\rerank_docs.jsonl (shared by all runs).
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import json
import time
import argparse
from collections import defaultdict


import jobstatus

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
EVAL = os.path.join(RESEARCH_DIR, "rag_eval")
MODEL = os.path.join(os.environ.get("SHAKTI_MODELS_DIR", "models"), "medcpt-cross-encoder")     # ncbi/MedCPT-Cross-Encoder (plain folder: no symlinks here)
MAX_LEN = 512
BATCH = 64
CHUNK = 200                          # questions per scoring chunk (one jsonl flush, one status update)


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + "[rerank] " + str(s)
    with open(os.path.join(EVAL, "rag_eval.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True)
    ap.add_argument("--depth", type=int, default=100)
    a = ap.parse_args()
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; reranking runs on the GPU only.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rag_score
    from rag_agents import DocStore, load_questions

    out = a.hits.replace(".hits.parquet", "_medcpt.hits.parquet")
    cache = out + ".scores.jsonl"
    Q = load_questions()
    ranked = {q: ids[:a.depth] for q, ids in rag_score.load_ranked(a.hits).items()}
    done = {}
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                done[r["qid"]] = r["s"]
    todo = [q for q in sorted(ranked) if q not in done]
    name = os.path.basename(out).split(".")[0]
    job = jobstatus.Job("rag_rerank", f"RAG eval: MedCPT cross-encoder rerank of top-{a.depth} (GPU)",
                        unit="questions", total=len(ranked), order=52)
    log(f"{name}: {len(ranked):,} questions, {len(done):,} already scored")

    if todo:
        job.update("running", done=len(done), note=f"{name}: fetching paper texts")
        docs = DocStore(os.path.join(EVAL, "rerank_docs.jsonl"))
        docs.fetch([i for q in todo for i in ranked[q]])
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("cuda").half().eval()
        t0, pairs = time.time(), 0
        for c in range(0, len(todo), CHUNK):
            qs = todo[c:c + CHUNK]
            items = []                                   # (qid, pos, question, paper text)
            for q in qs:
                for p, i in enumerate(ranked[q]):
                    d = docs.get(i)
                    text = " ".join(x for x in (d.get("title"), d.get("abstract")) if x) or "(no text)"
                    items.append((q, p, Q[q]["question"], text))
            items.sort(key=lambda x: len(x[3]))          # similar lengths per batch -> less padding
            sc = defaultdict(dict)
            with torch.inference_mode():
                for b in range(0, len(items), BATCH):
                    bt = items[b:b + BATCH]
                    enc = tok([x[2] for x in bt], [x[3] for x in bt], truncation="only_second",
                              max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
                    logits = model(**enc).logits.squeeze(-1).float().cpu().tolist()
                    for (q, p, _, _), s in zip(bt, logits):
                        sc[q][p] = s
            with open(cache, "a", encoding="utf-8") as f:
                for q in qs:
                    s = [sc[q][p] for p in range(len(ranked[q]))]
                    done[q] = s
                    f.write(json.dumps({"qid": q, "s": s}) + "\n")
            pairs += len(items)
            el = time.time() - t0
            job.update("running", done=len(done), items=pairs,
                       note=f"{name}: {pairs / el:,.0f} pairs/s")
            if (c // CHUNK) % 10 == 0:
                log(f"{name}: {len(done):,}/{len(ranked):,} questions, {pairs / el:,.0f} pairs/s, "
                    f"ETA {(len(todo) - c - len(qs)) / max(1, c + len(qs)) * el / 60:.0f} min")

    rows = {"qid": [], "rank": [], "score": [], "id": []}
    for q, ids in ranked.items():
        order = sorted(range(len(ids)), key=lambda p: -done[q][p])
        for r, p in enumerate(order, 1):
            rows["qid"].append(q); rows["rank"].append(r)
            rows["score"].append(done[q][p]); rows["id"].append(ids[p])
    pq.write_table(pa.table(rows), out)
    job.update("done", done=len(ranked), note=f"{name}: reranked -> {os.path.basename(out)}")
    log(f"{name}: wrote {out}")


if __name__ == "__main__":
    main()
