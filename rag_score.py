"""
rag_score.py: retrieval metrics for every method's ranked paper list against the expert gold papers.

A gold paper ("gkey": one PubMed / NFCorpus / SciFact document) may sit on several index rows
(OpenAlex duplicates, preprint versions); retrieving ANY of them retrieves that gold paper, and each
gold paper counts once. Gold papers missing from the index are excluded from the denominators (the
"in index" coverage is reported separately in coverage.json), so the metrics measure the retriever,
not the corpus.

  Recall@10, Recall@100  share of the question's gold papers in the top 10 / 100
  nDCG@10                graded (NFCorpus 2 = direct, 1 = indirect; others 1)
  MRR@100                1 / rank of the first gold paper
  Hit@10                 at least one gold paper in the top 10

  python rag_score.py round1.minilm.hits.parquet round1.specter2.hits.parquet ...   (prints + writes json)
  python rag_score.py --fuse OUT.hits.parquet A.hits.parquet B.hits.parquet          (RRF hybrid)
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import json
import math
from collections import defaultdict

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
EVAL = os.path.join(RESEARCH_DIR, "rag_eval")
RRF_K = 60


def load_gold():
    """qid -> {gkey: (rel, set(ids))} for gold papers present in the index."""
    import pyarrow.parquet as pq
    gold = defaultdict(dict)
    for r in pq.read_table(os.path.join(EVAL, "gold.parquet"), columns=["qid", "gkey", "rel", "id"]).to_pylist():
        rel, ids = gold[r["qid"]].get(r["gkey"], (r["rel"], set()))
        ids.add(r["id"])
        gold[r["qid"]][r["gkey"]] = (rel, ids)
    return gold


def load_ranked(path):
    import pyarrow.parquet as pq
    ranked = defaultdict(list)
    for r in sorted(pq.read_table(path, columns=["qid", "rank", "id"]).to_pylist(),
                    key=lambda r: (r["qid"], r["rank"])):
        if r["id"]:
            ranked[r["qid"]].append(r["id"])
    return ranked


def score_question(ids, g):
    """ids: ranked W ids; g: {gkey: (rel, set(ids))}."""
    id2key = {i: k for k, (_, s) in g.items() for i in s}
    seen, gains, first = set(), [], None
    for rank, i in enumerate(ids[:100], 1):
        k = id2key.get(i)
        if k and k not in seen:                      # each gold paper counts once
            seen.add(k)
            gains.append((rank, g[k][0]))
            first = first or rank
    n = len(g)
    dcg = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in gains if rank <= 10)
    ideal = sorted((rel for rel, _ in g.values()), reverse=True)[:10]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return {"recall@10": sum(1 for r, _ in gains if r <= 10) / n, "recall@100": len(gains) / n,
            "ndcg@10": dcg / idcg if idcg else 0.0, "mrr@100": 1 / first if first else 0.0,
            "hit@10": float(bool(gains and gains[0][0] <= 10))}


def score(path, gold=None):
    gold = gold or load_gold()
    ranked = load_ranked(path)
    per_ds = defaultdict(lambda: defaultdict(list))
    for qid, g in gold.items():
        ds = qid.split(":", 1)[0]
        for m, v in score_question(ranked.get(qid, []), g).items():
            per_ds[ds][m].append(v)
            per_ds["all"][m].append(v)
    return {ds: {"questions": len(ms["hit@10"]), **{m: round(sum(v) / len(v), 4) for m, v in ms.items()}}
            for ds, ms in per_ds.items()}


def fuse(out, paths, k=100):
    """Reciprocal-rank fusion of several ranked lists (the MiniLM + SPECTER2 hybrid)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    acc = defaultdict(lambda: defaultdict(float))
    for p in paths:
        for qid, ids in load_ranked(p).items():
            for rank, i in enumerate(ids, 1):
                acc[qid][i] += 1 / (RRF_K + rank)
    rows = []
    for qid, sc in acc.items():
        for rank, (i, s) in enumerate(sorted(sc.items(), key=lambda x: -x[1])[:k], 1):
            rows.append({"qid": qid, "rank": rank, "score": s, "id": i})
    pq.write_table(pa.Table.from_pylist(rows), out)


def main():
    if sys.argv[1] == "--fuse":
        fuse(sys.argv[2], sys.argv[3:])
        print("fused ->", sys.argv[2])
        return
    gold = load_gold()
    res = {}
    for p in sys.argv[1:]:
        name = os.path.basename(p).replace(".hits.parquet", "")
        res[name] = score(p, gold)
        print(f"== {name}")
        for ds, m in sorted(res[name].items()):
            print(f"  {ds:9s} n={m['questions']:5d}  R@10 {m['recall@10']:.3f}  R@100 {m['recall@100']:.3f}  "
                  f"nDCG@10 {m['ndcg@10']:.3f}  MRR {m['mrr@100']:.3f}  Hit@10 {m['hit@10']:.3f}")
    out = os.path.join(EVAL, "retrieval_scores.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res)
    json.dump(old, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
