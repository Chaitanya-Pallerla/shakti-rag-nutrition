"""
rag_stats.py: 95% bootstrap confidence intervals and paired significance tests for the paper.

  retrieval : per-question R@10 / nDCG@10 / MRR of every hits file in rag_eval (rag_score.score_question);
              paired bootstrap of (run B - run A) for the comparisons in PAIRS_RET
  agents    : agents\\per_question.jsonl (rag_agents.py score); paired bootstrap of (method - vanilla)
-> rag_eval\\stats.json, also printed.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
from collections import defaultdict
import numpy as np

import rag_score

EVAL = rag_score.EVAL
B = 10000
RNG = np.random.default_rng(2026)
PAIRS_RET = [("round1.minilm", "round1.hybrid"), ("round1.specter2", "round1.hybrid"),
             ("round1.hybrid", "round1.hybrid_medcpt"), ("round1abs.hybrid", "round1abs.hybrid_medcpt"),
             ("round1.hybrid", "round1abs.hybrid")]
RET_METRICS = ["recall@10", "ndcg@10", "mrr@100"]
AG_METRICS = ["verdict_acc", "answer_correct", "judged_precision", "has_direct_evidence",
              "citation_precision", "proof_valid", "gold_hit"]


def ci(x):
    x = np.asarray(x, float)
    bs = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def paired(a, b):
    """mean(b - a), 95% CI, two-sided bootstrap p (share of resampled means on the other side of 0)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    bs = d[RNG.integers(0, len(d), (B, len(d)))].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": round(float(d.mean()), 4), "ci": [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)], "p": round(float(max(p, 1 / B)), 4), "n": int(len(d))}


def retrieval():
    gold = rag_score.load_gold()
    runs = {}
    for f in sorted(os.listdir(EVAL)):
        if f.endswith(".hits.parquet") and f.startswith("round1"):
            name = f[:-len(".hits.parquet")]
            ranked = rag_score.load_ranked(os.path.join(EVAL, f))
            runs[name] = {q: rag_score.score_question(ranked.get(q, []), g) for q, g in gold.items()}
    out = {"ci": {}, "paired": {}}
    for name, per in runs.items():
        by = defaultdict(list)
        for q, s in per.items():
            by[q.split(":")[0]].append(s)
            by["all"].append(s)
        out["ci"][name] = {ds: {m: ci([s[m] for s in v]) for m in RET_METRICS} for ds, v in by.items()}
    for a, b in PAIRS_RET:
        if a in runs and b in runs:
            qs = sorted(runs[a])
            by = defaultdict(list)
            for q in qs:
                by[q.split(":")[0]].append(q)
                by["all"].append(q)
            out["paired"][f"{b} vs {a}"] = {ds: {m: paired([runs[a][q][m] for q in v], [runs[b][q][m] for q in v])
                                                 for m in RET_METRICS} for ds, v in by.items()}
    return out


def agents():
    rows = [json.loads(ln) for ln in open(os.path.join(EVAL, "agents", "per_question.jsonl"), encoding="utf-8")]
    val = defaultdict(dict)                         # (method, metric) -> qid -> value
    ds_of = {}
    for r in rows:
        ds_of[r["qid"]] = r["dataset"]
        for k in AG_METRICS:
            if k in r:
                val[(r["method"], k)][r["qid"]] = r[k]
    out = {"ci": defaultdict(dict), "paired": defaultdict(dict)}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        for k in AG_METRICS:
            v = val[(m, k)]
            if v:
                out["ci"][m][k] = ci(list(v.values()))
    for m in methods:
        if m == "vanilla":
            continue
        for k in AG_METRICS:
            a, b = val[("vanilla", k)], val[(m, k)]
            qs = sorted(set(a) & set(b))
            if qs:
                out["paired"][f"{m} vs vanilla"][k] = paired([a[q] for q in qs], [b[q] for q in qs])
    return out


if __name__ == "__main__":
    res = {"retrieval": retrieval(), "agents": agents()}
    json.dump(res, open(os.path.join(EVAL, "stats.json"), "w"), indent=1)
    for name, v in res["retrieval"]["paired"].items():
        print(name, {m: v["all"][m] for m in RET_METRICS})
    for name, v in res["agents"]["paired"].items():
        print(name, v)
    for m, v in res["agents"]["ci"].items():
        print(m, v)
