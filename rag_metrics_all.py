"""
rag_metrics_all.py: the full metric set for the RAG benchmark comparison (4 RAG methods x 1,000 questions), in four families.

  1 retrieval   Precision@10, Recall@10/100, Hit@10, MRR@100, MAP@100, nDCG@10 for every retriever on all 9,952
                scored questions (gold papers in the index), unfiltered and abstract-only index
  2 generation  verdict Exact Match + macro-F1 (PubMedQA yes/no/maybe, SciFact SUPPORT/CONTRADICT/NEI);
                text Exact Match, token F1 (SQuAD), ROUGE-1/2/L, BLEU (sacreBLEU), BERTScore (roberta-large,
                baseline-rescaled; CPU) against the expert reference answers (BioASQ ideal answer, PubMedQA long answer)
  3 grounding   context precision (rank-aware, RAGAS-style, from the pooled 0/1/2 relevance judgements),
                context recall (share of gold papers in the evidence), citation accuracy (cited paper is direct
                evidence) and proof validity (verbatim quote from a relevant paper);
                faithfulness, hallucination rate and answer relevancy from an Opus claim-level judge (`judge`,
                stratified 50 questions per dataset = 200 per method, the same questions for every method)
  4 efficiency  Claude calls, input / output tokens, list-price cost and Claude time per question for each method
                (replayed exactly from the call cache, which stores each call's usage); retrieval time per query
                (batch exact search over 484M papers, from the search log)

  python rag_metrics_all.py offline   -> rag_eval\\metrics_all.json   (families 1, 2, 3-offline, 4)
  python rag_metrics_all.py judge     -> rag_eval\\agents\\grounding_judged.jsonl  (Opus calls, resumable)
  python rag_metrics_all.py offline   (again, to fold the judge results in)
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import re
import sys
import json
import math
import random
import string
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import rag_llm
import rag_score
import rag_agents as RA

EVAL = RA.EVAL
AG = RA.AG
OUT = os.path.join(EVAL, "metrics_all.json")
JUDGED = os.path.join(AG, "grounding_judged.jsonl")
JUDGE = "opus"                         # stronger than, and different from, the answering model (sonnet)
PER_DS_JUDGE = 50
B = 2000                               # bootstrap resamples


# ---------------------------------------------------------------- helpers
def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def boot_ci(xs, seed=7):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(B))
    return [round(ms[int(.025 * B)], 4), round(ms[int(.975 * B)], 4)]


def stat(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": mean(xs), "ci95": boot_ci(xs), "n": len(xs)}


# ---------------------------------------------------------------- 1 retrieval
def retrieval_extra(ids, g):
    """Precision@10 and AP@100 (gold papers in the index as the relevant set; each gold paper counts once)."""
    id2key = {i: k for k, (_, s) in g.items() for i in s}
    seen, hits_at, found = set(), [], 0
    for rank, i in enumerate(ids[:100], 1):
        k = id2key.get(i)
        if k and k not in seen:
            seen.add(k)
            found += 1
            hits_at.append(found / rank)
    top10 = len({id2key[i] for i in ids[:10] if i in id2key})
    return {"precision@10": top10 / 10, "map@100": sum(hits_at) / len(g)}


def family_retrieval(gold):
    res = {}
    for name in ["round1.minilm", "round1.specter2", "round1.hybrid", "round1.hybrid_medcpt",
                 "round1abs.minilm", "round1abs.specter2", "round1abs.hybrid", "round1abs.hybrid_medcpt"]:
        p = os.path.join(EVAL, name + ".hits.parquet")
        if not os.path.exists(p):
            continue
        ranked = rag_score.load_ranked(p)
        per = defaultdict(lambda: defaultdict(list))
        for qid, g in gold.items():
            ids = ranked.get(qid, [])
            m = {**rag_score.score_question(ids, g), **retrieval_extra(ids, g)}
            for ds in (qid.split(":", 1)[0], "all"):
                for k, v in m.items():
                    per[ds][k].append(v)
        res[name] = {ds: {"questions": len(v["hit@10"]), **{k: round(sum(x) / len(x), 4) for k, x in v.items()}}
                     for ds, v in per.items()}
        print("retrieval", name, res[name]["all"], flush=True)
    return res


# ---------------------------------------------------------------- 2 generation
CITE = re.compile(r"\s*\[(?:F?\d+(?:\s*[,–-]\s*F?\d+)*)\]")


def strip_cites(s):
    return CITE.sub("", s or "").strip()


def norm_answer(s):            # SQuAD normalisation
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, ref):
    p, r = norm_answer(pred).split(), norm_answer(ref).split()
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if not p or not r or not same:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def macro_f1(pairs, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for p, g in pairs if p == c and g == c)
        fp = sum(1 for p, g in pairs if p == c and g != c)
        fn = sum(1 for p, g in pairs if p != c and g == c)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return round(sum(f1s) / len(f1s), 4) if f1s else None


def family_generation(Q, S, sample):
    from rouge_score import rouge_scorer
    import sacrebleu
    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    res = {m: {} for m in RA.METHODS}
    pairs = []                                              # (method, ds, qid, pred, ref) for the text metrics
    for m in RA.METHODS:
        for ds, labels in (("pubmedqa", RA.VERDICTS["pubmedqa"]), ("scifact", RA.VERDICTS["scifact"])):
            vp = [((S[m][q].get("answer") or {}).get("verdict"), Q[q]["label"]) for q in sample
                  if Q[q]["dataset"] == ds and Q[q]["label"] in labels]
            res[m][ds] = {"verdict_exact_match": stat([float(p == g) for p, g in vp]),
                          "verdict_macro_f1": macro_f1(vp, labels), "n": len(vp)}
        for q in sample:
            if Q[q]["dataset"] in ("bioasq", "pubmedqa") and Q[q].get("answer"):
                a = (S[m][q].get("answer") or {}).get("answer") or ""
                pairs.append((m, Q[q]["dataset"], q, strip_cites(a), Q[q]["answer"]))
    print(f"generation: {len(pairs)} answer/reference pairs; BERTScore on CPU ...", flush=True)
    from bert_score import score as bscore
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    P, R_, F = bscore([p[3] or "." for p in pairs], [p[4] for p in pairs], lang="en", model_type="roberta-large",
                      rescale_with_baseline=True, device="cpu", batch_size=16, verbose=False)
    Fraw = bscore([p[3] or "." for p in pairs], [p[4] for p in pairs], lang="en", model_type="roberta-large",
                  rescale_with_baseline=False, device="cpu", batch_size=16, verbose=False)[2]
    per = defaultdict(lambda: defaultdict(list))
    for (m, ds, q, pred, ref), f, fr in zip(pairs, F.tolist(), Fraw.tolist()):
        r = rs.score(ref, pred)
        vals = {"exact_match": float(norm_answer(pred) == norm_answer(ref)), "token_f1": token_f1(pred, ref),
                "rouge1": r["rouge1"].fmeasure, "rouge2": r["rouge2"].fmeasure, "rougeL": r["rougeL"].fmeasure,
                "bleu_sentence": sacrebleu.sentence_bleu(pred, [ref]).score / 100,
                "bertscore_f1": f, "bertscore_f1_raw": fr}
        for k, v in vals.items():
            per[(m, ds)][k].append(v)
            per[(m, "both")][k].append(v)
        per[(m, ds)]["_p"].append(pred), per[(m, ds)]["_r"].append(ref)
        per[(m, "both")]["_p"].append(pred), per[(m, "both")]["_r"].append(ref)
    for (m, ds), v in per.items():
        out = {k: stat(x) for k, x in v.items() if not k.startswith("_")}
        out["bleu_corpus"] = round(sacrebleu.corpus_bleu(v["_p"], [v["_r"]]).score / 100, 4)
        out["n"] = len(v["_p"])
        res[m].setdefault("text_" + ds, out)
    return res


# ---------------------------------------------------------------- 3 grounding
def rel_of(S, gold, qid):
    rel = dict(S.get("judge", {}).get(qid, {}))
    for _, ids in gold.get(qid, {}).values():
        for i in ids:
            rel[i] = max(rel.get(i, 0), 2)
    return rel


def context_precision(ev, rel):
    """RAGAS context precision: mean of precision@k over the ranks k that hold a relevant paper (relevance >= 1)."""
    v = [1 if rel.get(d, 0) >= 1 else 0 for d in ev]
    if not sum(v):
        return 0.0 if ev else None
    return sum(sum(v[:k]) / k for k in range(1, len(v) + 1) if v[k - 1]) / sum(v)


SYS_CLAIMS_RAG = (
    "You are a strict fact-checking annotator for a biomedical question-answering system. You get the question, "
    "the system's answer and the only sources it was allowed to use: numbered papers [1], [2], ... (title and "
    "abstract, exactly as the system saw them). Split the answer into atomic factual claims: one checkable fact "
    "each; include numbers, effects, populations and conclusions; skip hedges and restatements of the question. "
    "For each claim give the paper numbers cited for it in the answer ([] if none) and a label:\n"
    "  supported: fully entailed by the sources (the cited ones, or any source if uncited);\n"
    "  partial: the gist is in the sources but a detail is overstated, more general or not in them;\n"
    "  unsupported: not found in any source, even if it may be true in general;\n"
    "  contradicted: a source says otherwise.\n"
    "cited_ok: true if at least one cited paper supports the claim, false if none does, null if uncited. Be strict "
    "and literal. Finally rate answer relevancy: how well the answer addresses the question asked: 2 fully, "
    "1 partly, 0 not (declining for lack of evidence counts as 1 if the sources really lack it).")
SCH_CLAIMS_RAG = {"type": "object", "properties": {
    "claims": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "cites": {"type": "array", "items": {"type": "integer"}},
        "label": {"type": "string", "enum": ["supported", "partial", "unsupported", "contradicted"]},
        "cited_ok": {"type": ["boolean", "null"]}},
        "required": ["claim", "cites", "label", "cited_ok"]}},
    "relevance": {"type": "integer", "enum": [0, 1, 2]}}, "required": ["claims", "relevance"]}


def judge_sample(Q, sample):
    rng = random.Random(29)
    by = defaultdict(list)
    for q in sample:
        by[Q[q]["dataset"]].append(q)
    return sorted(q for ds in sorted(by) for q in rng.sample(sorted(by[ds]), min(PER_DS_JUDGE, len(by[ds]))))


def judge(Q, S, sample):
    js = judge_sample(Q, sample)
    have = set()
    if os.path.exists(JUDGED):
        have = {(r["method"], r["qid"]) for r in map(json.loads, open(JUDGED, encoding="utf-8"))}
    items = [(m, q) for q in js for m in RA.METHODS if (m, q) not in have and (S[m][q].get("answer") or {}).get("answer")]
    RA.DOCS.fetch([d for m, q in items for d in S[m][q].get("evidence", [])])
    print(f"judging {len(items)} answers ({len(js)} questions x {len(RA.METHODS)} methods) with {JUDGE}", flush=True)
    import threading
    lock = threading.Lock()

    def one(it):
        m, q = it
        st = S[m][q]
        out = rag_llm.ask(RA.task_text(Q[q]) + f"\n\nSYSTEM ANSWER:\n{st['answer']['answer']}\n\nSOURCES:\n"
                          + RA.papers_text(st.get("evidence", [])), system=SYS_CLAIMS_RAG, schema=SCH_CLAIMS_RAG,
                          model=JUDGE, timeout=300)
        with lock, open(JUDGED, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": m, "qid": q, "dataset": Q[q]["dataset"], **out}, ensure_ascii=False) + "\n")
        print("judged", m, q, len(out["claims"]), "claims", flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, items))


def family_grounding(Q, S, sample, gold):
    res = {}
    judged = defaultdict(dict)
    if os.path.exists(JUDGED):
        for r in map(json.loads, open(JUDGED, encoding="utf-8")):
            judged[r["method"]][r["qid"]] = r
    for m in RA.METHODS:
        per = defaultdict(lambda: defaultdict(list))
        for q in sample:
            st, ds = S[m][q], Q[q]["dataset"]
            ev, a = st.get("evidence", []), st.get("answer") or {}
            rel = rel_of(S, gold, q)
            g = gold.get(q, {})
            vals = {"context_precision": context_precision(ev, rel)}
            if g:
                vals["context_recall"] = rag_score.score_question(ev, g)["recall@100"]
            cited = [ev[n - 1] for n in a.get("citations", []) if 1 <= n <= len(ev)]
            if cited:
                vals["citation_accuracy"] = sum(rel.get(d, 0) == 2 for d in cited) / len(cited)
                vals["citation_relevant"] = sum(rel.get(d, 0) >= 1 for d in cited) / len(cited)
            j = judged[m].get(q)
            if j:
                cl = j["claims"]
                lab = Counter(c["label"] for c in cl)
                n = len(cl)
                vals.update(answer_relevancy=j["relevance"] / 2)
                if n:
                    vals.update(faithfulness=(lab["supported"] + .5 * lab["partial"]) / n,
                                supported_rate=lab["supported"] / n,
                                hallucination_rate=(lab["unsupported"] + lab["contradicted"]) / n,
                                contradiction_rate=lab["contradicted"] / n,
                                hallucination_free=float(lab["unsupported"] + lab["contradicted"] == 0))
                    ck = [c["cited_ok"] for c in cl if c["cited_ok"] is not None]
                    if ck:
                        vals["claim_citation_precision"] = sum(ck) / len(ck)
                    vals["claim_citation_coverage"] = sum(1 for c in cl if c["cites"]) / n
                    vals["_claims"] = n
                    for k in ("supported", "partial", "unsupported", "contradicted"):
                        vals["_" + k] = lab[k]
            for d in (ds, "all"):
                for k, v in vals.items():
                    per[d][k].append(v)
        res[m] = {}
        for d, v in per.items():
            res[m][d] = {k: stat(x) for k, x in v.items() if not k.startswith("_")}
            res[m][d]["claims"] = {k[1:]: sum(x) for k, x in v.items() if k.startswith("_")}
    return res


# ---------------------------------------------------------------- 4 efficiency
class NotCached(Exception):
    pass


def cached_only(prompt, system, schema=None, model="sonnet", **kw):
    if not os.path.exists(rag_llm._key(model, system, prompt, schema)):
        raise NotCached()
    return rag_llm.ask(prompt, system=system, schema=schema, model=model)


def family_efficiency(Q, sample):
    RA.ask = cached_only                       # replay every method from the call cache: same calls, no new ones
    res, per = {}, defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    incomplete = Counter()
    for m in RA.METHODS:
        for q in sample:
            st = {}
            with rag_llm.collect() as calls:
                try:
                    RA.advance(Q, m, q, st)
                except (NotCached, KeyError):
                    incomplete[m] += 1
                    continue
            u = lambda k: sum((c["usage"].get(k) or 0) for c in calls)
            inp = u("input_tokens") + u("cache_creation_input_tokens") + u("cache_read_input_tokens")
            vals = {"claude_calls": len(calls), "input_tokens": inp, "output_tokens": u("output_tokens"),
                    "total_tokens": inp + u("output_tokens"),
                    "cost_list_usd": sum(c.get("original_cost_list_usd") or 0 for c in calls),
                    "claude_secs": sum(c.get("original_secs") or 0 for c in calls),
                    "follow_up_searches": sum(1 for k in RA.R if k.startswith(q + "|" + m + "|"))}
            for d in (Q[q]["dataset"], "all"):
                for k, v in vals.items():
                    per[m][d][k].append(v)
        print("efficiency", m, "replayed", len(per[m]["all"]["claude_calls"]), "incomplete", incomplete[m], flush=True)
    for m in RA.METHODS:
        res[m] = {d: {k: {"mean": round(sum(x) / len(x), 4), "p50": sorted(x)[len(x) // 2], "n": len(x)}
                      for k, x in v.items()} for d, v in per[m].items()}
        a = per[m]["all"]
        res[m]["all"]["cost_per_1000_questions_usd"] = round(1000 * sum(a["cost_list_usd"]) / len(a["cost_list_usd"]), 2)
        res[m]["incomplete_replays"] = incomplete[m]
    # retrieval: exact batch search over all 484M papers, from the search log
    lines = open(os.path.join(EVAL, "rag_eval.log"), encoding="utf-8").read().splitlines()
    rx = re.compile(r"\[search\] (\w+): (encoded|searched) ([\d,]+) queries.* in (\d+)(s| min) -> .*\\(round1abs|round1)\.\w+\.")
    t = defaultdict(float)
    nq = {}
    for ln in lines:
        mm = rx.search(ln)
        if mm:
            secs = int(mm.group(4)) * (60 if mm.group(5) == " min" else 1)
            t[mm.group(6)] += secs
            nq[mm.group(6)] = int(mm.group(3).replace(",", ""))
    res["retrieval_batch"] = {k: {"queries": nq[k], "wall_secs": t[k], "secs_per_query": round(t[k] / nq[k], 3),
                                  "queries_per_min": round(60 * nq[k] / t[k], 1),
                                  "note": "MiniLM + SPECTER2 encode + exact scan of all 484M papers, both models, GPU"}
                              for k in t}
    return res


# ---------------------------------------------------------------- main
def main():
    cmd = sys.argv[1]
    Q = RA.load_questions()
    gold = rag_score.load_gold()
    S = json.load(open(RA.state_path(), encoding="utf-8"))
    sample = RA.make_sample(Q, gold, 250)
    RA.R.update(json.load(open(os.path.join(AG, "retrieved.json"), encoding="utf-8")))
    RA.DOCS = RA.DocStore()
    if cmd == "judge":
        judge(Q, S, sample)
        return
    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    only = sys.argv[2:] or ["retrieval", "generation", "grounding", "efficiency"]
    if "retrieval" in only:
        out["retrieval"] = family_retrieval(gold)
    if "generation" in only:
        out["generation"] = family_generation(Q, S, sample)
    if "grounding" in only:
        out["grounding"] = family_grounding(Q, S, sample, gold)
        out["grounding_judge"] = {"model": JUDGE, "questions_per_method": len(judge_sample(Q, sample)),
                                  "per_dataset": PER_DS_JUDGE}
    if "efficiency" in only:
        out["efficiency"] = family_efficiency(Q, sample)
    out["sample_questions"] = len(sample)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
