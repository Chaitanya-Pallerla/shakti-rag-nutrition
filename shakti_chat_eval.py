"""
shakti_chat_eval.py: scripted evaluation of the live SHAKTI chat (context awareness + hallucination).

  (an optional second argument names the version folder; default "after")
  python shakti_chat_eval.py run       -> asks the 12 scripted conversations through http://127.0.0.1:8780
  python shakti_chat_eval.py judge     -> an independent judge (Claude Opus) sees exactly the sources SHAKTI saw
  python shakti_chat_eval.py score     -> metrics with 95% bootstrap CIs over conversations -> scores.json

Every step is resumable. The sessions are created with role "eval", so shakti_chat_analyze.py leaves
them out of the participant statistics. The gold labels below were written before any answer was seen.

Context awareness (per turn, against the gold labels):
  * follow-up detection: the system's is_followup vs the gold label (accuracy, precision, recall, F1)
  * concept recall: share of the gold concept groups (synonym lists) present in the system's standalone
    rewrite; the raw user message is the no-context baseline
  * topic leakage: on a topic switch, the rewrite wrongly carries a concept of the previous topic
  * rewrite fidelity (judge, 0-2): the rewrite keeps the meaning of the gold standalone question
  * answer relevance (judge, 0-2): the answer addresses the gold standalone question
Hallucination (per answer; the judge sees the full abstracts and USDA values the system was given):
  * the answer is split into atomic claims, each labelled supported / partial / unsupported / contradicted
  * faithfulness = (supported + 0.5 partial) / claims; hallucination rate = (unsupported + contradicted) / claims
  * citation precision = cited claims whose cited source supports them / cited claims
  * plus the online checks (verbatim quotes, citation validity, number grounding, dead links removed)
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import re
import sys
import json
import time
import random
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from rag_llm import ask

BASE = "http://127.0.0.1:8780"
ROOT = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "shakti_chat_eval")
# one folder per evaluated version: "before" (8 food nutrients, first-listed portions), "after" (food fixes)
OUT = os.path.join(ROOT, sys.argv[2] if len(sys.argv) > 2 else "after")
JUDGE = "opus"                       # a different, stronger model than the one answering (sonnet)
B = 5000

K, CAL, FIB, PROT = ["calori", "kcal", "energy"], ["calori", "kcal", "energy"], ["fiber", "fibre"], ["protein"]
# turn: (message, gold is_followup, gold standalone question, concept groups, leak groups)
CONVS = [
    ("rice", [
        ("How many calories are in a cup of cooked white rice?", False,
         "How many calories are in a cup of cooked white rice?", [["white rice"], CAL], []),
        ("And brown rice?", True, "How many calories are in a cup of cooked brown rice?", [["brown rice"], CAL], []),
        ("Which of the two has more fiber?", True, "Does cooked white rice or brown rice have more fiber?",
         [["white rice"], ["brown rice"], FIB], []),
    ]),
    ("fasting", [
        ("Does intermittent fasting help with weight loss?", False,
         "Does intermittent fasting help with weight loss?", [["intermittent fasting"], ["weight"]], []),
        ("Is it safe for people with type 2 diabetes?", True,
         "Is intermittent fasting safe for people with type 2 diabetes?",
         [["fasting", "time-restricted"], ["type 2 diabetes", "diabet"]], []),
        ("What about during pregnancy?", True, "Is intermittent fasting safe during pregnancy?",
         [["fasting", "time-restricted"], ["pregnan"]], []),
    ]),
    ("coffee", [
        ("Is coffee bad for blood pressure?", False, "Is coffee bad for blood pressure?",
         [["coffee"], ["blood pressure", "hypertension"]], []),
        ("How much per day is considered safe?", True,
         "How much coffee per day is considered safe for blood pressure?",
         [["coffee", "caffeine"], ["per day", "daily", "a day", "cups", "intake", "how much"]], []),
        ("Does decaf have the same effect?", True,
         "Does decaffeinated coffee have the same effect on blood pressure?",
         [["decaf"], ["blood pressure", "hypertension"]], []),
    ]),
    ("nuts-vitd", [
        ("What are the health benefits of eating nuts?", False, "What are the health benefits of eating nuts?",
         [["nut"]], []),
        ("Which ones have the most protein?", True, "Which nuts have the most protein?", [["nut"], PROT], []),
        ("Different question: can vitamin D supplements prevent colds?", False,
         "Can vitamin D supplements prevent colds?", [["vitamin d"], ["cold", "respiratory"]], [["nut"]]),
    ]),
    ("keto", [
        ("Is the ketogenic diet effective for epilepsy in children?", False,
         "Is the ketogenic diet effective for epilepsy in children?", [["ketogenic", "keto"], ["epilep"], ["child", "pediatric", "paediatric"]], []),
        ("What side effects does it have?", True,
         "What side effects does the ketogenic diet have in children with epilepsy?",
         [["ketogenic", "keto"], ["side effect", "adverse"]], []),
        ("Would it also work for adults?", True, "Is the ketogenic diet effective for epilepsy in adults?",
         [["ketogenic", "keto"], ["adult"], ["epilep", "seizure"]], []),
    ]),
    ("eggs", [
        ("Compare the protein in eggs and chicken breast.", False,
         "How much protein is in eggs compared with chicken breast?", [["egg"], ["chicken"], PROT], []),
        ("Which one has less fat?", True, "Do eggs or chicken breast have less fat?",
         [["egg"], ["chicken"], ["fat"]], []),
        ("And is eating them daily bad for cholesterol?", True, "Is eating eggs every day bad for cholesterol?",
         [["egg"], ["cholesterol"]], []),
    ]),
    ("green-tea", [
        ("Does green tea help burn fat?", False, "Does green tea help burn fat?", [["green tea"], ["fat", "weight"]], []),
        ("What about matcha?", True, "Does matcha help burn fat?", [["matcha"], ["fat", "weight", "metabol"]], []),
        ("Can it interact with any medications?", True, "Can matcha or green tea interact with medications?",
         [["matcha", "green tea"], ["interact", "medication", "drug"]], []),
    ]),
    ("breakfast", [
        ("I had 2 slices of whole wheat toast with peanut butter and a banana for breakfast. How many calories was that?",
         False, "How many calories are in 2 slices of whole wheat toast with peanut butter and a banana?",
         [["toast", "bread"], ["peanut butter"], ["banana"], CAL], []),
        ("How much protein was in it?", True,
         "How much protein is in 2 slices of whole wheat toast with peanut butter and a banana?",
         [PROT, ["toast", "bread", "peanut butter", "breakfast", "meal"]], []),
        ("Is that a good breakfast for someone with prediabetes?", True,
         "Is whole wheat toast with peanut butter and a banana a good breakfast for someone with prediabetes?",
         [["prediabet"], ["toast", "bread", "peanut butter", "banana"]], []),
    ]),
    ("probiotics", [
        ("Can probiotics help with irritable bowel syndrome?", False,
         "Can probiotics help with irritable bowel syndrome?", [["probiotic"], ["irritable bowel", "ibs"]], []),
        ("Which strains work best?", True, "Which probiotic strains work best for irritable bowel syndrome?",
         [["probiotic"], ["strain"], ["irritable bowel", "ibs"]], []),
        ("How long before you see results?", True,
         "How long do probiotics take to improve irritable bowel syndrome symptoms?",
         [["probiotic"], ["irritable bowel", "ibs"], ["how long", "duration", "week", "time"]], []),
    ]),
    ("cola-creatine", [
        ("How much sugar is in a can of cola?", False, "How much sugar is in a can of cola?",
         [["cola", "soda", "soft drink"], ["sugar"]], []),
        ("Unrelated, but is creatine safe for teenagers who lift weights?", False,
         "Is creatine safe for teenagers who lift weights?", [["creatine"], ["teen", "adolesc"]], [["cola", "soda", "sugar"]]),
        ("What dose do studies use?", True, "What creatine dose do studies in teenagers use?",
         [["creatine"], ["dose", "dosage", "g/day", "gram"]], []),
    ]),
    ("mediterranean", [
        ("Does the Mediterranean diet lower the risk of heart disease?", False,
         "Does the Mediterranean diet lower the risk of heart disease?",
         [["mediterranean"], ["heart", "cardiovascular", "coronary"]], []),
        ("And dementia?", True, "Does the Mediterranean diet lower the risk of dementia?",
         [["mediterranean"], ["dementia", "cognitive", "alzheimer"]], []),
        ("Is olive oil the main reason?", True,
         "Is olive oil the main reason the Mediterranean diet lowers the risk of dementia?",
         [["olive oil"], ["mediterranean"]], []),
    ]),
    ("iron", [
        ("How much iron is in 100 g of spinach?", False, "How much iron is in 100 g of spinach?",
         [["iron"], ["spinach"]], []),
        ("How much vitamin C is in an orange?", False, "How much vitamin C is in an orange?",
         [["vitamin c", "ascorbic"], ["orange"]], [["spinach", "iron"]]),
        ("Does eating them together help absorb the iron?", True,
         "Does eating an orange with spinach increase iron absorption from the spinach?",
         [["orange", "vitamin c", "ascorbic"], ["spinach"], ["iron"], ["absor"]], []),
    ]),
]


def path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def read(name):
    p = path(name)
    return [json.loads(ln) for ln in open(p, encoding="utf-8")] if os.path.exists(p) else []


def append(name, rec):
    with open(path(name), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def call(p, body=None):
    req = urllib.request.Request(BASE + p, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")


# ---------------------------------------------------------------- 1) run the conversations

def run_conv(ci):
    name, turns = CONVS[ci]
    sid = call("/api/session", {"consent": True, "adult": True, "role": "eval", "background": "eval"})["sid"]
    for ti, (q, *_rest) in enumerate(turns):
        t = time.time()
        for _ in range(20):
            j = call("/api/ask", {"sid": sid, "conv": f"eval-{name}", "question": q})
            if "job" in j:
                break
            time.sleep(15)                                  # queue full: wait
        while True:
            time.sleep(3)
            s = call(f"/api/job?id={j['job']}&sid={sid}")
            if s["status"] in ("done", "error", "unknown"):
                break
        r = s.get("result") or {}
        print(f"[{name} {ti + 1}] {s['status']} {time.time() - t:.0f}s  {r.get('context', {}).get('standalone', '')}",
              flush=True)
        if s["status"] != "done":
            raise RuntimeError(f"{name} turn {ti + 1}: {s.get('error')}")
        append("runs.jsonl", {"conv": name, "turn": ti, "job": j["job"], "sid": sid, "question": q,
                              "secs": round(time.time() - t, 1), "result": r})


def run():
    done = {r["conv"] for r in read("runs.jsonl") if r["turn"] == len(dict(CONVS)[r["conv"]]) - 1}
    # a conversation interrupted half-way is rerun from its first turn in a new session
    keep = [r for r in read("runs.jsonl") if r["conv"] in done]
    with open(path("runs.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
    todo = [i for i, (n, _) in enumerate(CONVS) if n not in done]
    with ThreadPoolExecutor(2) as ex:                       # the server runs 2 questions at a time
        list(ex.map(run_conv, todo))
    # the arm of each answer, from the server log
    arms = {}
    for ln in open(os.path.join(os.path.dirname(ROOT), "shakti_chat", "turns.jsonl"), encoding="utf-8"):
        t = json.loads(ln)
        arms[t["turn"]] = {"arm": t["arm"], "understand": t["understand"], "context_metrics": t["context_metrics"],
                           "dead_sources_removed": t["dead_sources_removed"]}
    rows = read("runs.jsonl")
    with open(path("runs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            r.update(arms.get(r["job"], {}))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 2) judge

SYS_CLAIMS = (
    "You are a strict fact-checking annotator for a health and nutrition question-answering system. You get the "
    "system's answer and the only sources it was allowed to use: numbered research papers [1], [2], ... (title "
    "and abstract) and USDA food values [F1], [F2], .... Split the answer into atomic factual claims: one checkable "
    "fact each; include numbers, effects, populations and recommendations; skip pure hedges, 'consult a doctor' "
    "advice and restatements of the question. For each claim give the citation labels attached to it in the "
    "answer (e.g. ['2','F1']; [] if none) and a label:\n"
    "  supported: fully entailed by the sources (the cited ones, or any source if the claim is uncited);\n"
    "  partial: the gist is in the sources but a detail is overstated, more general or not in them "
    "(a number, the population, the strength of the conclusion);\n"
    "  unsupported: not found in any source, even if it may be true in general;\n"
    "  contradicted: a source says otherwise.\n"
    "cited_ok: true if at least one cited source supports the claim, false if none does, null if uncited. "
    "any_source: true if any source supports it. general_knowledge: true if the claim is common, uncontroversial "
    "background (e.g. 'fiber aids digestion') rather than a specific finding. Arithmetic on source values "
    "(totals, differences, unit conversions) counts as supported when it is correct. Be strict and literal.\n"
    "Finally rate how well the answer addresses the REFERENCE question (what the user meant): 2 fully, 1 partly, "
    "0 not.")
SCH_CLAIMS = {"type": "object", "properties": {
    "claims": {"type": "array", "items": {"type": "object", "properties": {
        "claim": {"type": "string"}, "cites": {"type": "array", "items": {"type": "string"}},
        "label": {"type": "string", "enum": ["supported", "partial", "unsupported", "contradicted"]},
        "cited_ok": {"type": ["boolean", "null"]}, "any_source": {"type": "boolean"},
        "general_knowledge": {"type": "boolean"}, "note": {"type": "string"}},
        "required": ["claim", "cites", "label", "cited_ok", "any_source", "general_knowledge"]}},
    "relevance": {"type": "integer", "enum": [0, 1, 2]}, "relevance_reason": {"type": "string"}},
    "required": ["claims", "relevance", "relevance_reason"]}

SYS_REWRITE = (
    "You annotate query rewriting in a multi-turn health chat. For each turn you see the conversation so far, "
    "the user's message, the REFERENCE standalone question (what the user meant, written by a researcher) and "
    "the SYSTEM's standalone rewrite. Score the system's rewrite: 2 = same meaning as the reference (wording may "
    "differ; extra harmless detail is fine), 1 = partly right (a referent missing, wrong or a wrong topic carried "
    "over, but the main subject is right), 0 = wrong meaning.")
SCH_REWRITE = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "turn": {"type": "integer"}, "score": {"type": "integer", "enum": [0, 1, 2]}, "reason": {"type": "string"}},
    "required": ["turn", "score", "reason"]}}}, "required": ["items"]}


DOCS = None


def sources_text(r):
    """The papers exactly as SHAKTI saw them (same index records, same 1,500-character cut), plus the USDA values."""
    global DOCS
    if DOCS is None:
        import rag_agents as RA
        DOCS = RA.DocStore(path("docs.jsonl"))
        DOCS.cut = RA.ABSTRACT_CHARS
    papers = r["result"]["papers"]
    DOCS.fetch([p["id"] for p in papers])
    parts = []
    for p in papers:
        d = DOCS.get(p["id"])
        parts.append(f"[{p['n']}] {d.get('title') or p.get('title') or '(untitled)'} ({d.get('year') or p.get('year') or 'n.d.'})\n"
                     f"{(d.get('abstract') or '(no abstract)')[:DOCS.cut]}")
    for i, f in enumerate(r["result"]["foods"], 1):
        parts.append(f"[F{i}] " + json.dumps({k: f.get(k) for k in ("name", "portion", "grams", "serving", "per100")
                                              if f.get(k) is not None}, ensure_ascii=False))
    return "\n\n".join(parts) or "(no sources)"


def judge():
    rows = read("runs.jsonl")
    gold = {(n, i): t for n, ts in CONVS for i, t in enumerate(ts)}
    have = {(r["conv"], r["turn"]) for r in read("judged.jsonl")}
    srcs = {(r["conv"], r["turn"]): sources_text(r) for r in rows}      # index reads once, before the threads

    def one(r):
        if (r["conv"], r["turn"]) in have or r.get("arm") == "none":
            return
        g = gold[(r["conv"], r["turn"])]
        earlier = "\n\n".join(f"User: {e['question']}\nSHAKTI: {e['result']['answer']}" for e in
                              sorted((e for e in rows if e["conv"] == r["conv"] and e["turn"] < r["turn"]), key=lambda e: e["turn"]))
        out = ask(f"REFERENCE question: {g[2]}\n\n"
                  + (f"EARLIER IN THIS CONVERSATION (values SHAKTI already gave from its sources may be reused; "
                     f"treat them as a source labelled 'earlier'):\n{earlier}\n\n" if earlier else "")
                  + f"SYSTEM ANSWER:\n{r['result']['answer']}\n\n"
                  f"SOURCES (abstracts exactly as the system saw them, cut at {DOCS.cut} characters):\n{srcs[(r['conv'], r['turn'])]}",
                  system=SYS_CLAIMS, schema=SCH_CLAIMS, model=JUDGE, timeout=300)
        append("judged.jsonl", {"conv": r["conv"], "turn": r["turn"], **out})
        print("judged", r["conv"], r["turn"], len(out["claims"]), "claims", flush=True)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(one, rows))
    have = {r["conv"] for r in read("rewrites.jsonl")}
    for n, ts in CONVS:
        if n in have:
            continue
        rs = sorted((r for r in rows if r["conv"] == n), key=lambda r: r["turn"])
        txt = []
        for r in rs:
            hist = " | ".join(t[0] for t in ts[:r["turn"]]) or "(start of conversation)"
            txt.append(f"Turn {r['turn']}:\n  earlier user messages: {hist}\n  user message: {r['question']}\n"
                       f"  REFERENCE: {ts[r['turn']][2]}\n  SYSTEM: {r['result']['context']['standalone']}")
        out = ask("\n\n".join(txt), system=SYS_REWRITE, schema=SCH_REWRITE, model=JUDGE, timeout=300)
        append("rewrites.jsonl", {"conv": n, **out})
        print("rewrites judged", n, flush=True)


# ---------------------------------------------------------------- 3) score

def has(text, group):
    t = text.lower()
    return any(s in t for s in group)


def boot(rows, f, seed=0):
    """rows: list of (conversation, value). Cluster bootstrap over conversations."""
    vals = [v for _, v in rows if v is not None]
    if not vals:
        return None
    by = {}
    for c, v in rows:
        if v is not None:
            by.setdefault(c, []).append(v)
    keys, rnd, bs = list(by), random.Random(seed), []
    for _ in range(B):
        pick = [v for k in (rnd.choice(keys) for _ in keys) for v in by[k]]
        bs.append(f(pick))
    bs.sort()
    return {"mean": round(f(vals), 3), "ci": [round(bs[int(0.025 * B)], 3), round(bs[int(0.975 * B) - 1], 3)],
            "n": len(vals)}


avg = lambda x: sum(x) / len(x)


def ratio(pairs):                       # pairs of (numerator, denominator): pooled ratio
    d = sum(b for _, b in pairs)
    return sum(a for a, _ in pairs) / d if d else 0.0


def score():
    rows = {(r["conv"], r["turn"]): r for r in read("runs.jsonl")}
    jd = {(r["conv"], r["turn"]): r for r in read("judged.jsonl")}
    rw = {(r["conv"], it["turn"]): it["score"] for r in read("rewrites.jsonl") for it in r["items"]}
    per = []
    for n, ts in CONVS:
        for i, (q, g_fu, g_sq, concepts, leaks) in enumerate(ts):
            r = rows.get((n, i))
            if not r:
                continue
            sq = r["result"]["context"]["standalone"]
            fu = bool(r["result"]["context"]["is_followup"])
            j = jd.get((n, i))
            cl = j["claims"] if j else []
            lab = [c["label"] for c in cl]
            cited = [c for c in cl if c["cites"]]
            v = r["result"]["verification"]
            per.append({
                "conv": n, "turn": i, "arm": r.get("arm"), "secs": r["secs"], "question": q, "gold_followup": g_fu,
                "gold_standalone": g_sq, "standalone": sq, "followup": fu,
                "concept_recall": avg([has(sq, c) for c in concepts]),
                "concept_recall_raw": avg([has(q, c) for c in concepts]),
                "leak": (any(has(sq, c) for c in leaks) if leaks else None),
                "rewrite_score": rw.get((n, i)), "relevance": j["relevance"] if j else None,
                "claims": len(cl), "supported": lab.count("supported"), "partial": lab.count("partial"),
                "unsupported": lab.count("unsupported"), "contradicted": lab.count("contradicted"),
                "unsupported_specific": sum(1 for c in cl if c["label"] == "unsupported" and not c["general_knowledge"]),
                "cited_claims": len(cited), "cited_ok": sum(1 for c in cited if c["cited_ok"]),
                "verification": v, "risk": v.get("risk"),
                "dead_removed": len(r.get("dead_sources_removed") or []),
                "links": [p["link_status"] for p in r["result"]["papers"]],
                "papers": len(r["result"]["papers"]), "foods": len(r["result"]["foods"]),
                "judge_claims": cl,
            })
    fus = [p for p in per if p["gold_followup"]]
    tp = sum(1 for p in per if p["followup"] and p["gold_followup"])
    fp = sum(1 for p in per if p["followup"] and not p["gold_followup"])
    fn = sum(1 for p in per if not p["followup"] and p["gold_followup"])
    prec, rec = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
    C = lambda key, ps=per, f=avg: boot([(p["conv"], p[key]) for p in ps], f)
    ctx = {
        "turns": len(per), "followup_turns": len(fus), "topic_switch_turns": sum(1 for p in per if p["leak"] is not None),
        "followup_detection_accuracy": boot([(p["conv"], float(p["followup"] == p["gold_followup"])) for p in per], avg),
        "followup_precision": round(prec, 3), "followup_recall": round(rec, 3),
        "followup_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "concept_recall_followups": C("concept_recall", fus),
        "concept_recall_followups_raw_message": C("concept_recall_raw", fus),
        "concept_recall_all": C("concept_recall"),
        "topic_leak_rate": boot([(p["conv"], float(p["leak"])) for p in per if p["leak"] is not None], avg),
        "rewrite_fidelity_followups": boot([(p["conv"], p["rewrite_score"] / 2) for p in fus if p["rewrite_score"] is not None], avg),
        "rewrite_exact_followups": boot([(p["conv"], float(p["rewrite_score"] == 2)) for p in fus if p["rewrite_score"] is not None], avg),
        "answer_relevance_followups": boot([(p["conv"], p["relevance"] / 2) for p in fus if p["relevance"] is not None], avg),
        "answer_relevance_all": boot([(p["conv"], p["relevance"] / 2) for p in per if p["relevance"] is not None], avg),
    }
    # Context-Awareness Score: mean of five [0,1] components (detection, concept recall, rewrite fidelity,
    # answer relevance on follow-ups, no topic leakage)
    comp = [ctx["followup_detection_accuracy"]["mean"], ctx["concept_recall_followups"]["mean"],
            ctx["rewrite_fidelity_followups"]["mean"], ctx["answer_relevance_followups"]["mean"],
            1 - ctx["topic_leak_rate"]["mean"]]
    ctx["context_awareness_score"] = round(avg(comp), 3)
    J = [p for p in per if p["claims"]]
    H = lambda f: boot([(p["conv"], (f(p), p["claims"])) for p in J], ratio)
    hal = {
        "answers_judged": len(J), "claims": sum(p["claims"] for p in J),
        "labels": {k: sum(p[k] for p in J) for k in ("supported", "partial", "unsupported", "contradicted")},
        "faithfulness": H(lambda p: p["supported"] + 0.5 * p["partial"]),
        "supported_rate": H(lambda p: p["supported"]),
        "hallucination_rate": H(lambda p: p["unsupported"] + p["contradicted"]),
        "hallucination_rate_specific": H(lambda p: p["unsupported_specific"] + p["contradicted"]),
        "contradiction_rate": H(lambda p: p["contradicted"]),
        "answers_with_any_hallucination": boot([(p["conv"], float(p["unsupported"] + p["contradicted"] > 0)) for p in J], avg),
        "answers_hallucination_free": boot([(p["conv"], float(p["unsupported"] + p["contradicted"] == 0)) for p in J], avg),
        "citation_precision": boot([(p["conv"], (p["cited_ok"], p["cited_claims"])) for p in J if p["cited_claims"]], ratio),
        "citation_coverage": boot([(p["conv"], (p["cited_claims"], p["claims"])) for p in J], ratio),
        "online_quote_verbatim": boot([(p["conv"], (p["verification"]["quotes_verbatim"], p["verification"]["quotes_total"])) for p in per if p["verification"].get("quotes_total")], ratio),
        "online_invalid_citations": boot([(p["conv"], (p["verification"]["refs_invalid"], p["verification"]["refs_total"])) for p in per if p["verification"].get("refs_total")], ratio),
        "online_number_grounding": boot([(p["conv"], (p["verification"]["numbers_total"] - p["verification"]["numbers_unverified"], p["verification"]["numbers_total"])) for p in per if p["verification"].get("numbers_total")], ratio),
        "links_shown": sum(len(p["links"]) for p in per), "links_ok": sum(1 for p in per for s in p["links"] if s in ("doi", "openalex")),    # resolved before showing
        "dead_sources_removed": sum(p["dead_removed"] for p in per),
        "risk_counts": {k: sum(1 for p in per if p["risk"] == k) for k in ("low", "medium", "high")},
    }
    # does the online risk flag agree with the judge? (an answer is 'clean' when the judge finds no hallucination)
    hal["online_flag_vs_judge"] = {k: {"answers": len(g), "judge_clean": sum(1 for p in g if p["unsupported"] + p["contradicted"] == 0)}
                                   for k in ("low", "medium", "high") for g in [[p for p in J if p["risk"] == k]]}
    arms = {}
    for a in sorted({p["arm"] for p in per}):
        ps = [p for p in J if p["arm"] == a]
        cl = sum(p["claims"] for p in ps)
        arms[a] = {"answers": sum(1 for p in per if p["arm"] == a), "claims": cl,
                   "faithfulness": round(sum(p["supported"] + 0.5 * p["partial"] for p in ps) / cl, 3) if cl else None,
                   "hallucination_rate": round(sum(p["unsupported"] + p["contradicted"] for p in ps) / cl, 3) if cl else None,
                   "median_secs": sorted(p["secs"] for p in per if p["arm"] == a)[len([p for p in per if p["arm"] == a]) // 2]}
    out = {"context": ctx, "hallucination": hal, "per_arm": arms,
           "latency": {"median_secs": sorted(p["secs"] for p in per)[len(per) // 2], "max_secs": max(p["secs"] for p in per)},
           "judge_model": JUDGE, "turns": per}
    with open(path("scores.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "turns"}, indent=1))


if __name__ == "__main__":
    {"run": run, "judge": judge, "score": score}[sys.argv[1]]()
