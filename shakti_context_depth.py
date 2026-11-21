"""
shakti_context_depth.py: how many unrelated questions can come between a first question and a call-back to it
before SHAKTI loses the context?

Each conversation: an anchor question, N questions on other topics, then a call-back that only makes sense
with the anchor ("Going back to my very first question, ..."). The call-back is scored by whether SHAKTI's
standalone rewrite, and its answer, name the anchor's topic. N = 1, 3, 4, 5, 6, 8, 12; two anchors per N.

  python shakti_context_depth.py run     -> <SHAKTI_RESEARCH_DIR>\\shakti_chat_eval\\depth\\runs.jsonl
  python shakti_context_depth.py score   -> depth\\scores.json
  python shakti_context_depth.py run after / score after -> depth_after\\ (after the earlier-questions fix; adds N = 16)
Sessions use role "eval", so the participant statistics leave them out.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor

import shakti_chat_eval as EV

OUT = os.path.join(EV.ROOT, "depth" + (f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""))   # depth = before the fix
DEPTHS = [1, 3, 4, 5, 6, 8, 12] + ([16] if len(sys.argv) > 2 else [])          # 16 + 2 stays under the 20-question cap
ANCHORS = {
    "rice": ("How many calories are in a cup of cooked brown rice?",
             "Going back to my very first question: how much fiber does that food have?",
             [["brown rice"], ["fiber", "fibre"]]),
    "fasting": ("Is intermittent fasting safe for people with type 2 diabetes?",
                "Going back to the first thing I asked about: does it also help with weight loss?",
                [["fasting", "time-restricted"], ["weight"]]),
}
# other topics, none mentioning rice, fiber, fasting, diabetes or weight
OTHER = ["How much protein is in a boiled egg?", "Is green tea good for sleep?", "How much vitamin C is in a kiwi?",
         "Does creatine help older adults build muscle?", "How much sugar is in a can of cola?",
         "Are probiotics useful after a course of antibiotics?", "How much calcium is in a glass of milk?",
         "Does dark chocolate lower blood pressure?", "How many calories are in an avocado?",
         "Is coffee safe during pregnancy?", "How much iron is in cooked lentils?", "Does vitamin D help with depression?",
         "Is olive oil healthier than butter?", "How much potassium is in a banana?",
         "Does turmeric reduce inflammation?", "How much caffeine is in black tea?"]


def path(n):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, n)


def ask_turn(sid, conv, q):
    for _ in range(40):
        j = EV.call("/api/ask", {"sid": sid, "conv": conv, "question": q})
        if "job" in j:
            break
        time.sleep(15)                                      # queue full
    else:
        raise RuntimeError(f"could not ask: {j}")
    while True:
        time.sleep(3)
        s = EV.call(f"/api/job?id={j['job']}&sid={sid}")
        if s["status"] in ("done", "error", "unknown"):
            break
    if s["status"] != "done":
        raise RuntimeError(f"{conv}: {s.get('error')}")
    return j["job"], s["result"]


def run_one(key):
    anchor, n = key
    first, back, _ = ANCHORS[anchor]
    sid = EV.call("/api/session", {"consent": True, "adult": True, "role": "eval", "background": "eval"})["sid"]
    conv = f"depth-{anchor}-{n}"
    for q in [first] + OTHER[:n]:
        ask_turn(sid, conv, q)
    t = time.time()
    job, r = ask_turn(sid, conv, back)
    rec = {"anchor": anchor, "n_between": n, "job": job, "question": back, "secs": round(time.time() - t, 1),
           "standalone": r["context"]["standalone"], "is_followup": r["context"]["is_followup"],
           "context_used": r["context"].get("context_used", ""), "answer": r["answer"],
           "foods": [f["name"] for f in r["foods"]]}
    with open(path("runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{anchor} N={n}] {rec['standalone']}", flush=True)


def run():
    p = path("runs.jsonl")
    done = {(r["anchor"], r["n_between"]) for r in map(json.loads, open(p, encoding="utf-8"))} if os.path.exists(p) else set()
    todo = [(a, n) for n in DEPTHS for a in ANCHORS if (a, n) not in done]
    with ThreadPoolExecutor(2) as ex:                       # the server answers 2 at a time
        list(ex.map(run_one, todo))


def score():
    rows = [json.loads(ln) for ln in open(path("runs.jsonl"), encoding="utf-8")]
    out = []
    for r in sorted(rows, key=lambda r: (r["n_between"], r["anchor"])):
        groups = ANCHORS[r["anchor"]][2]
        hit = lambda text: all(EV.has(text, g) for g in groups)
        out.append({**r, "rewrite_remembers": hit(r["standalone"]),
                    "answer_remembers": hit(r["answer"] + " " + " ".join(r["foods"]))})
    by_n = {}
    for r in out:
        b = by_n.setdefault(r["n_between"], {"conversations": 0, "rewrite_remembers": 0, "answer_remembers": 0})
        b["conversations"] += 1
        b["rewrite_remembers"] += r["rewrite_remembers"]
        b["answer_remembers"] += r["answer_remembers"]
    ok = [n for n in sorted(by_n) if by_n[n]["rewrite_remembers"] == by_n[n]["conversations"]]
    worst = [n for n in sorted(by_n) if by_n[n]["rewrite_remembers"] < by_n[n]["conversations"]]
    res = {"depths": by_n, "max_between_all_remembered": max([n for n in ok if all(m in ok for m in by_n if m <= n)], default=0),
           "first_failure": min(worst) if worst else None, "turns": out}
    with open(path("scores.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "turns"}, indent=1))
    for r in out:
        print(r["n_between"], r["anchor"], r["rewrite_remembers"], "|", r["standalone"])


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
