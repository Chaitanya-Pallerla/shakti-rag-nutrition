"""
shakti_load_test.py: efficiency of the live SHAKTI server: latency, throughput, tokens and cost per answer.

Virtual users (role "eval", left out of participant statistics) each open a session and ask fresh questions one
after another, as a person would. Concurrency levels 1, 2, 4 and 8 users. Per level:
  throughput   answered questions per minute of wall time
  latency      submit -> answer seen by the client (includes queueing); p50 / p95 / max
  tokens, cost from the server's per-turn LLM log (turns.jsonl "llm": every Claude call of the turn, list price)
The server answers at most 2 questions at a time (MAX_RUNNING), so throughput saturates by design.

  python shakti_load_test.py run     -> <SHAKTI_RESEARCH_DIR>\\shakti_chat_eval\\load\\runs.jsonl
  python shakti_load_test.py score   -> load\\scores.json
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import shakti_chat_eval as EV

OUT = os.path.join(EV.ROOT, "load")
TURNS = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "shakti_chat", "turns.jsonl")
LEVELS = [(1, 4), (2, 3), (4, 3), (8, 2)]          # (concurrent users, questions per user) -> 38 questions
POOL = [  # fresh questions (not asked before, so no cached Claude calls), health evidence and food data mixed
    "Does eating walnuts improve memory in older adults?", "How much protein is in 100 g of tofu?",
    "Is sparkling water bad for your teeth?", "Do omega-3 supplements lower triglycerides?",
    "How many calories are in a medium baked potato?", "Does cinnamon lower blood sugar in type 2 diabetes?",
    "Is it safe to eat raw sprouts during pregnancy?", "How much vitamin K is in a cup of cooked kale?",
    "Do nitrates in beetroot juice improve running performance?", "How much sodium is in a slice of pizza?",
    "Does magnesium help with migraine prevention?", "Are artificial sweeteners linked to weight gain?",
    "How much fiber is in a cup of cooked oats?", "Does garlic lower blood pressure?",
    "Is a high-protein diet harmful for healthy kidneys?", "How much sugar is in a cup of orange juice?",
    "Does zinc shorten the common cold?", "Is red meat linked to colorectal cancer?",
    "How many calories are in a tablespoon of olive oil?", "Does melatonin help with jet lag?",
    "Do soy foods affect testosterone in men?", "How much calcium is in a cup of plain yogurt?",
    "Does fish oil help with depression?", "Is skipping breakfast bad for metabolism?",
    "How much potassium is in a medium sweet potato?", "Do ketone supplements improve endurance?",
    "Does vitamin C prevent colds in athletes?", "Is MSG harmful to health?",
    "How much caffeine is in a can of energy drink?", "Does a low-salt diet reduce stroke risk?",
    "Are eggs safe for people with high cholesterol?", "How much iron is in 100 g of beef liver?",
    "Does collagen powder improve skin elasticity?", "Is chewing gum good for digestion after surgery?",
    "How many grams of carbs are in a banana?", "Does folic acid prevent neural tube defects?",
    "Do polyphenols in berries improve cognition?", "Is coconut oil bad for LDL cholesterol?",
]
_lock = threading.Lock()


def path(n):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, n)


def ask_timed(sid, conv, q):
    t = time.time()
    retries = 0
    while True:
        j = EV.call("/api/ask", {"sid": sid, "conv": conv, "question": q})
        if "job" in j:
            break
        retries += 1                                     # queue full: the client backs off, counted
        time.sleep(5)
    while True:
        time.sleep(1)
        s = EV.call(f"/api/job?id={j['job']}&sid={sid}")
        if s["status"] in ("done", "error", "unknown"):
            break
    return j["job"], s["status"], round(time.time() - t, 1), retries


def user(level, u, qs):
    sid = EV.call("/api/session", {"consent": True, "adult": True, "role": "eval", "background": "eval"})["sid"]
    for q in qs:
        job, status, secs, retries = ask_timed(sid, f"load-{level}-{u}", q)
        with _lock, open(path("runs.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"level": level, "user": u, "job": job, "question": q, "status": status,
                                "client_secs": secs, "queue_full_retries": retries}) + "\n")
        print(f"[{level} users] u{u} {status} {secs}s  {q}", flush=True)


def run():
    pool = list(POOL)
    phases = []
    for users, per in LEVELS:
        qs = [[pool.pop(0) for _ in range(per)] for _ in range(users)]
        t = time.time()
        with ThreadPoolExecutor(users) as ex:
            list(ex.map(lambda a: user(users, *a), enumerate(qs)))
        phases.append({"users": users, "per_user": per, "wall_secs": round(time.time() - t, 1)})
        print(f"== level {users}: {phases[-1]['wall_secs']} s", flush=True)
    with open(path("phases.json"), "w") as f:
        json.dump(phases, f, indent=1)


def pct(xs, p, nd=1):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), nd)


def score():
    runs = EV.read(path("runs.jsonl"))
    phases = {p["users"]: p for p in json.load(open(path("phases.json")))}
    turns = {}
    for ln in open(TURNS, encoding="utf-8"):
        r = json.loads(ln)
        turns[r["turn"]] = r
    out, all_llm = {"levels": []}, []
    for users in sorted(phases):
        rs = [r for r in runs if r["level"] == users]
        ok = [r for r in rs if r["status"] == "done"]
        tr = [turns[r["job"]] for r in ok if r["job"] in turns]
        llm = [t["llm"] for t in tr if t.get("llm")]
        all_llm += [(t, t["llm"]) for t in tr if t.get("llm")]
        lat = [r["client_secs"] for r in ok]
        out["levels"].append({
            "users": users, "questions": len(rs), "answered": len(ok), "errors": len(rs) - len(ok),
            "wall_secs": phases[users]["wall_secs"],
            "throughput_q_per_min": round(60 * len(ok) / phases[users]["wall_secs"], 2),
            "latency_p50": pct(lat, .5), "latency_p95": pct(lat, .95), "latency_max": max(lat) if lat else None,
            "server_processing_p50": pct([t["secs"] for t in tr], .5),
            "queue_wait_p50": pct([t.get("total_secs", t["secs"]) - t["secs"] for t in tr], .5),
            "queue_full_retries": sum(r["queue_full_retries"] for r in rs)})
    n = len(all_llm)
    tok = lambda k: [l[k] for _, l in all_llm]
    tot_in = [l["input_tokens"] + l["cache_write_tokens"] + l["cache_read_tokens"] for _, l in all_llm]
    out["per_answer"] = {
        "answers": n, "claude_calls_mean": round(sum(tok("calls")) / n, 2),
        "input_tokens_mean": round(sum(tot_in) / n), "output_tokens_mean": round(sum(tok("output_tokens")) / n),
        "total_tokens_mean": round((sum(tot_in) + sum(tok("output_tokens"))) / n),
        "input_tokens_p50": pct(tot_in, .5), "output_tokens_p50": pct(tok("output_tokens"), .5),
        "cost_list_usd_mean": round(sum(tok("cost_list_usd")) / n, 4), "cost_list_usd_p50": pct(tok("cost_list_usd"), .5, 4),
        "cost_per_1000_answers_usd": round(1000 * sum(tok("cost_list_usd")) / n, 2),
        "llm_secs_mean": round(sum(tok("llm_secs")) / n, 1), "cached_calls": sum(tok("cached_calls"))}
    by_arm = {}
    for t, l in all_llm:
        by_arm.setdefault(t["arm"], []).append((t, l))
    out["per_arm"] = {a: {"answers": len(v), "total_tokens_mean": round(sum(l["input_tokens"] + l["cache_write_tokens"]
                                                                              + l["cache_read_tokens"] + l["output_tokens"] for _, l in v) / len(v)),
                          "cost_list_usd_mean": round(sum(l["cost_list_usd"] for _, l in v) / len(v), 4),
                          "latency_p50": pct([t["secs"] for t, _ in v], .5)} for a, v in by_arm.items()}
    out["stage_secs_p50"] = {}
    stages = {}
    for t, _ in all_llm:
        st = sorted(t["stage_times"].items(), key=lambda kv: kv[1]) + [("end", t["secs"])]
        for (a, x), (_, y) in zip(st, st[1:]):
            stages.setdefault(a, []).append(y - x)
    out["stage_secs_p50"] = {k: pct(v, .5) for k, v in stages.items()}
    with open(path("scores.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
