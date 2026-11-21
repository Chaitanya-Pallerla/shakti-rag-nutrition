"""
shakti_chat_analyze.py: statistics of the SHAKTI chat study (<SHAKTI_RESEARCH_DIR>\\shakti_chat).

  python shakti_chat_analyze.py            -> prints a summary, writes shakti_chat\\report.json

Satisfaction (emoji 1-5, the latest reaction per answer), hallucination metrics from the automatic
verification, context awareness (follow-ups, query rewriting, satisfaction on follow-up vs first
turns, the final 'understood my follow-ups' item) and problem reports, per method. 95% CIs are
cluster bootstraps over participants, since one person's answers are not independent.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
import random
from collections import defaultdict, Counter

DATA = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "shakti_chat")
B = 5000


def read(name):
    p = os.path.join(DATA, name)
    return [json.loads(ln) for ln in open(p, encoding="utf-8")] if os.path.exists(p) else []


def mean_ci(rows, f, seed=0):
    """rows: list of (participant, value); f: aggregation of values. Cluster bootstrap over participants."""
    vals = [v for _, v in rows if v is not None]
    if not vals:
        return None
    by = defaultdict(list)
    for s, v in rows:
        if v is not None:
            by[s].append(v)
    keys, rnd, bs = list(by), random.Random(seed), []
    for _ in range(B if len(keys) > 1 else 0):
        pick = [v for k in (rnd.choice(keys) for _ in keys) for v in by[k]]
        if pick:
            bs.append(f(pick))
    bs.sort()
    est = f(vals)
    lo, hi = (bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]) if bs else (est, est)
    return {"mean": round(est, 3), "ci": [round(lo, 3), round(hi, 3)], "n": len(vals), "participants": len(keys)}


avg = lambda x: sum(x) / len(x)


def main():
    turns, reacts, reps, fin = read("turns.jsonl"), read("reactions.jsonl"), read("reports.jsonl"), read("final.jsonl")
    # scripted test / evaluation sessions (shakti_chat_eval.py) are not participants
    staff = {s["sid"] for s in read("sessions.jsonl") if s.get("role") in ("test", "eval")}
    turns, reacts, reps, fin = ([r for r in rows if r.get("sid") not in staff] for rows in (turns, reacts, reps, fin))
    score = {}
    for r in reacts:
        score[r["turn"]] = r["score"]
    rep_by = defaultdict(list)
    for r in reps:
        rep_by[r["turn"]] += r["reasons"] or ["other"]
    out = {"n_turns": len(turns), "n_participants": len({t["sid"] for t in turns}), "n_rated": len(score),
           "n_final": len(fin), "per_method": {}, "context": {}, "final": {}}
    groups = defaultdict(list)
    for t in turns:
        groups[t["arm"]].append(t)
        groups["all"].append(t)
    for arm, ts in sorted(groups.items()):
        v = lambda t: t["verification"]
        m = {
            "answers": len(ts),
            "satisfaction": mean_ci([(t["sid"], score.get(t["turn"])) for t in ts], avg),
            "satisfied_share(4-5)": mean_ci([(t["sid"], None if t["turn"] not in score else float(score[t["turn"]] >= 4)) for t in ts], avg),
            "quote_verbatim_rate": mean_ci([(t["sid"], v(t)["quotes_verbatim"] / v(t)["quotes_total"] if v(t)["quotes_total"] else None) for t in ts], avg),
            "invalid_citation_rate": mean_ci([(t["sid"], v(t)["refs_invalid"] / v(t)["refs_total"] if v(t)["refs_total"] else None) for t in ts], avg),
            "number_grounding_rate": mean_ci([(t["sid"], 1 - v(t)["numbers_unverified"] / v(t)["numbers_total"] if v(t)["numbers_total"] else None) for t in ts], avg),
            "cited_sentence_support": mean_ci([(t["sid"], v(t)["cited_supported"] / v(t)["cited_checked"] if v(t)["cited_checked"] else None) for t in ts], avg),
            "uncited_factual_per_answer": mean_ci([(t["sid"], float(v(t)["sentences_uncited_factual"])) for t in ts], avg),
            "risk": dict(Counter(v(t)["risk"] for t in ts)),
            "dead_sources_removed_per_answer": mean_ci([(t["sid"], float(len(t.get("dead_sources_removed", [])))) for t in ts], avg),
            "reported_share": mean_ci([(t["sid"], float(t["turn"] in rep_by)) for t in ts], avg),
            "report_reasons": dict(Counter(x for t in ts for x in rep_by.get(t["turn"], []))),
            "median_seconds": sorted(t["secs"] for t in ts)[len(ts) // 2] if ts else None,
        }
        out["per_method"][arm] = m
    ts = groups["all"]
    fol = [t for t in ts if t["understand"].get("is_followup")]
    first = [t for t in ts if t["turn_index"] > 0 and not t["understand"].get("is_followup")]
    out["context"] = {
        "turns_with_history": sum(1 for t in ts if t["turn_index"] > 0),
        "followup_share_of_turns_with_history": mean_ci([(t["sid"], float(bool(t["understand"].get("is_followup"))))
                                                         for t in ts if t["turn_index"] > 0], avg),
        "rewrite_changed_share_on_followups": mean_ci([(t["sid"], float(t["context_metrics"].get("rewrite_changed", False))) for t in fol], avg),
        "cos_prev_raw_vs_standalone_on_followups": {
            "raw": mean_ci([(t["sid"], t["context_metrics"].get("cos_raw_prev")) for t in fol], avg),
            "standalone": mean_ci([(t["sid"], t["context_metrics"].get("cos_standalone_prev")) for t in fol], avg)},
        "satisfaction_followup": mean_ci([(t["sid"], score.get(t["turn"])) for t in fol], avg),
        "satisfaction_new_topic_later_turn": mean_ci([(t["sid"], score.get(t["turn"])) for t in first], avg),
        "followups_reported_off_topic": sum(1 for t in fol if "off_topic" in rep_by.get(t["turn"], [])),
    }
    for k in ("umux_capabilities", "umux_easy", "context_understood", "trust_sources", "charts_helpful", "use_again"):
        out["final"][k] = mean_ci([(r["sid"], r.get(k)) for r in fin], avg)
    um = [(r["sid"], ((r["umux_capabilities"] - 1) + (r["umux_easy"] - 1)) / 12 * 100) for r in fin
          if "umux_capabilities" in r and "umux_easy" in r]
    out["final"]["umux_lite_0_100"] = mean_ci(um, avg)
    with open(os.path.join(DATA, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
