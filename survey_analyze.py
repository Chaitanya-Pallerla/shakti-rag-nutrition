"""
survey_analyze.py: results of the human survey (survey_server.py) for the paper.

Per method (arm): mean of each 1-5 rating with a 95% cluster-bootstrap CI (participants resampled, since
each participant rates several answers), share flagged wrong/harmful, share of verbatim-verified quotes
and answer time; pairwise arm differences (cluster bootstrap, two-sided p); UMUX-Lite (0-100).
-> <SHAKTI_RESEARCH_DIR>\\survey\\survey_report.json, also printed.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
from collections import defaultdict
import numpy as np

DATA = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "survey")
LIKERT = ["relevance", "evidence", "clarity", "trust", "satisfaction"]
B = 10000
RNG = np.random.default_rng(2026)


def read(name):
    p = os.path.join(DATA, name)
    return [json.loads(ln) for ln in open(p, encoding="utf-8")] if os.path.exists(p) else []


def cluster_boot(by_sid, stat):
    sids = list(by_sid)
    vals = []
    for _ in range(B):
        pick = RNG.choice(len(sids), len(sids))
        v = stat([r for i in pick for r in by_sid[sids[i]]])
        if v is not None:
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None


def main():
    ratings, answers, final = read("ratings.jsonl"), read("answers.jsonl"), read("final.jsonl")
    ans = {a["job"]: a for a in answers}
    arms = sorted({r["arm"] for r in ratings})
    by_sid = defaultdict(list)
    for r in ratings:
        by_sid[r["sid"]].append(r)
    rep = {"participants": len(by_sid), "ratings": len(ratings), "answers": len(answers), "arms": {}}
    for arm in arms:
        rs = [r for r in ratings if r["arm"] == arm]
        out = {"n": len(rs)}
        for k in LIKERT:
            f = lambda rows, k=k, arm=arm: (np.mean([r[k] for r in rows if r["arm"] == arm])
                                            if any(r["arm"] == arm for r in rows) else None)
            out[k] = [round(float(np.mean([r[k] for r in rs])), 3), cluster_boot(by_sid, f)]
        out["flagged_wrong_or_harmful"] = round(float(np.mean([r["wrong_or_harmful"] for r in rs])), 3)
        q = [x["verbatim"] for r in rs if r["job"] in ans for x in ans[r["job"]].get("quotes", [])]
        out["quotes_verbatim"] = round(float(np.mean(q)), 3) if q else None
        out["answer_seconds_median"] = float(np.median([ans[r["job"]]["secs"] for r in rs if r["job"] in ans] or [0]))
        rep["arms"][arm] = out
    rep["pairwise"] = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            for k in LIKERT:
                def diff(rows, k=k, a=a, b=b):
                    xa = [r[k] for r in rows if r["arm"] == a]
                    xb = [r[k] for r in rows if r["arm"] == b]
                    return np.mean(xb) - np.mean(xa) if xa and xb else None
                obs = diff(ratings)
                sids = list(by_sid)
                bs = [d for d in (diff([r for j in RNG.choice(len(sids), len(sids)) for r in by_sid[sids[j]]])
                                  for _ in range(2000)) if d is not None]
                p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0)) if bs else None
                rep["pairwise"][f"{b} - {a}: {k}"] = {"diff": round(float(obs), 3) if obs is not None else None,
                                                     "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)] if bs else None,
                                                     "p": round(float(max(p, 1 / 2000)), 4) if p is not None else None}
    u = [((f["umux_capabilities"] - 1) + (f["umux_easy"] - 1)) / 12 * 100 for f in final
         if "umux_capabilities" in f and "umux_easy" in f]
    rep["umux_lite"] = {"n": len(u), "mean": round(float(np.mean(u)), 1) if u else None}
    json.dump(rep, open(os.path.join(DATA, "survey_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
