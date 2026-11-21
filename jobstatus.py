"""
jobstatus.py: tiny shared progress reporter for the research-index jobs (stdlib only, so every env
can import it). Each job writes <SHAKTI_RESEARCH_DIR>\\jobs\\<job>.json; full_index_ui.py shows them.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
import time

JOBS_DIR = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "jobs")


class Job:
    def __init__(self, key, label, unit="shards", total=0, order=50):
        self.key, self.label, self.unit, self.order = key, label, unit, order
        self.total, self.done = total, 0
        self.started = time.time()
        self._t0_done = None
        self._t0 = None
        self.items = 0                      # e.g. papers embedded this run (for a rate)
        os.makedirs(JOBS_DIR, exist_ok=True)
        self.path = os.path.join(JOBS_DIR, key + ".json")

    def update(self, state, done=None, total=None, note="", items=None, extra=None):
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        if items is not None:
            self.items = items
        now = time.time()
        if state == "running" and self._t0 is None:
            self._t0, self._t0_done = now, self.done
        rate = eta = None
        if self._t0 is not None and now - self._t0 > 30 and self.done > self._t0_done:
            per = (now - self._t0) / (self.done - self._t0_done)
            eta = per * max(0, self.total - self.done)
            rate = self.items / (now - self._t0) if self.items else None
        obj = {"key": self.key, "label": self.label, "state": state, "done": self.done,
               "total": self.total, "unit": self.unit, "note": note, "rate_items": rate,
               "eta_s": eta, "started": self.started, "updated": now, "order": self.order}
        if extra:
            obj.update(extra)
        # Windows refuses the replace while a reader (the dashboard) has the file open: retry briefly,
        # then skip this update. A progress report must never kill the job it reports on.
        tmp = self.path + ".tmp"
        for attempt in range(5):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(obj, f)
                os.replace(tmp, self.path)
                return
            except OSError:
                time.sleep(0.2 * (attempt + 1))


def read_all():
    out = []
    try:
        names = os.listdir(JOBS_DIR)
    except Exception:
        return out
    for n in names:
        if n.endswith(".json"):
            try:
                out.append(json.load(open(os.path.join(JOBS_DIR, n), encoding="utf-8")))
            except Exception:
                pass
    return sorted(out, key=lambda j: (j.get("order", 50), j.get("key", "")))


def read(key):
    try:
        return json.load(open(os.path.join(JOBS_DIR, key + ".json"), encoding="utf-8"))
    except Exception:
        return None
