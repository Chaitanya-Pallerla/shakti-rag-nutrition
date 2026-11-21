"""
citation_graph.py: build the paper graph for GraphRAG-style retrieval from the raw OpenAlex files.

For every work: who it cites, OpenAlex's related works, and its topic hierarchy (topic -> subfield
-> field -> domain), plus the flags needed to report the corpus honestly (is_xpac = OpenAlex's
2025 "expansion pack" of non-core records, is_paratext, is_retracted).

  full\\graph\\<name>.parquet   id, year, type, cites, is_xpac, is_paratext, is_retracted,
                               refs list<str>, related list<str>, topic_id, topic, subfield, field, domain

No embedding: this is CPU parsing only (gunzip + JSON). The citation edges + topic communities
are the graph that GraphRAG-style retrieval expands over. Resumable per shard; pause with
full\\graph\\STOP. Run with the fullindex environment.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import time
import glob
import gzip

import jobstatus

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
SRC = os.path.join(RESEARCH_DIR, "openalex_works")
FULL = os.path.join(RESEARCH_DIR, "full")
OUT = os.path.join(FULL, "graph")
LOG = os.path.join(FULL, "graph.log")
STOP = os.path.join(OUT, "STOP")
WORKERS = int(os.environ.get("GRAPH_WORKERS", "3"))
ROW_GROUP = 50_000


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def _wid(u):
    return (u or "").rsplit("/", 1)[-1]


def _schema():
    import pyarrow as pa
    return pa.schema([("id", pa.string()), ("year", pa.int32()), ("type", pa.string()),
                      ("cites", pa.int32()), ("is_xpac", pa.bool_()), ("is_paratext", pa.bool_()),
                      ("is_retracted", pa.bool_()), ("refs", pa.list_(pa.string())),
                      ("related", pa.list_(pa.string())), ("topic_id", pa.string()),
                      ("topic", pa.string()), ("subfield", pa.string()), ("field", pa.string()),
                      ("domain", pa.string())])


def parse_file(fp):
    """Worker: one raw .gz -> one graph shard. Returns (name, rows, edges, error); never raises."""
    name = os.path.basename(fp)[:-3]
    tmp = os.path.join(OUT, name + ".parquet.tmp")
    try:
        import orjson
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = _schema()
        cols = [f.name for f in schema]
        buf = {c: [] for c in cols}
        rows = edges = 0

        def put(w):
            w.write_table(pa.table(buf, schema=schema))
            for c in cols:
                buf[c].clear()

        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer, gzip.open(fp, "rb") as fh:
            for line in fh:
                try:
                    w = orjson.loads(line)
                except Exception:
                    continue
                wid = _wid(w.get("id"))
                if not wid:
                    continue
                pt = w.get("primary_topic") or {}
                refs = [_wid(r) for r in (w.get("referenced_works") or []) if r]
                year, cites = w.get("publication_year"), w.get("cited_by_count")
                buf["id"].append(wid)
                buf["year"].append(year if isinstance(year, int) and -5000 < year < 3000 else None)
                buf["type"].append(w.get("type"))
                buf["cites"].append(min(cites, 2**31 - 1) if isinstance(cites, int) else 0)
                buf["is_xpac"].append(bool(w.get("is_xpac")))
                buf["is_paratext"].append(bool(w.get("is_paratext")))
                buf["is_retracted"].append(bool(w.get("is_retracted")))
                buf["refs"].append(refs)
                buf["related"].append([_wid(r) for r in (w.get("related_works") or []) if r])
                buf["topic_id"].append(_wid(pt.get("id")) or None)
                buf["topic"].append(pt.get("display_name"))
                buf["subfield"].append((pt.get("subfield") or {}).get("display_name"))
                buf["field"].append((pt.get("field") or {}).get("display_name"))
                buf["domain"].append((pt.get("domain") or {}).get("display_name"))
                rows += 1
                edges += len(refs)
                if len(buf["id"]) >= ROW_GROUP:
                    put(writer)
            if buf["id"] or rows == 0:
                put(writer)
        fd = os.open(tmp, os.O_RDWR | getattr(os, "O_BINARY", 0))
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, os.path.join(OUT, name + ".parquet"))
        return name, rows, edges, None
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return name, 0, 0, repr(e)[:300]


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.tmp")):
        os.remove(f)
    if os.path.exists(STOP):
        os.remove(STOP)
    job = jobstatus.Job("graph", "Citation graph for GraphRAG (CPU parse)", unit="files", order=30)
    files = sorted(glob.glob(os.path.join(SRC, "*.gz")))
    have = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(OUT, "*.parquet"))}
    todo = [f for f in files if os.path.basename(f)[:-3] not in have]
    todo.sort(key=os.path.getsize, reverse=True)
    done = len(files) - len(todo)
    log(f"=== citation graph start: {done}/{len(files)} files done, workers={WORKERS} ===")
    job.update("running", done=done, total=len(files))
    rows_run = edges_run = 0
    failed = []
    from multiprocessing import Pool
    with Pool(WORKERS, maxtasksperchild=20) as pool:
        for name, rows, edges, err in pool.imap_unordered(parse_file, todo):
            if err:
                failed.append(name)
                log(f"FAILED {name}: {err}")
            else:
                done += 1
                rows_run += rows
                edges_run += edges
                log(f"file {done}/{len(files)} {name}: {rows} works, {edges} citation edges")
            job.update("running", done=done, total=len(files), items=rows_run,
                       note=f"{rows_run:,} works, {edges_run:,} citation edges this run")
            if os.path.exists(STOP):
                pool.terminate()
                log(f"=== paused at {done}/{len(files)} ===")
                job.update("paused", done=done, total=len(files))
                return
    state = "done" if not failed else "done_with_errors"
    job.update(state, done=done, total=len(files),
               note=(f"{len(failed)} files failed: {failed[:5]}" if failed else "graph complete"))
    log(f"=== citation graph {state}: {done}/{len(files)} files; failed {len(failed)} ===")


if __name__ == "__main__":
    import faulthandler
    import traceback
    os.makedirs(OUT, exist_ok=True)
    faulthandler.enable(open(os.path.join(FULL, "graph_crash.log"), "a", encoding="utf-8"))
    try:
        main()
    except BaseException:
        log("CRASH: " + traceback.format_exc())
        raise
