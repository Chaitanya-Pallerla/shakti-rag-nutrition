"""
full_index.py: index EVERY OpenAlex work (no domain filter) into <SHAKTI_RESEARCH_DIR>\\full.

Unlike bulk_corpus.index_openalex_parallel (domain-filtered, one flat FAISS index held in RAM and
rewritten every checkpoint, which cannot scale past ~20M papers on 32 GB RAM), this writes one
shard per source .gz file, so memory stays flat and resume is just "which shards exist":

  full\\meta\\<name>.parquet  id, title, year, doi, cites, type, lang, venue, authors, abstract
  full\\vec\\<name>.npy       float16 [n, 384] all-MiniLM-L6-v2 vectors, row i == parquet row i

Works with a title are kept even without an abstract (embedded from the title alone).
CPU worker processes only gunzip/parse JSON into Parquet; ALL embedding runs on the GPU
(refuses to start without CUDA, never falls back to CPU).

The search index (IVF over the shard vectors) is built afterwards by a separate step.

Run:   python full_index.py
Pause: create <SHAKTI_RESEARCH_DIR>\\full\\STOP  (finishes the current shard, then exits)
Log:   <SHAKTI_RESEARCH_DIR>\\full\\full.log
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import time
import glob
import gzip

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
SRC = os.path.join(RESEARCH_DIR, "openalex_works")
OUT = os.path.join(RESEARCH_DIR, "full")
META = os.path.join(OUT, "meta")
VEC = os.path.join(OUT, "vec")
LOG = os.path.join(OUT, "full.log")
STOP = os.path.join(OUT, "STOP")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # same model as bulk/litrag -> same query space
WORKERS = int(os.environ.get("SHAKTI_WORKERS", "3"))    # one worker parses ~3x faster than the GPU embeds
AHEAD = int(os.environ.get("SHAKTI_AHEAD", "6"))        # max parsed shards waiting for the GPU; more just
                                                        # thrashes the disk the GPU feeder reads from
ROW_GROUP = 50_000
ABS_MAX = 4000

COLS = ["id", "title", "year", "doi", "cites", "type", "lang", "venue", "authors", "abstract"]


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def _name(fp):
    return os.path.basename(fp)[:-3]                     # strip ".gz"


def _schema():
    import pyarrow as pa
    return pa.schema([("id", pa.string()), ("title", pa.string()), ("year", pa.int32()),
                      ("doi", pa.string()), ("cites", pa.int32()), ("type", pa.string()),
                      ("lang", pa.string()), ("venue", pa.string()), ("authors", pa.string()),
                      ("abstract", pa.string())])


def _fsync(path):
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def parse_file(fp):
    """Worker: one .gz -> one Parquet shard. Returns (name, rows, error); never raises, so one bad
    source file can't kill the run."""
    name = _name(fp)
    try:
        return name, _parse_file(fp), None
    except Exception as e:
        tmp = os.path.join(META, name + ".parquet.tmp")
        if os.path.exists(tmp):
            os.remove(tmp)
        return name, -1, repr(e)[:300]


def _parse_file(fp):
    import orjson
    import pyarrow as pa
    import pyarrow.parquet as pq
    name = _name(fp)
    dst = os.path.join(META, name + ".parquet")
    tmp = dst + ".tmp"
    schema = _schema()
    rows = 0
    buf = {c: [] for c in COLS}

    def put(writer):
        writer.write_table(pa.table(buf, schema=schema))
        for c in COLS:
            buf[c].clear()

    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer, gzip.open(fp, "rb") as fh:
        for line in fh:
            try:
                w = orjson.loads(line)
            except Exception:
                continue
            title = (w.get("display_name") or w.get("title") or "").strip()
            if not title:
                continue
            inv = w.get("abstract_inverted_index")
            abstract = ""
            if inv:
                pos = {}
                for word, idxs in inv.items():
                    for i in idxs:
                        pos[i] = word
                abstract = " ".join(pos[i] for i in sorted(pos))[:ABS_MAX]
            src = ((w.get("primary_location") or {}).get("source") or {})
            auth = [((a.get("author") or {}).get("display_name") or "")
                    for a in (w.get("authorships") or [])[:6]]
            year = w.get("publication_year")
            cites = w.get("cited_by_count")
            buf["id"].append((w.get("id") or "").rsplit("/", 1)[-1])
            buf["title"].append(title[:1000])
            buf["year"].append(year if isinstance(year, int) and -5000 < year < 3000 else None)
            buf["doi"].append((w.get("doi") or "").replace("https://doi.org/", "") or None)
            buf["cites"].append(min(cites, 2**31 - 1) if isinstance(cites, int) else 0)
            buf["type"].append(w.get("type"))
            buf["lang"].append(w.get("language"))
            buf["venue"].append(src.get("display_name"))
            buf["authors"].append("; ".join(a for a in auth if a) or None)
            buf["abstract"].append(abstract or None)
            rows += 1
            if len(buf["id"]) >= ROW_GROUP:
                put(writer)
        if buf["id"] or rows == 0:
            put(writer)
    _fsync(tmp)                       # on disk before it counts as done (survives a forced reboot)
    os.replace(tmp, dst)
    return rows


def embed_shard(model, name):
    """GPU: parquet shard -> float16 vectors (written atomically)."""
    import numpy as np
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(META, name + ".parquet"), columns=["title", "abstract"])
    titles = t.column("title").to_pylist()
    abstracts = t.column("abstract").to_pylist()
    texts = [f"{ti}. {ab}" if ab else ti for ti, ab in zip(titles, abstracts)]
    if texts:
        vecs = model.encode(texts, batch_size=512, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    else:
        vecs = np.zeros((0, 384), np.float16)
    dst = os.path.join(VEC, name + ".npy")
    with open(dst + ".tmp", "wb") as f:
        np.save(f, vecs)
        f.flush()
        os.fsync(f.fileno())
    os.replace(dst + ".tmp", dst)
    return len(texts), sum(1 for a in abstracts if a)


def main():
    os.makedirs(META, exist_ok=True)
    os.makedirs(VEC, exist_ok=True)
    for d in (META, VEC):                                  # leftovers of a killed run
        for f in glob.glob(os.path.join(d, "*.tmp")):
            os.remove(f)
    if os.path.exists(STOP):
        os.remove(STOP)

    import torch
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available; this indexer embeds on the GPU only. Not starting.")
        sys.exit(1)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cuda")
    model.half()
    log(f"=== FULL indexing start: GPU={torch.cuda.get_device_name(0)} workers={WORKERS} ===")

    files = sorted(glob.glob(os.path.join(SRC, "*.gz")))
    have_meta = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(META, "*.parquet"))}
    have_vec = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(VEC, "*.npy"))}
    todo_parse = [f for f in files if _name(f) not in have_meta]
    todo_embed = sorted(have_meta - have_vec)
    total = len(files)
    done = len(have_vec)
    log(f"{done}/{total} shards complete; {len(todo_embed)} parsed awaiting GPU; "
        f"{len(todo_parse)} to parse")

    t0 = time.time()
    n_done_run = 0
    papers_run = 0

    by_name = {_name(f): f for f in files}

    def do_embed(name):
        """Embed one parsed shard. An unreadable parquet (cut off by a crash) is deleted and
        queued for re-parsing instead of stopping the run."""
        nonlocal done, n_done_run, papers_run
        s = time.time()
        try:
            n, n_abs = embed_shard(model, name)
        except Exception as e:
            if "parquet" not in repr(e).lower() and "arrow" not in type(e).__module__:
                raise
            log(f"unreadable shard {name} ({repr(e)[:120]}); deleting it to re-parse")
            os.remove(os.path.join(META, name + ".parquet"))
            todo_parse.append(by_name[name])
            return
        done += 1
        n_done_run += 1
        papers_run += n
        el = time.time() - t0
        eta_h = (total - done) * el / n_done_run / 3600
        log(f"shard {done}/{total} {name}: {n} papers ({n_abs} w/ abstract), gpu {time.time() - s:.0f}s"
            f" | run {papers_run} papers, {papers_run / el:.0f}/s, ETA {eta_h:.1f} h")

    for name in todo_embed:
        if os.path.exists(STOP):
            break
        do_embed(name)

    if todo_parse and not os.path.exists(STOP):
        from multiprocessing import Pool
        from collections import deque
        # biggest files first so the long tail at the end is small ones
        todo_parse.sort(key=os.path.getsize, reverse=True)
        queue = deque(todo_parse)
        todo_parse.clear()                     # do_embed appends re-parse jobs here
        failed = []
        with Pool(WORKERS, maxtasksperchild=10) as pool:
            pending = []                       # AsyncResults in flight or parsed-but-not-embedded
            while queue or pending or todo_parse:
                queue.extend(todo_parse)
                todo_parse.clear()
                while queue and len(pending) < WORKERS + AHEAD:
                    pending.append(pool.apply_async(parse_file, (queue.popleft(),)))
                ready = next((r for r in pending if r.ready()), None)
                if ready is None:
                    time.sleep(0.5)
                    continue
                pending.remove(ready)
                name, rows, err = ready.get()
                if err:
                    failed.append(name)
                    log(f"PARSE FAILED {name}: {err} (skipped; re-run to retry)")
                else:
                    do_embed(name)
                if os.path.exists(STOP):
                    log("STOP file seen: stopping workers (partial shards are discarded next run)")
                    pool.terminate()
                    break
        if failed:
            log(f"{len(failed)} source files failed to parse: {failed[:20]}")

    if done >= total:
        log(f"=== FULL indexing DONE: {done}/{total} shards ===")
    else:
        log(f"=== paused at {done}/{total} shards; run again to resume ===")


if __name__ == "__main__":
    # runs under pythonw (no console): record Python errors in full.log and native crashes
    # (CUDA / faiss / arrow segfaults) in crash.log, otherwise a dead run leaves no trace
    import faulthandler
    import traceback
    os.makedirs(OUT, exist_ok=True)
    _crash = open(os.path.join(OUT, "crash.log"), "a", encoding="utf-8")
    faulthandler.enable(_crash)
    try:
        main()
    except BaseException:
        log("CRASH: " + traceback.format_exc())
        raise
