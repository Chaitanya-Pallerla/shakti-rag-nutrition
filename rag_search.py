"""
rag_search.py: EXACT dense retrieval over the full index (all 484M papers) for a batch of queries.

No approximate index: every shard's vectors are streamed from E: to the GPU once per batch and every
query is scored against every paper, so the top-k is the true top-k (no ANN recall loss to explain in
the paper). One pass costs the disk read (MiniLM 372 GB, SPECTER2 693 GB), whatever the number of
queries, so the evaluation batches queries: all first-round queries in one pass, all second-hop
sub-queries in the next, and so on.

  encode : queries.jsonl (qid, text) -> <out>.<model>.q.npy        (MiniLM: the fullindex environment,
                                                                     SPECTER2: the specter2 environment)
  search : <out>.<model>.q.npy -> <out>.<model>.hits.parquet       (qid, rank, score, shard, row, id)

  python rag_search.py encode  --model minilm   --queries Q.jsonl --out RUN
  python rag_search.py search  --model specter2 --queries Q.jsonl --out RUN [--k 100]

GPU only. Resumable: per-shard top-k is checkpointed every 100 shards.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import json
import time
import glob
import argparse
import threading
import queue as _queue

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
FULL = os.path.join(RESEARCH_DIR, "full")
VEC_DIR = {"minilm": os.path.join(FULL, "vec"), "specter2": os.path.join(FULL, "specter2")}
DIM = {"minilm": 384, "specter2": 768}
LOG = os.path.join(RESEARCH_DIR, "rag_eval", "rag_eval.log")
Q_CHUNK = 4096                       # queries scored at once (scores matrix = Q_CHUNK x shard rows, fp16)
SCORE_ELEMS = 1_500_000_000          # cap on that matrix: 3 GB of fp16, well inside the 16 GB card


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + "[search] " + str(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def read_queries(path):
    with open(path, encoding="utf-8-sig") as f:
        qs = [json.loads(ln) for ln in f if ln.strip()]
    return [q["qid"] for q in qs], [q.get("text") or q["question"] for q in qs]


def encode(model_name, texts):
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; retrieval runs on the GPU only.")
    if model_name == "minilm":                       # same model + normalisation as full_index.py
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        return m.encode(texts, batch_size=512, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    import specter2_index                            # queries use the adhoc_query adapter
    return specter2_index.encode_queries(texts)


def _loader(names, vdir, q, mask_dir=None):
    """Read-ahead thread. A failure is handed to the main thread (else it would wait forever)."""
    import numpy as np
    try:
        for i, name in enumerate(names):
            v = np.load(os.path.join(vdir, name + ".npy"))
            m = np.load(os.path.join(mask_dir, name + ".npy")) if mask_dir else None
            if m is not None and len(m) != len(v):
                raise ValueError(f"mask has {len(m)} rows, vectors {len(v)}")
            q.put((i, v, m))
        q.put(None)
    except BaseException as e:
        q.put(RuntimeError(f"cannot read {name}.npy: {e!r}"))


def search(model_name, qvec, k, ckpt, max_shards=0, mask=None):
    """Stream every shard through the GPU; keep a running top-k per query. mask="abstract": only
    papers with an abstract can be returned (full\\mask_abstract, see rag_mask.py)."""
    import numpy as np
    import torch
    vdir = VEC_DIR[model_name]
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(vdir, "*.npy")))
    if max_shards:                                   # smoke tests only
        names = names[:max_shards]
    Q = torch.from_numpy(qvec).to("cuda", torch.float16)
    nq = Q.shape[0]
    best_s = torch.full((nq, k), -2.0, device="cuda")
    best_i = torch.full((nq, k), -1, dtype=torch.int64, device="cuda")   # shard_idx << 32 | row
    start = 0
    if os.path.exists(ckpt):
        with np.load(ckpt) as z:                     # closed again, else Windows can't replace it later
            if list(z["names"]) == names and z["s"].shape == (nq, k):
                best_s, best_i = torch.from_numpy(z["s"]).cuda(), torch.from_numpy(z["i"]).cuda()
                start = int(z["next"])
                log(f"{model_name}: resuming at shard {start}/{len(names)}")
    import jobstatus
    job = jobstatus.Job(f"rag_search_{model_name}", f"RAG eval: exact {model_name.upper()} search "
                        f"({nq:,} queries x all papers, GPU)", unit="shards", total=len(names), order=51)
    job.update("running", done=start, note=os.path.basename(ckpt).split(".")[0])
    q = _queue.Queue(maxsize=3)                      # read-ahead: the disk is the bottleneck
    mask_dir = os.path.join(FULL, "mask_" + mask) if mask else None
    threading.Thread(target=_loader, args=(names[start:], vdir, q, mask_dir), daemon=True).start()
    t0, rows = time.time(), 0
    with torch.inference_mode():
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                job.update("failed", note=str(item)[:200])
                raise item
            j, v, msk = item
            si = start + j
            if len(v):
                V = torch.from_numpy(v).to("cuda", non_blocking=True)
                drop = None if msk is None else ~torch.from_numpy(msk).to("cuda")   # True = not searchable
                kk = min(k, V.shape[0])
                qc = max(64, min(Q_CHUNK, SCORE_ELEMS // V.shape[0]))   # big shards -> fewer queries at once
                for a in range(0, nq, qc):
                    sc = Q[a:a + qc] @ V.T
                    if drop is not None:          # -inf never beats the -2 placeholders, so never kept
                        sc.masked_fill_(drop, float("-inf"))
                    s, r = sc.topk(kk, dim=1)
                    cs = torch.cat([best_s[a:a + qc], s.float()], 1)
                    ci = torch.cat([best_i[a:a + qc], (si << 32) | r], 1)
                    top = cs.topk(k, dim=1)
                    best_s[a:a + qc] = top.values
                    best_i[a:a + qc] = ci.gather(1, top.indices)
                rows += V.shape[0]
                del V
            if (si + 1) % 10 == 0:
                job.update("running", done=si + 1, items=rows,
                           note=f"{os.path.basename(ckpt).split('.')[0]}: {nq:,} queries, top-{k}")
            if (si + 1) % 100 == 0 or si + 1 == len(names):
                el = time.time() - t0
                log(f"{model_name}: shard {si + 1}/{len(names)}, {rows / max(el, 1e-6):,.0f} papers/s, "
                    f"ETA {(len(names) - si - 1) * el / max(1, j + 1) / 60:.0f} min")
                np.savez(ckpt + ".tmp.npz", s=best_s.cpu().numpy(), i=best_i.cpu().numpy(),
                         names=np.array(names), next=si + 1)
                os.replace(ckpt + ".tmp.npz", ckpt)
    job.update("done", done=len(names), items=rows,
               note=f"{os.path.basename(ckpt).split('.')[0]}: top-{k} for {nq:,} queries")
    return names, best_s.cpu().numpy(), best_i.cpu().numpy()


def attach_ids(names, idx):
    """(shard_idx, row) -> OpenAlex W id, reading only the id column of the shards that are needed."""
    import numpy as np
    import pyarrow.parquet as pq
    sh, rw = (idx >> 32).astype(np.int64), (idx & 0xFFFFFFFF).astype(np.int64)
    ids = np.empty(idx.shape, dtype=object)
    for s in np.unique(sh[sh >= 0]):
        col = pq.read_table(os.path.join(FULL, "meta", names[s] + ".parquet"), columns=["id"]).column("id")
        m = sh == s
        ids[m] = np.asarray(col.take(rw[m]).to_pylist(), dtype=object)
    return sh, rw, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "search"])
    ap.add_argument("--model", choices=["minilm", "specter2"], required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-shards", type=int, default=0, help="smoke test: only the first N shards")
    ap.add_argument("--mask", choices=["abstract"], help="only return papers that pass this filter")
    a = ap.parse_args()
    import numpy as np
    qids, texts = read_queries(a.queries)
    qfile = f"{a.out}.{a.model}.q.npy"
    if a.cmd == "encode":
        s = time.time()
        v = encode(a.model, texts)
        np.save(qfile, v)
        log(f"{a.model}: encoded {len(texts):,} queries in {time.time() - s:.0f}s -> {qfile}")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    qvec = np.load(qfile)
    assert qvec.shape == (len(qids), DIM[a.model]), (qvec.shape, len(qids))
    s = time.time()
    names, sc, idx = search(a.model, qvec, a.k, f"{a.out}.{a.model}.ckpt.npz", a.max_shards, a.mask)
    sh, rw, ids = attach_ids(names, idx)
    nq, k = sc.shape
    pq.write_table(pa.table({
        "qid": np.repeat(np.array(qids, dtype=object), k), "rank": np.tile(np.arange(1, k + 1), nq),
        "score": sc.ravel(), "shard": [names[i] if i >= 0 else None for i in sh.ravel()],
        "row": rw.ravel(), "id": ids.ravel()}), f"{a.out}.{a.model}.hits.parquet")
    log(f"{a.model}: searched {nq:,} queries x all shards in {(time.time() - s) / 60:.0f} min -> "
        f"{a.out}.{a.model}.hits.parquet")


if __name__ == "__main__":
    main()
