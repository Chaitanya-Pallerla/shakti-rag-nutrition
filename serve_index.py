"""
serve_index.py: a compact, fast search index for the live chat / human-survey server (survey_server.py).

The evaluation uses exact brute force (rag_search.py, ~30 min per pass): right for a benchmark, far
too slow for a person waiting in a chat. This builds, for the papers WITH an abstract (rag_mask.py,
~230M), on the NVMe drive:

  pq.npz            product-quantisation codebooks for MiniLM: 48 sub-spaces x 256 centroids x 8 dims
  codes\\<shard>.npy uint8 (n, 48): one 48-byte code per paper (all shards ~11 GB -> fits on the GPU)
  text\\<shard>.parquet  id, title, abstract (first 2000 chars), year, doi, venue for the same rows,
                    1024-row groups so one paper costs one small read

Row j of codes\\<shard>.npy is row j of text\\<shard>.parquet. Search = asymmetric PQ distance of the
query against all codes on the GPU (~1 s), then MedCPT cross-encoder rerank of the top few hundred.
GPU only (k-means and encoding run in torch on CUDA). Resumable per shard.

  python serve_index.py build
  python serve_index.py check        recall of the PQ search against the exact eval results
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import glob
import time
import threading
import queue as _queue

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import jobstatus

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
FULL = os.path.join(RESEARCH_DIR, "full")
OUT = os.environ.get("SHAKTI_SERVE_INDEX", "shakti_serve_index")
M, KS, DSUB = 48, 256, 8                     # 48 x 8 = 384 = MiniLM dimension
TEXT_CHARS = 2000
LOG = os.path.join(OUT, "serve_index.log")


def log(s):
    os.makedirs(OUT, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + "[serve_index] " + str(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def shard_names():
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(FULL, "vec", "*.npy")))
    smoke = int(os.environ.get("SERVE_SMOKE_SHARDS", "0"))          # smoke tests only
    return names[:smoke] if smoke else names


def train_pq(names, per_shard=8000, n_shards=300, iters=25):
    import numpy as np
    import torch
    path = os.path.join(OUT, "pq.npz")
    if os.path.exists(path):
        return torch.from_numpy(np.load(path)["C"]).cuda()
    rng = np.random.default_rng(7)
    pick = sorted(rng.choice(len(names), min(n_shards, len(names)), replace=False))
    xs = []
    for k, i in enumerate(pick):
        v = np.load(os.path.join(FULL, "vec", names[i] + ".npy"), mmap_mode="r")
        m = np.load(os.path.join(FULL, "mask_abstract", names[i] + ".npy"))
        rows = np.flatnonzero(m)
        if len(rows):
            xs.append(np.asarray(v[np.sort(rng.choice(rows, min(per_shard, len(rows)), replace=False))]))
        if k % 50 == 0:
            log(f"PQ training sample: {k}/{len(pick)} shards")
    X = torch.from_numpy(np.concatenate(xs)).cuda().float().view(-1, M, DSUB).transpose(0, 1).contiguous()
    n = X.shape[1]
    log(f"PQ training: {n:,} vectors, {M} x {KS} centroids, {iters} k-means iterations (GPU)")
    C = X[:, torch.randperm(n, device="cuda")[:KS]].clone()            # (M, KS, DSUB)
    for it in range(iters):
        for m in range(M):
            x = X[m]
            a = (x @ C[m].T).mul_(-2).add_((C[m] ** 2).sum(1)).argmin(1)   # nearest centroid
            s = torch.zeros(KS, DSUB, device="cuda").index_add_(0, a, x)
            c = torch.bincount(a, minlength=KS).float()
            ok = c > 0
            C[m][ok] = s[ok] / c[ok, None]
            if (~ok).any():                                              # re-seed empty centroids
                C[m][~ok] = x[torch.randint(0, n, (int((~ok).sum()),), device="cuda")]
    np.savez(path, C=C.cpu().numpy())
    return C


def encode(C, v):
    import torch
    x = torch.from_numpy(v).cuda().float().view(-1, M, DSUB)
    codes = torch.empty(x.shape[0], M, dtype=torch.uint8, device="cuda")
    for a in range(0, x.shape[0], 500_000):
        xb = x[a:a + 500_000].transpose(0, 1)                          # (M, b, DSUB)
        d = torch.baddbmm((C ** 2).sum(2)[:, None, :], xb, C.transpose(1, 2), alpha=-2)
        codes[a:a + 500_000] = d.argmin(2).T.to(torch.uint8)
    return codes.cpu().numpy()


def _reader(names, q):
    import numpy as np
    import pyarrow.parquet as pq
    try:
        for name in names:
            m = np.load(os.path.join(FULL, "mask_abstract", name + ".npy"))
            v = np.load(os.path.join(FULL, "vec", name + ".npy"))
            t = pq.read_table(os.path.join(FULL, "meta", name + ".parquet"),
                              columns=["id", "title", "abstract", "year", "doi", "venue"])
            q.put((name, m, v, t))
        q.put(None)
    except BaseException as e:
        q.put(RuntimeError(f"cannot read {name}: {e!r}"))


def build():
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; the index is built on the GPU only.")
    names = shard_names()
    for d in ("codes", "text"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    job = jobstatus.Job("serve_index", "Survey server: fast search index (PQ codes + text store, GPU)",
                        unit="shards", total=len(names), order=70)
    job.update("running", done=0, note="training the PQ codebooks")
    C = train_pq(names)
    todo = [n for n in names if not os.path.exists(os.path.join(OUT, "text", n + ".parquet"))]
    log(f"encoding {len(todo):,} of {len(names):,} shards")
    q = _queue.Queue(maxsize=3)
    threading.Thread(target=_reader, args=(todo, q), daemon=True).start()
    t0, papers, k = time.time(), 0, len(names) - len(todo)
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            job.update("failed", done=k, note=str(item)[:200])
            raise item
        name, m, v, t = item
        rows = np.flatnonzero(m)
        np.save(os.path.join(OUT, "codes", name + ".npy"), encode(C, v[rows]) if len(rows) else
                np.zeros((0, M), np.uint8))
        t = t.take(pa.array(rows))
        t = t.set_column(t.column_names.index("abstract"), "abstract",
                         pc.utf8_slice_codeunits(t.column("abstract"), 0, TEXT_CHARS))
        tmp = os.path.join(OUT, "text", name + ".parquet.tmp")
        pq.write_table(t, tmp, row_group_size=1024, compression="zstd")
        os.replace(tmp, os.path.join(OUT, "text", name + ".parquet"))   # written last = shard done
        papers += len(rows)
        k += 1
        if k % 10 == 0:
            el = time.time() - t0
            job.update("running", done=k, items=papers, note=f"{papers:,} papers encoded this run")
            if k % 100 == 0:
                log(f"shard {k}/{len(names)}, {papers / el:,.0f} papers/s, "
                    f"ETA {(len(names) - k) * el / max(1, k - (len(names) - len(todo))) / 60:.0f} min")
    job.update("done", done=len(names), note="PQ codes + text store ready")
    log("build done")


class Searcher:
    """Loads all codes onto the GPU. search(qvecs, k) -> (scores, global rows); text(rows) -> records."""

    def __init__(self):
        import numpy as np
        import torch
        import pyarrow.parquet as pq
        self.names = shard_names()
        C = torch.from_numpy(np.load(os.path.join(OUT, "pq.npz"))["C"]).cuda()
        self.C = C
        parts, starts, acc = [], [], 0
        for n in self.names:
            c = np.load(os.path.join(OUT, "codes", n + ".npy"))
            starts.append(acc)
            acc += len(c)
            parts.append(c)
        self.starts = np.array(starts + [acc], dtype=np.int64)
        self.codes = torch.from_numpy(np.concatenate(parts)).cuda()      # (N, M) uint8, ~11 GB
        self.off = (torch.arange(M, device="cuda", dtype=torch.int32) * KS)[None, :]
        self.files = {}
        self.lock = threading.Lock()
        self.n = acc

    def search(self, qv, k=300):
        """qv: (Q, 384) float, normalised. Asymmetric PQ inner product over every paper."""
        import torch
        import torch.nn.functional as F
        q = torch.as_tensor(qv, device="cuda").float().view(-1, M, DSUB)
        nq = q.shape[0]
        lut = torch.einsum("qmd,mkd->mkq", q, self.C).reshape(M * KS, nq).half()   # (M*KS, Q)
        chunk = max(200_000, min(4_000_000, 600_000_000 // (M * nq)))   # int32 idx <= 0.8 GB, (chunk, M, Q) fp16 <= 1.2 GB
        best_s = torch.full((nq, k), -1e4, device="cuda")
        best_i = torch.zeros((nq, k), dtype=torch.int64, device="cuda")
        with self.lock, torch.inference_mode():
            for a in range(0, self.n, chunk):
                idx = self.codes[a:a + chunk].int() + self.off
                s = F.embedding(idx, lut).sum(1, dtype=torch.float32).T       # (Q, chunk)
                ts, ti = s.topk(min(k, s.shape[1]), dim=1)
                cs = torch.cat([best_s, ts], 1)
                ci = torch.cat([best_i, ti + a], 1)
                top = cs.topk(k, dim=1)
                best_s, best_i = top.values, ci.gather(1, top.indices)
        return best_s.cpu().numpy(), best_i.cpu().numpy()

    def text(self, rows):
        import numpy as np
        import pyarrow.parquet as pq
        out = []
        for g in rows:
            s = int(np.searchsorted(self.starts, g, side="right") - 1)
            r = int(g - self.starts[s])
            name = self.names[s]
            pf = self.files.get(name)
            if pf is None:
                pf = self.files[name] = pq.ParquetFile(os.path.join(OUT, "text", name + ".parquet"))
            out.append(pf.read_row_group(r // 1024).slice(r % 1024, 1).to_pylist()[0])
        return out


def check(n=1000):
    """Recall of the PQ top-k against the exact MiniLM top-10 of the evaluation (abstract-only run)."""
    import json
    import numpy as np
    import pyarrow.parquet as pq
    from collections import defaultdict
    ev = os.path.join(RESEARCH_DIR, "rag_eval")
    qv = np.load(os.path.join(ev, "round1abs.minilm.q.npy"))
    with open(os.path.join(ev, "questions.jsonl"), encoding="utf-8-sig") as f:
        qids = [json.loads(ln)["qid"] for ln in f if ln.strip()]
    sel = np.random.default_rng(1).choice(len(qids), n, replace=False)
    want = {qids[i] for i in sel}
    S = Searcher()
    sidx = {name: i for i, name in enumerate(S.names)}
    ranks = {}                                   # shard -> rank of each row among its abstract rows
    exact = defaultdict(set)                     # qid -> global PQ rows of the exact MiniLM top-10
    for r in pq.read_table(os.path.join(ev, "round1abs.minilm.hits.parquet"),
                           columns=["qid", "rank", "shard", "row"]).to_pylist():
        if r["qid"] in want and r["rank"] <= 10 and r["shard"]:
            if r["shard"] not in ranks:
                ranks[r["shard"]] = np.cumsum(np.load(os.path.join(FULL, "mask_abstract", r["shard"] + ".npy"))) - 1
            exact[r["qid"]].add(int(S.starts[sidx[r["shard"]]] + ranks[r["shard"]][r["row"]]))
    res = {}
    for k in (100, 300, 1000):
        hit, tot, t = 0, 0, 0.0
        for b in range(0, n, 16):
            idx = sel[b:b + 16]
            s0 = time.time()
            _, rows = S.search(qv[idx], k=k)
            t += time.time() - s0
            for qi, rr in zip(idx, rows):
                e = exact[qids[qi]]
                hit += len(e & set(rr.tolist()))
                tot += len(e)
        res[k] = {"recall_of_exact_top10": hit / max(1, tot), "sec_per_16_queries": t / (n / 16)}
        log(f"PQ check k={k}: {res[k]}")
    json.dump(res, open(os.path.join(OUT, "pq_check.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "check": check}[cmd]()
