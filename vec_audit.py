"""
vec_audit.py: check that every vector file still matches its papers (row i == meta row i).

For each shard: header shape == (meta rows, dim) and file size agrees; then re-embed a few sampled
rows (first, last, random) on the GPU and compare with the stored vectors (cosine >= 0.98).
Catches files whose content was damaged without the header showing it (seen after the E: disconnect:
two MiniLM files held another version's vectors).

  the fullindex environment\\python vec_audit.py minilm
  the specter2 environment\\python  vec_audit.py specter2
Writes full\\vec_audit_<model>.json and reports to the dashboard.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import json
import glob
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pyarrow.parquet as pq
import jobstatus

FULL = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "full")
CFG = {"minilm": ("vec", 384), "specter2": ("specter2", 768)}
SAMPLES = 6


def main(model_name):
    sub, dim = CFG[model_name]
    if model_name == "minilm":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        enc = lambda texts: m.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        fmt = lambda ti, ab, sep: f"{ti}. {ab}" if ab else ti
        sep = None
    else:
        import specter2_index
        tok, m = specter2_index.load_model("proximity")
        enc = lambda texts: specter2_index._encode(tok, m, texts).astype(np.float32)
        fmt = lambda ti, ab, sep: f"{ti}{sep}{ab}" if ab else ti
        sep = tok.sep_token
    names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(FULL, "meta", "*.parquet")))
    job = jobstatus.Job(f"vec_audit_{model_name}", f"Vector audit: {model_name.upper()} rows match papers",
                        unit="shards", total=len(names), order=45)
    bad, rng, s = [], np.random.default_rng(1), time.time()
    for i, name in enumerate(names):
        f = os.path.join(FULL, sub, name + ".npy")
        try:
            pf = pq.ParquetFile(os.path.join(FULL, "meta", name + ".parquet"))
            n = pf.metadata.num_rows
            v = np.load(f, mmap_mode="r")
            if v.shape != (n, dim) or os.path.getsize(f) != 128 + n * dim * 2:
                bad.append({"shard": name, "why": f"shape {v.shape} / size {os.path.getsize(f)} vs {n} rows"})
            elif n:
                rows = sorted({0, n - 1, *rng.integers(0, n, SAMPLES - 2).tolist()})
                t = pf.read(columns=["title", "abstract"]).take(rows)
                texts = [fmt(ti, ab, sep) for ti, ab in zip(t.column("title").to_pylist(),
                                                            t.column("abstract").to_pylist())]
                cos = (enc(texts).astype(np.float32) * np.asarray(v[rows], np.float32)).sum(1)
                if cos.min() < 0.98:
                    bad.append({"shard": name, "why": "content mismatch",
                                "cos": [round(float(c), 3) for c in cos], "rows": rows})
        except Exception as e:
            bad.append({"shard": name, "why": repr(e)[:200]})
        if (i + 1) % 10 == 0 or i + 1 == len(names):
            job.update("running", done=i + 1, note=f"bad so far: {len(bad)}")
    json.dump({"checked": len(names), "bad": bad, "secs": round(time.time() - s)},
              open(os.path.join(FULL, f"vec_audit_{model_name}.json"), "w"), indent=1)
    job.update("done" if not bad else "done_with_errors", done=len(names),
               note=f"{len(bad)} bad of {len(names)}" + (": " + ", ".join(b["shard"][-25:] for b in bad[:4]) if bad else ""))
    print(f"{model_name}: {len(bad)} bad of {len(names)} in {time.time() - s:.0f}s")
    for b in bad[:30]:
        print(" ", b)


if __name__ == "__main__":
    main(sys.argv[1])
