"""
rag_mask.py: per-paper "has an abstract" flag, row-aligned with full\\meta and the vector files.

Search over all 484M records is dominated, for short questions, by title-only records ("Deafness",
"Sunbathing", dataset / paratext entries): a bare title that matches a word is closer to the query
than a full paper. A paper without an abstract cannot serve as proof anyway, so rag_search.py
--mask abstract scores only papers that have one (>= MIN_CHARS characters).

  full\\mask_abstract\\<shard>.npy   bool [rows]
Run with the fullindex environment. Reads only the abstract column.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

import jobstatus

FULL = os.path.join(os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research"), "full")
OUT = os.path.join(FULL, "mask_abstract")
MIN_CHARS = 100


def one(name):
    dst = os.path.join(OUT, name + ".npy")
    if os.path.exists(dst):
        m = np.load(dst)
        return len(m), int(m.sum())
    col = pq.read_table(os.path.join(FULL, "meta", name + ".parquet"), columns=["abstract"]).column("abstract")
    m = pc.fill_null(pc.greater_equal(pc.utf8_length(col), MIN_CHARS), False).to_numpy(zero_copy_only=False)
    np.save(dst + ".tmp.npy", m.astype(bool))
    os.replace(dst + ".tmp.npy", dst)
    return len(m), int(m.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(FULL, "meta", "*.parquet")))
    job = jobstatus.Job("rag_mask", "RAG eval: has-abstract filter", unit="shards", total=len(names), order=52)
    rows = kept = 0
    s = time.time()
    with ThreadPoolExecutor(4) as ex:
        for i, (n, k) in enumerate(ex.map(one, names), 1):
            rows, kept = rows + n, kept + k
            if i % 20 == 0:
                job.update("running", done=i, note=f"{kept:,} of {rows:,} papers have an abstract so far")
    note = f"{kept:,} of {rows:,} papers ({kept / rows:.1%}) have an abstract >= {MIN_CHARS} chars"
    job.update("done", done=len(names), note=note)
    print(note, f"({time.time() - s:.0f}s)")


if __name__ == "__main__":
    main()
