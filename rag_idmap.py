"""
rag_idmap.py: OpenAlex W id -> (shard, row) for all 484M papers, so GraphRAG can jump from a cited /
related id straight to its vectors and metadata.

  full\\idmap\\keys.npy   int64, sorted W numbers ("W2961833822" -> 2961833822)
  full\\idmap\\pos.npy    int64, shard_idx << 32 | row   (same order as keys)
  full\\idmap\\names.json shard names in shard_idx order (= sorted full\\meta names)

Lookup = np.searchsorted on the memory-mapped keys (7.7 GB on disk, nothing held in RAM).
The sort runs on the GPU (torch.sort). Run with the fullindex environment.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
import glob
import time

import numpy as np

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
FULL = os.path.join(RESEARCH_DIR, "full")
OUT = os.path.join(FULL, "idmap")


def log(s):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + "[idmap] " + str(s), flush=True)


def build():
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; the id map is sorted on the GPU only.")
    import jobstatus
    os.makedirs(OUT, exist_ok=True)
    names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(FULL, "meta", "*.parquet")))
    job = jobstatus.Job("rag_idmap", "RAG eval: paper id -> row map (for GraphRAG)", unit="shards",
                        total=len(names), order=53)
    job.update("running", done=0, note="reading ids")
    keys, pos = [], []
    s = time.time()
    for si, name in enumerate(names):
        col = pq.read_table(os.path.join(FULL, "meta", name + ".parquet"), columns=["id"]).column("id")
        k = pc.cast(pc.utf8_slice_codeunits(col, 1), "int64").to_numpy()
        keys.append(k)
        pos.append((np.int64(si) << 32) | np.arange(len(k), dtype=np.int64))
        if (si + 1) % 20 == 0:
            job.update("running", done=si + 1, note="reading ids")
        if (si + 1) % 200 == 0:
            log(f"read {si + 1}/{len(names)} shards ({time.time() - s:.0f}s)")
    keys, pos = np.concatenate(keys), np.concatenate(pos)
    log(f"{len(keys):,} ids; sorting on the GPU")
    job.update("running", done=len(names), note=f"sorting {len(keys):,} ids on the GPU")
    with torch.no_grad():
        order = torch.sort(torch.from_numpy(keys).cuda(), stable=True).indices.cpu().numpy()
    torch.cuda.empty_cache()
    keys, pos = keys[order], pos[order]
    dup = int((keys[1:] == keys[:-1]).sum())
    for nm, arr in (("keys", keys), ("pos", pos)):
        np.save(os.path.join(OUT, nm + ".tmp.npy"), arr)
        os.replace(os.path.join(OUT, nm + ".tmp.npy"), os.path.join(OUT, nm + ".npy"))
    json.dump(names, open(os.path.join(OUT, "names.json"), "w"))
    log(f"done in {time.time() - s:.0f}s; duplicate ids: {dup:,}")
    job.update("done", note=f"{len(keys):,} ids mapped; duplicates: {dup:,}")


class IdMap:
    """W id strings -> (shard name, row), or None when the id is not in the index."""

    def __init__(self):
        self.keys = np.load(os.path.join(OUT, "keys.npy"), mmap_mode="r")
        self.pos = np.load(os.path.join(OUT, "pos.npy"), mmap_mode="r")
        self.names = json.load(open(os.path.join(OUT, "names.json")))

    def lookup(self, ids):
        q = np.array([int(i.rsplit("/", 1)[-1][1:]) for i in ids], dtype=np.int64)
        j = np.searchsorted(self.keys, q)
        out = []
        for qi, ji in zip(q, j):
            if ji < len(self.keys) and self.keys[ji] == qi:
                p = int(self.pos[ji])
                out.append((self.names[p >> 32], p & 0xFFFFFFFF))
            else:
                out.append(None)
        return out


if __name__ == "__main__":
    build()
