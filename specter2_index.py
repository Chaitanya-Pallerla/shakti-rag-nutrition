"""
specter2_index.py: embed every paper of the full index with SPECTER2 (allenai/specter2_base +
the "proximity" adapter, the document-side encoder) for the retrieval-model comparison.

Reads the SAME shards as the MiniLM index (full\\meta\\<name>.parquet) and writes
full\\specter2\\<name>.npy (float16, L2-normalised [CLS] vectors, 768-d), so row i of every
vector file is the same paper -> a like-for-like comparison. Queries must be encoded with the
"adhoc_query" adapter (see encode_queries()).

Input per SPECTER2's model card: title + [SEP] + abstract, max 512 tokens. GPU only (no CPU
fallback). Resumable per shard. Pause: create full\\specter2\\STOP.
Run with the specter2 environment (needs transformers 4.57 + adapters, separate from the MiniLM env).
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import sys
import time
import glob
import threading
import queue as _queue

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import jobstatus

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
FULL = os.path.join(RESEARCH_DIR, "full")
META = os.path.join(FULL, "meta")
OUT = os.path.join(FULL, "specter2")
LOG = os.path.join(FULL, "specter2.log")
STOP = os.path.join(OUT, "STOP")
MODELS = os.path.join(os.environ.get("SHAKTI_MODELS_DIR", "models"), "specter2")
MAX_LEN = 512
TOKENS_PER_BATCH = int(os.environ.get("SPECTER2_TOKENS", "65536"))   # padded tokens per GPU batch
# Hard cap on PyTorch's VRAM share. Without it the Windows driver silently spills past 16 GB into
# system RAM ("sysmem fallback") and throughput drops ~25x; with it we get an OOM and split the batch.
MEM_FRACTION = float(os.environ.get("SPECTER2_MEM_FRACTION", "0.70"))


def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def load_model(adapter="proximity"):
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; SPECTER2 indexing runs on the GPU only.")
    tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "base"))
    model = AutoAdapterModel.from_pretrained(os.path.join(MODELS, "base"))
    model.load_adapter(os.path.join(MODELS, adapter), load_as=adapter, set_active=True)
    model = model.to("cuda").half().eval()
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    return tok, model


def _forward(model, enc):
    """[CLS] vectors for one batch; on OOM, free the cache and split the batch in half."""
    import torch
    try:
        return model(**enc).last_hidden_state[:, 0, :].float()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        n = enc["input_ids"].shape[0]
        if n == 1:
            raise
        h = n // 2
        return torch.cat([_forward(model, {k: v[:h] for k, v in enc.items()}),
                          _forward(model, {k: v[h:] for k, v in enc.items()})])


def _encode(tok, model, texts):
    """texts -> float16 [n,768] L2-normalised, in input order. Length-sorted, token-budget batches;
    a background thread tokenises the next batch while the GPU runs the current one."""
    import numpy as np
    import torch
    n = len(texts)
    out = np.zeros((n, 768), np.float16)
    if not n:
        return out
    order = sorted(range(n), key=lambda i: len(texts[i]))
    batches, i = [], 0
    while i < n:                                     # est. tokens ~ chars/4, capped at MAX_LEN
        est = min(MAX_LEN, len(texts[order[min(n - 1, i + 31)]]) // 4 + 8)
        bs = max(8, min(1024, TOKENS_PER_BATCH // max(16, est)))
        batches.append(order[i:i + bs])
        i += bs
    q = _queue.Queue(maxsize=4)

    def producer():
        for b in batches:
            enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt", return_token_type_ids=False)
            q.put((b, enc))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    with torch.inference_mode():
        while True:
            item = q.get()
            if item is None:
                break
            b, enc = item
            enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
            cls = _forward(model, enc)
            cls = torch.nn.functional.normalize(cls, dim=1)
            out[b] = cls.half().cpu().numpy()
    return out


def encode_queries(queries):
    """For retrieval experiments: short queries use the adhoc_query adapter (same 768-d space)."""
    tok, model = load_model("adhoc_query")
    return _encode(tok, model, list(queries))


def embed_shard(tok, model, name):
    import numpy as np
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(META, name + ".parquet"), columns=["title", "abstract"])
    titles = t.column("title").to_pylist()
    abstracts = t.column("abstract").to_pylist()
    sep = tok.sep_token
    texts = [f"{ti}{sep}{ab}" if ab else ti for ti, ab in zip(titles, abstracts)]
    vecs = _encode(tok, model, texts)
    dst = os.path.join(OUT, name + ".npy")
    with open(dst + ".tmp", "wb") as f:
        np.save(f, vecs)
        f.flush()
        os.fsync(f.fileno())
    os.replace(dst + ".tmp", dst)
    return len(texts)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.tmp")):
        os.remove(f)
    if os.path.exists(STOP):
        os.remove(STOP)
    job = jobstatus.Job("specter2", "SPECTER2 embeddings (GPU)", unit="shards", order=20)
    job.update("starting", note="loading model")
    import torch
    tok, model = load_model("proximity")
    names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(META, "*.parquet")))
    have = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(OUT, "*.npy"))}
    todo = [n for n in names if n not in have]
    done = len(names) - len(todo)
    log(f"=== SPECTER2 start: GPU={torch.cuda.get_device_name(0)}; {done}/{len(names)} shards done ===")
    papers = 0
    for name in todo:
        if os.path.exists(STOP):
            log(f"=== paused at {done}/{len(names)} shards ===")
            job.update("paused", done=done, total=len(names), items=papers)
            return
        job.update("running", done=done, total=len(names), items=papers, note=name)
        s = time.time()
        n = embed_shard(tok, model, name)
        torch.cuda.empty_cache()
        done += 1
        papers += n
        log(f"shard {done}/{len(names)} {name}: {n} papers in {time.time() - s:.0f}s "
            f"({n / max(1e-6, time.time() - s):.0f}/s)")
    job.update("done", done=done, total=len(names), items=papers, note="all shards embedded")
    log(f"=== SPECTER2 DONE: {done}/{len(names)} shards ===")


if __name__ == "__main__":
    import faulthandler
    import traceback
    os.makedirs(OUT, exist_ok=True)
    faulthandler.enable(open(os.path.join(FULL, "specter2_crash.log"), "a", encoding="utf-8"))
    try:
        main()
    except BaseException:
        log("CRASH: " + traceback.format_exc())
        try:
            jobstatus.Job("specter2", "SPECTER2 embeddings (GPU)", order=20).update(
                "failed", note=traceback.format_exc().strip().splitlines()[-1][:200])
        except Exception:
            pass
        raise
