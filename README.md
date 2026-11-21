# SHAKTI: evidence-cited nutrition and health answers from 230 million research papers

SHAKTI (Smart Health Advisor for Knowledge-based Tailored Interventions) is a retrieval-augmented chat system for nutrition, food and health questions. It answers only from research abstracts it has retrieved and from USDA food-composition data. Every answer is then checked automatically, without calling a model, for fabricated citations, non-verbatim quotes, numbers with no source, and weakly supported sentences.

This repository contains the code for the paper's system and for its evaluation. You can use it with **Claude** (Claude Code CLI), **ChatGPT / OpenAI** (Codex CLI or the API), **any MCP client**, or **fully offline models** such as **Qwen2.5-VL** through Ollama, vLLM or LM Studio.

> Not medical advice. SHAKTI summarises published research automatically and can be wrong or incomplete.

---

## Contents

1. [How it works](#how-it-works)
2. [Quick start](#quick-start)
3. [Your settings in one file: `shakti.env`](#your-settings-in-one-file-shaktienv)
4. [Connecting a language model](#connecting-a-language-model): Claude CLI, Codex / ChatGPT, OpenAI API, offline Qwen2.5-VL
5. [Using SHAKTI from an MCP client](#using-shakti-from-an-mcp-client)
6. [What has been tested, and with which models](#what-has-been-tested-and-with-which-models)
7. [Data: the indexed OpenAlex corpus (≈1.4 TB, available on request)](#data)
8. [Building everything from scratch](#building-everything-from-scratch)
9. [Reproducing the evaluation](#reproducing-the-evaluation)
10. [Files](#files)

---

## How it works

Each message passes through six stages (`shakti_chat_server.py`):

| # | Stage | What happens |
|---|-------|--------------|
| 1 | Understand | The model rewrites the message into a standalone question, using the conversation so far. It also classifies the question and lists any foods that need nutrition data. |
| 2 | Food data | Exact values come from a local copy of USDA FoodData Central. No values are estimated by the model. |
| 3 | Literature | A product-quantised search over the MiniLM vectors of about 230M papers with abstracts runs on the GPU and takes about 1 s. A MedCPT cross-encoder then reranks the results. The retrieval method is single-pass, corrective (CRAG) or deep research. |
| 4 | Link check | Each source's DOI or OpenAlex record is resolved, and sources that no longer resolve are dropped. |
| 5 | Answer | The model writes the answer from the numbered papers and food rows only, citing them as [1] or [F1] and quoting sentences word for word. |
| 6 | Verification | Runs without any model call. Every citation must point to a provided source, and every quote must occur verbatim in the cited paper. Every number must appear in the sources, the food data or the conversation, or be a sum or difference of such numbers. Cited sentences must be semantically close to a sentence of the cited paper (MiniLM cosine ≥ 0.45). The result is shown to the user as a risk level of low, medium or high, with the reasons. |

---

## Quick start

Requirements:

- Python 3.10.
- An NVIDIA GPU. 16 GB is enough: the chat server needs about 12 GB.
- The **serving index** (113 GB). Either build it yourself (see [below](#building-everything-from-scratch)) or request it (see [Data](#data)).

```bash
git clone https://github.com/<your-account>/shakti-rag-nutrition.git
cd shakti-rag-nutrition

python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cp shakti.env.example shakti.env                         # Windows: copy shakti.env.example shakti.env
#   -> open shakti.env and paste your paths / model choice / keys
python shakti_config.py                                  # shows what was loaded (keys masked)

python rag_llm.py                                        # 1-question smoke test of your language model
python shakti_chat_server.py --port 8780                 # then open http://127.0.0.1:8780
```

The server listens on `127.0.0.1` only. When it starts it prints an admin link (`/admin?token=...`) to a summary of all chats.

---

## Your settings in one file: `shakti.env`

All the folders, the model choice and any API keys live in **one file**. To set it up:

1. Copy `shakti.env.example` to `shakti.env`.
2. Paste your values after the `=` signs. Leave a value empty to keep the default.
3. Every script loads the file automatically, through `shakti_config.py`, which each script imports first.

`shakti.env` is listed in `.gitignore`, so **your keys and local paths are never committed**. A variable already set in your shell takes precedence over the file, which is handy for a one-off override. To keep the file somewhere else, set `SHAKTI_ENV_FILE=/path/to/file`.

The most important entries:

| Setting | What to paste |
|---------|---------------|
| `SHAKTI_RESEARCH_DIR` | Folder for the corpus, vectors, benchmark runs and chat logs (≈1.4 TB for the full corpus) |
| `SHAKTI_SERVE_INDEX` | Folder with the 113 GB serving index (an SSD or NVMe drive is strongly recommended) |
| `SHAKTI_FOOD_DB` | Path to `food.duckdb` (built with `build_food_db.py`) |
| `SHAKTI_MODELS_DIR` | Folder containing `medcpt-cross-encoder/` and `specter2/` |
| `SHAKTI_LLM_BACKEND` | `claude`, `codex` or `openai` (see the next section) |
| `SHAKTI_LLM_MODEL` | Model name for the codex and openai backends, e.g. `qwen2.5vl:7b` |
| `SHAKTI_OPENAI_BASE_URL` | Your OpenAI-compatible server, e.g. `http://127.0.0.1:11434/v1` for Ollama |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `SHAKTI_OPENAI_API_KEY`, `HF_TOKEN` | Optional keys. Leave them empty if you sign in to a CLI or run models locally. |

---

## Connecting a language model

All model calls go through `rag_llm.py`. Choose a backend with `SHAKTI_LLM_BACKEND` in `shakti.env`. Calls are stateless, use no tools, and are cached on disk, so a rerun never pays twice for the same call.

| Backend | Uses | Key needed? | Status |
|---------|------|-------------|--------|
| `claude` | Claude Code CLI (`claude -p`) | No: sign in with your Claude account. `ANTHROPIC_API_KEY` also works. | Used for all results in the paper |
| `codex` | OpenAI Codex CLI (`codex exec`) | No: sign in with your ChatGPT account. `OPENAI_API_KEY` also works. | Experimental |
| `openai` | Any OpenAI-compatible `/v1/chat/completions` server | Only for the OpenAI API or a server protected by a key | Tested against a mock server; tested in the paper with Claude only |

The code uses three model aliases: `sonnet` for answering and grading, `opus` for the evaluation judge, and `haiku`. For the `claude` backend they are passed through unchanged. For the other backends, `SHAKTI_LLM_MODEL` sets the model for all of them, and `SHAKTI_LLM_MODEL_OPUS` (or `_SONNET`, `_HAIKU`) overrides one alias.

### A. Claude (Claude Code CLI): the default

```bash
npm install -g @anthropic-ai/claude-code
claude                      # sign in once, then exit
```

```ini
# shakti.env
SHAKTI_LLM_BACKEND=claude
```

### B. ChatGPT account (OpenAI Codex CLI)

```bash
npm install -g @openai/codex
codex login                 # sign in with ChatGPT (or: codex login --api-key ...)
```

```ini
SHAKTI_LLM_BACKEND=codex
SHAKTI_LLM_MODEL=gpt-5
```

Codex runs read-only, in an empty folder, with the JSON schema written into the prompt.

### C. OpenAI API (or another hosted OpenAI-compatible API)

```ini
SHAKTI_LLM_BACKEND=openai
SHAKTI_OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
SHAKTI_LLM_MODEL=gpt-5
```

### D. Offline: Qwen2.5-VL (or any local model)

No account, no key, and no data leaves your machine. SHAKTI sends many abstracts per question, so **give the model a context of at least 32k tokens**. Local models are also slower, so raise the timeouts.

**Ollama**

```bash
ollama pull qwen2.5vl:7b
OLLAMA_CONTEXT_LENGTH=32768 ollama serve      # Windows: set OLLAMA_CONTEXT_LENGTH=32768 then: ollama serve
```

```ini
SHAKTI_LLM_BACKEND=openai
SHAKTI_OPENAI_BASE_URL=http://127.0.0.1:11434/v1
SHAKTI_LLM_MODEL=qwen2.5vl:7b
SHAKTI_T_STEP=300
SHAKTI_T_ANSWER=600
```

**vLLM**

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 32768
```

```ini
SHAKTI_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
SHAKTI_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

**LM Studio:** start its local server, then set `SHAKTI_OPENAI_BASE_URL=http://127.0.0.1:1234/v1` and `SHAKTI_LLM_MODEL` to the model's identifier.

**Notes on local models:**

- **GPU memory.** The chat server already uses about 12 GB of GPU memory for the search index and MedCPT, so a 7B model on the same 16 GB GPU will not fit. Run the model on a second GPU or another machine (set `SHAKTI_OPENAI_BASE_URL` to that machine), or use a smaller quantised model.
- **JSON output.** SHAKTI asks for structured JSON output (`response_format: json_schema`). If your server rejects that, SHAKTI retries automatically with the schema written into the prompt. You can also set `SHAKTI_OPENAI_JSON_SCHEMA=0` to skip the first attempt.
- **Answer quality.** Smaller models follow the citation and quote rules less reliably. Stage 6 still checks every answer and reports the problems it finds, so you can see how well your model does.

Test your choice with `python rag_llm.py`, which asks one yes/no question and prints the reply.

---

## Using SHAKTI from an MCP client

`mcp_server.py` exposes SHAKTI as two MCP tools:

- `ask_shakti` returns a cited answer, the sources and the verification result.
- `ask_shakti_json` returns the full JSON result.

It is a thin client of a running chat server, so **start `shakti_chat_server.py` first**. The MCP server connects to `SHAKTI_URL`, which defaults to `http://127.0.0.1:8780`.

**Claude Code**

```bash
claude mcp add shakti -- python /full/path/to/shakti-rag-nutrition/mcp_server.py
```

**Claude Desktop:** add this to `claude_desktop_config.json` (Settings → Developer → Edit config):

```json
{
  "mcpServers": {
    "shakti": {
      "command": "/full/path/to/.venv/bin/python",
      "args": ["/full/path/to/shakti-rag-nutrition/mcp_server.py"],
      "env": { "SHAKTI_URL": "http://127.0.0.1:8780" }
    }
  }
}
```

**OpenAI Codex CLI:** add this to `~/.codex/config.toml`:

```toml
[mcp_servers.shakti]
command = "/full/path/to/.venv/bin/python"
args = ["/full/path/to/shakti-rag-nutrition/mcp_server.py"]
```

Any other MCP client that can start a stdio server (Cursor, Windsurf, Continue and others) is configured the same way: give it the command plus the path to `mcp_server.py`.

The MCP route keeps SHAKTI's own model, set by `SHAKTI_LLM_BACKEND`, for writing the answer. Your assistant then receives the cited, verified result and can discuss it with you.

---

## What has been tested, and with which models

### In the paper (all results)

| What | Models | Scale |
|------|--------|-------|
| Retrieval benchmark: exact search over all 484M works, hybrid fusion, reranking | all-MiniLM-L6-v2, SPECTER2 (proximity + adhoc_query adapters), MedCPT cross-encoder | 10,065 questions (NFCorpus, BioASQ, SciFact, PubMedQA) |
| RAG methods: vanilla, multi-hop, CRAG, deep research | Answering: **Claude Sonnet**. Judge: **Claude Opus**. Both through the Claude Code CLI (`claude` backend, `sonnet` / `opus` aliases), November 2026. | 1,000 questions per method (999 scored); claim-level judging on 200 per method |
| Live SHAKTI chat (stages 1–6) | Claude Sonnet answering, Claude Opus as the claim / rewrite / relevance judge, the same retrieval models | 12 scripted conversations, 36 turns; load test with 38 fresh questions |
| Serving index | PQ 48 × 8 over MiniLM vectors, GPU search | 229.8M papers; its top-300 contains 98.2% of the exact top-10 |

The hardware was one workstation: an Intel Core i9-13900KF, 32 GB RAM, an NVIDIA RTX 4080 (16 GB), Windows 11 and Python 3.10. Package versions are pinned in `requirements.txt` and `requirements-specter2.txt`. Every Claude response is cached on disk, so the scoring can be replayed exactly.

The CLI aliases point to the newest Sonnet and Opus versions available at the time the evaluation ran. Running the evaluation later, or with another backend, may give different numbers.

### Checks on this public copy (November 2026)

| Component | Result |
|-----------|--------|
| All Python files compile; no personal paths, names or keys | Passed |
| `shakti_config.py` loads `shakti.env` (relative paths resolved, keys masked when printed) | Passed |
| `openai` backend: structured JSON, automatic fallback when a server rejects `json_schema`, token counts, disk cache | Passed against a mock OpenAI-compatible server |
| `build_food_db.py` on CSV files in the FoodData Central format; the chat server's food queries run on the result | Passed |
| `flatten_openalex.py` on the snapshot folder layout | Passed |

### Not tested yet

- The `codex` backend (Codex CLI / ChatGPT sign-in).
- A real local model on the `openai` backend: Qwen2.5-VL on Ollama, vLLM or LM Studio. The OpenAI API itself is also untested.
- `mcp_server.py` end to end with an MCP client.
- Linux and macOS. Everything above ran on Windows.

If you run one of these, please open an issue with your results, both successes and failures.

---

## Data

### The indexed and embedded OpenAlex corpus

The paper searches the **complete OpenAlex works snapshot: 484,368,086 works**, of which 229,801,842 have an abstract. Every work was embedded on the GPU with all-MiniLM-L6-v2 (384-d) and with SPECTER2 (768-d). Measured sizes, in GB (10⁹ bytes):

| Part | Folder | Size | Needed for |
|------|--------|-----:|------------|
| Paper metadata (id, title, abstract, year, DOI, venue, authors…) | `full/meta/*.parquet` | 132.0 GB | Everything |
| MiniLM vectors, fp16, 484M × 384 | `full/vec/*.npy` | 372.0 GB | Benchmark search; building the serving index |
| SPECTER2 vectors, fp16, 484M × 768 | `full/specter2/*.npy` | 744.0 GB | SPECTER2 / hybrid benchmark runs |
| Citation graph (references, related works, topics) | `full/graph/*.parquet` | 35.8 GB | GraphRAG / citation features |
| Abstract mask | `full/mask_abstract` | 0.5 GB | Abstract-only runs |
| **Serving index**: PQ codes plus title/abstract text for 229.8M papers | `shakti_serve_index/` | **112.9 GB** | **The chat server and the MCP tools** |
| **Total** | | **≈1,397 GB (≈1.4 TB)** | |

The data comes in 2,127 shards, one per OpenAlex snapshot file. The raw OpenAlex snapshot itself is not included because anyone can download it (see below).

**Available on request.** This data is too large to host on GitHub. To get a copy, **open an issue on this repository** and say which parts you need. For example, only the 113 GB serving index is enough to run the chat. Large transfers are usually easiest on a drive you send, or through a cloud bucket you provide. OpenAlex data is released under CC0, so the derived index can be shared freely.

### Models

| Model | Where to put it | Source |
|-------|-----------------|--------|
| all-MiniLM-L6-v2 | Downloaded automatically into the Hugging Face cache | `sentence-transformers/all-MiniLM-L6-v2` |
| MedCPT cross-encoder | `$SHAKTI_MODELS_DIR/medcpt-cross-encoder/` | `ncbi/MedCPT-Cross-Encoder` |
| SPECTER2 (base + adapters) | `$SHAKTI_MODELS_DIR/specter2/{base,proximity,adhoc_query}/` | `allenai/specter2_base`, `allenai/specter2`, `allenai/specter2_adhoc_query` |

```bash
huggingface-cli download ncbi/MedCPT-Cross-Encoder --local-dir models/medcpt-cross-encoder
huggingface-cli download allenai/specter2_base       --local-dir models/specter2/base
huggingface-cli download allenai/specter2            --local-dir models/specter2/proximity
huggingface-cli download allenai/specter2_adhoc_query --local-dir models/specter2/adhoc_query
```

### Food data

Download **"Full Download of All Data Types" (CSV)** from [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets), unzip it, then run:

```bash
python build_food_db.py /path/to/FoodData_Central_csv_folder      # -> food.duckdb (SHAKTI_FOOD_DB)
```

---

## Building everything from scratch

Expect about 1.4 TB of disk space and several days of GPU time. All steps are resumable: rerun a step after any interruption and it continues where it stopped. Embedding runs on the GPU only, and the scripts refuse to fall back to the CPU.

```bash
# 1. OpenAlex works snapshot (free, no account), then flatten it to the layout full_index.py reads
aws s3 sync s3://openalex/data/works ./openalex_download --no-sign-request
python flatten_openalex.py ./openalex_download            # -> $SHAKTI_RESEARCH_DIR/openalex_works/

# 2. metadata + MiniLM vectors for every work (GPU)
python full_index.py

# 3. SPECTER2 vectors (second environment: requirements-specter2.txt)
python specter2_index.py

# 4. optional extras: citation graph, corpus statistics, vector audit
python citation_graph.py
python corpus_stats.py
python vec_audit.py

# 5. abstract mask + compact serving index for the chat (GPU k-means, PQ 48 x 8)
python rag_mask.py
python serve_index.py build
python serve_index.py check            # recall of the PQ search against exact search
```

---

## Reproducing the evaluation

The benchmark set has **10,065 questions** from NFCorpus, BioASQ, SciFact and PubMedQA. Gold papers are mapped to OpenAlex IDs.

```bash
# benchmark questions + gold mapping
python rag_benchmarks.py

# exact search over all shards (MiniLM in the main env, SPECTER2 in the specter2 env)
python rag_search.py encode --model minilm   --queries $SHAKTI_RESEARCH_DIR/rag_eval/questions.jsonl --out $SHAKTI_RESEARCH_DIR/rag_eval/round1
python rag_search.py encode --model specter2 --queries ... --out ...      # specter2 env
python rag_search.py search --model minilm   --queries ... --out ...
python rag_search.py search --model specter2 --queries ... --out ...
python rag_idmap.py

# hybrid fusion (RRF), scoring, MedCPT rerank
python rag_score.py --fuse RUN.hybrid.hits.parquet RUN.minilm.hits.parquet RUN.specter2.hits.parquet
python rag_score.py RUN.minilm.hits.parquet RUN.specter2.hits.parquet RUN.hybrid.hits.parquet
python rag_rerank.py --hits RUN.hybrid.hits.parquet

# LLM methods (vanilla, multi-hop, CRAG, deep research) on the 1,000-question sample, then all metrics
python rag_agents.py run --per-dataset 250
python rag_agents.py score
python rag_metrics_all.py offline
python rag_metrics_all.py judge
python rag_metrics_all.py offline
python rag_stats.py

# live system: scripted conversations, judge, load test, context depth, HTML report
python shakti_chat_eval.py run
python shakti_chat_eval.py judge
python shakti_chat_eval.py score
python shakti_load_test.py run
python shakti_context_depth.py run
python shakti_eval_report.py
```

Add `--mask abstract` to `rag_search.py search` for the abstract-only runs. Each script's docstring lists its options and outputs.

---

## Files

| File | Purpose |
|------|---------|
| `shakti_chat_server.py`, `shakti_chat.html` | The chat app (stages 1–6) and its web page |
| `survey_server.py`, `survey_page.html`, `survey_analyze.py` | The blinded human-evaluation survey: search engine, page and analysis |
| `rag_agents.py` | Retrieval and answering methods: vanilla, multi-hop, CRAG, deep research; prompts and schemas |
| `rag_llm.py` | Language-model backends: Claude CLI, Codex CLI, OpenAI-compatible |
| `mcp_server.py` | SHAKTI as MCP tools |
| `shakti_config.py`, `shakti.env.example` | Your settings and keys in one file |
| `full_index.py`, `specter2_index.py`, `serve_index.py`, `rag_mask.py`, `rag_idmap.py`, `flatten_openalex.py` | Corpus and index building |
| `citation_graph.py`, `corpus_stats.py`, `vec_audit.py`, `jobstatus.py` | Graph, statistics, integrity checks, job status |
| `rag_benchmarks.py`, `rag_search.py`, `rag_score.py`, `rag_rerank.py`, `rag_metrics_all.py`, `rag_stats.py` | The retrieval and answer benchmark |
| `shakti_chat_eval.py`, `shakti_chat_analyze.py`, `shakti_load_test.py`, `shakti_context_depth.py`, `shakti_eval_report.py` | Evaluation of the live system |
| `build_food_db.py` | Builds the USDA FoodData Central database |

## License

The code is released under the [MIT License](LICENSE). Data and models carry their own terms:

- OpenAlex: CC0.
- USDA FoodData Central: public domain.
- all-MiniLM-L6-v2, SPECTER2 and MedCPT: see each model card.
- Claude and OpenAI services: their providers' terms of use.

## Citation

If you use SHAKTI, please cite the paper (citation to be added on publication).
