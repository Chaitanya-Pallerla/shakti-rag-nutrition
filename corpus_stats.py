"""
corpus_stats.py: exact corpus numbers for the paper (run after the citation graph is built).

Counts rows vs DISTINCT work ids (duplicates across OpenAlex update partitions), core vs xpac
(OpenAlex's 2025 expansion pack), works with an abstract, and citation edges. Result goes to
jobs\\corpus_stats.json (shown on the dashboard) and full\\corpus_stats.json. Run with the fullindex environment.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
import time

import jobstatus

RESEARCH_DIR = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
FULL = os.path.join(RESEARCH_DIR, "full")


def main():
    import duckdb
    job = jobstatus.Job("corpus_stats", "Corpus statistics (dedup, core vs xpac)", unit="steps",
                        total=3, order=40)
    job.update("running", done=0, note="counting the embedded (titled) papers")
    tmp = os.path.join(FULL, "duckdb_tmp")
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='14GB'; SET temp_directory='{tmp}'; SET preserve_insertion_order=false")
    meta = os.path.join(FULL, "meta", "*.parquet").replace("\\", "/")
    graph = os.path.join(FULL, "graph", "*.parquet").replace("\\", "/")
    r = {}
    r["indexed_rows"], r["indexed_distinct_ids"], r["with_abstract"] = con.execute(
        f"SELECT count(*), count(DISTINCT id), count(abstract) FROM read_parquet('{meta}')").fetchone()
    job.update("running", done=1, note="counting the citation graph")
    (r["graph_rows"], r["graph_distinct_ids"], r["xpac_works"], r["core_works"], r["paratext"],
     r["retracted"], r["citation_edges"]) = con.execute(
        f"SELECT count(*), count(DISTINCT id), count(*) FILTER (is_xpac), count(*) FILTER (NOT is_xpac), "
        f"count(*) FILTER (is_paratext), count(*) FILTER (is_retracted), sum(len(refs)) "
        f"FROM read_parquet('{graph}')").fetchone()
    job.update("running", done=2, note="core vs xpac among embedded papers")
    r["indexed_core_distinct"] = con.execute(
        f"SELECT count(DISTINCT m.id) FROM read_parquet('{meta}') m "
        f"JOIN (SELECT DISTINCT id FROM read_parquet('{graph}') WHERE NOT is_xpac) g USING (id)").fetchone()[0]
    r["duplicate_rows"] = r["indexed_rows"] - r["indexed_distinct_ids"]
    r["computed"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(r, open(os.path.join(FULL, "corpus_stats.json"), "w"), indent=1)
    job.update("done", done=3, note=(f"{r['indexed_distinct_ids']:,} distinct papers indexed "
                                     f"({r['indexed_core_distinct']:,} core, rest xpac); "
                                     f"{r['duplicate_rows']:,} duplicate rows"), extra={"stats": r})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        jobstatus.Job("corpus_stats", "Corpus statistics (dedup, core vs xpac)", order=40).update(
            "failed", note=repr(e)[:200])
        raise
