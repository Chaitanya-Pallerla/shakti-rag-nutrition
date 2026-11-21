"""
shakti_eval_report.py: one self-contained HTML page with the SHAKTI pipeline, the question sources, what
was tested, every metric (with its meaning) and the results:
  * retrieval over 484M OpenAlex works (rag_eval\\retrieval_scores.json, stats.json)
  * the four query-time RAG methods on 1,000 questions (rag_eval\\agents\\scores.json)
  * the scripted live-chat evaluation: context awareness + claim-level hallucination (shakti_chat_eval\\scores.json)

  python shakti_eval_report.py   -> <SHAKTI_RESEARCH_DIR>\\shakti_chat_eval\\SHAKTI_evaluation.html
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import json
import html

R = os.environ.get("SHAKTI_RESEARCH_DIR", "shakti_research")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(R, "shakti_chat_eval", "SHAKTI_evaluation.html")


def load(*p):
    with open(os.path.join(R, *p), encoding="utf-8") as f:
        return json.load(f)


RET, AG, ST, COV, EV, EV0 = (load("rag_eval", "retrieval_scores.json"), load("rag_eval", "agents", "scores.json"),
                             load("rag_eval", "stats.json"), load("rag_eval", "coverage.json"),
                             load("shakti_chat_eval", "after", "scores.json"), load("shakti_chat_eval", "before", "scores.json"))
RUNS = {(r["conv"], r["turn"]): r for r in map(json.loads, open(os.path.join(R, "shakti_chat_eval", "after", "runs.jsonl"), encoding="utf-8"))}
E = html.escape
pct = lambda x, d=1: "–" if x is None else f"{100 * x:.{d}f}%"
f3 = lambda x: "–" if x is None else f"{x:.3f}"


def ci(m, as_pct=True):
    if not m:
        return "–"
    fmt = (lambda x: f"{100 * x:.1f}") if as_pct else f3
    return f"{fmt(m['mean'])}{'%' if as_pct else ''} <span class=ci>[{fmt(m['ci'][0])}–{fmt(m['ci'][1])}]</span>"


def bars(items, vmax=1.0, fmt=lambda v: f"{v:.3f}", color="var(--s1)", unit=""):
    """items: (label, value, tooltip). Horizontal bars, label left, value right."""
    rows = []
    for lab, v, tip in items:
        w = 0 if v is None else max(0.4, 100 * v / vmax)
        rows.append(f'<div class=brow data-tip="{E(tip)}"><span class=blab>{E(lab)}</span>'
                    f'<span class=btrack><span class=bfill style="width:{w:.1f}%;background:{color}"></span></span>'
                    f'<span class=bval>{"–" if v is None else fmt(v)}{unit}</span></div>')
    return '<div class=bars>' + "".join(rows) + "</div>"


def pair_bars(groups, names, colors, vmax=1.0, fmt=lambda v: f"{v:.3f}"):
    """groups: (label, [v1, v2]). Two series per row, with a legend."""
    leg = "".join(f'<span class=lg><i style="background:{c}"></i>{E(n)}</span>' for n, c in zip(names, colors))
    rows = []
    for lab, vals in groups:
        inner = "".join(
            f'<div class=brow data-tip="{E(lab)} · {E(n)}: {fmt(v)}"><span class=blab>{E(lab) if k == 0 else ""}</span>'
            f'<span class=btrack><span class=bfill style="width:{max(0.4, 100 * v / vmax):.1f}%;background:{c}"></span></span>'
            f'<span class=bval>{fmt(v)}</span></div>' for k, (v, n, c) in enumerate(zip(vals, names, colors)))
        rows.append(f"<div class=bgroup>{inner}</div>")
    return f'<div class=legend>{leg}</div><div class=bars>{"".join(rows)}</div>'


def tile(big, label, sub=""):
    return f'<div class=tile><div class=big>{big}</div><div class=tl>{E(label)}</div><div class=ts>{sub}</div></div>'


# ---------------------------------------------------------------- sections

ctx, hal, arms, turns = EV["context"], EV["hallucination"], EV["per_arm"], EV["turns"]
n_conv = len({t["conv"] for t in turns})

hero = f"""
<header class=hero>
  <div class=brand><div class=mark>S</div><div><h1>SHAKTI</h1>
  <div class=sub>Smart Health Advisor for Knowledge-based Tailored Interventions</div></div></div>
  <p class=lede>An evidence-grounded health and nutrition assistant that answers only from research papers it
  retrieves from <b>484 million</b> OpenAlex works and from USDA food data. This page shows how it is built, what we tested,
  which questions we used, what every metric means, and the results, including a new scripted evaluation of
  <b>context awareness</b> and <b>claim-level hallucination</b> in the live chat.</p>
</header>
<section class=tiles>
  {tile("484.4M", "OpenAlex works indexed", "exact GPU search, 2 encoders")}
  {tile("10,065", "benchmark questions", "NFCorpus · BioASQ · SciFact · PubMedQA")}
  {tile("1,000", "questions × 4 RAG methods", "250 per benchmark")}
  {tile(f"{len(turns)}", "live-chat turns, scripted", f"{n_conv} conversations, {ctx['followup_turns']} follow-ups")}
  {tile(pct(ctx['context_awareness_score']), "Context-Awareness Score", f"scored on {len(turns)} turns ({ctx['followup_turns']} follow-ups)")}
  {tile(pct(hal['hallucination_rate']['mean']), "claim hallucination rate", f"scored on {hal['claims']} claims from {hal['answers_judged']} answers")}
</section>
"""

step = lambda t, d, cls="": f'<div class="step {cls}"><b>{E(t)}</b><span>{d}</span></div>'
arrow = '<div class=arrow aria-hidden=true>→</div>'
pipeline = f"""
<section id=pipeline><h2>1 · The pipeline</h2>
<p class=note>Two parts: a one-time offline index of the scholarly record, and the online chat that answers each message.
GPU work is marked <span class=gpu>GPU</span>.</p>
<h3>Offline: index the literature</h3>
<div class=flow>
 {step("OpenAlex snapshot", "484,368,086 works (to 30 Mar 2026): title, abstract, year, DOI, venue, authors")}{arrow}
 {step("Embed", "MiniLM (384-d) and SPECTER2 (768-d), title + abstract; audited by re-embedding samples", "g")}{arrow}
 {step("Exact search", "every query against every paper, true top-100 (no approximate search)", "g")}{arrow}
 {step("Hybrid fusion", "reciprocal-rank fusion of the two rankings, k = 60")}{arrow}
 {step("MedCPT rerank", "medical cross-encoder rereads question + abstract, top-100", "g")}
</div>
<p class=note>For the live chat, the 229.8 million works with an abstract (≥100 characters) are served from a compressed
GPU index (product quantisation, 48 bytes per paper, 11 GB). Title-only records cannot be quoted as proof, and all 484M would not fit in
16 GB of GPU memory.</p>
<h3>Online: one chat turn</h3>
<div class=flow>
 {step("1 · Understand", "rewrites the message into a standalone question using the last 4 turns; detects follow-ups, intent, foods")}{arrow}
 {step("2 · Food data", "USDA FoodData Central: 13,601 foods, exact calories and nutrients per portion")}{arrow}
 {step("3 · Retrieve", "one of three methods, randomised and blinded: single-shot, Corrective RAG, deep research", "g")}{arrow}
 {step("4 · Check links", "every DOI / OpenAlex record is resolved; dead sources are removed before answering")}{arrow}
 {step("5 · Answer", "Claude Sonnet, closed-book: only the numbered sources, a citation for every claim, one verbatim quote per key claim")}{arrow}
 {step("6 · Verify", "quotes checked word-for-word, numbers traced to sources, citations matched to their papers; risk badge", "g")}{arrow}
 {step("7 · Show", "answer, collapsible sources, charts computed from the data, follow-up suggestions, 1–5 emoji rating")}
</div>
<div class=two>
 <div class=card><h4>How hallucination is avoided</h4><ul>
  <li><b>Closed sources:</b> the model sees only retrieved abstracts and USDA values and must cite them by number.</li>
  <li><b>No invented links:</b> links come from the index, not the model, and each is resolved before it is shown.</li>
  <li><b>Proof quotes:</b> every key claim carries a sentence copied from the paper, checked verbatim.</li>
  <li><b>Server-side numbers:</b> food values and every chart are computed from the data, never written by the model.</li>
  <li><b>Insufficient-evidence verdict:</b> the model is told to say so when the papers do not settle the question.</li></ul></div>
 <div class=card><h4>How context awareness works</h4><ul>
  <li>The last 4 turns are given to an understanding step that writes a <b>standalone question</b>
      (“And brown rice?” → “How many calories are in a cup of cooked brown rice?”).</li>
  <li>Retrieval, food lookup and the answer use the standalone question, so follow-ups find the right papers.</li>
  <li>A new topic is detected as <b>not a follow-up</b>, so the earlier topic is not carried over.</li>
  <li>The rewrite is shown under each answer (“Follow-up understood as …”) so users can see and correct it.</li></ul></div>
</div>
</section>
"""

ds_info = [
    ("NFCorpus", "nfcorpus", "Lay nutrition questions from NutritionFacts.org, paired with the medical articles those pages cite (graded relevance).", "none (evidence only)"),
    ("BioASQ", "bioasq", "Biomedical questions written by experts, with supporting PubMed articles and an expert ideal answer (rag-mini-bioasq).", "free text"),
    ("SciFact", "scifact", "Expert-written scientific claims with abstracts that support or refute them.", "SUPPORT / CONTRADICT / NEI"),
    ("PubMedQA", "pubmedqa", "1,000 expert-labelled yes/no/maybe research questions, each derived from one PubMed article.", "yes / no / maybe"),
]
ds_rows = "".join(
    f"<tr><td><b>{n}</b></td><td>{E(d)}</td><td class=num>{COV[k]['questions']:,}</td>"
    f"<td class=num>{COV[k]['questions_with_gold_in_index']:,}</td><td class=num>{COV[k]['gold_papers']:,}</td>"
    f"<td class=num>{COV[k]['gold_papers_in_index']:,}</td><td>{E(a)}</td></tr>" for n, k, d, a in ds_info)
T = COV["total"]
questions = f"""
<section id=questions><h2>2 · Where the 10,000 questions come from</h2>
<p>We did not write these questions. They are the union of four <b>public, expert-labelled benchmarks</b>, each listing the papers
that answer every question (the “gold” papers). They were downloaded from their public releases (Hugging Face and the SciFact release),
and every gold paper was located in our 484M-work index.</p>
<div class=scroll><table><thead><tr><th>Benchmark</th><th>What it is</th><th class=num>Questions</th><th class=num>Scored*</th>
<th class=num>Gold links</th><th class=num>Found in index</th><th>Answer type</th></tr></thead><tbody>{ds_rows}
<tr class=tot><td><b>Total</b></td><td></td><td class=num>{T['questions']:,}</td><td class=num>{T['questions_with_gold_in_index']:,}</td>
<td class=num>{T['gold_papers']:,}</td><td class=num>{T['gold_papers_in_index']:,}</td><td></td></tr></tbody></table></div>
<p class=note>* Scored = questions with at least one gold paper found in the index. Mapping: PubMed IDs → DOI and title (NCBI E-utilities)
→ OpenAlex work by DOI, else by normalised title (titles shared by more than 5 works discarded). {T['gold_papers_in_index'] / T['gold_papers']:.1%} of gold links matched.</p>
<div class=two>
 <div class=card><h4>Sample for the RAG methods</h4><p>A stratified random sample of <b>1,000</b> questions (250 per benchmark, fixed seed)
 was answered by all four query-time methods, because each answer costs 1–3 LLM calls plus judging.</p></div>
 <div class=card><h4>Scripted live-chat set (new)</h4><p><b>{n_conv} conversations × 3 turns = {len(turns)} turns</b>, written by us before any answer was seen:
 calorie and nutrient questions, evidence questions and safety questions, with {ctx['followup_turns']} follow-ups (pronouns, “and …?”, “which of the two”) and
 {ctx['topic_switch_turns']} topic switches. Each turn has a gold standalone question, a gold follow-up label and gold concept lists.</p></div>
</div>
</section>
"""

tested = """
<section id=tested><h2>3 · What we tested</h2>
<ol class=exp>
 <li><b>Retrieval at web scale</b> (9,952 questions): MiniLM vs SPECTER2 vs their hybrid fusion vs hybrid + MedCPT reranking,
   on the full corpus and on the abstract-only corpus. Scored against the benchmark gold papers.</li>
 <li><b>Query-time RAG methods</b> (1,000 questions): single-shot RAG, multi-hop (IRCoT / Self-Ask), Corrective RAG and a deep-research
   agent, all with the same retriever and the same answering model (Claude Sonnet). Scored for correctness, evidence relevance, citations and proof quotes.</li>
 <li><b>Live chat: context awareness and hallucination</b> (36 scripted turns, run before and after fixing the food data it exposed): the deployed SHAKTI server end-to-end (understanding,
   USDA lookup, retrieval, link check, answer, verification). Context is scored against gold labels; every answer is split into atomic claims
   that an independent judge (Claude Opus, a different model from the one answering) checks against exactly the abstracts and USDA values SHAKTI was given.</li>
 <li class=planned><b>Human study</b> (planned, pending ethics approval): participants chat with SHAKTI, rate each answer 1–5 and fill in a short
   questionnaire (UMUX-Lite, trust, perceived context awareness). The three methods are randomised and blinded.</li>
</ol>
</section>
"""

# ---- retrieval results
cfg = [("MiniLM", "minilm"), ("SPECTER2", "specter2"), ("Hybrid (RRF)", "hybrid"), ("Hybrid + MedCPT", "hybrid_medcpt")]
ret_rows = ""
for corp, pre in (("Full corpus", "round1"), ("Abstract-only", "round1abs")):
    for name, k in cfg:
        a = RET[f"{pre}.{k}"]
        ret_rows += (f"<tr><td>{corp if k == 'minilm' else ''}</td><td>{name}</td>" +
                     "".join(f"<td class=num>{a['all'][m]:.3f}</td>" for m in ("recall@10", "ndcg@10", "mrr@100", "hit@10", "recall@100")) +
                     "".join(f"<td class=num>{a[d]['recall@10']:.3f}</td>" for d in ("bioasq", "scifact", "pubmedqa", "nfcorpus")) + "</tr>")
ret_chart = pair_bars([(n, [RET[f"round1.{k}"]["all"]["recall@10"], RET[f"round1abs.{k}"]["all"]["recall@10"]]) for n, k in cfg],
                      ["Full corpus", "Abstract-only"], ["var(--s1)", "var(--s2)"], vmax=0.25)
mrr_chart = pair_bars([(n, [RET[f"round1.{k}"]["all"]["mrr@100"], RET[f"round1abs.{k}"]["all"]["mrr@100"]]) for n, k in cfg],
                      ["Full corpus", "Abstract-only"], ["var(--s1)", "var(--s2)"], vmax=0.25)
retrieval = f"""
<section id=retrieval><h2>5 · Results: retrieval (9,952 questions)</h2>
<div class=two>
 <figure class=card><figcaption>Recall@10, all questions</figcaption>{ret_chart}</figure>
 <figure class=card><figcaption>MRR@100, all questions</figcaption>{mrr_chart}</figure>
</div>
<p><b>Hybrid fusion beats either encoder, and MedCPT reranking adds the largest gain</b>: Recall@10 0.166 → 0.192 and MRR 0.169 → 0.205
(p &lt; 0.001). Absolute scores look low because the gold lists are incomplete at web scale: in the judged pool, 57.6% of single-shot top-10 papers
were relevant, yet only 27.3% of questions retrieved a listed gold paper. PubMedQA questions paraphrase the title of their single gold paper, which is why it scores near 0.95.</p>
<details><summary>Full retrieval table</summary><div class=scroll><table><thead><tr><th>Corpus</th><th>Method</th><th class=num>R@10</th><th class=num>nDCG@10</th>
<th class=num>MRR@100</th><th class=num>Hit@10</th><th class=num>R@100</th><th class=num>BioASQ R@10</th><th class=num>SciFact R@10</th>
<th class=num>PubMedQA R@10</th><th class=num>NFCorpus R@10</th></tr></thead><tbody>{ret_rows}</tbody></table></div></details>
</section>
"""

# ---- all metrics, four families (rag_metrics_all.py + shakti_load_test.py)
MA_P, LT_P = os.path.join(R, "rag_eval", "metrics_all.json"), os.path.join(R, "shakti_chat_eval", "load", "scores.json")
MA = load("rag_eval", "metrics_all.json") if os.path.exists(MA_P) else {}
LT = load("shakti_chat_eval", "load", "scores.json") if os.path.exists(LT_P) else {}
MM = [("Single-shot RAG", "vanilla"), ("Multi-hop RAG", "multihop"), ("Corrective RAG", "crag"), ("Deep-research agent", "deep")]
f2 = lambda x: "–" if x is None else f"{x:.2f}"
testset = lambda items: ('<div class=testset><b>Tested on</b>' + "".join(f"<span><b>{v}</b> {E(k)}</span>" for k, v in items) + "</div>")
st_mean = lambda d: None if not d else (d["mean"] if isinstance(d, dict) else d)


def cell(d, fmt=f3, with_ci=False):
    if d is None:
        return "<td class=num>–</td>"
    if isinstance(d, dict):
        v, c = d.get("mean"), d.get("ci95")
        txt = fmt(v) + (f" <span class=ci>[{fmt(c[0])}–{fmt(c[1])}]</span>" if with_ci and c else "")
        return f"<td class=num>{txt}</td>"
    return f"<td class=num>{fmt(d)}</td>"


def table(head, rows, scored):
    return (f'<p class=scored><b>Scored on:</b> {scored}</p><div class=scroll><table><thead><tr>'
            + "".join(f"<th{' class=num' if i else ''}>{E(h)}</th>" for i, h in enumerate(head))
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


allm = ""
if MA:
    RT = MA.get("retrieval", {})
    rnames = [("Full", "round1"), ("Abstract-only", "round1abs")]
    rmeth = [("MiniLM", "minilm"), ("SPECTER2", "specter2"), ("Hybrid (RRF)", "hybrid"), ("Hybrid + MedCPT rerank", "hybrid_medcpt")]
    rk = ["precision@10", "recall@10", "recall@100", "hit@10", "mrr@100", "map@100", "ndcg@10"]
    r_rows = [f"<tr><td>{c} · {n}</td>" + "".join(cell(RT[f'{p}.{k}']['all'].get(m)) for m in rk) + "</tr>"
              for c, p in rnames for n, k in rmeth if f"{p}.{k}" in RT]
    fam1 = table(["Corpus · retriever", "Precision@10", "Recall@10", "Recall@100", "Hit Rate@10", "MRR@100", "MAP@100", "nDCG@10"], r_rows,
                 f"{RT.get('round1.hybrid', {}).get('all', {}).get('questions', 9952):,} questions with a gold paper in the index (all four benchmarks)")
    GN = MA.get("generation", {})
    g_rows = []
    for n, k in MM:
        g = GN.get(k, {})
        t = g.get("text_both", {})
        g_rows.append(f"<tr><td>{n}</td>" + cell(g.get("pubmedqa", {}).get("verdict_exact_match")) + cell(g.get("pubmedqa", {}).get("verdict_macro_f1"))
                      + cell(g.get("scifact", {}).get("verdict_exact_match")) + cell(g.get("scifact", {}).get("verdict_macro_f1"))
                      + cell(AG[k]["bioasq"].get("answer_correct"))
                      + "".join(cell(t.get(m)) for m in ("exact_match", "token_f1", "rouge1", "rouge2", "rougeL"))
                      + cell(t.get("bleu_corpus")) + cell(t.get("bertscore_f1")) + "</tr>")
    nt = GN.get("vanilla", {}).get("text_both", {}).get("n")
    fam2 = table(["Method", "Verdict EM · PubMedQA", "Verdict macro-F1 · PubMedQA", "Verdict EM · SciFact", "Verdict macro-F1 · SciFact",
                  "Correct (judge) · BioASQ", "Text EM", "Token F1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore F1"], g_rows,
                 f"verdicts: 250 PubMedQA + {GN.get('vanilla', {}).get('scifact', {}).get('n', '–')} SciFact questions per method; text metrics: "
                 f"{nt or '–'} answers per method with an expert reference answer (250 BioASQ ideal answers + 250 PubMedQA long answers); "
                 "BioASQ correctness: 250 answers graded against the expert answer by Claude") if GN else "<p class=note>Generation metrics are still being computed.</p>"
    GR, GJ = MA.get("grounding", {}), MA.get("grounding_judge", {})
    gr_rows = []
    for n, k in MM:
        a = GR.get(k, {}).get("all", {})
        gr_rows.append(f"<tr><td>{n}</td>" + "".join(cell(a.get(m)) for m in ("context_precision", "context_recall", "citation_accuracy"))
                       + cell(AG[k]["all"].get("proof_valid"))
                       + "".join(cell(a.get(m), pct) for m in ("faithfulness", "hallucination_rate", "answer_relevancy"))
                       + cell(a.get("claim_citation_precision"), pct) + "</tr>")
    hl = EV["hallucination"]
    gr_rows.append("<tr class=tot><td>SHAKTI live chat</td><td class=num>–</td><td class=num>–</td><td class=num>–</td>"
                   f"<td class=num>{pct(EV['online_quote_verbatim']['mean']) if isinstance(EV.get('online_quote_verbatim'), dict) else '–'}</td>"
                   f"<td class=num>{pct(hl['faithfulness']['mean'])}</td><td class=num>{pct(hl['hallucination_rate']['mean'])}</td>"
                   f"<td class=num>{pct(EV['context']['answer_relevance_all']['mean'])}</td><td class=num>{pct(hl['citation_precision']['mean'])}</td></tr>")
    njq = GR.get("vanilla", {}).get("all", {}).get("faithfulness", {}).get("n")
    ncl = sum(GR.get(k, {}).get("all", {}).get("claims", {}).get(x, 0) for _, k in MM for x in ("supported", "partial", "unsupported", "contradicted"))
    fam3 = table(["System", "Context precision", "Context recall", "Citation accuracy", "Proof validity", "Faithfulness", "Hallucination rate",
                  "Answer relevancy", "Claim citation precision"], gr_rows,
                 "context precision, context recall, citation accuracy and proof validity: all 1,000 questions per method; faithfulness, hallucination "
                 f"and answer relevancy: {njq or '–'} answers per method ({GJ.get('per_dataset', 50)} per benchmark, the same questions for every method; "
                 f"{ncl:,} claims, Claude {E(GJ.get('model', 'opus')).capitalize()} judge); live chat: {hl['answers_judged']} answers, {hl['claims']} claims")
    EF = MA.get("efficiency", {})
    e_rows = []
    for n, k in MM:
        a = EF.get(k, {}).get("all", {})
        g = lambda key: (a.get(key) or {}).get("mean")
        e_rows.append(f"<tr><td>{n}</td><td class=num>{f2(g('claude_calls'))}</td><td class=num>{g('input_tokens') or 0:,.0f}</td>"
                      f"<td class=num>{g('output_tokens') or 0:,.0f}</td><td class=num>{g('total_tokens') or 0:,.0f}</td>"
                      f"<td class=num>${g('cost_list_usd') or 0:.4f}</td><td class=num>${a.get('cost_per_1000_questions_usd', 0):,.2f}</td>"
                      f"<td class=num>{g('claude_secs') or 0:.1f} s</td><td class=num>{f2(g('follow_up_searches'))}</td></tr>")
    rb = EF.get("retrieval_batch", {}).get("round1abs", {})
    fam4 = table(["Method", "Claude calls", "Input tokens", "Output tokens", "Total tokens", "Cost / question", "Cost / 1,000 questions",
                  "Claude time / question", "Extra searches"], e_rows,
                 f"{EF.get('vanilla', {}).get('all', {}).get('claude_calls', {}).get('n', '–')} questions per method, every Claude call replayed from the call cache "
                 "(each stores its own token usage); cost at Anthropic list prices (the study runs on a subscription)")
    if rb:
        fam4 += (f"<p class=note>Retrieval (batch, exact search of all 484M papers with both encoders on the GPU): {rb['queries']:,} questions in "
                 f"{rb['wall_secs'] / 60:.0f} min = <b>{rb['secs_per_query']:.2f} s per question</b>, <b>{rb['queries_per_min']:.0f} questions/min</b>. "
                 "A follow-up search pass costs the same ~85 min whether it holds 2 or 4,527 queries, so the methods' follow-up searches are batched.</p>")
    lt_html = ""
    if LT:
        PA = LT["per_answer"]
        l_rows = [f"<tr><td>{l['users']}</td><td class=num>{l['answered']}/{l['questions']}</td><td class=num>{l['throughput_q_per_min']:.1f}</td>"
                  f"<td class=num>{l['latency_p50']:.0f} s</td><td class=num>{l['latency_p95']:.0f} s</td><td class=num>{l['server_processing_p50']:.0f} s</td>"
                  f"<td class=num>{l['queue_wait_p50']:.0f} s</td></tr>" for l in LT["levels"]]
        arm_name = {"single": "Single-shot", "crag": "Corrective RAG", "deep": "Deep research", "food": "Food data only"}
        a_rows = [f"<tr><td>{arm_name.get(a, a)}</td><td class=num>{v['answers']}</td><td class=num>{v['total_tokens_mean']:,}</td>"
                  f"<td class=num>${v['cost_list_usd_mean']:.4f}</td><td class=num>{v['latency_p50']:.0f} s</td></tr>"
                  for a, v in sorted(LT["per_arm"].items(), key=lambda kv: kv[1]["total_tokens_mean"])]
        n_lt = sum(l["questions"] for l in LT["levels"])
        stg = "; ".join(f"{E(k.lower())} {v:.1f} s" for k, v in LT["stage_secs_p50"].items() if v)
        lt_html = f"""
<h3>Live SHAKTI server: latency, throughput, tokens and cost</h3>
{testset([("questions", n_lt), ("concurrency levels", len(LT['levels'])), ("errors", sum(l['errors'] for l in LT['levels']))])}
<div class=two>
 <div>{table(["Concurrent users", "Answered", "Throughput (q/min)", "Latency p50", "Latency p95", "Processing p50", "Queue wait p50"], l_rows,
             f"{n_lt} fresh questions from virtual users at 1, 2, 4 and 8 concurrent users; latency is what the user waits, from sending to seeing the answer")}</div>
 <div>{table(["Answer path", "Answers", "Tokens / answer", "Cost / answer", "Latency p50"], a_rows,
             f"the same {PA['answers']} answers, from the server's per-turn Claude usage log")}</div>
</div>
<div class=tiles>
 {tile(f"{PA['total_tokens_mean']:,}", "tokens per answer (mean)", f"{PA['input_tokens_mean']:,} in · {PA['output_tokens_mean']:,} out · {PA['claude_calls_mean']:.1f} Claude calls")}
 {tile(f"${PA['cost_list_usd_mean']:.3f}", "cost per answer (list price)", f"${PA['cost_per_1000_answers_usd']:.0f} per 1,000 answers")}
 {tile(f"{max(l['throughput_q_per_min'] for l in LT['levels']):.1f}/min", "peak throughput", "2 answers in parallel by design")}
 {tile(f"{LT['levels'][0]['latency_p50']:.0f} s", "latency p50, 1 user", f"p95 {LT['levels'][0]['latency_p95']:.0f} s")}
</div>
<p class=note>Median time per stage: {stg}. Throughput levels off at about 5 questions per minute because the server answers at most 2 questions at a time
(a guard on the Claude subscription); beyond 2 users the extra time is queueing, not slower answers.</p>"""
    check = [("Retrieval", ["Precision", "Recall", "Hit Rate", "MRR", "MAP", "nDCG"]),
             ("Generation", ["Exact Match", "F1", "ROUGE", "BLEU", "BERTScore"]),
             ("Grounding & factuality", ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Citation Accuracy"]),
             ("Efficiency", ["Latency", "Tokens", "Cost", "Throughput"])]
    chk = "".join(f'<div class=card><h4>{i + 1} · {E(h)}</h4><div class=chips>' + "".join(f"<span class=chip>✓ {E(x)}</span>" for x in xs)
                  + "</div></div>" for i, (h, xs) in enumerate(check))
    allm = f"""
<section id=allmetrics><h2>4 · All metrics, four families</h2>
<p>The standard RAG metric set, every one measured. Families 1–3 compare the retrievers and the four RAG methods on the benchmark questions (the
expert answers and gold papers make reference-based metrics possible); grounding and efficiency are also measured on the live SHAKTI chat.</p>
<div class=grid4>{chk}</div>
<h3>1 · Retrieval: did the system find the right papers?</h3>{fam1}
<p class=note>Precision@10 is low by construction: most questions have only 1–3 listed gold papers, so even a perfect top 10 scores at most 0.1–0.3.
The judged precision of the papers the methods actually used is much higher (context precision below).</p>
<h3>2 · Generation: did the model give the correct answer?</h3>{fam2}
<p class=note>Verdict EM is exact agreement with the expert yes/no/maybe or SUPPORT/CONTRADICT/NOT ENOUGH INFO label. Text Exact Match is near zero by nature:
the answers are explanations, not copies of the reference text; token F1, ROUGE and BLEU measure word overlap, BERTScore (roberta-large, baseline-rescaled,
so 0 ≈ unrelated text) measures meaning overlap, and the Claude grade judges correctness of the key facts.</p>
<h3>3 · Grounding and factuality: is the answer supported by the evidence?</h3>{fam3}
<h3>4 · Efficiency: how fast and how expensive?</h3>{fam4}{lt_html}
</section>
"""

# ---- agent results
M = [("Single-shot RAG", "vanilla"), ("Multi-hop RAG", "multihop"), ("Corrective RAG", "crag"), ("Deep-research agent", "deep")]
small = lambda key, title, sub: (f'<figure class="card mini"><figcaption>{title}<small>{sub}</small></figcaption>' +
                                 bars([(n, AG[k]["all"].get(key) if key != "answer_correct" else AG[k]["bioasq"][key],
                                        f"{n}: {(AG[k]['all'].get(key) if key != 'answer_correct' else AG[k]['bioasq'][key]):.3f}") for n, k in M]) + "</figure>")
ag_rows = "".join(
    f"<tr><td>{n}</td><td class=num>{AG[k]['all']['verdict_acc']:.3f}</td><td class=num>{AG[k]['bioasq']['answer_correct']:.3f}</td>"
    f"<td class=num>{AG[k]['all']['judged_precision']:.3f}</td><td class=num>{AG[k]['all']['has_direct_evidence']:.3f}</td>"
    f"<td class=num>{AG[k]['all']['citation_precision']:.3f}</td><td class=num>{AG[k]['all']['proof_valid']:.3f}</td>"
    f"<td class=num>{AG[k]['all']['gold_hit']:.3f}</td><td class=num>{AG[k]['all']['calls']:.1f}</td></tr>" for n, k in M)
agents = f"""
<section id=agents><h2>6 · Results: four RAG methods (1,000 questions)</h2>
<div class=grid4>
 {small("answer_correct", "BioASQ answer correct", "vs expert ideal answer, n = 250")}
 {small("judged_precision", "Judged evidence precision", "share of evidence papers relevant")}
 {small("citation_precision", "Citation precision", "cited papers that are direct evidence")}
 {small("proof_valid", "Proof validity", "quotes verbatim in a relevant cited paper")}
</div>
<p><b>Every multi-step method improved every evidence metric</b> (p &lt; 0.001 against single-shot). The deep-research agent gave the most correct
BioASQ answers (0.732 vs 0.592) and the most verifiable evidence (97.3% valid proofs); Corrective RAG had the cleanest evidence (93.2% relevant).
Yes/no/maybe verdict accuracy changed little (0.592–0.620), so verdicts are limited by reasoning rather than retrieval. The live chat uses
single-shot, Corrective RAG and deep research.</p>
<details><summary>Full method table</summary><div class=scroll><table><thead><tr><th>Method</th><th class=num>Verdict acc.</th><th class=num>BioASQ correct</th>
<th class=num>Judged prec.</th><th class=num>Direct evidence</th><th class=num>Citation prec.</th><th class=num>Proof valid</th><th class=num>Gold hit</th>
<th class=num>LLM calls</th></tr></thead><tbody>{ag_rows}</tbody></table></div></details>
</section>
"""

# ---- live chat: context
comp = [("Follow-up detection accuracy", ctx["followup_detection_accuracy"]["mean"], "is_followup matches the gold label, all turns"),
        ("Concept recall of rewrite", ctx["concept_recall_followups"]["mean"], "gold concepts present in the standalone question, follow-ups"),
        ("Rewrite fidelity (judge)", ctx["rewrite_fidelity_followups"]["mean"], "rewrite has the meaning of the gold question, 0–2 scaled"),
        ("Answer relevance (judge)", ctx["answer_relevance_followups"]["mean"], "answer addresses what the user meant, 0–2 scaled"),
        ("No topic leakage", 1 - ctx["topic_leak_rate"]["mean"], "topic switches that did not carry the old topic over")]
abl = pair_bars([("Concept recall, follow-ups", [ctx["concept_recall_followups_raw_message"]["mean"], ctx["concept_recall_followups"]["mean"]])],
                ["Raw message (no context)", "SHAKTI rewrite (with context)"], ["var(--s2)", "var(--s1)"], fmt=lambda v: f"{100 * v:.0f}%")
testset = lambda items: ('<div class=testset><b>Tested on</b>' + "".join(f"<span><b>{v}</b> {E(k)}</span>" for k, v in items) + "</div>")
DP = os.path.join(R, "shakti_chat_eval", "depth", "scores.json")
depth_html = ""
if os.path.exists(DP):
    D = load("shakti_chat_eval", "depth", "scores.json")
    DA_P = os.path.join(R, "shakti_chat_eval", "depth_after", "scores.json")
    DA = load("shakti_chat_eval", "depth_after", "scores.json") if os.path.exists(DA_P) else None
    sets = [("Before fix", D)] + ([("After fix", DA)] if DA else [])
    count = lambda S: (sum(v["conversations"] for v in S["depths"].values()),
                       sum((int(k) + 2) * v["conversations"] for k, v in S["depths"].items()))
    n_dconv, n_dq = (sum(count(S)[i] for _, S in sets) for i in (0, 1))
    all_n = sorted({int(k) for _, S in sets for k in S["depths"]})

    def cells(S):
        out = ""
        for n in all_n:
            v = S["depths"].get(str(n))
            if v is None:
                out += f'<div class="dcell na" data-tip="{n} in between: not run before the fix"><b>{n}</b><span>–</span></div>'
                continue
            r, c = v["rewrite_remembers"], v["conversations"]
            out += (f'<div class="dcell {"ok" if r == c else "no"}" data-tip="{n} other questions in between: {r} of {c} remembered">'
                    f'<b>{n}</b><span>{r}/{c}</span></div>')
        return out
    drow_sets = "".join(f'<div class=depthrow><span class=dlab>{E(lab)}</span>{cells(S)}</div>' for lab, S in sets)
    tot = lambda S: (sum(v["rewrite_remembers"] for v in S["depths"].values()), sum(v["conversations"] for v in S["depths"].values()))
    after_turn = {(t["anchor"], t["n_between"]): t for t in DA["turns"]} if DA else {}
    topic = lambda a: "brown rice calories" if a == "rice" else "intermittent fasting and type 2 diabetes"
    mark = lambda t: "" if t is None else ('<span class=ok>✓</span>' if t["rewrite_remembers"] else '<span class=no>✗</span>')
    keys = sorted({(t["anchor"], t["n_between"]) for _, S in sets for t in S["turns"]}, key=lambda k: (k[1], k[0]))
    before_turn = {(t["anchor"], t["n_between"]): t for t in D["turns"]}
    drows = ""
    for k in keys:
        b, a = before_turn.get(k), after_turn.get(k)
        drows += (f"<tr><td class=num>{k[1]}</td><td>{E(topic(k[0]))}</td><td>{E(b['standalone']) if b else '–'}</td><td class=c>{mark(b)}</td>"
                  + (f"<td>{E(a['standalone']) if a else '–'}</td><td class=c>{mark(a)}</td>" if DA else "") + "</tr>")
    dhead = ("<th>Before fix: SHAKTI's rewrite of the call-back</th><th>OK</th>"
             + ("<th>After fix: SHAKTI's rewrite</th><th>OK</th>" if DA else ""))
    if DA:
        ta, tb = tot(DA), tot(D)
        deepest = max(int(k) for k in DA["depths"])
        verdict = (f"remembers at every depth tested (up to {deepest} questions back)" if ta[0] == ta[1]
                   else f"remembers up to {DA['max_between_all_remembered']} questions back")
        cards = f"""
 <div class=card><h4>Before: remembered up to {D['max_between_all_remembered']} questions back ({tb[0]}/{tb[1]})</h4><p>From
 {D['first_failure']} unrelated questions onwards no call-back was resolved: SHAKTI took “my first question” to mean the oldest question it
 could still see (for example “How much fiber is in a boiled egg?”), because the understanding step read only the last 4 turns.</p></div>
 <div class=card><h4>After: {verdict} ({ta[0]}/{ta[1]})</h4><p>The understanding step now also gets a numbered list of every earlier question
 in the conversation (its standalone form, answers left out), followed by the last 4 turns in full. A conversation is capped at 20
 questions, so the list stays short. The test was rerun unchanged, plus a deeper case of 16 questions in between.</p></div>"""
    else:
        cards = f"""
 <div class=card><h4>Result: remembers up to {D['max_between_all_remembered']} questions back</h4><p>From {D['first_failure']} onwards
 SHAKTI took “my first question” to mean the oldest question it could still see, because the understanding step reads only the last 4 turns.</p></div>
 <div class=card><h4>Fix for longer memory</h4><p>Add a short list of every earlier question to the last 4 turns, then rerun this test.</p></div>"""
    depth_html = f"""
<h3 id=depth>How far back does SHAKTI remember?</h3>
{testset([("conversations", n_dconv), ("questions asked", n_dq), ("call-backs scored", n_dconv), ("depths tested", len(all_n))])}
<p>Each conversation asks a first question, then N questions on unrelated topics (eggs, green tea, kiwi, creatine, cola, …), then refers
back: “Going back to my very first question: how much fiber does that food have?”. It counts as remembered when SHAKTI's rewrite names the
first topic (brown rice + fiber, or intermittent fasting + weight loss). Two first questions per depth{", run before and after the fix" if DA else ""}.
Each cell shows N, the number of unrelated questions in between, and how many of the call-backs were remembered.</p>
{drow_sets}
<div class=two>{cards}
</div>
<details><summary>All {n_dconv} call-backs and how SHAKTI rewrote them</summary><div class=scroll><table class=small>
<thead><tr><th class=num>In between</th><th>First question about</th>{dhead}</tr></thead>
<tbody>{drows}</tbody></table></div></details>
"""

n_first = sum(1 for t in turns if t["turn"] == 0)
n_new_mid = len(turns) - ctx["followup_turns"] - n_first
context = f"""
<section id=context><h2>7 · Results: context awareness (live chat, {len(turns)} turns)</h2>
{testset([("conversations", n_conv), ("questions (turns) in total", len(turns)), ("follow-up questions", ctx['followup_turns']),
          ("first questions", n_first), ("topic switches", ctx['topic_switch_turns'])])}
<p class=note>The {len(turns)} questions are the scripted set in section 2: {n_conv} conversations of 3 turns. {ctx['followup_turns']} turns depend on an earlier
turn (“And brown rice?”), {n_first} open a conversation and {n_new_mid} start a new topic mid-conversation ({ctx['topic_switch_turns']} of them checked for leakage).
Each score below states which of these turns it uses.</p>
<section class=tiles>
 {tile(pct(ctx['context_awareness_score']), "Context-Awareness Score", "mean of the 5 components below")}
 {tile(f"{ctx['followup_f1']:.2f}", "follow-up detection F1", f"precision {ctx['followup_precision']:.2f} · recall {ctx['followup_recall']:.2f}")}
 {tile(pct(ctx['rewrite_exact_followups']['mean'], 0), "follow-ups rewritten exactly right", "judge score 2 of 2")}
 {tile(pct(ctx['topic_leak_rate']['mean'], 0), "topic leakage", f"{ctx['topic_switch_turns']} topic switches")}
</section>
<div class=two>
 <figure class=card><figcaption>The five components (0–100%)</figcaption>{bars([(a, b, c) for a, b, c in comp], fmt=lambda v: f"{100 * v:.0f}%")}</figure>
 <figure class=card><figcaption>Ablation: what the rewrite adds<small>share of the gold concepts a search would see</small></figcaption>{abl}
 <p class=note>Without context, a follow-up such as “Which of the two has more fiber?” does not name the foods it is about (raw messages carry {pct(ctx['concept_recall_followups_raw_message']['mean'], 0)} of the gold concepts); the rewritten
 question names {pct(ctx['concept_recall_followups']['mean'], 0)} of the gold concepts. This is the information retrieval and the answer actually use.</p></figure>
</div>
<table class=kv><thead><tr><th>Score</th><th>Result [95% CI]</th><th class=num>Scored on</th></tr></thead><tbody>
<tr><td><b>Context-Awareness Score</b> (mean of the 5 rows below)</td><td><b>{pct(ctx['context_awareness_score'])}</b></td><td class=num>{len(turns)} turns</td></tr>
<tr><td>Follow-up detection accuracy</td><td>{ci(ctx['followup_detection_accuracy'])}</td><td class=num>all {ctx['followup_detection_accuracy']['n']} turns</td></tr>
<tr><td>Concept recall, follow-ups (raw message → rewrite)</td><td>{ci(ctx['concept_recall_followups_raw_message'])} → {ci(ctx['concept_recall_followups'])}</td><td class=num>{ctx['concept_recall_followups']['n']} follow-ups</td></tr>
<tr><td>Rewrite fidelity (judge)</td><td>{ci(ctx['rewrite_fidelity_followups'])}</td><td class=num>{ctx['rewrite_fidelity_followups']['n']} follow-ups</td></tr>
<tr><td>Answer relevance (judge)</td><td>{ci(ctx['answer_relevance_followups'])}</td><td class=num>{ctx['answer_relevance_followups']['n']} follow-ups</td></tr>
<tr><td>Answer relevance, every turn (not part of the score)</td><td>{ci(ctx['answer_relevance_all'])}</td><td class=num>all {ctx['answer_relevance_all']['n']} turns</td></tr>
<tr><td>Topic leakage on topic switches</td><td>{ci(ctx['topic_leak_rate'])}</td><td class=num>{ctx['topic_leak_rate']['n']} topic switches</td></tr>
</tbody></table>
<p class=note>Brackets: 95% bootstrap confidence intervals, resampling whole conversations (5,000 resamples).</p>
{depth_html}
</section>
"""

# ---- live chat: hallucination
L = hal["labels"]
spec = sum(t["unsupported_specific"] + t["contradicted"] for t in turns)
vsum = lambda k: sum(t["verification"].get(k) or 0 for t in turns)
n_quotes, n_nums, n_refs = vsum("quotes_total"), vsum("numbers_total"), vsum("refs_total")
n_cited = sum(t["cited_claims"] for t in turns)
low_spec = sum(1 for t in turns if t["risk"] == "low" and t["unsupported_specific"] + t["contradicted"])
tot = sum(L.values())
seg_c = {"supported": "var(--good)", "partial": "var(--warn)", "unsupported": "var(--serious)", "contradicted": "var(--crit)"}
stack = "".join(f'<span class=seg data-tip="{k}: {v} claims ({100 * v / tot:.1f}%)" style="width:{100 * v / tot:.2f}%;background:{seg_c[k]}"></span>'
                for k, v in L.items() if v)
stack_leg = "".join(f'<span class=lg><i style="background:{seg_c[k]}"></i>{k} <b>{v}</b> ({100 * v / tot:.1f}%)</span>' for k, v in L.items())
fj = hal["online_flag_vs_judge"]
flag_rows = "".join(f"<tr><td>{k}</td><td class=num>{fj[k]['answers']}</td><td class=num>{fj[k]['judge_clean']}</td></tr>" for k in ("low", "medium", "high"))
arm_name = {"single": "Single-shot RAG", "crag": "Corrective RAG", "deep": "Deep research", "food": "Food data only", "none": "No retrieval"}
arm_rows = "".join(f"<tr><td>{arm_name.get(a, a)}</td><td class=num>{v['answers']}</td><td class=num>{v['claims']}</td>"
                   f"<td class=num>{pct(v['faithfulness'])}</td><td class=num>{pct(v['hallucination_rate'])}</td><td class=num>{v['median_secs']:.0f}s</td></tr>"
                   for a, v in arms.items())
hallu = f"""
<section id=hallucination><h2>8 · Results: hallucination (live chat, {hal['answers_judged']} answers, {hal['claims']} claims)</h2>
{testset([("questions answered", len(turns)), ("answers judged", hal['answers_judged']), ("atomic claims checked", hal['claims']),
          ("quotes checked", n_quotes), ("numbers checked", n_nums), ("source links checked", hal['links_shown'])])}
<p class=note>The same {len(turns)} questions as section 7. The judge split the {hal['answers_judged']} answers into {hal['claims']} single, checkable claims
(about {hal['claims'] / hal['answers_judged']:.0f} per answer) and checked each one against the sources SHAKTI was given. Claim-level rates are
out of {hal['claims']} claims; answer-level rates are out of {hal['answers_judged']} answers.</p>
<section class=tiles>
 {tile(pct(hal['faithfulness']['mean']), "claim faithfulness", "supported + ½ partial")}
 {tile(pct(hal['hallucination_rate']['mean']), "hallucination rate", "unsupported + contradicted claims")}
 {tile(pct(hal['contradiction_rate']['mean']), "contradicted claims", "a source says otherwise")}
 {tile(pct(hal['answers_hallucination_free']['mean'], 0), "answers with zero hallucinated claims", "")}
 {tile(pct(hal['citation_precision']['mean']), "citation precision", "cited claims backed by their citation")}
 {tile(f"{hal['links_ok']}/{hal['links_shown']}", "source links working", f"{hal['dead_sources_removed']} dead sources removed first")}
</section>
<figure class=card><figcaption>Every atomic claim, labelled by the judge against the sources SHAKTI was given</figcaption>
<div class=stack>{stack}</div><div class=legend>{stack_leg}</div></figure>
<table class=kv><thead><tr><th>Score</th><th>Result [95% CI]</th><th class=num>Scored on</th></tr></thead><tbody>
<tr><td>Claim faithfulness</td><td>{ci(hal['faithfulness'])}</td><td class=num>{hal['claims']} claims</td></tr>
<tr><td>Fully supported claims</td><td>{ci(hal['supported_rate'])}</td><td class=num>{hal['claims']} claims</td></tr>
<tr><td><b>Hallucination rate</b> (unsupported + contradicted)</td><td>{ci(hal['hallucination_rate'])}</td><td class=num>{hal['claims']} claims</td></tr>
<tr><td>… excluding general background knowledge</td><td>{spec} of {hal['claims']} claims{f" (95% upper bound {pct(3 / hal['claims'])}, rule of three)" if spec == 0 else " " + ci(hal['hallucination_rate_specific'])}</td><td class=num>{hal['claims']} claims</td></tr>
<tr><td>Answers with at least one hallucinated claim</td><td>{ci(hal['answers_with_any_hallucination'])}</td><td class=num>{hal['answers_judged']} answers</td></tr>
<tr><td>Citation precision (claim level)</td><td>{ci(hal['citation_precision'])}</td><td class=num>{n_cited} cited claims</td></tr>
<tr><td>Citation coverage (claims carrying a citation)</td><td>{ci(hal['citation_coverage'])}</td><td class=num>{hal['claims']} claims</td></tr>
<tr><td>Online check: quotes verbatim in the paper</td><td>{ci(hal['online_quote_verbatim'])}</td><td class=num>{n_quotes} quotes in {hal['online_quote_verbatim']['n']} answers</td></tr>
<tr><td>Online check: citations pointing to no source</td><td>{ci(hal['online_invalid_citations'])}</td><td class=num>{n_refs} citations in {hal['online_invalid_citations']['n']} answers</td></tr>
<tr><td>Online check: numbers traced to a source</td><td>{ci(hal['online_number_grounding'])}</td><td class=num>{n_nums} numbers in {hal['online_number_grounding']['n']} answers</td></tr>
</tbody></table>
<div class=two>
 <div class=card><h4>Does the live risk badge agree with the judge?</h4>
 <table><thead><tr><th>Badge shown to user</th><th class=num>Answers</th><th class=num>Judge: no hallucinated claim</th></tr></thead><tbody>{flag_rows}</tbody></table>
 <p class=note>The badge is computed instantly from mechanical checks; the judge reads every claim. {low_spec} low-badge answer{"" if low_spec == 1 else "s"} had a specific (non-background) claim the judge could not find in the sources.</p></div>
 <div class=card><h4>By method (descriptive; small n)</h4>
 <table><thead><tr><th>Method</th><th class=num>Answers</th><th class=num>Claims</th><th class=num>Faithful</th><th class=num>Halluc.</th><th class=num>Median time</th></tr></thead>
 <tbody>{arm_rows}</tbody></table></div>
</div>
</section>
"""

c0, h0, T0 = EV0["context"], EV0["hallucination"], {(t["conv"], t["turn"]): t for t in EV0["turns"]}
T1 = {(t["conv"], t["turn"]): t for t in turns}
food_keys = [k for k in T1 if T1[k]["arm"] == "food" or (k in T0 and T0[k]["arm"] == "food")]
rel = lambda T, ks: sum(T[k]["relevance"] for k in ks if T[k]["relevance"] is not None) / (2 * len(ks))
spec0 = sum(t["unsupported_specific"] + t["contradicted"] for t in EV0["turns"])
cmp_rows = [
    ("Context-Awareness Score", c0["context_awareness_score"], ctx["context_awareness_score"], True),
    ("Answer relevance, all 36 turns", c0["answer_relevance_all"]["mean"], ctx["answer_relevance_all"]["mean"], True),
    (f"Answer relevance, {len(food_keys)} food-data turns", rel(T0, food_keys), rel(T1, food_keys), True),
    ("Claim faithfulness", h0["faithfulness"]["mean"], hal["faithfulness"]["mean"], True),
    ("Hallucination rate (all claims)", h0["hallucination_rate"]["mean"], hal["hallucination_rate"]["mean"], False),
    ("Answers with zero hallucinated claims", h0["answers_hallucination_free"]["mean"], hal["answers_hallucination_free"]["mean"], True),
    ("Citation precision (claim level)", h0["citation_precision"]["mean"], hal["citation_precision"]["mean"], True),
]


def delta(a, b, up):
    d = 100 * (b - a)
    if abs(d) < 0.05:
        return "<span class=ci>no change</span>"
    good = (d > 0) == up
    return f'<span class={"ok" if good else "no"}>{"+" if d > 0 else "−"}{abs(d):.1f} pts</span>'


cmp_table = "".join(f"<tr><td>{E(n)}</td><td class=num>{pct(a)}</td><td class=num>{pct(b)}</td><td class=num>{delta(a, b, up)}</td></tr>"
                    for n, a, b, up in cmp_rows)
cmp_table += (f"<tr><td>Specific claims unsupported or contradicted</td><td class=num>{spec0} of {h0['claims']}</td>"
              f"<td class=num>{spec} of {hal['claims']}</td><td></td></tr>"
              f"<tr><td>Median answer time</td><td class=num>{EV0['latency']['median_secs']:.0f} s</td>"
              f"<td class=num>{EV['latency']['median_secs']:.0f} s</td><td></td></tr>")
cmp_chart = pair_bars([(n, [a, b]) for n, a, b, up in cmp_rows if up][:4], ["Before", "After"], ["var(--s2)", "var(--s1)"],
                      fmt=lambda v: f"{100 * v:.0f}%")
was_partial = [k for k in sorted(T0, key=lambda k: (list(dict.fromkeys(t['conv'] for t in EV0['turns'])).index(k[0]), k[1]))
               if T0[k]["relevance"] is not None and T0[k]["relevance"] < 2 and k in T1]
fix_rows = "".join(
    f"<tr><td>{E(T1[k]['question'])}</td><td class=num>{T0[k]['relevance']}/2</td><td class=num>{T1[k]['relevance']}/2</td>"
    f"<td>{E(RUNS[k]['result']['answer'][:260])}{'…' if len(RUNS[k]['result']['answer']) > 260 else ''}</td></tr>" for k in was_partial)

found = f"""
<section id=found><h2>9 · Before vs after: fixing what the test found</h2>
<p>The first run exposed two gaps in the food data (not hallucinations: SHAKTI said honestly that it lacked the data). We fixed them and
asked the same 36 messages again, scored by the same judge with the same rules.</p>
<div class="two wide">
 <div class=card><h4>Fix 1 · Micronutrients</h4><p><b>Before:</b> the food table held calories, macronutrients, fiber, sugars, sodium and saturated fat,
 so “How much iron is in spinach?” got “I can't give you an iron number”. <b>After:</b> iron, vitamin C, calcium, potassium, magnesium, zinc,
 vitamin D, B12, folate, vitamin A, cholesterol and caffeine are loaded from the same USDA database, and the asked-about nutrient gets its own
 column in the food table.</p></div>
 <div class=card><h4>Fix 2 · Typical portions</h4><p><b>Before:</b> the first listed portion was used: 1 cup of peanut butter (1,543 kcal),
 “snack-size” toast, a 16 fl oz cola, so SHAKTI declined to total the breakfast. <b>After:</b> typical household portions are preferred
 (tablespoons, a regular slice, a 12 fl oz can) and SHAKTI states which portion it assumed and adds up the meal (402 kcal, assuming
 1 tbsp of peanut butter). Of the 7 answers below, the 3 still partly relevant are evidence gaps: the retrieved papers do not name side effects,
 best strains or teen doses, and SHAKTI says so rather than guessing.</p></div>
</div>
<div class=two>
 <figure class=card><figcaption>Before vs after (higher is better)</figcaption>{cmp_chart}</figure>
 <div class=card><table><thead><tr><th>Metric</th><th class=num>Before</th><th class=num>After</th><th class=num>Change</th></tr></thead>
 <tbody>{cmp_table}</tbody></table>
 <p class=note>Same messages and judge. Answers are generated afresh, and each conversation's method is re-randomised, so small changes
 outside the food turns are run-to-run variation.</p></div>
</div>
<details open><summary>The {len(was_partial)} answers that were only partly relevant before</summary><div class=scroll><table class=small>
<thead><tr><th>User message</th><th class=num>Before</th><th class=num>After</th><th>Answer after the fix (start)</th></tr></thead>
<tbody>{fix_rows}</tbody></table></div></details>
<div class="two wide">
 <div class=card><h4>Ambiguous references</h4><p>“Is eating <i>them</i> daily bad for cholesterol?” after eggs and chicken was read as both foods
 (the reference meant eggs); “Is olive oil the main reason?” brought back heart disease as well as dementia. Both are defensible readings,
 and the rewrite is shown to the user, who can correct it.</p></div>
 <div class=card><h4>Evaluation set-up matters</h4><p>A first judging pass gave the judge different abstract text from what SHAKTI saw and
 no conversation history, which inflated “unsupported” claims (5.2% hallucination rate, including 3 “contradicted” claims that were
 artefacts of the different text). The final judge sees exactly the same index records (same 1,500-character cut) and SHAKTI's earlier
 answers, as a fair test requires. Both passes are kept for transparency.</p></div>
</div>
</section>
"""

# ---- per-turn detail
def turn_row(t):
    bad = [c for c in t["judge_claims"] if c["label"] in ("unsupported", "contradicted")]
    bad_txt = "".join(f"<li><b>{E(c['label'])}</b>: {E(c['claim'])}{(' <i>(' + E(c.get('note') or '') + ')</i>') if c.get('note') else ''}</li>" for c in bad)
    ok = lambda b: '<span class=ok>✓</span>' if b else '<span class=no>✗</span>'
    return (f"<tr><td>{E(t['conv'])} {t['turn'] + 1}</td><td>{E(t['question'])}</td><td>{E(t['standalone'])}</td>"
            f"<td class=c>{ok(t['followup'] == t['gold_followup'])}</td><td class=num>{100 * t['concept_recall']:.0f}%</td>"
            f"<td class=num>{t['rewrite_score'] if t['rewrite_score'] is not None else '–'}/2</td><td>{arm_name.get(t['arm'], t['arm'])}</td>"
            f"<td class=num>{t['supported']}/{t['claims']}</td><td>{t['risk']}</td><td>{('<ul>' + bad_txt + '</ul>') if bad else '–'}</td></tr>")


detail = f"""
<section id=turns><h2>10 · Every turn of the live-chat test</h2>
<details><summary>Show all {len(turns)} turns (question, rewrite, scores, flagged claims)</summary><div class=scroll><table class=small>
<thead><tr><th>Turn</th><th>User message</th><th>SHAKTI's standalone question</th><th>Follow-up label</th><th class=num>Concepts</th>
<th class=num>Rewrite</th><th>Method</th><th class=num>Supported claims</th><th>Badge</th><th>Unsupported / contradicted claims</th></tr></thead>
<tbody>{"".join(turn_row(t) for t in turns)}</tbody></table></div></details>
</section>
"""

G = [
    ("Retrieval (vs gold papers)", [
        ("Recall@10", "Share of a question's gold papers that appear in the top 10 results."),
        ("Hit@10", "Whether at least one gold paper is in the top 10."),
        ("MRR@100", "1 / rank of the first gold paper (0 if none in the top 100); rewards putting it first."),
        ("nDCG@10", "Rank-weighted gain of the top 10, with graded relevance for NFCorpus."),
        ("Recall@100", "Share of gold papers in the top 100: what a reranker can work with."),
        ("Precision@10", "Share of the top 10 results that are gold papers (capped by how few gold papers a question has)."),
        ("MAP@100", "Mean average precision: precision at each rank where a gold paper appears, averaged over the question's gold papers.")]),
    ("Generation (vs expert answers)", [
        ("Verdict Exact Match / macro-F1", "The yes/no/maybe or SUPPORT/CONTRADICT/NOT ENOUGH INFO verdict equals the expert label; macro-F1 averages F1 over the classes."),
        ("Text Exact Match", "The normalised answer text equals the reference text (SQuAD normalisation)."),
        ("Token F1", "Word-overlap F1 between answer and reference (SQuAD)."),
        ("ROUGE-1 / 2 / L", "Overlap of words, word pairs and the longest common word sequence with the reference (F-measure, stemmed)."),
        ("BLEU", "n-gram precision against the reference with a brevity penalty (sacreBLEU, corpus level, 0–1)."),
        ("BERTScore F1", "Similarity of meaning between answer and reference from roberta-large token embeddings, rescaled so unrelated text ≈ 0.")]),
    ("Grounding and efficiency", [
        ("Context precision", "Rank-aware share of relevant papers in the evidence given to the model (RAGAS-style; precision@k averaged over the relevant ranks)."),
        ("Context recall", "Share of the question's gold papers that made it into the evidence."),
        ("Citation accuracy", "Share of cited papers that are direct evidence (judged 2 of 2, or a gold paper)."),
        ("Answer relevancy", "Judge score 0–2 for how well the answer addresses the question, scaled to 0–1."),
        ("Latency / throughput", "Seconds from sending a question to seeing the answer (p50, p95); answered questions per minute under load."),
        ("Tokens / cost", "Input + output tokens of every Claude call behind one answer; cost at Anthropic list prices.")]),
    ("Answers and evidence (1,000 questions)", [
        ("Verdict accuracy", "Yes/no/maybe (PubMedQA) or support/contradict (SciFact) equals the expert label."),
        ("BioASQ answer correct", "An LLM grader compares the answer's meaning with the expert ideal answer: correct, partial, wrong."),
        ("Judged precision", "Share of evidence papers a judge labels relevant; needed because gold lists miss many relevant papers."),
        ("Direct-evidence rate", "Share of questions with at least one paper that directly answers them."),
        ("Citation precision", "Share of cited papers that are direct evidence or gold."),
        ("Proof validity", "A quoted proof sentence occurs word-for-word in the cited paper, and that paper is relevant.")]),
    ("Context awareness (live chat)", [
        ("Follow-up detection", "SHAKTI's follow-up / new-topic decision equals the gold label (accuracy, precision, recall, F1)."),
        ("Concept recall", "Share of the gold concepts (e.g. ‘brown rice’, ‘fiber’) that appear in the standalone question SHAKTI searches with. Compared with the raw message as a no-context baseline."),
        ("Rewrite fidelity", "Judge score 0–2: the standalone question has the same meaning as the researcher's reference question."),
        ("Answer relevance", "Judge score 0–2: the answer addresses what the user meant, not only the literal words."),
        ("Topic leakage", "On a topic switch, the rewrite wrongly carries over the previous topic."),
        ("Context-Awareness Score", "Mean of detection accuracy, concept recall, rewrite fidelity, answer relevance and (1 − leakage); 0–100%.")]),
    ("Hallucination (live chat)", [
        ("Atomic claim", "One checkable statement of fact from the answer (a number, an effect, a population, a recommendation)."),
        ("Supported / partial / unsupported / contradicted", "Whether the sources SHAKTI was given entail the claim fully, in part (a detail overstated), not at all, or say the opposite."),
        ("Claim faithfulness", "(supported + ½ partial) / all claims."),
        ("Hallucination rate", "(unsupported + contradicted) / all claims. ‘Excluding general knowledge’ ignores uncontroversial background statements."),
        ("Citation precision (claim)", "Cited claims for which at least one cited source supports the claim."),
        ("Online checks", "Run on every live answer in milliseconds: verbatim quotes, citations pointing to real sources, numbers traced to sources or USDA data, links resolved; they drive the risk badge.")]),
]
glossary = '<section id=metrics><h2>11 · What every metric means</h2><div class=two>' + "".join(
    f'<div class=card><h4>{E(h)}</h4><dl>' + "".join(f"<dt>{E(a)}</dt><dd>{E(b)}</dd>" for a, b in items) + "</dl></div>" for h, items in G) + "</div></section>"

limits = f"""
<section id=limits><h2>12 · How to read these results</h2><ul>
<li>The live-chat test is small ({len(turns)} turns, {n_conv} conversations) and scripted by us; it shows the behaviour and validates the metrics.
The human study will provide real questions at scale.</li>
<li>The hallucination judge is an LLM (Claude Opus, a different and stronger model than the Sonnet answering model, given exactly the text SHAKTI saw).
A human double-check of a sample of claims is recommended before publication to report judge–human agreement.</li>
<li>“Unsupported” means not found in the sources given to SHAKTI. Such a claim may still be true; it is counted because the system promises
to answer only from its sources.</li>
<li>Reference-overlap metrics (Exact Match, token F1, ROUGE, BLEU) penalise correct answers worded differently from the expert reference, so
they are low for explanatory answers and are reported next to BERTScore and the judged correctness, not instead of them. Costs are list-price
estimates; the study itself runs on a Claude subscription. The load test is small (one machine, 38 questions) and measures this deployment, not a scaled one.</li>
<li>The retrieval and method results come from the paper's experiments (Claude Sonnet, Nov 2026), with 95% bootstrap CIs and paired tests reported there.</li>
</ul><p class=note>Judge model: {E(EV['judge_model'])}. Median answer time in the test: {EV['latency']['median_secs']:.0f} s (max {EV['latency']['max_secs']:.0f} s).</p></section>
"""

CSS = """
:root{--bg:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;--grid:#e1e0d9;--base:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--crit:#d03b3b;--track:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surf:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--base:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--track:#2c2c2a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surf:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--base:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--track:#2c2c2a;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding-block:28px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px))}.two>*,.grid4>*,.tiles>*{min-width:0}
[data-tip]:hover::after{max-width:min(320px,80vw);white-space:normal}
h1{margin:0;font-size:30px;letter-spacing:.5px}h2{font-size:21px;margin:44px 0 12px;padding-top:6px;border-top:1px solid var(--grid)}
h3{font-size:15px;margin:18px 0 8px;color:var(--ink2);text-transform:uppercase;letter-spacing:.6px}h4{margin:0 0 8px;font-size:15px}
.hero .brand{display:flex;gap:14px;align-items:center}.mark{width:48px;height:48px;border-radius:12px;background:var(--s1);color:#fff;
display:grid;place-items:center;font-weight:700;font-size:24px}.sub{color:var(--ink2)}.lede{max-width:820px;color:var(--ink2);font-size:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--surf);border:1px solid var(--grid);border-radius:10px;padding:12px 14px}.big{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.tl{font-size:13px;font-weight:600}.ts{font-size:12px;color:var(--mut)}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px}.step{flex:1 1 130px;background:var(--surf);border:1px solid var(--grid);
border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:4px;font-size:13px}.step b{font-size:13.5px}.step span{color:var(--ink2)}
.step.g{border-color:var(--s3);box-shadow:inset 3px 0 0 var(--s3)}.arrow{align-self:center;color:var(--mut);font-size:18px}
.gpu{border:1px solid var(--s3);border-radius:4px;padding:0 5px;font-size:12px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;margin:12px 0}
.testset{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline;background:var(--surf);border:1px solid var(--grid);
border-radius:10px;padding:10px 14px;margin:6px 0 4px;font-size:14px}.testset>b{color:var(--ink2);font-size:12px;text-transform:uppercase;letter-spacing:.6px}
.testset span b{font-size:18px;font-variant-numeric:tabular-nums}table.kv th:last-child,table.kv td:last-child{white-space:nowrap}
.depthrow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:12px 0}.dlab{font-size:13px;color:var(--ink2);margin-right:6px}
.dcell{position:relative;display:flex;flex-direction:column;align-items:center;min-width:54px;padding:6px 8px;border-radius:8px;border:1px solid var(--grid);background:var(--surf)}
.dcell b{font-size:18px;font-variant-numeric:tabular-nums}.dcell span{font-size:12px}.dcell.ok{box-shadow:inset 0 -3px 0 var(--good)}.dcell.no{box-shadow:inset 0 -3px 0 var(--crit)}
.dcell.ok span{color:var(--good)}.dcell.no span{color:var(--crit)}.dcell.na{opacity:.55}.dlab{min-width:84px}
.two.wide{grid-template-columns:repeat(auto-fit,minmax(440px,1fr))}.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.card{background:var(--surf);border:1px solid var(--grid);border-radius:10px;padding:14px 16px;margin:0}
.card ul{margin:0;padding-left:18px}.card li{margin:4px 0}figcaption{font-weight:600;margin-bottom:10px;font-size:14px}
figcaption small{display:block;font-weight:400;color:var(--mut);font-size:12px}
.note{color:var(--mut);font-size:13px}.ci{color:var(--mut);font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12.5px}td.num,th.num{text-align:right}td.c{text-align:center}tr.tot td{border-top:1px solid var(--base)}
table.kv{margin:12px 0}table.kv td:first-child{color:var(--ink2);width:55%}table.small{font-size:12.5px}.scroll{overflow-x:auto}
details{margin:10px 0}summary{cursor:pointer;color:var(--s1);font-weight:600}
.bars{display:flex;flex-direction:column;gap:7px}.bgroup{display:flex;flex-direction:column;gap:2px;margin-bottom:4px}
.brow{display:grid;grid-template-columns:150px 1fr 54px;align-items:center;gap:8px;font-size:13px;position:relative}
.mini .brow{grid-template-columns:118px 1fr 44px}.blab{color:var(--ink2)}.bval{text-align:right;font-variant-numeric:tabular-nums}
.btrack{height:12px;background:var(--track);border-radius:0 4px 4px 0;overflow:hidden}.bfill{display:block;height:100%;border-radius:0 4px 4px 0}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12.5px;color:var(--ink2);margin:8px 0}.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.stack{display:flex;gap:2px;height:26px}.seg{display:block;height:100%;position:relative}.seg:first-child{border-radius:4px 0 0 4px}.seg:last-child{border-radius:0 4px 4px 0}
[data-tip]:hover::after{content:attr(data-tip);position:absolute;left:50%;bottom:calc(100% + 6px);transform:translateX(-50%);background:var(--ink);color:var(--bg);
padding:4px 8px;border-radius:6px;font-size:12px;white-space:nowrap;z-index:5;pointer-events:none}
.exp li{margin:8px 0}.exp .planned{color:var(--ink2)}dl{margin:0}dt{font-weight:600;font-size:13.5px;margin-top:8px}dd{margin:2px 0 0;color:var(--ink2);font-size:13.5px}
.ok{color:var(--good);font-weight:700}.no{color:var(--crit);font-weight:700}td ul{margin:0;padding-left:16px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin:10px 0 0}nav a{color:var(--s1);text-decoration:none}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{font-size:12.5px;border:1px solid var(--grid);border-radius:999px;padding:2px 9px;color:var(--ink2)}
.chip::first-letter{color:var(--good);font-weight:700}.scored{font-size:13px;color:var(--ink2);margin:6px 0}
@media (max-width:640px){.arrow{display:none}.brow{grid-template-columns:110px 1fr 48px}}
"""
nav = ('<nav>' + "".join(f'<a href="#{a}">{b}</a>' for a, b in [("pipeline", "Pipeline"), ("questions", "Questions"), ("tested", "What we tested"),
       ("allmetrics", "All metrics"), ("retrieval", "Retrieval"), ("agents", "RAG methods"), ("context", "Context awareness"), ("hallucination", "Hallucination"), ("found", "Before vs after"),
       ("turns", "All turns"), ("metrics", "Metric glossary"), ("limits", "Caveats")]) + "</nav>")
TITLE = "SHAKTI Evaluation Report"
body = ("<main>" + hero.replace("</header>", nav + "</header>", 1) + pipeline + questions + tested + allm + retrieval + agents
        + context + hallu + found + detail + glossary + limits + "</main>")
page = (f'<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
        f'<title>{TITLE}</title><style>{CSS}</style></head><body>{body}</body></html>')
with open(OUT, "w", encoding="utf-8") as f:
    f.write(page)
# claude.ai artifact version: the publisher adds the document skeleton
ART = OUT.replace(".html", "_artifact.html")
with open(ART, "w", encoding="utf-8") as f:
    f.write(f'<title>{TITLE}</title><meta name="claude:viewport-fit-sides" content="cover"><style>{CSS}</style>{body}')
print(OUT, f"{len(page) / 1024:.0f} KB", "|", ART)

