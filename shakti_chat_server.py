"""
shakti_chat_server.py: SHAKTI (Smart Health Advisor for Knowledge-based Tailored Interventions),
a context-aware, evidence-cited health and food chat for the human evaluation.

Per message:
  1. understand  - Claude rewrites the message into a standalone question using the conversation
                   (context awareness), classifies it, and lists foods whose nutrition is needed.
  2. food data   - exact values from the local USDA FoodData Central copy (no model estimates).
  3. literature  - PQ search over ~230M papers + MedCPT rerank, by one of the compared methods
                   (single / crag / deep; random, balanced per session, blinded).
  4. link check  - every source's DOI / OpenAlex record is resolved; dead records are dropped.
  5. answer      - Claude, from the numbered papers and food data only.
  6. verification (hallucination metrics) - fabricated citations, verbatim quotes, numbers grounded
                   in the sources, semantic support of cited sentences, uncited claims, links.
Charts are built here from the food data and the verified evidence, never drawn by the model.

  python shakti_chat_server.py [--port 8780]     (needs <SHAKTI_SERVE_INDEX>)

Binds to 127.0.0.1; Data: <SHAKTI_RESEARCH_DIR>\\shakti_chat
(no IP addresses or names; random session ids). Admin: /admin?token=<shakti_chat\\admin_token.txt>.
"""
import shakti_config  # noqa: F401  (loads shakti.env: your paths and keys, before anything reads them)
import os
import re
import json
import time
import uuid
import random
import secrets
import argparse
import threading
import urllib.request
import urllib.parse
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import survey_server as SV                 # Engine (PQ search + MedCPT) and papers_text, shared with the survey
import rag_agents as RA
from rag_llm import ask, RateLimited, collect

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SV.RESEARCH_DIR, "shakti_chat")
STATIC = os.path.join(HERE, "static_survey")
FOOD_DB = os.environ.get("SHAKTI_FOOD_DB", "food.duckdb")
ARMS = ["single", "crag", "deep"]
MAX_TURNS = int(os.environ.get("SHAKTI_MAX_TURNS", "20"))          # per participant
DAILY_CAP = int(os.environ.get("SHAKTI_DAILY_CAP", "300"))         # protects the Claude subscription
MAX_RUNNING, MAX_QUEUE, MAX_CHARS, HISTORY = 2, 12, 600, 4
T_STEP, T_ANSWER = int(os.environ.get("SHAKTI_T_STEP", "60")), int(os.environ.get("SHAKTI_T_ANSWER", "90"))                # s; a Claude call past this is a stall: killed and retried (rag_llm)
SUPPORT_COS = 0.45                   # cited sentence counts as supported at this MiniLM cosine to a cited sentence (pair)


def log(s):
    os.makedirs(DATA, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s)
    with open(os.path.join(DATA, "server.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


_wlock = threading.Lock()


def append(name, rec):
    with _wlock:
        with open(os.path.join(DATA, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(name):
    p = os.path.join(DATA, name)
    return [json.loads(ln) for ln in open(p, encoding="utf-8")] if os.path.exists(p) else []

# ---------------------------------------------------------------- food data (USDA FoodData Central)


class FoodDB:
    NUT = {1008: "kcal", 1003: "protein", 1005: "carbs", 1004: "fat", 1079: "fiber", 2000: "sugars",
           1093: "sodium", 1258: "satfat", 1253: "cholesterol",
           1089: "iron", 1162: "vitamin_c", 1087: "calcium", 1092: "potassium", 1090: "magnesium", 1095: "zinc",
           1114: "vitamin_d", 1178: "vitamin_b12", 1190: "folate", 1106: "vitamin_a", 1057: "caffeine"}
    ATWATER = (2047, 2048)                           # energy for foundation foods without 1008
    FOLATE_TOTAL = 1177                              # folate when DFE is not given
    UNITS = {"kcal": "kcal", "protein": "g", "carbs": "g", "fat": "g", "fiber": "g", "sugars": "g",
             "sodium": "mg", "satfat": "g", "cholesterol": "mg", "iron": "mg", "vitamin_c": "mg", "calcium": "mg",
             "potassium": "mg", "magnesium": "mg", "zinc": "mg", "vitamin_d": "µg", "vitamin_b12": "µg",
             "folate": "µg", "vitamin_a": "µg", "caffeine": "mg"}
    LABELS = {"kcal": "Calories", "carbs": "Carbohydrate", "satfat": "Saturated fat", "vitamin_c": "Vitamin C",
              "vitamin_d": "Vitamin D", "vitamin_b12": "Vitamin B12", "vitamin_a": "Vitamin A (RAE)",
              "folate": "Folate (DFE)"}
    # portion choice when several fit: typical household servings first, unusual sizes last
    PREFER = ("not specified", "medium", "regular", "12 fl oz", "tbsp", "serving")
    AVOID = ("snack", "small", "thin", "thick", "crust", "baby", "extra", "jumbo", "package", "bunch", "drink",
             "mashed", "chopped", "pieces", "pkg", "container")
    MEASURES = {"cup", "tbsp", "tsp", "oz", "fl", "g", "ml", "lb", "quart", "pint"}
    UNIT_SYN = {"tablespoon": "tbsp", "tablespoons": "tbsp", "teaspoon": "tsp", "teaspoons": "tsp", "ounce": "oz",
                "ounces": "oz", "slices": "slice", "cups": "cup", "cans": "can", "eggs": "egg", "pieces": "piece"}
    PREF = {"survey_fndds_food": 0, "sr_legacy_food": 1, "foundation_food": 2}
    # specialty variants: chosen only when the user's words ask for them ("white rice" is not glutinous rice)
    SPECIAL = {"glutinous", "baby", "infant", "toddler", "formula", "imitation", "dehydrated", "dry", "canned",
               "frozen", "ingredient", "made", "sauce", "sweetened", "flavored", "chocolate", "strawberry",
               "reduced", "lactose", "fried", "breaded", "battered", "candied", "dried", "powder", "concentrate"}

    def __init__(self):
        import duckdb
        t = time.time()
        con = duckdb.connect(FOOD_DB, read_only=True)
        types = "('survey_fndds_food','sr_legacy_food','foundation_food')"
        self.foods = {f: {"fdc": f, "type": ty, "desc": d, "words": set(re.findall(r"[a-z]+", d.lower())),
                          "n": {}, "portions": []}
                      for f, ty, d in con.execute(f"select fdc_id, data_type, description from usda_food "
                                                  f"where data_type in {types}").fetchall()}
        ids = ",".join(map(str, list(self.NUT) + list(self.ATWATER) + [self.FOLATE_TOTAL]))
        for f, nid, amt in con.execute(f"select fn.fdc_id, fn.nutrient_id, fn.amount from usda_food_nutrient fn "
                                       f"join usda_food u on u.fdc_id = fn.fdc_id where u.data_type in {types} "
                                       f"and fn.nutrient_id in ({ids})").fetchall():
            if amt is None:
                continue
            n = self.foods[f]["n"]
            if nid in self.NUT:
                n[self.NUT[nid]] = float(amt)
            elif nid == self.FOLATE_TOTAL:
                n["_folate"] = float(amt)
            elif "kcal" not in n:
                n.setdefault("_atwater", float(amt))
        for x in self.foods.values():
            if "kcal" not in x["n"] and "_atwater" in x["n"]:
                x["n"]["kcal"] = x["n"]["_atwater"]
            if "folate" not in x["n"] and "_folate" in x["n"]:
                x["n"]["folate"] = x["n"]["_folate"]
            x["n"].pop("_atwater", None)
            x["n"].pop("_folate", None)
        for f, amt, desc, mod, gw in con.execute(
                f"select p.fdc_id, p.amount, p.portion_description, p.modifier, p.gram_weight from usda_food_portion p "
                f"join usda_food u on u.fdc_id = p.fdc_id where u.data_type in {types} and p.gram_weight > 0").fetchall():
            label = (desc or "").strip() or f"{(amt or 1):g} {(mod or '').strip()}".strip()
            if label and not re.match(r"^\d+$", label):
                self.foods[f]["portions"].append((label, float(gw)))
        con.close()
        self.foods = {k: v for k, v in self.foods.items() if "kcal" in v["n"]}
        log(f"food DB ready: {len(self.foods):,} USDA foods, {time.time() - t:.0f}s")

    @staticmethod
    def _stem(w):
        return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w

    def find(self, query):
        toks = [self._stem(w) for w in re.findall(r"[a-z]+", (query or "").lower()) if len(w) > 1]
        if not toks:
            return None
        for keep in range(len(toks), 0, -1):              # relax trailing words until something matches
            want = toks[:keep]
            hits = [x for x in self.foods.values()
                    if all(any(w.startswith(t) for w in x["words"]) for t in want)]
            if hits:
                asked = set(toks)
                unasked = lambda ws: sum(1 for w in ws if not any(w.startswith(t) for t in asked))
                head = lambda x: re.findall(r"[a-z]+", x["desc"].split(",")[0].lower())   # "Oat milk" vs "Oats"
                return min(hits, key=lambda x: (unasked(x["words"] & self.SPECIAL), unasked(head(x)),
                                                self.PREF.get(x["type"], 3), len(x["desc"])))
        return None

    def _words(self, s):
        return {self.UNIT_SYN.get(w, w) for w in re.findall(r"[a-z]+", (s or "").lower())}

    def _rank(self, p):
        """Typical household portion first: no unusual size words, a preferred word, then closest to 100 g."""
        label, gw = p[0].lower(), p[1]
        return (any(w in label for w in self.AVOID), "not specified" not in label,
                not any(w in label for w in self.PREFER), abs(gw - 100.0))

    @staticmethod
    def _per_item(label, gw):
        m = re.match(r"\s*([\d.]+)", label)
        return float(gw) / (float(m.group(1)) if m and float(m.group(1)) > 0 else 1.0), re.sub(r"^\s*[\d./]+\s*", "", label)

    def portion(self, x, grams, count, unit):
        if grams and grams > 0:
            return float(grams), f"{grams:g} g"
        ps = x["portions"]
        uw = self._words(unit)
        hits = [p for p in ps if uw & self._words(p[0])] if uw else []
        if hits:                                            # "2 slices": the typical slice, not the snack-size one
            per, unit_word = self._per_item(*min(hits, key=self._rank))
            c = count if count and count > 0 else 1
            return per * c, f"{c:g} {unit_word} ({per * c:.0f} g)"
        items = [p for p in ps if not self._words(p[0]) & self.MEASURES]
        if count and count > 0 and (items or ps):           # "2 eggs": one typical item each
            per, unit_word = self._per_item(*min(items or ps, key=self._rank))
            unit_word = "typical size" if "not specified" in unit_word.lower() else unit_word
            return per * count,f"{count:g} \u00d7 {unit_word} ({per * count:.0f} g)"
        if ps:                                              # no amount: a typical serving (2 tbsp, not 1 cup)
            label, gw = min(ps, key=self._rank)
            return gw, (f"typical serving ({gw:.0f} g)" if "not specified" in label.lower() else f"{label} ({gw:.0f} g)")
        return 100.0, "100 g"

    def lookup(self, items, meal=True):
        """meal: foods eaten together (typical portions, a meal total); otherwise a comparison, where
        foods without a stated amount are compared per 100 g so the numbers are like for like."""
        out = []
        for it in items[:6]:
            x = self.find(it.get("usda_query") or it.get("name"))
            if not x:
                out.append({"query": it.get("name"), "found": False})
                continue
            stated = (it.get("grams") or 0) > 0 or (it.get("count") or 0) > 0 or (it.get("unit") or "").strip()
            if meal or stated:
                g, label = self.portion(x, it.get("grams"), it.get("count"), it.get("unit"))
            else:
                g, label = 100.0, "100 g"
            per100 = {k: round(v, 2) for k, v in x["n"].items()}
            serving = {k: round(v * g / 100.0, 1) for k, v in x["n"].items()}
            out.append({"query": it.get("name"), "found": True, "fdc_id": x["fdc"], "name": x["desc"],
                        "data_type": x["type"], "grams": round(g, 1), "portion": label,
                        "per100": per100, "serving": serving,
                        "url": f"https://fdc.nal.usda.gov/food-details/{x['fdc']}/nutrients"})
        return out


def food_text(foods):
    lines = []
    for k, f in enumerate([f for f in foods if f["found"]], 1):
        s, p = f["serving"], f["per100"]
        fmt = lambda d: ", ".join(f"{n.replace('_', ' ')} {d[n]:g} {FoodDB.UNITS[n]}" for n in FoodDB.UNITS if n in d)
        lines.append(f"[F{k}] {f['name']} (USDA FoodData Central #{f['fdc_id']}). Portion: {f['portion']}: "
                     f"{fmt(s)}. Per 100 g: {fmt(p)}.")
    return "\n".join(lines)

# ---------------------------------------------------------------- prompts

SYS_UNDERSTAND = (
    "You are the understanding step of SHAKTI, a health, food and nutrition assistant. Given the "
    "conversation so far and the user's new message: (1) rewrite the message as a standalone question "
    "that can be understood without the conversation, resolving references such as 'it', 'that', "
    "'these foods' or 'what about children?' from earlier turns (a reference to 'my first question' or "
    "an earlier topic means the numbered question it points to, which may be in the earlier-questions "
    "list rather than the recent messages); keep it unchanged if it is already "
    "standalone; (2) say whether it is a follow-up that depends on earlier turns and, if so, briefly what "
    "was taken from them; (3) classify the intent; (4) list the foods whose nutrition values (calories, "
    "protein, carbs, fat, fiber, sugars, sodium, saturated fat, cholesterol, iron, vitamin C, calcium, "
    "potassium, magnesium, zinc, vitamin D, vitamin B12, folate, vitamin A, caffeine) are needed, each with "
    "a short USDA-style search phrase "
    "(for example 'egg whole boiled', 'rice white cooked', 'banana raw', 'lentils boiled'), the grams if "
    "the user gave a weight (otherwise 0), the count if the user gave a number of items or units "
    "(otherwise 0) and the unit word ('egg', 'cup', 'slice', 'tablespoon', or empty). For comparisons or "
    "'which foods are high in X' questions list up to 6 typical foods to compare. For health-effect "
    "questions list foods only if their nutrient amounts directly help answer. Leave foods empty when "
    "no nutrition values are needed. Say whether the foods are eaten together as one meal or snack "
    "(true) or are being compared or listed (false); (5) name the nutrient the user cares most about; (6) say whether "
    "research literature is needed (health effects, safety, diets, diseases, mechanisms: yes; a pure "
    "calorie or nutrient lookup: no).")
SCH_UNDERSTAND = {"type": "object", "properties": {
    "standalone_question": {"type": "string"},
    "is_followup": {"type": "boolean"},
    "context_used": {"type": "string"},
    "intent": {"type": "string", "enum": ["health_evidence", "food_nutrition", "food_and_health",
                                          "greeting_or_about", "out_of_scope", "emergency"]},
    "foods": {"type": "array", "maxItems": 6, "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "usda_query": {"type": "string"}, "grams": {"type": "number"},
        "count": {"type": "number"}, "unit": {"type": "string"}},
        "required": ["name", "usda_query", "grams", "count", "unit"]}},
    "foods_are_meal": {"type": "boolean"},
    "focus_nutrient": {"type": "string", "enum": list(FoodDB.UNITS)},
    "needs_literature": {"type": "boolean"}},
    "required": ["standalone_question", "is_followup", "context_used", "intent", "foods", "foods_are_meal",
                 "focus_nutrient", "needs_literature"]}

SYS_CHAT = (
    "You are SHAKTI, a Smart Health Advisor for Knowledge-based Tailored Interventions. Answer the user's "
    "health, food or nutrition question for a general audience in plain, friendly language, using ONLY "
    "the numbered research papers and the food data provided, never outside knowledge. Cite research "
    "papers like [2] or [1, 4] and food data like [F1] after every sentence that states a fact. Use only "
    "numbers that appear in the papers or the food data; never estimate or invent numbers. When the user "
    "gave no amount, the food data uses a typical portion: use it, say which portion you assumed, and "
    "give meal totals by adding the listed values. For every key "
    "research claim copy one exact sentence, verbatim, from the cited paper's title or abstract into "
    "quotes. Say how strong the evidence is and its limits (for example animal, small or observational "
    "studies). If the papers do not answer the question, say so honestly. Keep the conversation in mind "
    "so follow-up questions read naturally. Use 3 to 8 sentences; short bullet lists (lines starting with "
    "'- ') are fine. This is research information, not personal medical advice; for personal medical "
    "decisions, medications, pregnancy, children or symptoms, suggest talking to a clinician. Also label "
    "each cited paper's stance on the question, list numeric findings stated in the abstracts (value and "
    "unit exactly as written), and suggest up to 3 short follow-up questions the user might ask next.")
SCH_CHAT = {"type": "object", "properties": {
    "answer": {"type": "string"},
    "verdict": {"type": "string", "enum": ["answered", "partially_answered", "insufficient_evidence"]},
    "citations": {"type": "array", "items": {"type": "integer"}},
    "quotes": {"type": "array", "items": {"type": "object", "properties": {
        "paper": {"type": "integer"}, "quote": {"type": "string"}}, "required": ["paper", "quote"]}},
    "stances": {"type": "array", "items": {"type": "object", "properties": {
        "paper": {"type": "integer"},
        "stance": {"type": "string", "enum": ["supports", "mixed", "contradicts", "background"]}},
        "required": ["paper", "stance"]}},
    "findings": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
        "label": {"type": "string"}, "value": {"type": "number"}, "unit": {"type": "string"},
        "paper": {"type": "integer"}}, "required": ["label", "value", "unit", "paper"]}},
    "follow_ups": {"type": "array", "maxItems": 3, "items": {"type": "string"}}},
    "required": ["answer", "verdict", "citations", "quotes", "stances", "findings", "follow_ups"]}

ABOUT = ("I'm SHAKTI, a Smart Health Advisor for Knowledge-based Tailored Interventions. I answer health, "
         "diet and food questions from research papers (about 230 million scholarly works) and from USDA "
         "food data, and I show the sources and the exact sentences I relied on. You can ask, for example, "
         "whether a food or diet affects a health condition, how many calories or how much protein a meal "
         "has, or how two foods compare. I give research information, not personal medical advice.")
EMERGENCY = ("This sounds like it could be urgent. If you or someone else may be in danger or has severe "
             "symptoms (such as chest pain, trouble breathing, fainting, a severe allergic reaction or "
             "thoughts of self-harm), please call your local emergency number now or go to the nearest "
             "emergency department. I can share research information, but I can't help in an emergency.")
OUT_OF_SCOPE = ("I'm built for health, diet, food and nutrition questions, so I can't help with that one. "
                "Try asking about a food, a diet, calories or nutrients, or how something you eat affects health.")

# ---------------------------------------------------------------- retrieval methods (evidence only)


def ev_single(E, q, trace):
    return E.find(q, 10)


def ev_crag(E, q, trace):
    docs = E.find(q, 10)
    g = ask(SV.task(q) + "\n\nPapers:\n" + SV.papers_text(docs), system=RA.SYS_GRADE, schema=RA.SCH_GRADE, timeout=T_STEP)
    by_n = {x["paper"]: x["grade"] for x in g["grades"]}
    graded = [(d, by_n.get(n, "incorrect")) for n, d in enumerate(docs, 1)]
    trace["grades"] = [gr for _, gr in graded]
    if not any(gr == "correct" for _, gr in graded):
        rq = g["rewritten_query"].strip() or q
        trace["rewrite"] = rq
        more = E.find(rq, 10, exclude={d["id"] for d in docs})
        g2 = ask(SV.task(q) + "\n\nPapers:\n" + SV.papers_text(more), system=RA.SYS_GRADE, schema=RA.SCH_GRADE, timeout=T_STEP)
        by2 = {x["paper"]: x["grade"] for x in g2["grades"]}
        graded += [(d, by2.get(n, "incorrect")) for n, d in enumerate(more, 1)]
    good = [d for d, gr in graded if gr == "correct"]
    if len(good) < 3:
        good += [d for d, gr in graded if gr == "ambiguous"][:3 - len(good)]
    return good[:10]


def ev_deep(E, q, trace):
    p = ask(SV.task(q), system=RA.SYS_PLAN, schema=RA.SCH_PLAN, timeout=T_STEP)
    plan = [x.strip() for x in p["queries"] if x.strip()][:4] or [q]
    trace["plan"] = plan
    pool, seen = [], set()
    for d in E.find(q, 6) + [d for pq in plan for d in E.find(pq, 5, k=150)]:
        if d["id"] not in seen:
            seen.add(d["id"])
            pool.append(d)
    pool = pool[:25]
    r = ask(SV.task(q) + "\n\nPapers:\n" + SV.papers_text(pool), system=RA.SYS_READ, schema=RA.SCH_READ, timeout=T_STEP)
    kept = list({d["id"]: d for d in [pool[n - 1] for n in r["relevant"] if 1 <= n <= len(pool)]}.values())
    fu = [t.strip() for t in r["followup_queries"] if t.strip()][:2]
    trace["followups"] = fu
    for t in fu:
        kept += E.find(t, 5, exclude=seen | {d["id"] for d in kept}, k=150)
    return kept[:15]


EVIDENCE = {"single": ev_single, "crag": ev_crag, "deep": ev_deep}

# ---------------------------------------------------------------- link verification

_links, _links_lock = {}, threading.Lock()
_POOL = ThreadPoolExecutor(8)


def _get_json(url, timeout=6):
    req = urllib.request.Request(url, headers={"User-Agent": "SHAKTI-research-chat/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None                                  # network trouble: unknown, not dead


def check_link(d):
    """-> (url, status): doi (resolves), openalex (record exists), unchecked (network), dead."""
    key = d["id"]
    with _links_lock:
        if key in _links:
            return _links[key]
    doi = (d.get("doi") or "").strip()
    res = None
    if doi:
        code, js = _get_json("https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/"))
        if code == 200 and js and js.get("responseCode") == 1:
            res = ("https://doi.org/" + doi, "doi")
    if not res:
        code, _ = _get_json("https://api.openalex.org/works/" + key)
        if code == 200:
            res = ("https://openalex.org/" + key, "openalex")
        elif code in (404, 410):
            res = (None, "dead")
        else:
            res = (("https://doi.org/" + doi) if doi else ("https://openalex.org/" + key), "unchecked")
    if res[1] != "unchecked":
        with _links_lock:
            _links[key] = res
            append("link_cache.jsonl", {"id": key, "url": res[0], "status": res[1]})
    return res


def verify_links(ev):
    res = list(_POOL.map(check_link, ev))
    kept, dead = [], []
    for d, (url, st) in zip(ev, res):
        if st == "dead":
            dead.append(d["id"])
        else:
            kept.append(dict(d, link=url, link_status=st))
    return kept, dead

# ---------------------------------------------------------------- verification (hallucination metrics)

REF_GROUP = re.compile(r"\[((?:\s*(?:F?\d+)(?:\s*[\u2013-]\s*\d+)?\s*,?)+)\]")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
DISCLAIMER = re.compile(r"clinician|doctor|dietitian|medical advice|healthcare|emergency|not personal", re.I)


def parse_refs(group):
    papers, foods = [], []
    for part in group.split(","):
        part = part.strip()
        if part.startswith("F"):
            if part[1:].isdigit():
                foods.append(int(part[1:]))
        elif re.match(r"^\d+\s*[\u2013-]\s*\d+$", part):
            a, b = [int(x) for x in re.split(r"[\u2013-]", part)]
            papers += list(range(a, b + 1)) if 0 < b - a < 20 else [a, b]
        elif part.isdigit():
            papers.append(int(part))
    return papers, foods


_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
         "seventeen eighteen nineteen".split()
_TENS = {w: 10 * k for k, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if k > 1}
_NUMWORD = re.compile(r"\b(" + "|".join(list(_TENS) + _UNITS[::-1]) + r")(?:[\s-](" + "|".join(_UNITS[1:10]) +
                      r"))?(\s+hundred)?\b", re.I)


def _words_to_digits(text):
    """'Sixty obese adults', 'twenty-five trials', 'two hundred' -> digits, so they can be matched."""
    def rep(m):
        a = m.group(1).lower()
        v = _TENS.get(a, _UNITS.index(a) if a in _UNITS else 0) + (_UNITS.index(m.group(2).lower()) if m.group(2) else 0)
        return str(v * 100 if m.group(3) else v)
    return _NUMWORD.sub(rep, text)


def numbers_in(text):
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", _words_to_digits(text or ""))
    return [float(x) for x in NUM.findall(text)]


def sentences(text):
    """Sentences, with a citation written after the full stop ('...restriction. [2, 3] One review') kept with
    the sentence it belongs to, not the next one."""
    text = re.sub(r"([.!?])\s*((?:\[(?:\s*F?\d+(?:\s*[\u2013-]\s*\d+)?\s*,?)+\]\s*)+)", r" \2\1 ", text or "")
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])|\n+", text)
    return [p.strip(" -\u2022\t") for p in parts if len(p.strip(" -\u2022\t")) > 2]


def verify(E, a, ev, foods, allowed_text):
    text = a.get("answer", "")
    found = [f for f in foods if f["found"]]
    m = {"refs_total": 0, "refs_invalid": 0}
    for g in REF_GROUP.findall(text):
        ps, fs = parse_refs(g)
        m["refs_total"] += len(ps) + len(fs)
        m["refs_invalid"] += sum(1 for n in ps if not 1 <= n <= len(ev)) + sum(1 for n in fs if not 1 <= n <= len(found))
    # quotes
    qs = [q for q in a.get("quotes", []) if q.get("quote")]
    ok = []
    for q in qs:
        n = q.get("paper", 0)
        d = ev[n - 1] if 1 <= n <= len(ev) else None
        nq = RA._norm(q["quote"])
        ok.append(bool(d) and bool(nq) and nq in RA._norm(f"{d.get('title')} {d.get('abstract')}"))
    m["quotes_total"], m["quotes_verbatim"] = len(qs), sum(ok)
    # numbers: every number in the answer must appear in the sources, food data, question or conversation
    src_nums = set(numbers_in(" ".join(f"{d.get('title')} {d.get('abstract')} {d.get('year')}" for d in ev)
                              + " " + allowed_text))
    food_vals = [v for f in found for d in (f["per100"], f["serving"]) for v in d.values()] + [f["grams"] for f in found]
    food_vals += [sum(f["serving"].get(k, 0) for f in found) for k in FoodDB.UNITS] if len(found) > 1 else []
    close = lambda x, v: abs(x - v) <= max(0.5 if abs(v) >= 10 else 0.05, 0.01 * abs(v))
    body = REF_GROUP.sub(" ", text)
    nums = [x for x in numbers_in(body) if x > 3]          # 0-3 are mostly list/structure words
    good = [x for x in nums if x in src_nums or any(close(x, v) for v in food_vals)]
    bad = []
    for x in nums:
        if x in good:
            continue
        # the answer's own arithmetic on grounded numbers (totals, differences, e.g. 143 + 167 = 310)
        if any(close(x, p + q) or close(x, abs(p - q)) for i, p in enumerate(good) for q in good[i + 1:]):
            continue
        bad.append(x)
    m["numbers_total"], m["numbers_unverified"], m["unverified_numbers"] = len(nums), len(bad), bad[:10]
    # sentences: cited, semantically supported by a cited paper, or uncited factual
    sents = sentences(text)
    cited, uncited, uncited_other, sup_check = 0, 0, 0, []
    for s in sents:
        ps, fs = [], []
        for g in REF_GROUP.findall(s):
            p_, f_ = parse_refs(g)
            ps += p_
            fs += f_
        if ps or fs:
            cited += 1
            docs = [ev[n - 1] for n in ps if 1 <= n <= len(ev)]
            if docs:
                sup_check.append((REF_GROUP.sub("", s), docs))
        elif len(s.split()) >= 6 and not DISCLAIMER.search(s) and not s.endswith("?"):
            if any(x in bad for x in numbers_in(REF_GROUP.sub(" ", s)) if x > 3):
                uncited += 1                                # an unsourced figure: counts toward the risk
            else:
                uncited_other += 1                          # general or summary statement: reported only
    supported = 0
    if sup_check:
        import numpy as np
        for s, docs in sup_check:
            cand = []
            for d in docs:                          # single sentences and adjacent pairs (answers often merge two)
                ss = sentences(f"{d.get('title')}. {d.get('abstract') or ''}")
                cand += ss + [a_ + " " + b_ for a_, b_ in zip(ss, ss[1:])]
            cand = cand[:300]
            if not cand:
                continue
            with E.gpu:
                v = E.enc.encode([s] + cand, normalize_embeddings=True, convert_to_numpy=True)
            if float(np.max(v[1:] @ v[0])) >= SUPPORT_COS:
                supported += 1
    m.update(sentences_total=len(sents), sentences_cited=cited, sentences_uncited_factual=uncited,
             sentences_uncited_other=uncited_other, cited_checked=len(sup_check), cited_supported=supported)
    reasons = []
    if m["refs_invalid"]:
        reasons.append(f"{m['refs_invalid']} citation(s) point to no source")
    if m["quotes_total"] and m["quotes_verbatim"] < m["quotes_total"]:
        reasons.append(f"{m['quotes_total'] - m['quotes_verbatim']} quote(s) not found word-for-word in the paper")
    if bad:
        reasons.append(f"{len(bad)} number(s) not found in the sources: " + ", ".join(f"{x:g}" for x in bad[:4]))
    if sup_check and supported < len(sup_check):
        reasons.append(f"{len(sup_check) - supported} cited sentence(s) weakly matched to their papers")
    if uncited:
        reasons.append(f"{uncited} sentence(s) with unsourced numbers and no citation")
    qrate = m["quotes_verbatim"] / m["quotes_total"] if m["quotes_total"] else 1.0
    srate = supported / len(sup_check) if sup_check else 1.0
    # high needs hard evidence (mechanical checks); the semantic-similarity proxy can only raise to medium
    if m["refs_invalid"] or qrate < 0.5 or len(bad) >= 2:
        risk = "high"
    elif qrate < 0.9 or bad or srate < 0.5 or uncited:
        risk = "medium"
    else:
        risk = "low"
    m["risk"], m["risk_reasons"] = risk, reasons
    return m, ok

# ---------------------------------------------------------------- charts (built from data, never by the model)

STUDY = [("Meta-analysis / systematic review", r"meta-analys|systematic review|umbrella review|pooled analys"),
         ("Randomised trial", r"randomi[sz]ed|controlled trial|\brct\b|crossover"),
         ("Cohort / observational", r"cohort|prospective|cross-sectional|case-control|observational|population-based"),
         ("Animal / lab", r"\brats?\b|\bmice\b|\bmouse\b|in vitro|animal model|cell line"),
         ("Review", r"\breview\b|overview|perspective|commentary|editorial")]


def study_type(d):
    t = f"{d.get('title')} {(d.get('abstract') or '')[:600]}".lower()
    for name, pat in STUDY:
        if re.search(pat, t):
            return name
    return "Other"


def charts(foods, focus, a, ev, meal):
    out = []
    found = [f for f in foods if f["found"]]
    if found and (meal or len(found) == 1):          # a calorie split only makes sense for one food or one meal
        tot = {k: sum(f["serving"].get(k, 0) for f in found) for k in ("protein", "carbs", "fat")}
        kc = {"Protein": tot["protein"] * 4, "Carbohydrate": tot["carbs"] * 4, "Fat": tot["fat"] * 9}
        if sum(kc.values()) > 0:
            name = found[0]["name"] if len(found) == 1 else "this meal"
            out.append({"type": "donut", "title": f"Where the calories come from: {name}",
                        "subtitle": "kcal from protein (4/g), carbohydrate (4/g) and fat (9/g); USDA FoodData Central",
                        "unit": "kcal", "items": [{"label": k, "value": round(v)} for k, v in kc.items()]})
    if len(found) >= 2:
        nut = focus if focus in FoodDB.UNITS else "kcal"
        label = FoodDB.LABELS.get(nut, nut.capitalize())
        same = len({f["portion"] for f in found}) == 1
        out.append({"type": "bar", "title": f"{label} per {found[0]['portion'] if same else 'portion'}",
                    "subtitle": "portion sizes as listed in the food table; USDA FoodData Central",
                    "unit": FoodDB.UNITS[nut],
                    "items": [{"label": f"{f['query'] or f['name']}" + ("" if same else f" ({f['grams']:.0f} g)"),
                               "value": f["serving"].get(nut, 0)} for f in found]})
    if ev:
        cited = sorted({n for n in a.get("citations", []) if 1 <= n <= len(ev)}) or list(range(1, len(ev) + 1))
        st = Counter(s["stance"] for s in a.get("stances", []) if s.get("paper") in cited)
        if sum(st.values()) >= 2:
            order = ["supports", "mixed", "contradicts", "background"]
            out.append({"type": "donut", "title": "What the cited studies say",
                        "subtitle": "stance of each cited paper on your question, as read by SHAKTI from the abstract",
                        "unit": "papers", "items": [{"label": k.capitalize(), "value": st[k]} for k in order if st[k]]})
        ty = Counter(study_type(ev[n - 1]) for n in cited)
        if sum(ty.values()) >= 2:
            order = [s[0] for s in STUDY] + ["Other"]
            out.append({"type": "bar", "title": "Types of studies behind this answer",
                        "subtitle": "classified from each cited paper's title and abstract",
                        "unit": "papers", "items": [{"label": k, "value": ty[k]} for k in order if ty[k]]})
        fnd = []
        for f in a.get("findings", []):                     # keep a finding only if its value is in that paper
            n = f.get("paper", 0)
            if 1 <= n <= len(ev) and f.get("value") is not None:
                d = ev[n - 1]
                if float(f["value"]) in set(numbers_in(f"{d.get('title')} {d.get('abstract')}")):
                    fnd.append(f)
        units = Counter((f.get("unit") or "").strip() for f in fnd)
        if units:
            unit, c = units.most_common(1)[0]
            if c >= 2:
                out.append({"type": "bar", "title": "Results reported in the papers",
                            "subtitle": "values copied from the abstracts and checked against them",
                            "unit": unit, "items": [{"label": f"{f['label']} [{f['paper']}]", "value": f["value"]}
                                                    for f in fnd if (f.get("unit") or "").strip() == unit][:6]})
    return out[:4]

# ---------------------------------------------------------------- sessions, conversations, jobs


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions, self.convs, self.jobs = {}, defaultdict(list), {}
        self.queue = deque()
        self.today = (time.strftime("%Y-%m-%d"), 0)
        for r in read_jsonl("sessions.jsonl"):
            self.sessions[r["sid"]] = {"arms": r["arms"], "asked": 0}
        for r in read_jsonl("turns.jsonl"):
            if r["sid"] in self.sessions:
                self.sessions[r["sid"]]["asked"] += 1
            self.convs[(r["sid"], r["conv"])].append({"q": r["question"], "standalone": r.get("standalone"),
                                                      "a": (r.get("answer") or "")[:500], "papers": r.get("paper_ids", [])})
            self.jobs[r["turn"]] = {"id": r["turn"], "sid": r["sid"], "arm": r.get("arm"), "status": "done"}

    def count_today(self):
        d = time.strftime("%Y-%m-%d")
        if self.today[0] != d:
            self.today = (d, 0)
        return self.today[1]


ST = ENGINE = FOOD = None                    # set in main()
WORK = threading.Semaphore(MAX_RUNNING)


def history_text(hist):
    """The last HISTORY turns in full, preceded by every earlier question (sessions hold at most MAX_TURNS), so a
    call-back such as "going back to my first question" still resolves after many other topics."""
    if not hist:
        return "(no earlier messages)"
    old, recent = hist[:-HISTORY], hist[-HISTORY:]
    head = ""
    if old:
        head = ("Earlier questions in this conversation (oldest first; answers not shown):\n"
                + "\n".join(f"{i}. {h.get('standalone') or h['q']}" for i, h in enumerate(old, 1)) + "\n\n"
                + "Most recent messages:\n")
    first = len(old) + 1
    return head + "\n".join(f"[question {first + i}] User: {h['q']}\nSHAKTI: {h['a'][:400]}"
                            for i, h in enumerate(recent))


def llm_summary(calls):
    """Tokens, list-price cost and Claude time of one turn (efficiency metrics)."""
    fresh = [c for c in calls if not c.get("cached")]
    u = lambda k: sum(c["usage"].get(k, 0) or 0 for c in fresh)
    return {"calls": len(calls), "cached_calls": len(calls) - len(fresh), "input_tokens": u("input_tokens"),
            "cache_write_tokens": u("cache_creation_input_tokens"), "cache_read_tokens": u("cache_read_input_tokens"),
            "output_tokens": u("output_tokens"), "cost_list_usd": round(sum(c.get("cost_list_usd") or 0 for c in fresh), 5),
            "llm_secs": round(sum(c.get("secs") or 0 for c in fresh), 1), "models": sorted({c["model"] for c in calls})}


def process(job):
    with WORK, collect() as llm_calls:
        t0, times = time.time(), {}

        def stage(s):
            with ST.lock:
                job["stage"] = s
            times[s] = round(time.time() - t0, 1)
        with ST.lock:
            job["status"] = "running"
            hist = list(ST.convs[(job["sid"], job["conv"])])
        trace = {}
        try:
            stage("Understanding your question")
            u = ask(f"Conversation so far:\n{history_text(hist)}\n\nNew message: {job['question']}",
                    system=SYS_UNDERSTAND, schema=SCH_UNDERSTAND, timeout=T_STEP)
            sq = (u.get("standalone_question") or job["question"]).strip()
            intent = u.get("intent", "health_evidence")
            foods, ev, dead, arm = [], [], [], None
            a = {"answer": "", "verdict": "answered", "citations": [], "quotes": [], "stances": [],
                 "findings": [], "follow_ups": []}
            if intent in ("greeting_or_about", "out_of_scope", "emergency"):
                a["answer"] = {"greeting_or_about": ABOUT, "out_of_scope": OUT_OF_SCOPE, "emergency": EMERGENCY}[intent]
                arm = "none"
            else:
                if u.get("foods"):
                    stage("Looking up USDA food data")
                    foods = FOOD.lookup(u["foods"], meal=bool(u.get("foods_are_meal")))
                lit = u.get("needs_literature", True) or not any(f["found"] for f in foods)
                if lit:
                    arm = job["arm"]
                    stage("Searching 230 million research papers")
                    ev = EVIDENCE[arm](ENGINE, sq, trace)
                    stage("Checking that every source link works")
                    ev, dead = verify_links(ev)
                else:
                    arm = "food"
                stage("Writing the answer from the evidence")
                prompt = (f"Conversation so far:\n{history_text(hist)}\n\nUser's message: {job['question']}\n"
                          f"Standalone question: {sq}\n\n"
                          + (f"Food data (USDA FoodData Central, exact values; cite as [F1], [F2]):\n{food_text(foods)}\n\n"
                             if any(f["found"] for f in foods) else "")
                          + "Papers:\n" + (SV.papers_text(ev) if ev else "(no papers; answer from the food data only)"))
                a = ask(prompt, system=SYS_CHAT, schema=SCH_CHAT, timeout=T_ANSWER)
            stage("Checking the answer for hallucinations")
            allowed = job["question"] + " " + " ".join(h["q"] + " " + h["a"] for h in hist)
            ver, qok = verify(ENGINE, a, ev, foods, allowed)
            ch = charts(foods, u.get("focus_nutrient", "kcal"), a, ev, bool(u.get("foods_are_meal"))) if arm != "none" else []
            cited = set(a.get("citations", []))
            for g in REF_GROUP.findall(a.get("answer", "")):
                cited |= set(parse_refs(g)[0])
            stance = {s["paper"]: s["stance"] for s in a.get("stances", [])}
            fn = u.get("focus_nutrient") if u.get("focus_nutrient") in FoodDB.UNITS else "kcal"
            papers = [{"n": n, "id": d["id"], "title": d.get("title"), "year": d.get("year"), "venue": d.get("venue"),
                       "doi": d.get("doi"), "link": d.get("link"), "link_status": d.get("link_status"),
                       "type": study_type(d), "stance": stance.get(n), "cited": n in cited,
                       "abstract": (d.get("abstract") or "")[:900]} for n, d in enumerate(ev, 1)]
            quotes = [{"paper": q.get("paper"), "quote": q.get("quote"), "verbatim": ok}
                      for q, ok in zip([q for q in a.get("quotes", []) if q.get("quote")], qok)]
            res = {"answer": a.get("answer", ""), "verdict": a.get("verdict"), "papers": papers, "quotes": quotes,
                   "foods": [f for f in foods if f["found"]], "foods_missing": [f["query"] for f in foods if not f["found"]],
                   "charts": ch, "follow_ups": a.get("follow_ups", [])[:3], "verification": ver,
                   "focus": {"key": fn, "label": FoodDB.LABELS.get(fn, fn.capitalize()), "unit": FoodDB.UNITS[fn]},
                   "context": {"is_followup": bool(u.get("is_followup")), "standalone": sq,
                               "context_used": u.get("context_used", "")}}
            # context-awareness measures, logged for later analysis
            ctx = {"history_turns": len(hist), "rewrite_changed": RA._norm(sq) != RA._norm(job["question"])}
            if hist:
                with ENGINE.gpu:
                    v = ENGINE.enc.encode([job["question"], sq, hist[-1].get("standalone") or hist[-1]["q"]],
                                          normalize_embeddings=True, convert_to_numpy=True)
                ctx.update(cos_raw_prev=round(float(v[0] @ v[2]), 3), cos_standalone_prev=round(float(v[1] @ v[2]), 3))
                prev = set(hist[-1].get("papers", []))
                cur = {p["id"] for p in papers}
                ctx["paper_jaccard_prev"] = round(len(prev & cur) / len(prev | cur), 3) if prev | cur else None
            append("turns.jsonl", {"turn": job["id"], "sid": job["sid"], "conv": job["conv"], "turn_index": len(hist),
                                   "time": time.time(), "secs": round(time.time() - t0, 1), "stage_times": times,
                                   "total_secs": round(time.time() - job.get("t_submit", t0), 1),
                                   "arm": arm, "assigned_arm": job["arm"], "intent": intent, "question": job["question"],
                                   "standalone": sq, "understand": u, "context_metrics": ctx, "trace": trace,
                                   "answer": res["answer"], "verdict": res["verdict"], "paper_ids": [p["id"] for p in papers],
                                   "papers": [{k: p[k] for k in ("n", "id", "doi", "link", "link_status", "type", "stance", "cited")}
                                              for p in papers],
                                   "dead_sources_removed": dead, "quotes": quotes, "foods": res["foods"],
                                   "charts": ch, "verification": ver, "llm": llm_summary(llm_calls)})
            with ST.lock:
                ST.convs[(job["sid"], job["conv"])].append({"q": job["question"], "standalone": sq,
                                                            "a": res["answer"][:500], "papers": [p["id"] for p in papers]})
                job.update(status="done", result=res, arm=arm)
        except RateLimited:
            with ST.lock:
                job.update(status="error", error="SHAKTI has reached its usage limit for now. Please try again later.")
        except Exception as e:
            log(f"turn {job['id']} ({job['arm']}) failed: {e!r}")
            with ST.lock:
                job.update(status="error", error="Sorry, something went wrong with this question. Please try "
                                                 "rephrasing it.")

# ---------------------------------------------------------------- HTTP

PAGE_PATH = os.path.join(HERE, "shakti_chat.html")
ADMIN_TOKEN = None


def clean(s, n):
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(s or "")).strip()[:n]


class H(BaseHTTPRequestHandler):
    server_version = "shakti"

    def log_message(self, *a):            # no access log: it would hold participants' IP addresses
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", cache=False):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(b)

    def js(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False))

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 20000:
            raise ValueError("too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        args = dict(urllib.parse.parse_qsl(qs))
        if path == "/":
            return self.send(200, open(PAGE_PATH, encoding="utf-8").read(), "text/html; charset=utf-8")
        if path in ("/static/shakti_icon.png", "/favicon.ico"):
            p = os.path.join(STATIC, "shakti_icon.png")
            if os.path.exists(p):
                return self.send(200, open(p, "rb").read(), "image/png", cache=True)
            return self.send(404, "no icon", "text/plain")
        if path == "/api/job":
            with ST.lock:
                j = ST.jobs.get(args.get("id", ""))
                if not j or j.get("sid") != args.get("sid"):
                    return self.js({"status": "unknown"}, 404)
                pos = list(ST.queue).index(j["id"]) + 1 if j["id"] in ST.queue and j["status"] == "queued" else 0
                return self.js({"status": j["status"], "stage": j.get("stage"), "result": j.get("result"),
                                "error": j.get("error"), "queue": pos})
        if path == "/admin":
            if not ADMIN_TOKEN or not secrets.compare_digest(args.get("token", ""), ADMIN_TOKEN):
                return self.send(404, "not found", "text/plain")
            return self.send(200, admin_page(), "text/html; charset=utf-8")
        return self.send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            b = self.body()
        except Exception:
            return self.js({"error": "bad request"}, 400)
        sid = str(b.get("sid", ""))
        if self.path == "/api/session":
            if not (b.get("consent") is True and b.get("adult") is True):
                return self.js({"error": "consent required"}, 400)
            sid = uuid.uuid4().hex
            arms = ARMS[:]
            random.shuffle(arms)
            append("sessions.jsonl", {"sid": sid, "time": time.time(), "arms": arms, "role": clean(b.get("role"), 40),
                                      "nutrition_background": clean(b.get("background"), 40)})
            with ST.lock:
                ST.sessions[sid] = {"arms": arms, "asked": 0}
            return self.js({"sid": sid, "max_questions": MAX_TURNS})
        if self.path == "/api/ask":
            q, conv = clean(b.get("question"), MAX_CHARS), clean(b.get("conv"), 40)
            with ST.lock:
                s = ST.sessions.get(sid)
                if not s:
                    return self.js({"error": "Your session has expired. Please reload the page.", "reset": True}, 400)
                if len(q) < 2 or not conv:
                    return self.js({"error": "Please type a question."}, 400)
                if s["asked"] >= MAX_TURNS:
                    return self.js({"error": f"Thank you! You have reached the limit of {MAX_TURNS} questions. "
                                             "Please share your feedback with the button on the left."}, 400)
                if ST.count_today() >= DAILY_CAP:
                    return self.js({"error": "SHAKTI has reached today's question limit. Please come back tomorrow."}, 503)
                if len(ST.queue) >= MAX_QUEUE:
                    return self.js({"error": "Many people are asking right now. Please try again in a minute."}, 503)
                lit_n = sum(1 for j in ST.jobs.values() if j.get("sid") == sid and j.get("arm") in ARMS)
                arm = s["arms"][lit_n % len(s["arms"])]
                s["asked"] += 1
                ST.today = (ST.today[0], ST.today[1] + 1)
                jid = uuid.uuid4().hex
                job = {"id": jid, "sid": sid, "conv": conv, "arm": arm, "question": q, "status": "queued",
                       "stage": "Waiting for a free slot", "t_submit": time.time()}
                ST.jobs[jid] = job
                ST.queue.append(jid)

            def go():
                try:
                    process(job)
                finally:
                    with ST.lock:
                        if jid in ST.queue:
                            ST.queue.remove(jid)
            threading.Thread(target=go, daemon=True).start()
            return self.js({"job": jid, "remaining": MAX_TURNS - s["asked"]})
        if self.path == "/api/react":
            jid, score = str(b.get("turn", "")), b.get("score")
            with ST.lock:
                j = ST.jobs.get(jid)
            if not j or j.get("sid") != sid or score not in (1, 2, 3, 4, 5):
                return self.js({"error": "unknown answer"}, 400)
            append("reactions.jsonl", {"turn": jid, "sid": sid, "arm": j.get("arm"), "score": score, "time": time.time()})
            return self.js({"ok": True})
        if self.path == "/api/report":
            jid = str(b.get("turn", ""))
            with ST.lock:
                j = ST.jobs.get(jid)
            if not j or j.get("sid") != sid:
                return self.js({"error": "unknown answer"}, 400)
            reasons = [r for r in b.get("reasons", []) if r in ("incorrect", "broken_link", "made_up_source",
                                                                 "unsupported_number", "unsafe", "off_topic", "other")]
            append("reports.jsonl", {"turn": jid, "sid": sid, "arm": j.get("arm"), "reasons": reasons,
                                     "comment": clean(b.get("comment"), 1000), "time": time.time()})
            return self.js({"ok": True})
        if self.path == "/api/final":
            if sid not in ST.sessions:
                return self.js({"error": "unknown session"}, 400)
            rec = {"sid": sid, "time": time.time(), "comment": clean(b.get("comment"), 2000)}
            for k, hi in (("umux_capabilities", 7), ("umux_easy", 7), ("context_understood", 5), ("trust_sources", 5),
                          ("charts_helpful", 5), ("use_again", 5)):
                v = b.get(k)
                if isinstance(v, int) and 1 <= v <= hi:
                    rec[k] = v
            append("final.jsonl", rec)
            return self.js({"ok": True})
        return self.js({"error": "not found"}, 404)


def admin_page():
    import html
    turns, reacts, reps, fin = (read_jsonl(n) for n in ("turns.jsonl", "reactions.jsonl", "reports.jsonl", "final.jsonl"))
    last = {}
    for r in reacts:
        last[r["turn"]] = r                                  # the latest reaction per answer counts
    by = defaultdict(lambda: defaultdict(list))
    arm_of = {t["turn"]: t["arm"] for t in turns}
    for t in turns:
        v = t["verification"]
        by[t["arm"]]["risk"].append(v["risk"])
        if v.get("quotes_total"):
            by[t["arm"]]["quotes"].append(v["quotes_verbatim"] / v["quotes_total"])
        by[t["arm"]]["follow"].append(1 if t["understand"].get("is_followup") else 0)
    for r in last.values():
        by[arm_of.get(r["turn"], r.get("arm"))]["react"].append(r["score"])
    mean = lambda x: f"{sum(x) / len(x):.2f}" if x else "-"
    rows = "".join(
        f"<tr><td>{html.escape(str(a))}</td><td>{len(v['risk'])}</td><td>{mean(v['react'])} (n={len(v['react'])})</td>"
        f"<td>{mean(v['quotes'])}</td><td>{Counter(v['risk'])['low']}/{Counter(v['risk'])['medium']}/{Counter(v['risk'])['high']}</td>"
        f"<td>{mean(v['follow'])}</td></tr>" for a, v in sorted(by.items(), key=lambda kv: str(kv[0])))
    return (f"<!doctype html><meta charset=utf-8><title>SHAKTI admin</title><body style='font-family:sans-serif'>"
            f"<h2>SHAKTI chat study</h2><p>{len(ST.sessions)} participants, {len(turns)} answers, {len(last)} rated, "
            f"{len(reps)} problem reports, {len(fin)} final surveys; today {ST.count_today()}/{DAILY_CAP}; "
            f"queue {len(ST.queue)}</p><table border=1 cellpadding=4><tr><th>method</th><th>answers</th>"
            f"<th>satisfaction 1-5</th><th>verbatim quotes</th><th>risk low/med/high</th><th>follow-up share</th></tr>"
            f"{rows}</table><p>Data: {html.escape(DATA)}; analysis: shakti_chat_analyze.py</p>")


def main():
    global ENGINE, FOOD, ADMIN_TOKEN, ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    tp = os.path.join(DATA, "admin_token.txt")
    if not os.path.exists(tp):
        with open(tp, "w") as f:
            f.write(secrets.token_urlsafe(24))
    ADMIN_TOKEN = open(tp).read().strip()
    ST = State()
    for r in read_jsonl("link_cache.jsonl"):
        _links[r["id"]] = (r["url"], r["status"])
    FOOD = FoodDB()
    ENGINE = SV.Engine()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    log(f"SHAKTI chat on http://127.0.0.1:{a.port}/  (admin: /admin?token={ADMIN_TOKEN}); arms {ARMS}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
