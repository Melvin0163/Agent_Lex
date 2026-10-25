"""Pipeline B - JEV + OKF (Choice-based decisions).

Knowledge lives in an Open Knowledge Format bundle (knowledge/). The agent navigates it the
way OKF intends - root index -> topic / act index -> concept files -> cross-links - and every
navigation decision is a typed question answered by a System One decision model (Jev or a
Jev-compatible model). Decisions use Choice, the primitive small decision models answer best:

  route          Choice  doc_question | review | law_question | out_of_scope
  pick_clauses   Choice  which clause of the uploaded document is the question about? (+ "none")
  pick_topics    Choice  which of the 16 topic pages is it about? asked for the question and for each
                         selected clause (+ "none")
  pick_acts      Choice  which Act governs it?
  pick_sections  Choice  which candidate section is needed? repeated, removing each pick, until "none"
  gate           Noul    do the selected sources answer it?  -> if not, reply without the LLM
  verify         Choice  is each cited claim supported or not supported by its sources?

The generative LLM is called once (twice if verification fails badly), only to write the answer.
Every request is kept under ~1,000 tokens so small local models (1,024-token limit) can read it.
"""
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi

from .. import config, llm, metrics
from ..corpus import tokenize
from ..jev import Choice, Noul, get_engine
from ..okf import OKFBundle
from .common import REFUSAL, AgentResult, answer_messages, source_list
from .rag_agent import DOC_WORDS

ROUTE_OPTIONS = {
    "doc_question": "asks about a specific term, clause or situation in the user's own uploaded document",
    "review": "asks to review, check or find risks in the whole uploaded document",
    "law_question": "a general question about jobs, the workplace, salary, benefits, harassment or labour law",
    "out_of_scope": "unrelated to employment law and to the user's documents (for example food, travel or programming)",
}
ROUTE_HYPOTHESES = {
    "doc_question": "This question is about a clause or term in the user's own contract, agreement or policy.",
    "review": "This is a request to review or check an entire document.",
    "law_question": "This question is about employment, jobs, workplace rights, salary, benefits or labour law.",
    "out_of_scope": "This question is about food, travel, programming, entertainment or another topic unrelated to work or law.",
}

SECOND_PICK = 0.5        # also keep the runner-up when its probability is at least half the winner's
MIN_PICK = 0.15          # a pick below this probability is ignored
TRUST_TOPIC = 0.5        # topic chosen with at least this probability -> its primary governing section is kept
MAX_SECTIONS = 4         # rounds of "pick the next needed section"
MAX_CANDIDATES = 12
GATE_MIN = 0.15
UNSUPPORTED_MIN = 0.6    # flag a claim when "not supported" wins with at least this probability
CTX_CHARS = 900          # cap on the context string sent with each decision (token budget)


class State(TypedDict, total=False):
    question: str
    docs: list
    route: str
    route_probs: dict
    doc_hits: list
    topics: list
    acts: list
    candidates: list
    law_hits: list
    visited: list
    gated: bool
    draft: str
    issues: list
    retries: int
    retry_flag: bool
    answer: str


def _bundle():
    return OKFBundle(config.KNOWLEDGE_DIR)


def _clip(s, n):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."


def _focus(state, with_topics=False):
    """Question plus selected clause text (and later the chosen topic notes), capped for the token budget."""
    out = state["question"]
    clauses = " ".join(d["text"] for d in state.get("doc_hits", []))
    if clauses:
        out += "\nDocument clauses: " + _clip(clauses, 450)
    if with_topics and state.get("topics"):
        out += "\nBackground: " + _clip(" ".join(t.get("note", "") for t in state["topics"]), 350)
    return _clip(out, CTX_CHARS)


def _ranked_picks(answer, keys, none_key="none"):
    """Winner plus a close runner-up, ignoring 'none' and low-probability options."""
    probs = answer.probabilities or {answer.choice: 1.0}
    ranked = sorted((k for k in keys if k != none_key), key=lambda k: -probs.get(k, 0))
    if not ranked:
        return [], probs
    top = ranked[0]
    if answer.choice == none_key and probs.get(top, 0) < 0.25:
        return [], probs
    picks = [top] if probs.get(top, 0) >= MIN_PICK or answer.choice == top else []
    if picks and len(ranked) > 1:
        second = ranked[1]
        if probs.get(second, 0) >= max(MIN_PICK, SECOND_PICK * probs.get(top, 0)):
            picks.append(second)
    return picks, probs


# ---------------- nodes ----------------

def _route(state):
    eng = get_engine()
    has_doc = bool(state["docs"])
    keys = list(ROUTE_OPTIONS) if has_doc else ["law_question", "out_of_scope"]
    q = Choice("What kind of request is this?", {k: ROUTE_OPTIONS[k] for k in keys},
               hint={"premise": state["question"], "hypotheses": [ROUTE_HYPOTHESES[k] for k in keys]})
    st = {"question": state["question"],
          "uploaded_documents": ", ".join(d.name for d in state["docs"]) or "none"}
    a = eng.decide(st, {"route": q})["route"]
    r = a.choice
    if has_doc and r == "law_question" and DOC_WORDS.search(state["question"]):
        r = "doc_question"
    return {"route": r, "route_probs": a.probabilities, "retries": 0, "visited": ["index.md"]}


def _pick_clauses(state):
    eng, b = get_engine(), _bundle()
    hits, visited = [], list(state.get("visited", []))
    for doc in state["docs"]:
        entries = b.read_index(f"documents/{doc.okf_slug}")
        visited.append(f"documents/{doc.okf_slug}/index.md")
        clauses = [(e, b.read(e.path)) for e in entries if not e.path.endswith("overview.md")]
        if not clauses:
            continue
        bodies = [c.body.split("\n", 2)[-1].strip() for _e, c in clauses]
        if state["route"] == "review":
            chosen, probs = list(range(len(clauses)))[:10], {}
        else:
            idx = list(range(len(clauses)))
            if len(idx) > 10:                      # long documents: shortlist by keywords first
                bm = BM25Okapi([tokenize(clauses[i][1].title + " " + bodies[i]) for i in idx])
                sc = bm.get_scores(tokenize(state["question"]))
                idx = sorted(idx, key=lambda i: -sc[i])[:10]
            keys = [f"c{i}" for i in idx] + ["none"]
            criteria = {f"c{i}": f"{clauses[i][1].title}: {_clip(bodies[i], 160)}" for i in idx}
            criteria["none"] = "None of these clauses is about the question"
            a = eng.decide({"question": state["question"], "document": doc.name},
                           {"clause": Choice("Which clause of the user's document is the question about?", criteria,
                                             hint={"query": state["question"],
                                                   "passages": [clauses[i][1].title + ". " + bodies[i] for i in idx] + [None]})})["clause"]
            picks, probs = _ranked_picks(a, keys)
            chosen = [int(k[1:]) for k in picks]
        for i in sorted(chosen):
            e, c = clauses[i]
            visited.append(e.path)
            p = probs.get(f"c{i}")
            hits.append({"doc": doc.name, "title": c.title, "text": bodies[i],
                         "score": round(p, 3) if p is not None else None, "okf": e.path})
    return {"doc_hits": hits, "visited": visited}


def _topic_question(names, context):
    criteria = {k: f"{e.title}. {_clip(e.description, 110)}" for k, e in names.items()}
    criteria["none"] = "None of these topics; the question is about something else"
    return Choice("Which legal topic is this request about?", criteria,
                  hint={"query": context, "passages": [f"{e.title}. {e.description}" for e in names.values()] + [None]})


def _pick_topics(state):
    eng, b = get_engine(), _bundle()
    entries = b.read_index("topics")
    names = {f"t{i}": e for i, e in enumerate(entries)}
    keys = list(names) + ["none"]

    if state["route"] == "review" and state.get("doc_hits"):
        # one request per clause, sent together: which topic does this clause raise?
        reqs = [({"clause": _clip(d["title"] + ". " + d["text"], 600)},
                 {"topic": _topic_question(names, d["title"] + ". " + d["text"])}) for d in state["doc_hits"]]
        best, won = {}, set()
        for res in eng.decide_many(reqs):
            picks, probs = _ranked_picks(res["topic"], keys)
            for k in picks[:1]:
                best[k] = max(best.get(k, 0), probs.get(k, 0))
                won.add(k)
        chosen = sorted(best, key=lambda k: -best[k])[:5]
        probs = best
    else:
        # one topic decision for the question, plus one per selected clause (the clause is often the real subject)
        focus = _focus(state)
        reqs = [({"question": state["question"], "context": focus}, {"topic": _topic_question(names, focus)})]
        for d in state.get("doc_hits", [])[:3]:
            ctx = _clip(state["question"] + "\nClause: " + d["title"] + ". " + d["text"], CTX_CHARS)
            reqs.append(({"question": state["question"], "clause": _clip(d["title"] + ". " + d["text"], 600)},
                         {"topic": _topic_question(names, ctx)}))
        best, won = {}, set()
        for res in eng.decide_many(reqs):
            a = res["topic"]
            picks, probs = _ranked_picks(a, keys)
            if a.choice in names:
                won.add(a.choice)
            for k in picks:
                best[k] = max(best.get(k, 0), probs.get(k, 0))
        chosen = sorted(best, key=lambda k: (k not in won, -best[k]))[:3]
        probs = best

    topics = []
    for k in chosen:
        c = b.read(names[k].path)
        note = c.body.split("## Governing", 1)[0].replace("#", "").strip()
        links = [l for l in c.links if l.startswith("law/") and not l.endswith("overview.md")]
        topics.append({"title": names[k].title, "path": names[k].path, "p": round(probs.get(k, 0), 3),
                       "note": _clip(note, 500), "links": links, "won": k in won})
    return {"topics": topics, "visited": state.get("visited", []) + ["topics/index.md"] + [t["path"] for t in topics]}


def _pick_acts(state):
    eng, b = get_engine(), _bundle()
    entries = b.read_index("law")
    focus = _focus(state)
    keys = [f"a{i}" for i in range(len(entries))]
    criteria = {k: f"{entries[int(k[1:])].title}: {_clip(entries[int(k[1:])].description, 120)}" for k in keys}
    a = eng.decide({"question": state["question"], "context": focus},
                   {"act": Choice("Which Act governs this request?", criteria,
                                  hint={"query": focus, "passages": [f"{e.title}. {e.description}" for e in entries]})})["act"]
    chosen, probs = _ranked_picks(a, keys)
    chosen = chosen or [a.choice]

    # inside each chosen Act: search its index (titles + one-line descriptions), like grepping a wiki
    query = focus + " " + " ".join(t["title"] for t in state.get("topics", []))
    cands = []
    for k in chosen:
        act_dir = entries[int(k[1:])].path.rsplit("/", 1)[0]
        sec_entries = [e for e in b.read_index(act_dir) if not e.path.endswith("overview.md")]
        bm = BM25Okapi([tokenize(e.title + " " + e.description) for e in sec_entries])
        scores = bm.get_scores(tokenize(query))
        for i in sorted(range(len(sec_entries)), key=lambda i: -scores[i])[:5]:
            if scores[i] > 0:
                cands.append(sec_entries[i].path)
    # the sections each chosen topic page lists as governing come first
    for t in reversed(state.get("topics", [])):
        cands[0:0] = [l for l in t["links"] if l not in cands]
    uniq = list(dict.fromkeys(cands))[:MAX_CANDIDATES]
    acts = [{"title": entries[int(k[1:])].title, "p": round(probs.get(k, 0), 3)} for k in chosen]
    return {"acts": acts, "candidates": uniq,
            "visited": state.get("visited", []) + ["law/index.md"] + [entries[int(k[1:])].path for k in chosen]}


def _section_record(c, p, via):
    m = c.meta
    text = c.body.split("## Text", 1)[-1].split("## Related", 1)[0].strip()
    return {"act": m.get("act"), "section": str(m.get("section")), "status": m.get("status", ""),
            "replaced_by": m.get("replaced_by", ""), "title": c.title.split(": ", 1)[-1],
            "text": text[:1800], "score": round(p, 3) if p is not None else None, "okf": c.path, "via": via}


def _pick_sections(state):
    """Repeated Choice: pick the most needed section, remove it, ask again, until 'none' wins."""
    eng, b = get_engine(), _bundle()
    focus = _focus(state, with_topics=True)
    concepts = {p: b.read(p) for p in state.get("candidates", [])}
    remaining = list(concepts)
    kept = []
    for _round in range(MAX_SECTIONS):
        if not remaining:
            break
        keys = [f"s{i}" for i in range(len(remaining))] + ["none"]
        criteria, passages = {}, []
        for i, p in enumerate(remaining):
            c = concepts[p]
            short = f"{c.meta.get('act', '').split(',')[0]} s{c.meta.get('section')}: {c.title.split(': ', 1)[-1]}"
            criteria[f"s{i}"] = f"{short}. {_clip(c.description, 90)}"
            passages.append(c.title + ". " + c.body.split("## Text", 1)[-1][:1500])
        criteria["none"] = ("None of the remaining sections is needed to answer the question" if kept
                            else "None of these sections is relevant to the question")
        a = eng.decide({"question": state["question"], "context": focus},
                       {"section": Choice("Which law section is most needed to answer the question?", criteria,
                                          hint={"query": focus, "passages": passages + [None]})})["section"]
        probs = a.probabilities or {a.choice: 1.0}
        if a.choice == "none" or probs.get(a.choice, 0) < MIN_PICK:
            break
        path = remaining.pop(int(a.choice[1:]))
        kept.append(_section_record(concepts[path], probs.get(a.choice), "decision"))

    # curated OKF links are trusted: the primary section of a topic the model chose (its top pick in some
    # decision, or any topic above TRUST_TOPIC) is always kept. Ranking, not raw probability, decides, because
    # small decision models are not calibrated.
    have = {h["okf"] for h in kept}
    for t in state.get("topics", []):
        if (t["links"] and (t.get("won") or t["p"] >= TRUST_TOPIC) and t["links"][0] not in have
                and t["links"][0] in concepts):
            kept.append(_section_record(concepts[t["links"][0]], None, "topic link"))
            have.add(t["links"][0])
    kept.sort(key=lambda h: h["status"] != "active")
    return {"law_hits": kept, "visited": state.get("visited", []) + [h["okf"] for h in kept]}


def _gate(state):
    eng = get_engine()
    law, docs = state.get("law_hits", []), state.get("doc_hits", [])
    if state["route"] == "review" and docs:
        return {"gated": False}
    if not law and not docs:
        return {"gated": True}
    sources = " | ".join(f"{h['act'].split(',')[0]} s{h['section']} {h['title']}: {_clip(h['text'], 150)}" for h in law)
    best = max([h["score"] or 0.5 for h in law], default=0)
    a = eng.decide({"question": state["question"], "sources": _clip(sources, 1600) or "(none)"},
                   {"ok": Noul("The sources contain the information needed to answer the question.",
                               hint={"value": best})})["ok"]
    return {"gated": a.noul < GATE_MIN and not docs}


def _answer(state):
    msgs = answer_messages(state["question"], state.get("doc_hits", []), state.get("law_hits", []),
                           state["route"], state.get("issues"))
    return {"draft": llm.chat(msgs)}


_CLAIM = re.compile(r"[^.\n]*?\[(?:[DL]\d+)\][^.\n]*[.\n]?")


def _verify(state):
    eng = get_engine()
    docs, law = state.get("doc_hits", []), state.get("law_hits", [])
    src = {f"D{i}": d["text"] for i, d in enumerate(docs, 1)}
    src.update({f"L{i}": h["text"] for i, h in enumerate(law, 1)})
    claims = []
    for m in _CLAIM.finditer(state["draft"]):
        sent = re.sub(r"[*_`#]+", "", m.group(0)).strip(" -:\n")
        labels = [x for x in re.findall(r"\[([DL]\d+)\]", sent) if x in src]
        if labels and len(sent) > 25:
            claims.append((re.sub(r"\s*\[[DL]\d+\]", "", sent).strip(), labels))
    claims = claims[:10]
    if not claims:
        return {"issues": [], "retries": state.get("retries", 0), "retry_flag": False}
    reqs = []
    for c, labels in claims:
        sources = _clip(" ".join(src[l] for l in labels), 2400)
        reqs.append(({"claim": c, "sources": sources},
                     {"ok": Choice("Do the sources support the claim?",
                                   {"supported": "The sources state or directly support the claim",
                                    "not_supported": "The sources do not support the claim"},
                                   hint={"premise": sources,
                                         "hypotheses": [c, "The sources say nothing about this: " + c]})}))
    res = eng.decide_many(reqs)
    weak = []
    for (c, l), r in zip(claims, res):
        a = r["ok"]
        if a.choice == "not_supported" and (a.probabilities or {}).get("not_supported", 1) >= UNSUPPORTED_MIN:
            weak.append(f"Possibly unsupported: \"{c[:140]}\" (cites {', '.join(l)})")
    retry = len(weak) > len(claims) / 2 and state.get("retries", 0) == 0
    return {"issues": weak, "retries": state.get("retries", 0) + (1 if retry else 0), "retry_flag": retry}


def _after_verify(state):
    return "answer" if state.get("retry_flag") else "finalize"


def _finalize(state):
    return {"answer": state["draft"]}


def _gated_answer(state):
    topics = ", ".join(t["title"] for t in state.get("topics", [])) or "none found"
    msg = ("I could not find a provision in my legal knowledge base that answers this question.\n\n"
           f"- Topics checked: {topics}\n"
           "- The relevant law may not be in this knowledge base (for example the Industrial Relations Code, "
           "state Shops and Establishments Acts, or company-specific rules).\n\n"
           "Please consult a labour lawyer for this question.\n\n" + config.DISCLAIMER)
    return {"answer": msg}


def _refuse(state):
    return {"answer": REFUSAL, "doc_hits": [], "law_hits": []}


def build_graph():
    g = StateGraph(State)
    nodes = [
        ("route", _route, lambda o: f"{o.get('route')} {o.get('route_probs')}"),
        ("pick_clauses", _pick_clauses, lambda o: ", ".join(f"{d['title']} ({d['score']})" for d in o.get("doc_hits", []))),
        ("pick_topics", _pick_topics, lambda o: ", ".join(f"{t['title']} ({t['p']})" for t in o.get("topics", [])) or "none"),
        ("pick_acts", _pick_acts, lambda o: ", ".join(f"{a['title']} ({a['p']})" for a in o.get("acts", []))
         + f" -> {len(o.get('candidates', []))} candidate sections"),
        ("pick_sections", _pick_sections, lambda o: ", ".join(
            f"{h['act'].split(',')[0]} s{h['section']} ({h['score'] if h['score'] is not None else h['via']})"
            for h in o.get("law_hits", [])) or "none"),
        ("gate", _gate, lambda o: "insufficient sources - answer without LLM" if o.get("gated") else "pass"),
        ("answer", _answer, lambda o: f"{len(o.get('draft', ''))} chars"),
        ("verify", _verify, lambda o: "OK" if not o.get("issues") else f"{len(o['issues'])} weak claim(s)"),
        ("finalize", _finalize, None),
        ("gated_answer", _gated_answer, None),
        ("refuse", _refuse, None),
    ]
    for name, fn, desc in nodes:
        g.add_node(name, metrics.timed_node(name, fn, desc))
    g.set_entry_point("route")
    g.add_conditional_edges("route", lambda s: s["route"], {
        "doc_question": "pick_clauses", "review": "pick_clauses",
        "law_question": "pick_topics", "out_of_scope": "refuse"})
    g.add_edge("pick_clauses", "pick_topics")
    g.add_edge("pick_topics", "pick_acts")
    g.add_edge("pick_acts", "pick_sections")
    g.add_edge("pick_sections", "gate")
    g.add_conditional_edges("gate", lambda s: "gated_answer" if s.get("gated") else "answer",
                            {"gated_answer": "gated_answer", "answer": "answer"})
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", _after_verify, {"answer": "answer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    g.add_edge("gated_answer", END)
    g.add_edge("refuse", END)
    return g.compile()


_GRAPH = None


def run(question, docs=None, stop_after_retrieval=False) -> AgentResult:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    trace = metrics.start("jev_okf")
    final = {"question": question, "docs": list(docs or [])}
    for step in _GRAPH.stream(final, stream_mode="updates"):
        for _node, upd in step.items():
            final.update(upd or {})
        if stop_after_retrieval and ("law_hits" in final or final.get("route") == "out_of_scope"):
            break
    trace.finish()
    doc_hits, law_hits = final.get("doc_hits", []), final.get("law_hits", [])
    res = AgentResult("jev_okf", question, final.get("route", ""), final.get("answer", ""),
                      source_list(doc_hits, law_hits), final.get("issues", []),
                      trace.summary(), trace.steps, gated=bool(final.get("gated")))
    res.metrics["jev_backend"] = get_engine().label
    res.metrics["okf_files_read"] = len(dict.fromkeys(final.get("visited", [])))
    res.okf_path = list(dict.fromkeys(final.get("visited", [])))
    return res
