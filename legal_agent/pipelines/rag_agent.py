"""Pipeline A - agentic RAG.

route (LLM) -> retrieve_doc (embeddings) -> rewrite queries (LLM) -> retrieve_law (BM25 + dense)
            -> answer (LLM) -> verify (LLM fact-check, one retry) -> finalize
Every decision is made by the generative LLM; relevance comes from vector/BM25 similarity.
"""
import re
from typing import TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from .. import llm, metrics
from ..rag.retriever import embed_passages, embed_query, search
from .common import REFUSAL, AgentResult, answer_messages, build_sources, source_list

REVIEW_TOPICS = [
    "notice period and termination of employment",
    "non-compete or restriction after leaving the job",
    "training bond or service bond, liquidated damages",
    "salary, deductions and final settlement of dues",
    "probation period",
    "gratuity, provident fund and other benefits",
    "working hours, overtime and leave",
    "sexual harassment complaint procedure",
    "maternity leave",
]

DOC_WORDS = re.compile(r"\b(my|this|the|our)\s+(contract|agreement|offer|clause|document|letter|policy|bond)\b|\bclause\b"
                       r"|\b(in|under) (my|our|this|the) (agreement|contract|policy)\b", re.I)


class State(TypedDict, total=False):
    question: str
    docs: list
    route: str
    doc_hits: list
    queries: list
    law_hits: list
    draft: str
    issues: list
    retries: int
    answer: str


def _route(state):
    has_doc = bool(state["docs"])
    d = llm.ask_json(
        'Classify the user question. Reply ONLY with JSON: {"route": "..."}.\n'
        'Options: "doc_question" (asks about terms in their uploaded document), '
        '"review" (asks to review/check the whole document or its legality), '
        '"law_question" (general Indian employment law question), '
        '"out_of_scope" (not about employment law or the document).',
        f"Document uploaded: {has_doc}\nQuestion: {state['question']}")
    r = d.get("route", "law_question")
    if r not in ("doc_question", "review", "law_question", "out_of_scope"):
        r = "law_question"
    if has_doc and r == "law_question" and DOC_WORDS.search(state["question"]):
        r = "doc_question"
    if not has_doc and r in ("doc_question", "review"):
        r = "law_question"
    return {"route": r, "retries": 0}


def _retrieve_doc(state):
    queries = REVIEW_TOPICS if state["route"] == "review" else [state["question"]]
    per = 2 if state["route"] == "review" else 3
    hits, seen = [], set()
    for doc in state["docs"]:
        if doc.emb is None:
            doc.emb = embed_passages(doc.chunks)
        for q in queries:
            sims = doc.emb @ embed_query(q)
            for i in np.argsort(-sims)[:per]:
                key = (doc.name, int(i))
                if key not in seen:
                    seen.add(key)
                    hits.append({"doc": doc.name, "idx": int(i), "text": doc.chunks[i], "score": round(float(sims[i]), 3)})
    hits.sort(key=lambda h: (h["doc"], h["idx"]))
    return {"doc_hits": hits[:8]}


def _rewrite(state):
    excerpt = "\n---\n".join(h["text"] for h in state.get("doc_hits", []))[:3000]
    d = llm.ask_json(
        'You turn a question into 3 to 4 short search queries that use the wording of Indian statutes. '
        'Reply ONLY with JSON: {"queries": ["...", "..."]}.\n'
        'Vocabulary hints: non-compete -> "agreement in restraint of trade" (Indian Contract Act Section 27); '
        'notice period/termination -> "termination of employment"; '
        'final settlement -> "time limit for payment of wages on removal, dismissal, retrenchment or resignation, within two working days"; '
        'bond -> "compensation for breach of contract where penalty stipulated, liquidated damages"; '
        'deductions -> "deductions which may be made from wages"; gratuity -> "payment of gratuity continuous service fixed term"; '
        'harassment -> "sexual harassment internal committee complaint"; maternity -> "right to payment of maternity benefit".',
        f"Question: {state['question']}\n\nRelevant excerpts from the user's document (may be empty):\n{excerpt}")
    qs = [q for q in d.get("queries", []) if isinstance(q, str)][:4]
    return {"queries": qs + [state["question"]]}


def _retrieve_law(state):
    merged = {}
    for q in state["queries"]:
        for h in search(q, k=4):
            key = (h["act"], h["section"])
            if key in merged:
                if h["text"] not in merged[key]["text"]:
                    merged[key]["text"] += "\n" + h["text"]
                merged[key]["score"] = max(merged[key]["score"], h["score"])
            else:
                merged[key] = dict(h)
    hits = list(merged.values())[:8]
    for h in hits:
        h["text"] = h["text"][:1800]
    return {"law_hits": hits}


def _answer(state):
    msgs = answer_messages(state["question"], state.get("doc_hits", []), state.get("law_hits", []),
                           state["route"], state.get("issues"))
    return {"draft": llm.chat(msgs)}


def _verify(state):
    d = llm.ask_json(
        'You are a strict fact checker. Check whether every claim in the DRAFT is supported by the SOURCES, '
        'and whether any rule is stated as universal when the sources list exceptions. '
        'Reply ONLY with JSON: {"supported": true/false, "issues": ["short description", ...]}.',
        f"SOURCES:\n{build_sources(state.get('doc_hits', []), state.get('law_hits', []))}\n\nDRAFT:\n{state['draft']}")
    issues = [i for i in d.get("issues", []) if isinstance(i, str)]
    ok = bool(d.get("supported", True)) or not issues
    return {"issues": [] if ok else issues, "retries": state.get("retries", 0) + (0 if ok else 1)}


def _after_verify(state):
    return "answer" if state.get("issues") and state["retries"] <= 1 else "finalize"


def _finalize(state):
    return {"answer": state["draft"]}


def _refuse(state):
    return {"answer": REFUSAL, "doc_hits": [], "law_hits": []}


def build_graph():
    g = StateGraph(State)
    nodes = [
        ("route", _route, lambda o: o.get("route", "")),
        ("retrieve_doc", _retrieve_doc, lambda o: f"{len(o.get('doc_hits', []))} chunks"),
        ("rewrite", _rewrite, lambda o: " | ".join(o.get("queries", []))),
        ("retrieve_law", _retrieve_law, lambda o: ", ".join(f"{h['act'].split(',')[0]} s{h['section']}" for h in o.get("law_hits", []))),
        ("answer", _answer, lambda o: f"{len(o.get('draft', ''))} chars"),
        ("verify", _verify, lambda o: "OK" if not o.get("issues") else "issues: " + "; ".join(o["issues"])[:300]),
        ("finalize", _finalize, None),
        ("refuse", _refuse, None),
    ]
    for name, fn, desc in nodes:
        g.add_node(name, metrics.timed_node(name, fn, desc))
    g.set_entry_point("route")
    g.add_conditional_edges("route", lambda s: s["route"], {
        "doc_question": "retrieve_doc", "review": "retrieve_doc",
        "law_question": "rewrite", "out_of_scope": "refuse"})
    g.add_edge("retrieve_doc", "rewrite")
    g.add_edge("rewrite", "retrieve_law")
    g.add_edge("retrieve_law", "answer")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", _after_verify, {"answer": "answer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    g.add_edge("refuse", END)
    return g.compile()


_GRAPH = None


def run(question, docs=None, stop_after_retrieval=False) -> AgentResult:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    trace = metrics.start("rag")
    final = {"question": question, "docs": list(docs or [])}
    for step in _GRAPH.stream(final, stream_mode="updates"):
        for _node, upd in step.items():
            final.update(upd or {})
        if stop_after_retrieval and ("law_hits" in final or final.get("route") == "out_of_scope"):
            break
    trace.finish()
    doc_hits, law_hits = final.get("doc_hits", []), final.get("law_hits", [])
    return AgentResult("rag", question, final.get("route", ""), final.get("answer", ""),
                       source_list(doc_hits, law_hits), final.get("issues", []),
                       trace.summary(), trace.steps)
