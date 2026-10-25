"""Pieces shared by both pipelines so the comparison isolates retrieval and decision-making:
the same documents, the same answer-writing prompt, the same LLM and the same output format."""
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..documents import chunk_text, load_document

ROUTES = ("doc_question", "review", "law_question", "out_of_scope")

ANSWER_SYSTEM = f"""You are a careful legal assistant ("law agent") that explains Indian employment law and the
user's own documents (contracts, offer letters, HR policies) in plain, simple English.

You get two kinds of sources:
 [D1], [D2]... = excerpts from the user's own document(s)
 [L1], [L2]... = sections of Indian law (marked active or repealed)

Rules:
1. Use ONLY these sources. No outside knowledge.
2. Cite after every claim, e.g. [D2] or [L3]. Use just the label inside the brackets.
3. If a document clause may conflict with the law, say "this may conflict with" and explain why, quoting the
   relevant condition. Never call a clause simply "illegal" or "void".
4. State the conditions and exceptions found in the law text. Do not present a rule as universal if the text has exceptions.
5. Prefer active laws. If you use a repealed law, say it has been replaced.
6. If the sources are not enough, say exactly what is missing and suggest consulting a labour lawyer.
7. Do not predict court outcomes or tell the user what to do legally. Do not add tests or standards the text does not state.
8. Structure: a direct answer first (1-2 sentences), then short bullet points with citations.
9. End with exactly: {config.DISCLAIMER}"""

REVIEW_EXTRA = ("\n\nThis is a document review: go clause by clause through the [D] sources, and for each clause that "
                "may conflict with a cited law, explain the possible conflict. Say plainly which clauses raise no issue "
                "in these sources.")

REFUSAL = ("I can only help with questions about Indian employment law and the documents you upload "
           "(contracts, offer letters, HR policies).\n\n" + config.DISCLAIMER)


@dataclass
class LoadedDocument:
    name: str
    path: str
    text: str
    chunks: list = field(default_factory=list)
    emb: object = None            # chunk embeddings (RAG pipeline)
    okf_slug: str = ""            # documents/<slug> in the OKF bundle (JEV+OKF pipeline)
    n_clauses: int = 0


def load_user_document(path, display_name=None):
    path = str(path)
    name = display_name or Path(path).name
    text = load_document(path)
    return LoadedDocument(name=name, path=path, text=text, chunks=chunk_text(text))


def build_sources(doc_hits, law_hits):
    parts = []
    for i, d in enumerate(doc_hits, 1):
        parts.append(f"[D{i}] (user's document: {d['doc']}{', ' + d['title'] if d.get('title') else ''})\n{d['text']}")
    for i, h in enumerate(law_hits, 1):
        status = h["status"] + (f", replaced by {h['replaced_by']}" if h.get("replaced_by") else "")
        parts.append(f"[L{i}] {h['act']}, Section {h['section']} ({status})\n{h['text']}")
    return "\n\n".join(parts)


def answer_messages(question, doc_hits, law_hits, route, issues=None):
    system = ANSWER_SYSTEM + (REVIEW_EXTRA if route == "review" else "")
    extra = ""
    if issues:
        extra = "\n\nYour previous draft had these problems, fix them:\n- " + "\n- ".join(issues)
    if not law_hits:
        extra += ("\n\nNOTE: there are no [L] law sources for this question. Do not cite [L] labels and do not state "
                  "what the law says; explain only what the document says and that the governing law was not found.")
    user = f"{build_sources(doc_hits, law_hits)}\n\nQuestion: {question}{extra}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def source_list(doc_hits, law_hits):
    out = []
    for i, d in enumerate(doc_hits, 1):
        out.append({"label": f"D{i}", "kind": "document", "title": f"{d['doc']}" + (f" - {d['title']}" if d.get("title") else ""),
                    "text": d["text"], "score": d.get("score")})
    for i, h in enumerate(law_hits, 1):
        out.append({"label": f"L{i}", "kind": "law", "act": h["act"], "section": h["section"], "status": h["status"],
                    "title": f"{h['act']}, Section {h['section']}" + (f" - {h['title']}" if h.get("title") else ""),
                    "text": h["text"], "score": h.get("score")})
    return out


@dataclass
class AgentResult:
    pipeline: str
    question: str
    route: str
    answer: str
    sources: list
    issues: list
    metrics: dict
    steps: list
    gated: bool = False           # answered "not enough information" without calling the LLM
    okf_path: list = field(default_factory=list)   # OKF files the JEV+OKF agent read, in order
