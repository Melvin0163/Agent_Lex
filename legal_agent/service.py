"""Facade used by the UI, CLI and evaluation: upload documents once, ask either pipeline."""
import time
from pathlib import Path

from . import config
from .okf import build_law_bundle, compile_document
from .pipelines import jev_okf_agent, rag_agent
from .pipelines.common import load_user_document

PIPELINES = {
    "rag": ("Agentic RAG", rag_agent.run),
    "jev_okf": ("JEV + OKF", jev_okf_agent.run),
}


def ensure_bundle():
    if not (config.KNOWLEDGE_DIR / "law" / "index.md").exists():
        build_law_bundle(verbose=False)


def save_upload(name, data: bytes) -> Path:
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in Path(name).name)
    p = config.UPLOADS_DIR / safe
    p.write_bytes(data)
    return p


def prepare_document(path, display_name=None):
    """Load a document for both pipelines: text chunks for RAG, an OKF sub-bundle for JEV+OKF."""
    ensure_bundle()
    t0 = time.time()
    doc = load_user_document(path, display_name)
    doc.okf_slug, info = compile_document(path, display_name=doc.name)
    doc.n_clauses = info["clauses"]
    return doc, round(time.time() - t0, 2)


def ask(pipeline, question, docs=None, stop_after_retrieval=False):
    ensure_bundle()
    return PIPELINES[pipeline][1](question, docs or [], stop_after_retrieval=stop_after_retrieval)
