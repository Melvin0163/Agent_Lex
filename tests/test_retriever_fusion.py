"""Hybrid retrieval: reciprocal rank fusion of dense and BM25 rankings, tested with a tiny fake index
(no embedding model is loaded)."""
import numpy as np
from rank_bm25 import BM25Okapi

from legal_agent.corpus import tokenize
from legal_agent.rag import retriever

ROWS = [
    {"act": "Indian Contract Act, 1872", "section": "27", "status": "active", "text": "agreement in restraint of trade void"},
    {"act": "Code on Wages, 2019", "section": "17", "status": "active", "text": "time limit for payment of wages"},
    {"act": "Payment of Wages Act, 1936", "section": "5", "status": "repealed", "text": "time limit payment of wages"},
]


def fake_index(monkeypatch, query_vec, emb):
    bm25 = BM25Okapi([tokenize(f"{r['act']} Section {r['section']}. {r['text']}") for r in ROWS])
    monkeypatch.setattr(retriever, "_index", lambda: (ROWS, np.array(emb, float), bm25))
    monkeypatch.setattr(retriever, "embed_query", lambda q: np.array(query_vec, float))


def test_rank_agreement_wins_and_repealed_acts_are_excluded(monkeypatch):
    # dense and keyword both favour the wages sections; the repealed one must not be returned
    fake_index(monkeypatch, [0, 1], [[1, 0], [0, 1], [0, 1]])
    hits = retriever.search("time limit for payment of wages", k=3)
    assert [h["section"] for h in hits] == ["17", "27"]
    assert all(h["status"] == "active" for h in hits)


def test_include_repealed(monkeypatch):
    fake_index(monkeypatch, [0, 1], [[1, 0], [0, 1], [0, 1]])
    hits = retriever.search("time limit payment of wages", k=3, include_repealed=True)
    assert {h["section"] for h in hits} == {"17", "5", "27"}


def test_fused_scores_are_reciprocal_ranks(monkeypatch):
    fake_index(monkeypatch, [1, 0], [[1, 0], [0, 1], [0, 1]])
    top = retriever.search("restraint of trade", k=1)[0]
    assert top["section"] == "27"
    assert top["score"] == round(2 / 60, 4)    # ranked first by both dense and BM25: 1/60 + 1/60
