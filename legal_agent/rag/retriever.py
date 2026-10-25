"""Hybrid retrieval over the statute chunks: dense (multilingual-e5-small) + BM25, fused with RRF."""
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .. import config, metrics
from ..corpus import embedder, load_rows, tokenize


@lru_cache(maxsize=1)
def _index():
    rows = load_rows()
    texts = [f"{r['act']} Section {r['section']}. {r['text']}" for r in rows]
    model = embedder()
    emb = None
    if config.EMB_PATH.exists():
        e = np.load(config.EMB_PATH)
        if e.shape[0] == len(rows):
            emb = e
    if emb is None:
        emb = model.encode(["passage: " + t for t in texts], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=True)
        np.save(config.EMB_PATH, emb)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return rows, emb, bm25


def warm_up():
    _index()


def embed_query(text):
    metrics.record_embed()
    return embedder().encode("query: " + text, normalize_embeddings=True)


def embed_passages(texts):
    metrics.record_embed()
    return embedder().encode(["passage: " + t for t in texts], normalize_embeddings=True)


def search(query, k=5, include_repealed=False):
    rows, emb, bm25 = _index()
    q = embed_query(query)
    allowed = np.array([include_repealed or r["status"] == "active" for r in rows])
    dense = np.where(allowed, emb @ q, -np.inf)
    sparse = np.where(allowed, bm25.get_scores(tokenize(query)), -np.inf)
    score = {}
    for s in (dense, sparse):
        for pos, idx in enumerate(np.argsort(-s)[:50]):
            if allowed[idx]:
                score[idx] = score.get(idx, 0) + 1 / (60 + pos)
    best = sorted(score, key=score.get, reverse=True)[:k]
    return [dict(rows[i], score=round(score[i], 4)) for i in best]
