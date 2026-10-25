"""Embedding projector: see the knowledge base the way the RAG engine sees it, and compare what each engine picks.

Points on the map
  law chunk     - the 842 statute chunks in the RAG vector index (multilingual-e5-small, 384 dimensions)
  OKF topic     - the 16 curated topic pages of the OKF bundle, embedded the same way
  test question - the 18 benchmark questions (embedded as queries)

Projections: PCA (exact, new questions can be projected) and t-SNE (local neighbourhoods; a new question is
placed at the similarity-weighted position of its nearest chunks). Files for Google's Embedding Projector
(projector.tensorflow.org -> Load) are written to reports/projector/.
"""
import csv
import json
from functools import lru_cache

import numpy as np

from . import config
from .corpus import embedder, load_rows, section_title, sections
from .okf import OKFBundle
from .okf.builder import act_dir, sec_file
from .okf.topics import TOPICS

OUT = config.REPORTS_DIR / "projector"

SHORT_ACT = {
    "Code on Wages, 2019": "Code on Wages 2019",
    "Code on Social Security Act, 2020": "Social Security Code 2020",
    "Indian Contract Act, 1872": "Contract Act 1872",
    "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013": "POSH Act 2013",
    "Payment of Gratuity Act, 1972": "Gratuity Act 1972",
    "Maternity Benefit Act, 1961": "Maternity Benefit Act 1961",
    "Employees Provident Funds and Miscellaneous Provisions Act, 1952": "EPF Act 1952",
    "Payment of Wages Act, 1936": "Payment of Wages Act 1936",
    "Minimum Wages Act, 1948": "Minimum Wages Act 1948",
    "Industrial Disputes Act, 1947": "Industrial Disputes Act 1947",
    "Industrial Employment (Standing Orders) Act, 1946": "Standing Orders Act 1946",
}


def _topic_membership():
    """(act, section) -> [topic titles that list it as a governing section]."""
    m = {}
    for _slug, title, _d, _a, refs, _n in TOPICS:
        for ref in refs:
            m.setdefault(ref, []).append(title)
    return m


@lru_cache(maxsize=1)
def build(force=False):
    """Compute (or load) all points, vectors and projections. Returns a dict."""
    cache = OUT / "projection.npz"
    meta_path = OUT / "points.json"
    rows = load_rows()
    if cache.exists() and meta_path.exists() and not force:
        z = np.load(cache)
        points = json.loads(meta_path.read_text(encoding="utf-8"))
        n_real = sum(1 for l in open(config.DATA_DIR / "qa_real.jsonl", encoding="utf-8") if l.strip()) \
            if (config.DATA_DIR / "qa_real.jsonl").exists() else 0
        n_q = sum(p["kind"] == "test question" for p in points)
        if (len(points) == z["vectors"].shape[0] and sum(p["kind"] == "law chunk" for p in points) == len(rows)
                and n_q == 18 + n_real):
            return _pack(points, z)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    model = embedder()
    chunk_vecs = np.load(config.EMB_PATH)
    if chunk_vecs.shape[0] != len(rows):
        texts = [f"{r['act']} Section {r['section']}. {r['text']}" for r in rows]
        chunk_vecs = model.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=32)

    secs = sections()
    titles = {k: section_title(k[0], k[1], v["text"]) for k, v in secs.items()}
    membership = _topic_membership()
    points = []
    for i, r in enumerate(rows):
        key = (r["act"], str(r["section"]))
        points.append({
            "kind": "law chunk", "idx": i, "act": r["act"], "act_short": SHORT_ACT.get(r["act"], r["act"]),
            "section": str(r["section"]), "title": titles.get(key, ""), "status": r["status"],
            "topics": membership.get(key, []),
            "okf": f"law/{act_dir(r['act'])}/{sec_file(str(r['section']), titles.get(key, ''))}",
            "text": r["text"][:400].replace("\n", " "),
        })

    topic_texts = [f"{t}. {d} Also called: {', '.join(a)}" for _s, t, d, a, _r, _n in TOPICS]
    topic_vecs = model.encode(["passage: " + t for t in topic_texts], normalize_embeddings=True)
    for (_s, t, d, a, refs, _n), txt in zip(TOPICS, topic_texts):
        points.append({"kind": "OKF topic", "act": "", "act_short": "OKF topic page", "section": "", "title": t,
                       "status": "", "topics": [t], "okf": f"topics/{_s}.md", "text": txt[:400],
                       "links": [f"{x[0]}|{x[1]}" for x in refs]})

    tests = [json.loads(l) for l in open(config.SAMPLES_DIR / "qa_testset.jsonl", encoding="utf-8") if l.strip()]
    real_qs = config.DATA_DIR / "qa_real.jsonl"          # real-document test set, if present
    if real_qs.exists():
        tests += [json.loads(l) for l in open(real_qs, encoding="utf-8") if l.strip()]
    q_vecs = model.encode(["query: " + q["question"] for q in tests], normalize_embeddings=True)
    for q in tests:
        points.append({"kind": "test question", "act": q.get("expected_act") or "", "act_short": "Test question",
                       "section": q.get("expected_section") or "", "title": q["question"], "status": "",
                       "topics": [], "okf": "", "text": q["question"], "id": q["id"]})

    vectors = np.vstack([chunk_vecs, topic_vecs, q_vecs]).astype(np.float32)
    pca = PCA(n_components=3, random_state=0).fit(vectors)
    p3 = pca.transform(vectors)
    t2 = TSNE(n_components=2, perplexity=30, init="pca", random_state=42, metric="cosine").fit_transform(vectors)
    t3 = TSNE(n_components=3, perplexity=30, init="pca", random_state=42, metric="cosine").fit_transform(vectors)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vectors=vectors, pca3=p3, tsne2=t2, tsne3=t3,
                        pca_components=pca.components_, pca_mean=pca.mean_,
                        pca_var=pca.explained_variance_ratio_)
    meta_path.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")
    export_tsv(points, vectors)
    return _pack(points, np.load(cache))


def _pack(points, z):
    return {"points": points, "vectors": z["vectors"], "pca3": z["pca3"], "tsne2": z["tsne2"], "tsne3": z["tsne3"],
            "pca_components": z["pca_components"], "pca_mean": z["pca_mean"], "pca_var": z["pca_var"]}


def export_tsv(points, vectors):
    """Files for https://projector.tensorflow.org (Load -> vectors.tsv, then metadata.tsv)."""
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / "vectors.tsv", vectors, delimiter="\t", fmt="%.5f")
    with open(OUT / "metadata.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["label", "kind", "act", "section", "status", "okf_topics", "text"])
        for p in points:
            label = (f"{p['act_short']} s{p['section']}: {p['title']}" if p["kind"] == "law chunk" else p["title"])[:90]
            w.writerow([label, p["kind"], p["act_short"], p["section"], p["status"],
                        "; ".join(p.get("topics", [])) or "none", p["text"][:200].replace("\t", " ")])


def locate_query(data, question, k=8):
    """Embed a question, find its nearest chunks, and place it on each projection."""
    q = embedder().encode("query: " + question, normalize_embeddings=True).astype(np.float32)
    n_chunks = sum(p["kind"] == "law chunk" for p in data["points"])
    sims = data["vectors"][:n_chunks] @ q
    nn = np.argsort(-sims)[:k]
    w = np.clip(sims[nn], 1e-6, None) ** 8          # sharpen: the closest chunks dominate the placement
    w = w / w.sum()
    pos = {
        "pca3": (q - data["pca_mean"]) @ data["pca_components"].T,
        "tsne2": (data["tsne2"][nn] * w[:, None]).sum(0),
        "tsne3": (data["tsne3"][nn] * w[:, None]).sum(0),
    }
    return {"vector": q, "sims": sims, "nearest": nn.tolist(), "pos": pos}


def chunks_of(data, act, section):
    return [i for i, p in enumerate(data["points"]) if p["kind"] == "law chunk" and p["act"] == act and p["section"] == str(section)]


def comparison_table(data, located, rag_sources, okf_sources):
    """One row per statute section chosen by either engine, with its similarity to the question."""
    sims = located["sims"]
    order = {i: r for r, i in enumerate(np.argsort(-sims))}
    keys = []
    for s in rag_sources + okf_sources:
        if s["kind"] == "law" and (s["act"], str(s["section"])) not in keys:
            keys.append((s["act"], str(s["section"])))
    rag_k = {(s["act"], str(s["section"])) for s in rag_sources if s["kind"] == "law"}
    okf_k = {(s["act"], str(s["section"])) for s in okf_sources if s["kind"] == "law"}
    out = []
    for act, sec in keys:
        idx = chunks_of(data, act, sec)
        best = max(idx, key=lambda i: sims[i]) if idx else None
        out.append({
            "section": f"{SHORT_ACT.get(act, act)} s{sec}",
            "title": data["points"][idx[0]]["title"] if idx else "",
            "picked by RAG": (act, sec) in rag_k, "picked by JEV + OKF": (act, sec) in okf_k,
            "cosine to question": round(float(sims[best]), 3) if best is not None else None,
            "rank by similarity (of 842)": order.get(best, None) + 1 if best is not None else None,
        })
    return out
