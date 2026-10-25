"""Figures that document the RAG vs JEV + OKF comparison.

  python scripts/make_figures.py                       # uses reports/eval_full.json, else eval_retrieval.json
  python scripts/make_figures.py --eval reports/eval_retrieval.json

Needs reports/projector/ (python scripts/build_projector.py) and a benchmark report (python scripts/evaluate.py).
Writes PNGs to reports/figures/ (and an interactive knowledge_map.html if plotly is installed).

  fig1_knowledge_map.png     the 842 statute chunks in embedding space (t-SNE), coloured by Act, OKF topic pages marked
  fig2_question_maps.png     per question: the question, what each engine picked, and the expected section
  fig3_similarity_rank.png   how close (by cosine rank) each engine's picks are to the question
  fig4_benchmark.png         quality and cost metrics, one panel per metric
  fig5_step_time.png         average seconds per pipeline step
  fig6_projection_tsne.png   each engine's retrieval footprint, t-SNE projection
  fig7_projection_pca.png    the same on the PCA projection
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJ = ROOT / "reports" / "projector"
OUT = ROOT / "reports" / "figures"

# Reference palette (validated categorical order). Engines take slots 1-2; Acts take slots 1-4 with marker shapes.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ENGINE = {"rag": ("Agentic RAG", "#2a78d6", "o"), "jev_okf": ("JEV + OKF", "#eb6834", "D")}
ACT_STYLE = {
    "Code on Social Security Act, 2020": ("Social Security Code 2020", "#2a78d6", "o"),
    "Code on Wages, 2019": ("Code on Wages 2019", "#eb6834", "s"),
    "Indian Contract Act, 1872": ("Contract Act 1872", "#1baf7a", "^"),
    "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013": ("POSH Act 2013", "#6250d6", "D"),
}
REPEALED = ("Repealed Acts (7)", "#c3c2b7", "o")
SRC = re.compile(r"^(?P<act>.+?), Section (?P<sec>[0-9]+[A-Z]?)\b")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def load(eval_path):
    pts = json.loads((PROJ / "points.json").read_text(encoding="utf-8"))
    z = np.load(PROJ / "projection.npz")
    ev = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    return pts, z, ev


def chunk_index(pts):
    """(act, section) -> list of point indices of that section's chunks."""
    idx = {}
    for i, p in enumerate(pts):
        if p["kind"] == "law chunk":
            idx.setdefault((p["act"], p["section"]), []).append(i)
    return idx


def picks(row, pipe):
    out = []
    for t in row.get(f"{pipe}_sources") or []:
        m = SRC.match(t)
        if m and (m["act"], m["sec"]) not in out:
            out.append((m["act"], m["sec"]))
    return out


def question_point(pts, qid):
    for i, p in enumerate(pts):
        if p["kind"] == "test question" and p.get("id") == qid:
            return i
    return None


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def background(ax, pts, xy, size=6, alpha=0.9, grey=False):
    """Plot the 842 chunks: active Acts coloured (shape-coded), repealed Acts grey."""
    groups = {}
    for i, p in enumerate(pts):
        if p["kind"] != "law chunk":
            continue
        key = p["act"] if (p["status"] == "active" and p["act"] in ACT_STYLE) else "_repealed"
        groups.setdefault(key, []).append(i)
    handles = []
    for key in ["_repealed"] + list(ACT_STYLE):
        ids = groups.get(key, [])
        if not ids:
            continue
        label, colour, marker = REPEALED if key == "_repealed" else ACT_STYLE[key]
        if grey:
            colour = "#e1e0d9" if key == "_repealed" else "#d4d3cc"
        h = ax.scatter(xy[ids, 0], xy[ids, 1], s=size, c=colour, marker=marker, linewidths=0, alpha=alpha,
                       label=f"{label} ({len(ids)} chunks)", zorder=1)
        handles.append(h)
    return groups, handles


# ---------------------------------------------------------------- fig 1
def fig_knowledge_map(pts, z):
    xy = z["tsne2"]
    fig, ax = plt.subplots(figsize=(11, 8.2))
    groups, handles = background(ax, pts, xy, size=12)
    # direct labels at each active Act's centroid (identity is never colour alone)
    for key, (label, colour, _m) in ACT_STYLE.items():
        ids = groups.get(key, [])
        if ids:
            cx, cy = np.median(xy[ids], axis=0)
            ax.annotate(label, (cx, cy), fontsize=10, fontweight="bold", color=INK, ha="center",
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=colour, lw=1.2, alpha=0.92), zorder=5)
    topics = [i for i, p in enumerate(pts) if p["kind"] == "OKF topic"]
    ax.scatter(xy[topics, 0], xy[topics, 1], s=90, marker="*", c=INK, edgecolors=SURFACE, linewidths=0.8,
               label="OKF topic page (16)", zorder=4)
    for n, i in enumerate(topics, 1):
        ax.annotate(str(n), xy[i], xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK, zorder=6)
    key = "\n".join(f"{n:>2}  " + (pts[i]['title'] if len(pts[i]['title']) <= 42 else pts[i]['title'][:41] + "…") for n, i in enumerate(topics, 1))
    ax.text(1.01, 0.62, "OKF topic pages\n" + key, transform=ax.transAxes, fontsize=8, color=INK2, va="top",
            family="monospace", linespacing=1.35)
    ax.set_title("The knowledge base as the RAG engine sees it: 842 statute chunks in embedding space (t-SNE)", loc="left")
    ax.text(0, -0.04, "Each point is one chunk embedded with multilingual-e5-small (384 dimensions), projected to 2D. "
            "Chunks of the same Act cluster together. Stars are the 16 curated OKF topic pages the JEV + OKF engine "
            "navigates by; embedded the same way, they sit close to each other, near the Code on Wages, whichever Act they point to.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top", wrap=True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, markerscale=1.6, handletextpad=0.4)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_knowledge_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_question_maps(pts, z, rows):
    xy = z["tsne2"]
    cidx = chunk_index(pts)
    scored = [r for r in rows if r.get("expected_act")]
    n = len(scored)
    cols = 3
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 4.6, rws * 4.1))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        r = scored[k]
        background(ax, pts, xy, size=3, alpha=1, grey=True)
        qi = question_point(pts, r["id"])
        exp = cidx.get((r["expected_act"], str(r["expected_section"])), [])
        if exp:
            ax.scatter(xy[exp, 0], xy[exp, 1], s=170, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=3)
        focus = list(exp)
        for pipe, (label, colour, marker) in ENGINE.items():
            for key in picks(r, pipe):
                ids = cidx.get(key, [])
                if not ids:
                    continue
                pos = xy[ids].mean(axis=0)
                focus.extend(ids)
                if qi is not None:
                    ax.plot([xy[qi, 0], pos[0]], [xy[qi, 1], pos[1]], color=colour, lw=0.8, alpha=0.55, zorder=2)
                off = (-3, 0) if pipe == "rag" else (3, 0)
                ax.scatter(pos[0] + off[0], pos[1] + off[1], s=46, marker=marker, c=colour, edgecolors=SURFACE,
                           linewidths=1.2, zorder=4)
        if qi is not None:
            ax.scatter(xy[qi, 0], xy[qi, 1], s=200, marker="*", c=INK, edgecolors=SURFACE, linewidths=1, zorder=5)
            focus.append(qi)
        if focus:   # zoom to the region that matters, with margin
            f = xy[focus]
            lo, hi = f.min(axis=0), f.max(axis=0)
            pad = np.maximum((hi - lo) * 0.25, 8)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        hit = {p: r.get(f"{p}_hit") for p in ENGINE}
        mark = lambda v: "found" if v else "missed"  # noqa: E731
        q = r["question"] if len(r["question"]) <= 52 else r["question"][:50] + "..."
        ax.set_title(f"{r['id']}  {q}", loc="left", fontsize=9)
        sec = str(r["expected_section"])
        sec = "s" + sec if not sec.startswith("verify") else "(section to verify)"
        act_short = ACT_STYLE.get(r["expected_act"], (r["expected_act"].split(",")[0],))[0]
        ax.text(0.0, -0.02, f"Expected: {act_short} {sec}   "
                f"RAG {mark(hit['rag'])}, JEV + OKF {mark(hit['jev_okf'])}",
                transform=ax.transAxes, fontsize=7.8, color=INK2, va="top")
        clean_axes(ax)
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], marker="*", ls="", ms=13, color=INK, label="the question"),
              Line2D([], [], marker="o", ls="", ms=12, mfc="none", mec=INK, mew=1.6, label="expected section"),
              Line2D([], [], marker="o", ls="-", lw=0.8, ms=7, color=ENGINE["rag"][1], label="section picked by Agentic RAG"),
              Line2D([], [], marker="D", ls="-", lw=0.8, ms=6, color=ENGINE["jev_okf"][1], label="section picked by JEV + OKF"),
              Line2D([], [], marker="o", ls="", ms=4, color="#d4d3cc", label="other statute chunks")]
    fig.legend(handles=legend, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle("What each engine retrieved, shown on the embedding map (zoomed per question)", x=0.01, ha="left",
                 y=1.035, fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "fig2_question_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_similarity_rank(pts, z, rows):
    vec = z["vectors"]
    cidx = chunk_index(pts)
    n_chunks = sum(p["kind"] == "law chunk" for p in pts)
    data = {p: [] for p in ENGINE}
    for r in rows:
        qi = question_point(pts, r["id"])
        if qi is None:
            continue
        sims = vec[:n_chunks] @ vec[qi]
        rank = np.empty(n_chunks, int)
        rank[np.argsort(-sims)] = np.arange(1, n_chunks + 1)
        exp = (r.get("expected_act"), str(r.get("expected_section")))
        for pipe in ENGINE:
            for key in picks(r, pipe):
                ids = cidx.get(key, [])
                if ids:
                    data[pipe].append((int(rank[ids].min()), key == exp))
    if not any(data.values()):
        return
    fig, ax = plt.subplots(figsize=(10, 3.9))
    rng = np.random.default_rng(0)
    ylab = []
    for row_i, (pipe, (label, colour, marker)) in enumerate(ENGINE.items()):
        vals = data[pipe]
        if not vals:
            continue
        y = row_i + rng.uniform(-0.18, 0.18, len(vals))
        xs = np.array([v[0] for v in vals])
        good = np.array([v[1] for v in vals])
        ax.scatter(xs[~good], y[~good], s=30, marker=marker, c=colour, alpha=0.55, edgecolors=SURFACE, linewidths=0.8)
        ax.scatter(xs[good], y[good], s=80, marker=marker, c=colour, edgecolors=INK, linewidths=1.3)
        med = float(np.median(xs))
        ax.plot([med, med], [row_i - 0.3, row_i + 0.3], color=INK, lw=1.6)
        ax.annotate(f"median rank {med:.0f}", (med, row_i + 0.32), fontsize=8.5, color=INK, ha="center")
        ylab.append(f"{label}\n{len(vals)} sections picked")
    ax.set_xscale("log")
    ax.set_xlim(0.8, n_chunks * 1.3)
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab, color=INK, fontsize=9.5)
    ax.set_ylim(-0.6, len(ylab) - 0.35)
    ax.set_xlabel(f"Rank of the picked section by cosine similarity to the question (1 = nearest of {n_chunks} chunks, log scale)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("How close each picked section is to the question in embedding space (cosine rank)",
                 loc="left")
    ax.text(0, -0.3, "Outlined marks are the expected (correct) section for that question.", transform=ax.transAxes,
            fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_similarity_rank.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_benchmark(ev):
    summ = ev.get("summary") or {}
    pipes = [p for p in ENGINE if p in summ]
    if not pipes:
        return
    metrics = list(summ[pipes[0]].keys())
    quality = [m for m in metrics if "%" in m]
    cost = [m for m in metrics if m.startswith("Avg")]
    panels = [("Quality (higher is better)", quality, "%")] + [(m, [m], "") for m in cost]
    ncols = 3
    nrows = 1 + int(np.ceil(len(cost) / ncols))
    fig = plt.figure(figsize=(12, 3.2 + 2.3 * (nrows - 1)))
    gs = fig.add_gridspec(nrows, ncols, height_ratios=[1.5] + [1] * (nrows - 1), hspace=0.75, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    yy = np.arange(len(quality))
    h = 0.36
    for j, pipe in enumerate(pipes):
        label, colour, _m = ENGINE[pipe]
        vals = [summ[pipe][m] if summ[pipe][m] is not None else np.nan for m in quality]
        ax.barh(yy + (j - 0.5) * h, vals, height=h - 0.04, color=colour, label=label)
        for y, v in zip(yy + (j - 0.5) * h, vals):
            if not np.isnan(v):
                ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([m.replace(" (%)", "") for m in quality], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Quality (higher is better)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    for k, m in enumerate(cost):
        a = fig.add_subplot(gs[1 + k // ncols, k % ncols])
        vals = [summ[p][m] or 0 for p in pipes]
        a.bar(range(len(pipes)), vals, color=[ENGINE[p][1] for p in pipes], width=0.6)
        for x, v in enumerate(vals):
            a.text(x, v, f"{v:,.1f}" if v < 100 else f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        a.set_xticks(range(len(pipes)))
        a.set_xticklabels([ENGINE[p][0] for p in pipes], fontsize=8.5, color=INK2)
        t = m.replace("Avg ", "").replace(" per question", "")
        a.set_title(t[0].upper() + t[1:] + " per question", loc="left", fontsize=9.5)
        a.set_yticks([])
        a.spines["left"].set_visible(False)
        a.set_ylim(0, max(vals + [1]) * 1.25)
    meta = ev.get("meta", {})
    fig.suptitle(f"Benchmark: {meta.get('questions', '?')} questions ({meta.get('mode', '')} mode). "
                 f"Writer {meta.get('llm', '')}; decisions {meta.get('jev_backend', '')}",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=1.0)
    if meta.get("mode") == "retrieval":
        fig.text(0.01, -0.02, "Retrieval mode stops each engine once the law sources are chosen, before the answer is "
                 "written, so answer-writing LLM calls and tokens are not counted here. Run the full benchmark for those.",
                 fontsize=8.5, color=INK2)
    fig.savefig(OUT / "fig4_benchmark.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_step_time(rows):
    per = {p: {} for p in ENGINE}
    counts = {p: 0 for p in ENGINE}
    for r in rows:
        for p in ENGINE:
            steps = r.get(f"{p}_steps") or []
            if steps:
                counts[p] += 1
            for s in steps:
                per[p].setdefault(s["node"], []).append(s["secs"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for ax, (p, (label, colour, _m)) in zip(axes, ENGINE.items()):
        if not per[p]:
            ax.axis("off")
            continue
        names = list(per[p])
        avg = [sum(per[p][n]) / max(counts[p], 1) for n in names]
        y = np.arange(len(names))
        ax.barh(y, avg, color=colour, height=0.6)
        for yi, v in zip(y, avg):
            ax.text(v, yi, f" {v:.1f}s", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{label}: {sum(avg):.1f}s per question on average", loc="left")
        ax.set_xlabel("seconds per question (averaged over questions)")
    fig.suptitle("Where the time goes, step by step", x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_step_time.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 6 / 7: each engine's footprint
def fig_engine_projection(pts, z, rows, proj="tsne2", fname="fig6_projection_tsne.png", method="t-SNE"):
    xy = z[proj][:, :2] if proj == "pca3" else z[proj]
    cidx = chunk_index(pts)
    expected = {(r["expected_act"], str(r["expected_section"])) for r in rows
                if r.get("expected_act") and not str(r["expected_section"]).startswith("verify")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, (pipe, (label, colour, marker)) in zip(axes, ENGINE.items()):
        background(ax, pts, xy, size=5, alpha=0.35)
        counts = {}
        for r in rows:
            for key in picks(r, pipe):
                counts[key] = counts.get(key, 0) + 1
        for key, n in counts.items():
            ids = cidx.get(key, [])
            if not ids:
                continue
            pos = xy[ids].mean(axis=0)
            ax.scatter(*pos, s=30 + 45 * n, marker=marker, c=colour, edgecolors=SURFACE, linewidths=1.2,
                       alpha=0.9, zorder=4)
        for key in expected:
            ids = cidx.get(key, [])
            if ids:
                pos = xy[ids].mean(axis=0)
                hit = key in counts
                ax.scatter(*pos, s=260, facecolors="none", edgecolors=INK, linewidths=1.6 if hit else 1.0,
                           linestyle="-" if hit else "--", zorder=5)
        ids_here = {r["id"] for r in rows}
        qs = [i for i, p in enumerate(pts) if p["kind"] == "test question" and p.get("id") in ids_here]
        if qs:
            ax.scatter(xy[qs, 0], xy[qs, 1], s=40, marker="*", c=INK, linewidths=0, zorder=6)
        exp_found = sum(k in counts for k in expected)
        total = sum(counts.values())
        right_act = sum(n for (a, _s), n in counts.items()
                        if any(a == ea for ea, _ in expected)) if total else 0
        ax.set_title(f"{label}: {len(counts)} distinct sections, {total} picks", loc="left")
        ax.text(0, -0.03, f"Distinct expected sections reached in at least one question: {exp_found} of {len(expected)}. "
                f"Marker size = how many questions picked that section.", transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="top")
        clean_axes(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=5, color=c, label=l) for _k, (l, c, _m) in ACT_STYLE.items()]
    handles += [Line2D([], [], marker="o", ls="", ms=5, color=REPEALED[1], label=REPEALED[0]),
                Line2D([], [], marker="*", ls="", ms=9, color=INK, label="test question"),
                Line2D([], [], marker="o", ls="", ms=12, mfc="none", mec=INK, label="expected section (dashed: not reached)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle(f"Retrieval footprint of each architecture on the knowledge base ({method} projection)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- interactive map
def interactive_map(pts, z):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    xy = z["tsne3"]
    fig = go.Figure()
    groups = {}
    for i, p in enumerate(pts):
        if p["kind"] == "law chunk":
            key = p["act"] if (p["status"] == "active" and p["act"] in ACT_STYLE) else "_repealed"
        else:
            key = p["kind"]
        groups.setdefault(key, []).append(i)
    symbols = {"o": "circle", "s": "square", "^": "diamond-open", "D": "diamond"}
    for key, ids in groups.items():
        if key == "OKF topic":
            name, colour, sym, size = "OKF topic page", INK, "cross", 6
        elif key == "test question":
            name, colour, sym, size = "Test question", "#e34948", "x", 6
        elif key == "_repealed":
            name, colour, sym, size = REPEALED[0], "#9a998f", "circle", 2.5
        else:
            name, colour, m = ACT_STYLE[key]
            sym, size = symbols.get(m, "circle"), 3
        hover = [f"<b>{pts[i]['act_short']} s{pts[i]['section']}</b><br>{pts[i]['title']}<br>"
                 f"<span style='font-size:11px'>{pts[i]['text'][:160]}...</span>" if pts[i]["kind"] == "law chunk"
                 else f"<b>{pts[i]['kind']}</b><br>{pts[i]['title']}" for i in ids]
        fig.add_trace(go.Scatter3d(x=xy[ids, 0], y=xy[ids, 1], z=xy[ids, 2], mode="markers", name=f"{name} ({len(ids)})",
                                   marker=dict(size=size, color=colour, symbol=sym, line=dict(width=0)),
                                   hovertext=hover, hoverinfo="text"))
    fig.update_layout(title="Legal knowledge base in embedding space (t-SNE 3D). Drag to rotate, hover for the section text",
                      paper_bgcolor=SURFACE, font=dict(family="Segoe UI, sans-serif", color=INK),
                      scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor=SURFACE),
                      legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=50, b=0), height=780)
    fig.write_html(OUT / "knowledge_map_3d.html", include_plotlyjs="cdn")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out", default=None, help="output folder (default reports/figures)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    path = Path(args.eval) if args.eval else next(
        (p for p in (ROOT / "reports" / "eval_full.json", ROOT / "reports" / "eval_retrieval.json") if p.exists()), None)
    if not path:
        raise SystemExit("No benchmark report found. Run: python scripts/evaluate.py --retrieval-only")
    OUT.mkdir(parents=True, exist_ok=True)
    pts, z, ev = load(path)
    rows = ev.get("rows", [])
    fig_knowledge_map(pts, z)
    fig_question_maps(pts, z, rows)
    fig_similarity_rank(pts, z, rows)
    fig_benchmark(ev)
    fig_step_time(rows)
    fig_engine_projection(pts, z, rows, "tsne2", "fig6_projection_tsne.png", "t-SNE")
    fig_engine_projection(pts, z, rows, "pca3", "fig7_projection_pca.png", "PCA")
    html = interactive_map(pts, z)
    print(f"Figures from {path.name} written to {OUT}"
          + ("" if html else " (install plotly for knowledge_map_3d.html)"))


if __name__ == "__main__":
    main()
