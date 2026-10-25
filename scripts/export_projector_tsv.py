"""Export Embedding Projector files (projector.tensorflow.org) into visualization/.

visualization/
  all/            every point (842 law chunks, 16 topics, 31 questions) with columns saying which
                  engine retrieved each section, on the synthetic and the real test sets
  rag_synthetic/  only the chunks RAG retrieved on the 18 synthetic questions, plus those questions
  jev_synthetic/  the same for JEV + OKF
  rag_real/       only the chunks RAG retrieved on the 13 real-document questions, plus those questions
  jev_real/       the same for JEV + OKF

Each folder holds vectors.tsv (384-d e5-small embeddings, no header) and metadata.tsv (header row).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import json  # noqa: E402

from make_figures import chunk_index, picks  # noqa: E402

PROJ = ROOT / "reports" / "projector"
OUT = ROOT / "visualization"
RUNS = {"synthetic": ROOT / "reports" / "eval_full_rescored.json", "real": ROOT / "reports" / "eval_real_full.json"}
ENGINES = {"rag": "rag", "jev": "jev_okf"}

pts = json.loads((PROJ / "points.json").read_text(encoding="utf-8"))
vec_lines = (PROJ / "vectors.tsv").read_text(encoding="utf-8").splitlines()
meta_lines = (PROJ / "metadata.tsv").read_text(encoding="utf-8").splitlines()
header, meta_rows = meta_lines[0].split("\t"), meta_lines[1:]
assert len(pts) == len(vec_lines) == len(meta_rows), (len(pts), len(vec_lines), len(meta_rows))
idx = chunk_index(pts)

# how many questions each engine retrieved each point's section for, per run; and which questions belong to each run
counts = {(run, eng): [0] * len(pts) for run in RUNS for eng in ENGINES}
run_qids = {}
for run, path in RUNS.items():
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    run_qids[run] = {r["id"] for r in rows}
    for eng, pipe in ENGINES.items():
        for r in rows:
            for key in picks(r, pipe):
                for i in idx.get(key, []):
                    counts[(run, eng)][i] += 1


def picked_by(i, run):
    if pts[i]["kind"] != "law chunk":
        return pts[i]["kind"]
    a, b = counts[(run, "rag")][i] > 0, counts[(run, "jev")][i] > 0
    return {(True, True): "both", (True, False): "RAG only", (False, True): "JEV+OKF only"}.get((a, b), "neither")


def write(folder, keep, extra):
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "vectors.tsv").write_text("\n".join(vec_lines[i] for i in keep) + "\n", encoding="utf-8")
    lines = ["\t".join(header + [c for c, _ in extra])]
    for i in keep:
        lines.append("\t".join([meta_rows[i]] + [str(f(i)) for _, f in extra]))
    (d / "metadata.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{folder:15s} {len(keep):4d} points")


all_cols = []
for run in RUNS:
    all_cols += [(f"rag_{run}_picks", lambda i, run=run: counts[(run, "rag")][i]),
                 (f"jev_{run}_picks", lambda i, run=run: counts[(run, "jev")][i]),
                 (f"{run}_picked_by", lambda i, run=run: picked_by(i, run))]
write("all", list(range(len(pts))), all_cols)

for run in RUNS:
    for eng in ENGINES:
        keep = [i for i, p in enumerate(pts)
                if counts[(run, eng)][i] > 0 or (p["kind"] == "test question" and p.get("id") in run_qids[run])]
        write(f"{eng}_{run}", keep, [("times_picked", lambda i, run=run, eng=eng: counts[(run, eng)][i])])

(OUT / "README.txt").write_text(
    "Load in Google Embedding Projector: https://projector.tensorflow.org\n"
    "1. Click Load (left panel).\n"
    "2. Step 1: choose <folder>\\vectors.tsv. Step 2: choose <folder>\\metadata.tsv. Close the dialog.\n"
    "3. 'Color by': act, kind, synthetic_picked_by or real_picked_by. 'Label by': label.\n"
    "4. Choose T-SNE (bottom left) for clusters, or PCA for global distances. Search a label, e.g. 's27'.\n\n"
    "Folders:\n"
    "  all            all 889 points; colour by synthetic_picked_by / real_picked_by to compare the engines\n"
    "  rag_synthetic  chunks RAG retrieved on the 18 synthetic questions + the questions\n"
    "  jev_synthetic  chunks JEV + OKF retrieved on the 18 synthetic questions + the questions\n"
    "  rag_real       chunks RAG retrieved on the 13 real-document questions + the questions\n"
    "  jev_real       chunks JEV + OKF retrieved on the 13 real-document questions + the questions\n"
    "Vectors: multilingual-e5-small, 384 dimensions. Rebuild: python scripts\\export_projector_tsv.py\n",
    encoding="utf-8")
print("written to", OUT)
