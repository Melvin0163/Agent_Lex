"""Compute the embedding-projector data (PCA + t-SNE) and the files for Google's Embedding Projector.

  python scripts/build_projector.py

Then open https://projector.tensorflow.org -> Load -> choose reports/projector/vectors.tsv and metadata.tsv.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from legal_agent import projector  # noqa: E402

t0 = time.time()
d = projector.build(force=True)
kinds = {}
for p in d["points"]:
    kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
print(f"Done in {time.time() - t0:.1f}s: {kinds}")
print(f"PCA explained variance (3 components): {[round(float(v), 3) for v in d['pca_var']]}")
print(f"Files in {projector.OUT}: vectors.tsv, metadata.tsv, projection.npz, points.json")
