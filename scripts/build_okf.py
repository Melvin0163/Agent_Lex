"""Compile the statute corpus and curated topics into the OKF bundle (knowledge/), then validate it.

  python scripts/build_okf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from legal_agent.okf import OKFBundle, build_law_bundle  # noqa: E402

build_law_bundle()
n, problems = OKFBundle().validate()
print(f"OKF v0.1 conformance: {n} concepts checked, {len(problems)} problems")
for p in problems[:20]:
    print("  -", p)
