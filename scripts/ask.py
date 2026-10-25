"""Command-line law agent.

  python scripts/ask.py "Can my employer deduct salary for damage?"                  # both pipelines
  python scripts/ask.py --doc data/samples/02_noncompete_bond.docx "Is the bond valid?"
  python scripts/ask.py --pipeline jev_okf "What is the maternity leave duration?"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from legal_agent import service  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--doc", action="append", default=[])
ap.add_argument("--pipeline", choices=["rag", "jev_okf", "both"], default="both")
args = ap.parse_args()

docs = []
for d in args.doc:
    doc, secs = service.prepare_document(d)
    docs.append(doc)
    print(f"Loaded {doc.name}: {len(doc.text)} chars, {doc.n_clauses} clauses ({secs}s)")

for p in (["rag", "jev_okf"] if args.pipeline == "both" else [args.pipeline]):
    r = service.ask(p, args.question, docs)
    print("\n" + "=" * 30, service.PIPELINES[p][0], "=" * 30)
    print("route:", r.route)
    for s in r.steps:
        print(f"  [{s['secs']:>6.2f}s] {s['node']}: {s['info'][:160]}")
    print("\n" + r.answer)
    print("\nSources:")
    for s in r.sources:
        print(f"  [{s['label']}] {s['title']}  (score {s.get('score')})")
    if r.issues:
        print("Verification issues:", *r.issues, sep="\n  ")
    print("Metrics:", json.dumps(r.metrics))
