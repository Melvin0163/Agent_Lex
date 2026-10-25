"""Manual grading of answer quality (the part automatic metrics cannot judge).

  python scripts/grading_sheet.py make     # reports/grading_sheet.csv from reports/eval_full.json
  (open it in Excel, fill the score columns, save as CSV)
  python scripts/grading_sheet.py score    # totals per engine -> reports/grading_results.md

Rubric, each 0 / 1 / 2:
  law         cites the right governing law (2), partly or with irrelevant law mixed in (1), wrong or none (0)
  conclusion  the answer's conclusion is legally right for this question (2), partly (1), wrong (0)
  grounded    no invented facts, sections or conditions (2), minor (1), serious (0)
For questions that should be refused: 2/2/2 if it refused or pointed to a lawyer, otherwise 0/0/0.
Grade the two answers without looking at which engine wrote them if you can (the columns are labelled A and B
and the key is stored in a separate column at the far right).
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "eval_full.json"
SHEET = ROOT / "reports" / "grading_sheet.csv"
OUT = ROOT / "reports" / "grading_results.md"
NAMES = {"rag": "Agentic RAG", "jev_okf": "JEV + OKF"}
CRIT = ["law", "conclusion", "grounded"]


def make():
    if not SRC.exists():
        raise SystemExit("Run the full benchmark first: python scripts/evaluate.py")
    rows = json.loads(SRC.read_text(encoding="utf-8"))["rows"]
    rnd = random.Random(42)
    out = []
    for r in rows:
        order = ["rag", "jev_okf"]
        rnd.shuffle(order)      # blind: A/B order varies per question
        exp = "should refuse" if r["should_refuse"] else (
            f"{r['expected_act']} s{r['expected_section']}" if r.get("expected_act") else "no single section")
        rec = {"id": r["id"], "question": r["question"], "document": r.get("document") or "", "expected": exp}
        for tag, pipe in zip("AB", order):
            rec[f"answer {tag}"] = r.get(f"{pipe}_answer") or ""
            rec[f"sources {tag}"] = "; ".join(r.get(f"{pipe}_sources") or [])
            for c in CRIT:
                rec[f"{tag} {c} (0-2)"] = ""
        rec["notes"] = ""
        rec["key (do not look while grading)"] = f"A={order[0]} B={order[1]}"
        out.append(rec)
    with open(SHEET, "w", newline="", encoding="utf-8-sig") as f:   # BOM so Excel shows Unicode correctly
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {SHEET} ({len(out)} questions). Fill the (0-2) columns and save as CSV, then run: "
          f"python scripts/grading_sheet.py score")


def score():
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    tot = {p: {c: [] for c in CRIT} for p in NAMES}
    for r in rows:
        key = dict(kv.split("=") for kv in r["key (do not look while grading)"].split())
        for tag in "AB":
            pipe = key[tag]
            for c in CRIT:
                v = (r.get(f"{tag} {c} (0-2)") or "").strip()
                if v != "":
                    tot[pipe][c].append(float(v))
    lines = ["# Manual grading of answers", "", "Each criterion scored 0-2 per question; shown as % of the maximum.", "",
             "| Criterion | " + " | ".join(NAMES.values()) + " |", "|---|" + "---|" * len(NAMES)]
    for c in CRIT + ["overall"]:
        cells = []
        for p in NAMES:
            vals = sum((tot[p][k] for k in CRIT), []) if c == "overall" else tot[p][c]
            cells.append("n/a" if not vals else f"{100 * sum(vals) / (2 * len(vals)):.0f}% (n={len(vals)})")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"make": make, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
