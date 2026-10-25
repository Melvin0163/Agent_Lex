"""Copy of the real-document report with one invalid timing removed.

The laptop went to sleep during question r03, so its JEV + OKF time (6,110 s) is excluded.
Writes reports/eval_real_full_timing.{json,md,csv}; reports/eval_real_full.* is left untouched.

  python scripts/timing_clean.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate  # noqa: E402

src = ROOT / "reports" / "eval_real_full.json"
d = json.loads(src.read_text(encoding="utf-8"))
for r in d["rows"]:
    if r["id"] == "r03":
        r["jev_okf_secs_original"] = r["jev_okf_secs"]
        r["jev_okf_secs"] = None          # excluded from the average
        r["jev_okf_steps"] = []           # excluded from the per-step timing figure
meta = dict(d["meta"])
meta["note"] = ("r03 JEV + OKF timing excluded (6,110 s: the laptop went to sleep during that question). "
                "All other values are unchanged from eval_real_full.json.")
evaluate.write_reports(d["rows"], ["rag", "jev_okf"], "full", meta, "eval_real_full_timing")
