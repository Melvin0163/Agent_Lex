"""Compare Agentic RAG vs JEV + OKF on data/samples/qa_testset.jsonl.

  python scripts/evaluate.py                    # full run: answers generated, everything measured
  python scripts/evaluate.py --retrieval-only   # fast: routing + retrieval only (no answer LLM call)
  python scripts/evaluate.py --limit 5
  python scripts/evaluate.py --testset data/qa_real.jsonl --docs data/real --tag real
        # the real-document test set -> reports/eval_real_full.*
  python scripts/evaluate.py --rescore reports/eval_full.json
        # re-score saved answers with the current scoring rules (no model is called);
        # writes reports/eval_full_rescored.{json,md,csv} and leaves the original untouched

Writes reports/eval_<mode>.json, .csv and .md.

Scoring rules (applied identically to both pipelines)
  refused        the request was routed out of scope, the pipeline returned its "not found" answer, or the
                 answer OPENS (first 300 characters) by declining or saying the question cannot be determined.
                 An answer that answers and then also suggests a lawyer is NOT a refusal.
  citations      every [D#] / [L#] label inside square brackets counts, including forms such as
                 "[L1, Section 17(2)]" and "[L1][L3]".
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from legal_agent import config  # noqa: E402

NAMES = {"rag": "Agentic RAG", "jev_okf": "JEV + OKF"}
REFUSAL_OPENERS = ["i could not find", "i can only help", "cannot be determined", "can't be determined",
                   "not possible to determine", "i cannot determine", "not enough information",
                   "i cannot provide legal advice", "i can't provide legal advice"]
SRC = re.compile(r"^(?P<act>.+?), Section (?P<sec>[0-9]+[A-Z]?)\b")


# ---------------------------------------------------------------- scoring
def cite_labels(answer):
    """All [D#]/[L#] labels used in the answer, in order, including '[L1, Section 17(2)]' forms."""
    out = []
    for inside in re.findall(r"\[([^\[\]]{1,80})\]", answer or ""):
        out += re.findall(r"\b([DL]\d+)\b", inside)
    return out


def sec_matches(sec_expected, sec):
    return not sec_expected or str(sec_expected).startswith("verify") or str(sec) == str(sec_expected)


def hit(sources, act, sec):
    if not act:
        return None
    return any(s["kind"] == "law" and s["act"] == act and sec_matches(sec, s["section"]) for s in sources)


def cited_expected(answer, sources, act, sec):
    if not act or not answer:
        return None
    labels = {s["label"] for s in sources if s["kind"] == "law" and s["act"] == act and sec_matches(sec, s["section"])}
    return bool(labels & set(cite_labels(answer)))


def citation_validity(answer, sources):
    used = cite_labels(answer)
    if not used:
        return None
    valid = {s["label"] for s in sources}
    return sum(u in valid for u in used) / len(used)


def is_refusal(answer, route, gated=False):
    if route == "out_of_scope" or gated:
        return True
    head = (answer or "").lower()[:300]
    return any(p in head for p in REFUSAL_OPENERS)


def law_precision(sources, act):
    law = [s for s in sources if s["kind"] == "law"]
    if not act or not law:
        return None
    return sum(s["act"] == act for s in law) / len(law)


def sources_from_titles(titles):
    """Rebuild labelled sources from saved titles (documents are listed first, then law, as in source_list)."""
    out, d, l = [], 0, 0
    for t in titles or []:
        m = SRC.match(t)
        if m:
            l += 1
            out.append({"label": f"L{l}", "kind": "law", "act": m["act"], "section": m["sec"]})
        else:
            d += 1
            out.append({"label": f"D{d}", "kind": "document", "act": None, "section": None})
    return out


def pct(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else round(100 * sum(vals) / len(vals), 1)


def avg(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else round(sum(vals) / len(vals), 2)


# ---------------------------------------------------------------- reports
def summarise(rows, pipes, full):
    summary = {}
    for p in pipes:
        R = [r for r in rows if f"{p}_route" in r]
        summary[p] = {
            "Correct routing (%)": pct([r[f"{p}_route_ok"] for r in R]),
            "Expected section retrieved (%)": pct([r[f"{p}_hit"] for r in R]),
            "Law sources from expected Act (%)": pct([r[f"{p}_precision"] for r in R]),
            "Avg law sections sent to LLM": avg([r[f"{p}_n_law"] for r in R]),
            "Avg seconds per question": avg([r[f"{p}_secs"] for r in R]),
            "Avg LLM calls per question": avg([r[f"{p}_llm_calls"] for r in R]),
            "Avg LLM tokens per question": avg([r[f"{p}_llm_tokens"] for r in R]),
            "Avg Jev decision calls per question": avg([r[f"{p}_decision_calls"] for r in R]),
        }
        if full:
            summary[p].update({
                "Correct refuse/answer behaviour (%)": pct([r.get(f"{p}_refusal_ok") for r in R]),
                "Answer cites the expected section (%)": pct([r.get(f"{p}_cited_expected") for r in R]),
                "Valid citations (%)": pct([r.get(f"{p}_citation_valid") for r in R]),
            })
    return summary


def write_reports(rows, pipes, mode, meta, stem):
    summary = summarise(rows, pipes, full=(mode == "full"))
    out = {"meta": meta, "summary": summary, "rows": rows}
    (config.REPORTS_DIR / f"{stem}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                                                    encoding="utf-8")
    keys = list(next(iter(summary.values())).keys())
    lines = [f"# Evaluation: Agentic RAG vs JEV + OKF ({mode} mode)", "",
             f"{len(rows)} questions | LLM: {meta['llm']} | Decision model: {meta['jev_backend']} | {meta['date']}", ""]
    if meta.get("note"):
        lines += [f"> {meta['note']}", ""]
    lines += ["| Metric | " + " | ".join(NAMES[p] for p in pipes) + " |", "|---|" + "---|" * len(pipes)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join("n/a" if summary[p][k] is None else str(summary[p][k]) for p in pipes) + " |")
    cols = "route / hit / secs" + (" / refused / cites expected" if mode == "full" else "")
    lines += ["", "## Per question", "",
              "| id | question | expected | " + " | ".join(f"{NAMES[p]} {cols}" for p in pipes) + " |",
              "|---|---|---|" + "---|" * len(pipes)]
    for r in rows:
        exp = f"{r['expected_route']}; {(r['expected_act'] or '-').split(',')[0]} {r['expected_section'] or ''}"
        if r.get("should_refuse"):
            exp += "; should refuse"
        cells = []
        for p in pipes:
            c = f"{r.get(f'{p}_route')} / {r.get(f'{p}_hit')} / {r.get(f'{p}_secs')}"
            if mode == "full":
                c += f" / {r.get(f'{p}_refused')} / {r.get(f'{p}_cited_expected')}"
            cells.append(c)
        lines.append(f"| {r['id']} | {r['question']} | {exp} | " + " | ".join(cells) + " |")
    if str(meta.get("jev_backend", "")).startswith("jev-local"):
        lines += ["", "> Decisions were made by the offline Jev emulator (local cross-encoders), not the hosted Jev model."]
    (config.REPORTS_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(config.REPORTS_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        flat = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                for r in rows]
        w = csv.DictWriter(f, fieldnames=sorted({k for r in flat for k in r}))
        w.writeheader()
        w.writerows(flat)
    print("\n" + "\n".join(lines[:len(keys) + 8]))


def rescore(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows, meta = data["rows"], dict(data.get("meta", {}))
    pipes = [p for p in NAMES if any(f"{p}_route" in r for r in rows)]
    for r in rows:
        for p in pipes:
            if f"{p}_answer" not in r:
                continue
            src = sources_from_titles(r.get(f"{p}_sources"))
            ans, route = r.get(f"{p}_answer") or "", r.get(f"{p}_route")
            gated = ans.lower().startswith("i could not find a provision")
            r[f"{p}_refused_original"] = r.get(f"{p}_refused")
            r[f"{p}_refused"] = is_refusal(ans, route, gated)
            r[f"{p}_refusal_ok"] = r[f"{p}_refused"] == r["should_refuse"]
            r[f"{p}_cited_expected"] = cited_expected(ans, src, r.get("expected_act"), r.get("expected_section"))
            r[f"{p}_citation_valid"] = citation_validity(ans, src)
    meta["note"] = (f"Re-scored on {time.strftime('%Y-%m-%d %H:%M')} from the saved answers of the run on "
                    f"{meta.get('date', '?')} with the corrected refusal and citation rules; no model was called.")
    mode = meta.get("mode", "full")
    write_reports(rows, pipes, mode, meta, Path(path).stem + "_rescored")


# ---------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pipelines", default="rag,jev_okf")
    ap.add_argument("--rescore", default=None, help="path to a saved eval_*.json to re-score")
    ap.add_argument("--testset", default=None, help="questions file (default data/samples/qa_testset.jsonl)")
    ap.add_argument("--docs", default=None, help="folder with the documents named in the test set (default data/samples)")
    ap.add_argument("--tag", default="", help="report name prefix, e.g. 'real' -> reports/eval_real_full.*")
    args = ap.parse_args()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    if args.rescore:
        return rescore(args.rescore)

    from legal_agent import llm, service
    from legal_agent.jev import get_engine

    mode = "retrieval" if args.retrieval_only else "full"
    pipes = args.pipelines.split(",")
    testset = Path(args.testset) if args.testset else config.SAMPLES_DIR / "qa_testset.jsonl"
    doc_dir = Path(args.docs) if args.docs else config.SAMPLES_DIR
    stem = f"eval_{args.tag + '_' if args.tag else ''}{mode}"
    questions = [json.loads(l) for l in open(testset, encoding="utf-8") if l.strip()]
    if args.limit:
        questions = questions[:args.limit]
    service.ensure_bundle()

    docs_cache, rows = {}, []
    for qd in questions:
        docs = []
        if qd["document"]:
            if qd["document"] not in docs_cache:
                docs_cache[qd["document"]] = service.prepare_document(doc_dir / qd["document"])[0]
            docs = [docs_cache[qd["document"]]]
        print(f"[{qd['id']}] {qd['question']}", flush=True)
        row = dict(qd)
        for p in pipes:
            t0 = time.time()
            try:
                r = service.ask(p, qd["question"], docs, stop_after_retrieval=args.retrieval_only)
            except Exception as e:  # noqa: BLE001
                print("    ERROR", p, e, flush=True)
                row[f"{p}_error"] = str(e)
                continue
            m = r.metrics
            row.update({
                f"{p}_route": r.route,
                f"{p}_route_ok": r.route == qd["expected_route"],
                f"{p}_hit": hit(r.sources, qd["expected_act"], qd["expected_section"]),
                f"{p}_precision": law_precision(r.sources, qd["expected_act"]),
                f"{p}_n_law": sum(s["kind"] == "law" for s in r.sources),
                f"{p}_secs": round(time.time() - t0, 2),
                f"{p}_llm_calls": m["llm_calls"],
                f"{p}_llm_tokens": m["llm_tokens"],
                f"{p}_decision_calls": m["decision_calls"],
                f"{p}_answer": r.answer,
                f"{p}_sources": [s["title"] for s in r.sources],
                f"{p}_issues": r.issues,
                f"{p}_steps": r.steps,
            })
            if not args.retrieval_only:
                row[f"{p}_refused"] = is_refusal(r.answer, r.route, r.gated)
                row[f"{p}_refusal_ok"] = row[f"{p}_refused"] == qd["should_refuse"]
                row[f"{p}_cited_expected"] = cited_expected(r.answer, r.sources, qd["expected_act"], qd["expected_section"])
                row[f"{p}_citation_valid"] = citation_validity(r.answer, r.sources)
            print(f"    {NAMES[p]:12s} route={r.route:13s} hit={row[f'{p}_hit']}  llm_calls={m['llm_calls']} "
                  f"jev_calls={m['decision_calls']}  {row[f'{p}_secs']}s", flush=True)
        rows.append(row)
        (config.REPORTS_DIR / f"{stem}.json").write_text(
            json.dumps({"mode": mode, "rows": rows}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    meta = {"mode": mode, "questions": len(rows), "llm": llm.provider_info(), "jev_backend": get_engine().label,
            "date": time.strftime("%Y-%m-%d %H:%M"), "testset": str(testset.name)}
    write_reports(rows, pipes, mode, meta, stem)


if __name__ == "__main__":
    main()
