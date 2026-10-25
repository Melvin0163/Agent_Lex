"""Compile knowledge into an Open Knowledge Format (OKF v0.1) bundle.

Bundle layout (knowledge/):
  index.md                       entry point (reserved, no frontmatter)
  log.md                         change history (reserved)
  law/index.md                   list of Acts
  law/<act>/index.md             list of sections with one-line descriptions
  law/<act>/overview.md          type: Statute
  law/<act>/s-027-<title>.md     type: Statute Section
  topics/index.md, <topic>.md    type: Legal Topic  (curated, links to sections)
  documents/index.md             uploaded documents
  documents/<doc>/overview.md    type: Legal Document
  documents/<doc>/clause-03-*.md type: Contract Clause

Every non-reserved file has YAML frontmatter with a non-empty `type` (the only required
field); title/description/tags/timestamp are the recommended fields, and act/section/
status/replaced_by are extension fields. Concepts cross-link with relative markdown links.
"""
import datetime as dt
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

from .. import config
from ..corpus import section_title, sections, slug, tokenize
from ..documents import load_document, split_clauses
from .topics import ACT_INFO, TOPICS

STOP = set("""a an the of to in and or for by on at as is are be been being any such this that these those with
from shall may under which who whom whose where when than then there their its it his her he she they them
not no nor other said same section sub act code chapter provided clause made make every person persons
appropriate government central state rules prescribed notification notified within without into upon
order case cases thereof therein hereby provisions provision including include includes also
1 2 3 4 5 6 7 8 9 10 i ii iii iv v vi a b c d e f""".split())


def now():
    return dt.datetime.now().replace(microsecond=0).isoformat()


def _fm(meta: dict) -> str:
    meta = {k: v for k, v in meta.items() if v not in (None, "", [])}
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000) + "---\n\n"


def _clean(text):
    t = text.replace("\xa0", " ").replace("\r", "")
    t = re.sub(r"\n\s*\n", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def _describe(text, title, n=220):
    body = _clean(text)
    i = body.lower().find(title.lower()[:30]) if title else -1
    if i >= 0:
        body = body[i + len(title):]
    body = re.sub(r"^[\s.\-—:]+", "", body)
    body = re.sub(r"^\(?1\)?[.)]?\s*", "", body)
    body = re.sub(r"\s+", " ", body)
    return (body[:n].rsplit(" ", 1)[0] + "...") if len(body) > n else body


def sec_file(section, title):
    m = re.match(r"(\d+)(.*)", section)
    num = f"{int(m.group(1)):03d}{m.group(2).lower()}" if m else slug(section)
    return f"s-{num}-{slug(title, 48)}.md"


def act_dir(act):
    return slug(act, 70)


def _tfidf_tags(docs: dict, k=6):
    df = Counter()
    toks = {}
    for key, text in docs.items():
        ts = [t for t in tokenize(text) if t not in STOP and len(t) > 3 and not t.isdigit()]
        toks[key] = Counter(ts)
        df.update(set(ts))
    n = len(docs)
    out = {}
    for key, tf in toks.items():
        scored = sorted(tf, key=lambda t: -tf[t] * math.log(n / (1 + df[t])))
        out[key] = scored[:k]
    return out


def build_law_bundle(root: Path = config.KNOWLEDGE_DIR, verbose=True):
    root = Path(root)
    for sub in ("law", "topics"):
        if (root / sub).exists():
            shutil.rmtree(root / sub)
    (root / "law").mkdir(parents=True, exist_ok=True)
    (root / "topics").mkdir(parents=True, exist_ok=True)
    (root / "documents").mkdir(parents=True, exist_ok=True)
    ts = now()

    secs = sections()
    titles = {k: section_title(k[0], k[1], v["text"]) for k, v in secs.items()}
    tags = _tfidf_tags({k: titles[k] + " " + v["text"] for k, v in secs.items()})
    path_of = {k: f"law/{act_dir(k[0])}/{sec_file(k[1], titles[k])}" for k in secs}

    # topic back-links
    topic_links = {}
    for t_slug, t_title, *_rest in TOPICS:
        for ref in _rest[2]:
            topic_links.setdefault(ref, []).append((t_slug, t_title))

    by_act = {}
    for k in secs:
        by_act.setdefault(k[0], []).append(k)

    act_entries = []
    for act, keys in by_act.items():
        d = root / "law" / act_dir(act)
        d.mkdir(parents=True, exist_ok=True)
        status, replaced = secs[keys[0]]["status"], secs[keys[0]]["replaced_by"]
        keys.sort(key=lambda k: (int(re.match(r"\d+", k[1]).group()) if re.match(r"\d+", k[1]) else 9999, k[1]))
        lines = [f"# {act}", "",
                 f"Status: **{status}**" + (f" (replaced by {replaced})" if replaced else ""), "",
                 ACT_INFO.get(act, ""), "", "Overview: [overview](overview.md)", "", "## Sections", ""]
        for k in keys:
            v = secs[k]
            title = titles[k]
            desc = _describe(v["text"], title)
            refs = []
            for m in re.finditer(r"\bsections?\s+(\d+[A-Z]?)", v["text"]):
                rk = (act, m.group(1))
                if rk in secs and rk != k and rk not in refs:
                    refs.append(rk)
            meta = {
                "type": "Statute Section",
                "title": f"{act} - Section {k[1]}: {title}",
                "description": desc,
                "tags": tags[k],
                "timestamp": ts,
                "act": act,
                "section": k[1],
                "status": status,
                "replaced_by": replaced,
            }
            body = [f"# Section {k[1]}. {title}", "",
                    f"Part of [{act}](overview.md). Status: {status}" + (f", replaced by {replaced}." if replaced else "."),
                    "", "## Text", "", v["text"].replace("\xa0", " ").strip(), ""]
            rel = []
            for rk in refs[:8]:
                rel.append(f"- [Section {rk[1]}: {titles[rk]}]({Path(path_of[rk]).name})")
            for t_slug, t_title in topic_links.get(k, []):
                rel.append(f"- Topic: [{t_title}](../../topics/{t_slug}.md)")
            if rel:
                body += ["## Related", ""] + rel + [""]
            (root / path_of[k]).write_text(_fm(meta) + "\n".join(body), encoding="utf-8")
            lines.append(f"- [Section {k[1]}: {title}]({Path(path_of[k]).name}) - {desc}")
        (d / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        ov = _fm({"type": "Statute", "title": act, "description": ACT_INFO.get(act, act),
                  "tags": [act_dir(act)], "timestamp": ts, "status": status, "replaced_by": replaced,
                  "sections": len(keys)})
        ov += f"# {act}\n\n{ACT_INFO.get(act, '')}\n\nStatus: {status}." + \
              (f" This Act has been replaced by the {replaced}; prefer the new law." if replaced else "") + \
              f"\n\n{len(keys)} sections are in this knowledge base. See the [section index](index.md).\n"
        (d / "overview.md").write_text(ov, encoding="utf-8")
        act_entries.append((act, status, replaced, len(keys)))

    act_entries.sort(key=lambda e: (e[1] != "active", e[0]))
    lines = ["# Indian employment law", "",
             "Statutes in this bundle. Prefer **active** laws; repealed ones are kept for history.", ""]
    for act, status, replaced, n in act_entries:
        tail = f" (replaced by {replaced})" if replaced else ""
        lines.append(f"- [{act}]({act_dir(act)}/index.md) - {status}{tail}, {n} sections. {ACT_INFO.get(act, '')}")
    (root / "law" / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # topics
    tlines = ["# Legal topics", "",
              "Everyday employment questions mapped to the statute sections that govern them.", ""]
    for t_slug, t_title, desc, aliases, refs, note in TOPICS:
        links = []
        for ref in refs:
            if ref not in secs:
                if verbose:
                    print(f"  ! topic {t_slug}: {ref} not in corpus, skipped")
                continue
            links.append(f"- [{ref[0]}, Section {ref[1]}: {titles[ref]}](../{path_of[ref]})")
        meta = {"type": "Legal Topic", "title": t_title, "description": desc, "tags": aliases, "timestamp": ts}
        body = [f"# {t_title}", "", desc, ""]
        if note:
            body += [note, ""]
        body += ["Also called: " + ", ".join(aliases), "", "## Governing sections", ""] + links + [""]
        (root / "topics" / f"{t_slug}.md").write_text(_fm(meta) + "\n".join(body), encoding="utf-8")
        tlines.append(f"- [{t_title}]({t_slug}.md) - {desc} Also called: {', '.join(aliases[:6])}.")
    (root / "topics" / "index.md").write_text("\n".join(tlines) + "\n", encoding="utf-8")

    _write_root_index(root)
    if not (root / "documents" / "index.md").exists():
        (root / "documents" / "index.md").write_text("# Uploaded documents\n\n", encoding="utf-8")
    append_log(root, f"Built law bundle: {len(secs)} sections from {len(by_act)} Acts, {len(TOPICS)} topics.")
    if verbose:
        print(f"OKF bundle written to {root}: {len(secs)} sections, {len(by_act)} acts, {len(TOPICS)} topics")
    return root


def _write_root_index(root):
    (root / "index.md").write_text(
        "# Legal knowledge bundle\n\n"
        "Open Knowledge Format (OKF) v0.1 bundle for the legal decision-support agent.\n\n"
        "- [Legal topics](topics/index.md) - start here: everyday questions mapped to statute sections\n"
        "- [Indian employment law](law/index.md) - statutes and their sections\n"
        "- [Uploaded documents](documents/index.md) - contracts and policies compiled from user uploads\n",
        encoding="utf-8")


def append_log(root, message):
    p = Path(root) / "log.md"
    today = dt.date.today().isoformat()
    old = p.read_text(encoding="utf-8") if p.exists() else "# Change log\n"
    head, _, rest = old.partition("\n")
    entry = f"\n## {today}\n\n- {message}\n"
    if f"## {today}" in rest:
        rest = rest.replace(f"## {today}\n\n", f"## {today}\n\n- {message}\n", 1)
        p.write_text(head + "\n" + rest, encoding="utf-8")
    else:
        p.write_text(head + "\n" + entry + rest, encoding="utf-8")


_KIND = [("policy", "Policy"), ("letter", "Letter"), ("agreement", "Agreement"), ("contract", "Contract"),
         ("offer", "Offer Letter"), ("notice", "Notice")]


def compile_document(path, root: Path = config.KNOWLEDGE_DIR, display_name=None):
    """Turn an uploaded file into an OKF sub-bundle under documents/<slug>/. Returns (slug, info)."""
    root = Path(root)
    text = load_document(path)
    name = display_name or Path(path).name
    preamble, clauses = split_clauses(text)
    d_slug = slug(Path(name).stem, 50) or "document"
    d = root / "documents" / d_slug
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    ts = now()
    low = text.lower()[:600]
    kind = next((k for w, k in _KIND if w in low), "Document")
    first_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    heading = next((ln for ln in first_lines if ln.isupper() and 3 < len(ln) < 80 and "SAMPLE" not in ln), name)
    tags = _tfidf_tags({i: c["heading"] + " " + c["body"] for i, c in enumerate(clauses)}, k=5) if clauses else {}

    idx = [f"# {heading.title()}", "", f"Source file: {name}. {len(clauses)} clauses.", "",
           "Overview: [overview](overview.md)", "", "## Clauses", ""]
    files = []
    for i, c in enumerate(clauses):
        fname = f"clause-{i + 1:02d}-{slug(c['heading'], 40) or 'text'}.md"
        desc = re.sub(r"\s+", " ", c["body"])[:200]
        meta = {"type": "Contract Clause", "title": f"Clause {c['no']}: {c['heading']}", "description": desc,
                "tags": tags.get(i, []), "timestamp": ts, "document": name, "clause": c["no"]}
        body = f"# Clause {c['no']}. {c['heading']}\n\nFrom [{heading.title()}](overview.md).\n\n{c['body']}\n"
        (d / fname).write_text(_fm(meta) + body, encoding="utf-8")
        idx.append(f"- [Clause {c['no']}: {c['heading']}]({fname}) - {desc}")
        files.append(fname)
    (d / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    ov = _fm({"type": "Legal Document", "title": heading.title(), "description": f"{kind} uploaded by the user ({name}).",
              "tags": [kind.lower(), "uploaded"], "timestamp": ts, "source_file": name, "clauses": len(clauses)})
    ov += f"# {heading.title()}\n\n{preamble}\n\n{len(clauses)} clauses. See the [clause index](index.md).\n"
    (d / "overview.md").write_text(ov, encoding="utf-8")

    # documents/index.md
    di = root / "documents" / "index.md"
    lines = di.read_text(encoding="utf-8").splitlines() if di.exists() else ["# Uploaded documents", ""]
    lines = [ln for ln in lines if f"]({d_slug}/index.md)" not in ln]
    lines.append(f"- [{heading.title()}]({d_slug}/index.md) - {kind}, {len(clauses)} clauses, from {name}")
    di.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_log(root, f"Compiled uploaded document '{name}' into documents/{d_slug} ({len(clauses)} clauses).")
    return d_slug, {"name": name, "clauses": len(clauses), "chars": len(text), "kind": kind, "text": text}
