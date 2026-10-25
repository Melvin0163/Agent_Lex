"""The statute corpus (data/corpus.jsonl, built by scripts/ingest.py) and shared helpers."""
import json
import re
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def load_rows():
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=1)
def sections():
    """Merge chunks into whole sections: {(act, section): {...}} in corpus order."""
    out = {}
    for r in load_rows():
        key = (r["act"], str(r["section"]))
        if key not in out:
            out[key] = dict(act=r["act"], section=str(r["section"]), status=r["status"],
                            replaced_by=r["replaced_by"], text=r["text"])
        else:
            out[key]["text"] = _join(out[key]["text"], r["text"])
    return out


def _join(a, b, overlap=200):
    """Chunks were cut with a 200-char overlap; drop the repeated part when joining."""
    for k in range(min(overlap, len(a), len(b)), 20, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n" + b


_END = re.compile(
    r"\.\s*-|\.-|-\s*\n|\.\s*\n|\.\s+1\.\s|\n\s*\(?1[.)]|\s\d+\s*\[|\n(?=[A-Z(])|\s+1\.\s+[A-Z]")


def section_title(act, section, text):
    """Pull the marginal heading, e.g. '27. Agreement in restraint of trade, void' -> title."""
    body = text.replace("\xa0", " ").replace("\r", "")
    body = re.sub(r"\n\s*\n", " ", body)                 # headings are often broken by blank lines
    m = re.search(rf"(?:^|\n)\s*{re.escape(section)}\s*\.\s*(?=[A-Z\"'])", body)
    if m:
        rest = body[m.end():m.end() + 220]
        e = _END.search(rest)
        t = rest[:e.start()] if e else rest[:120]
        t = re.sub(r"\s+", " ", t).strip(" .-—:,;")
        t = re.sub(r"^\d+\s*\[", "", t).strip(" [")
        if 3 <= len(t) <= 140:
            return t[0].upper() + t[1:]
    return f"Section {section}"


def slug(s, n=60):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:n].strip("-")


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def tokenize(s):
    return re.findall(r"\w+", s.lower())
