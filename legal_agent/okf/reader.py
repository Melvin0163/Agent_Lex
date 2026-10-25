"""Read and validate an OKF bundle: frontmatter, index files (progressive disclosure) and links."""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .. import config

_LINK = re.compile(r"^-\s+\[(?P<title>[^\]]+)\]\((?P<href>[^)]+)\)\s*(?:-\s*(?P<desc>.*))?$")
_ANY_LINK = re.compile(r"\[(?P<title>[^\]]+)\]\((?P<href>[^)#]+\.md)\)")
RESERVED = ("index.md", "log.md")


@dataclass
class Concept:
    path: str                 # bundle-relative, forward slashes
    meta: dict
    body: str
    links: list = field(default_factory=list)   # bundle-relative paths this concept links to

    @property
    def type(self):
        return self.meta.get("type", "")

    @property
    def title(self):
        return self.meta.get("title", self.path)

    @property
    def description(self):
        return self.meta.get("description", "")


@dataclass
class IndexEntry:
    title: str
    path: str                 # bundle-relative target
    description: str


def parse_markdown(text):
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                return yaml.safe_load(parts[1]) or {}, parts[2].lstrip("\n")
            except yaml.YAMLError:
                return None, text
    return {}, text


class OKFBundle:
    def __init__(self, root=config.KNOWLEDGE_DIR):
        self.root = Path(root).resolve()

    def _rel(self, p: Path):
        return p.relative_to(self.root).as_posix()

    def _resolve(self, base_rel, href):
        target = (self.root / base_rel).parent / href
        try:
            return self._rel(target.resolve())
        except ValueError:
            return None

    def read_index(self, dir_rel=""):
        """Entries listed in <dir>/index.md, in order."""
        p = self.root / dir_rel / "index.md"
        if not p.exists():
            return []
        base = (Path(dir_rel) / "index.md").as_posix()
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            m = _LINK.match(line.strip())
            if m:
                tgt = self._resolve(base, m.group("href"))
                if tgt:
                    out.append(IndexEntry(m.group("title"), tgt, (m.group("desc") or "").strip()))
        return out

    def index_text(self, dir_rel=""):
        p = self.root / dir_rel / "index.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def read(self, rel) -> Concept:
        return _read_cached(str(self.root), rel, (self.root / rel).stat().st_mtime)

    def concepts(self, dir_rel=""):
        base = self.root / dir_rel
        for p in sorted(base.rglob("*.md")):
            if p.name not in RESERVED:
                yield self.read(self._rel(p))

    def validate(self):
        """OKF v0.1 conformance: every non-reserved .md has parseable frontmatter with a non-empty type."""
        problems, n = [], 0
        for p in sorted(self.root.rglob("*.md")):
            rel = self._rel(p)
            text = p.read_text(encoding="utf-8")
            if p.name == "index.md":
                if text.startswith("---"):
                    problems.append(f"{rel}: index.md must not have frontmatter")
                continue
            if p.name == "log.md":
                continue
            n += 1
            meta, _ = parse_markdown(text)
            if meta is None:
                problems.append(f"{rel}: unparseable YAML frontmatter")
            elif not str(meta.get("type", "")).strip():
                problems.append(f"{rel}: missing required 'type'")
        return n, problems


@lru_cache(maxsize=4096)
def _read_cached(root, rel, _mtime):
    p = Path(root) / rel
    meta, body = parse_markdown(p.read_text(encoding="utf-8"))
    links = []
    for m in _ANY_LINK.finditer(body):
        tgt = (p.parent / m.group("href")).resolve()
        try:
            links.append(tgt.relative_to(Path(root).resolve()).as_posix())
        except ValueError:
            pass
    return Concept(rel, meta or {}, body, links)
