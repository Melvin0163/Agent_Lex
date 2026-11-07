"""The Open Knowledge Format bundle the JEV + OKF engine navigates."""
from pathlib import Path

from legal_agent.okf.reader import OKFBundle, parse_markdown


def test_bundle_conforms_to_okf():
    n, problems = OKFBundle().validate()
    assert problems == []
    assert n >= 519            # 492 sections + 11 Act overviews + 16 topics (uploaded documents add more)


def test_index_entries_point_to_existing_files():
    b = OKFBundle()
    for d in ["", "topics", "law"]:
        entries = b.read_index(d)
        assert entries, f"{d or 'root'}/index.md lists nothing"
        for e in entries:
            if e.path.startswith("documents/"):     # created at run time from uploads (git-ignored)
                continue
            assert (b.root / e.path).exists(), f"{d}/index.md links to missing {e.path}"


def test_there_are_16_topic_pages_and_each_links_to_law():
    b = OKFBundle()
    topics = [c for c in b.concepts("topics")]
    assert len(topics) == 16
    for t in topics:
        assert any(link.startswith("law/") for link in t.links), f"{t.path} links to no statute section"


def test_parse_markdown_frontmatter():
    meta, body = parse_markdown("---\ntype: Statute Section\ntitle: S27\n---\nBody text")
    assert meta == {"type": "Statute Section", "title": "S27"}
    assert body == "Body text"
    assert parse_markdown("no frontmatter") == ({}, "no frontmatter")
    assert parse_markdown("---\n: : bad yaml [\n---\nx")[0] is None
