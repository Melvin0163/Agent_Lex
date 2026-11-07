"""Document loading and clause splitting: no models or network needed."""
import pytest

from legal_agent.documents import chunk_text, load_document, split_clauses
from legal_agent import config

NUMBERED = """EMPLOYMENT AGREEMENT
Between Example Pvt. Ltd. and A. Employee.

1. Probation
Probation is 3 months.

2. Notice Period
Either party shall give 60 days' written notice.

3. Non-Compete
The employee shall not join a competitor for 2 years after leaving.
"""

SECTIONED = """CONFIDENTIALITY AGREEMENT
Preamble text.

SECTION 1
DEFINITIONS
Confidential information means any information marked confidential.

SECTION 2
NON-COMPETE
The employee shall not compete for 12 months.
2.1 This applies within India.
"""


def test_numbered_headings_become_clauses():
    preamble, clauses = split_clauses(NUMBERED)
    assert [c["no"] for c in clauses] == ["1", "2", "3"]
    assert [c["heading"] for c in clauses] == ["Probation", "Notice Period", "Non-Compete"]
    assert "2 years" in clauses[2]["body"]
    assert "EMPLOYMENT AGREEMENT" in preamble


def test_section_headings_with_title_on_next_line():
    _, clauses = split_clauses(SECTIONED)
    assert [c["no"] for c in clauses] == ["1", "2"]
    assert clauses[0]["heading"] == "Definitions"           # upper-case title is title-cased
    assert "2.1 This applies within India." in clauses[1]["body"]   # sub-clause stays in its section


def test_unnumbered_text_falls_back_to_paragraph_chunks():
    text = "\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    _, clauses = split_clauses(text)
    assert len(clauses) >= 2
    assert all(c["body"] for c in clauses)


def test_chunk_text_respects_size():
    text = "\n".join("x" * 300 for _ in range(10))
    chunks = chunk_text(text, size=900)
    assert len(chunks) > 1
    assert all(len(c) <= 1800 for c in chunks)
    assert "".join(chunks).replace("\n", "") == "x" * 3000


def test_load_txt_and_html(tmp_path):
    (tmp_path / "a.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "b.html").write_text("<p>Hello</p><script>x()</script><div>World</div>", encoding="utf-8")
    assert load_document(tmp_path / "a.txt") == "plain text"
    html = load_document(tmp_path / "b.html")
    assert "Hello" in html and "World" in html and "x()" not in html


def test_unsupported_and_missing_files(tmp_path):
    (tmp_path / "c.xyz").write_text("?", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(tmp_path / "c.xyz")
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.txt")


@pytest.mark.parametrize("name, at_least", [("01_clean_agreement.txt", 3), ("02_noncompete_bond.docx", 3),
                                             ("05_leave_policy.txt", 3), ("07_termination_letter.txt", 2)])
def test_sample_documents_split_into_clauses(name, at_least):
    text = load_document(config.SAMPLES_DIR / name)
    _, clauses = split_clauses(text)
    assert len(clauses) >= at_least, f"{name} split into only {len(clauses)} clauses"
