"""The statute corpus behind both engines: sizes quoted in the README, and the helpers."""
from legal_agent.corpus import _join, load_rows, section_title, sections, slug, tokenize


def test_corpus_size_matches_readme():
    rows = load_rows()
    assert len(rows) == 842                                   # chunks
    assert len({r["act"] for r in rows}) == 11                # Acts
    assert len(sections()) == 492                             # whole sections


def test_every_chunk_has_status_and_repealed_acts_name_their_successor():
    for r in load_rows():
        assert r["status"] in ("active", "repealed")
        if r["status"] == "repealed":
            assert r["replaced_by"], f"{r['act']} is repealed but has no successor"


def test_four_acts_are_in_force():
    active = {r["act"] for r in load_rows() if r["status"] == "active"}
    assert len(active) == 4
    assert "Indian Contract Act, 1872" in active


def test_section_title_extracts_marginal_heading():
    assert section_title("X", "27", "27. Agreement in restraint of trade, void.- Every agreement ...") \
        == "Agreement in restraint of trade, void"
    assert section_title("X", "9", "no heading here") == "Section 9"


def test_contract_act_section_27_is_restraint_of_trade():
    s = sections()[("Indian Contract Act, 1872", "27")]
    assert "restraint" in s["text"].lower()


def test_join_removes_chunk_overlap():
    a = "The quick brown fox jumps over the lazy dog and keeps running"
    b = "over the lazy dog and keeps running far away"
    assert _join(a, b) == "The quick brown fox jumps over the lazy dog and keeps running far away"


def test_tokenize_and_slug():
    assert tokenize("Section 27: Restraint-of-Trade!") == ["section", "27", "restraint", "of", "trade"]
    assert slug("Indian Contract Act, 1872") == "indian-contract-act-1872"
