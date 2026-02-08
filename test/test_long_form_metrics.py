import pathlib

import pytest

from long_form_metrics import (
    near_verbatim_blocks,
    near_verbatim_metrics
)


def test_two_pass_merges_insertions_into_single_block():
    book_words = [f"w{i}" for i in range(150)]
    book_text = " ".join(book_words)

    # Same book passage with insertions in G (gaps in G only).
    gen_words = (
        book_words[:60]
        + ["X"] * 5
        + book_words[60:120]
        + ["Y"] * 5
        + book_words[120:]
    )
    gen_text = " ".join(gen_words)

    m = near_verbatim_metrics(book_text, gen_text)
    assert m.matched == 150
    assert m.additional == 10
    assert m.missing == 0
    assert m.nv_recall == 1.0
    assert len(m.blocks) == 1
    assert m.blocks[0][2] == 150


def test_alignment_constraint_blocks_large_deletions_from_merging():
    book_words = [f"w{i}" for i in range(150)]
    book_text = " ".join(book_words)

    # Delete 5 book-words between blocks: delta_B=5, delta_G=0.
    gen_words = book_words[:60] + book_words[65:]
    gen_text = " ".join(gen_words)

    blocks = near_verbatim_blocks(
        book_text,
        gen_text,
        min_len_1=1,
        min_len_2=1,
    )

    # Without merging, we keep the two blocks.
    assert len(blocks) == 2
    assert blocks[0][2] == 60
    assert blocks[1][2] == 85


def test_frankenstein_short_preprocessed_excerpt_matches_after_merging():
    book_path = pathlib.Path("data/frankenstein_short_preprocessed.txt")
    book_text = book_path.read_text(encoding="utf-8")
    book_words = book_text.split()
    assert len(book_words) > 250

    excerpt = book_words[30:200]
    gen_words = excerpt[:80] + ["NOISE"] * 4 + excerpt[80:140] + ["NOISE2"] * 4 + excerpt[140:]
    gen_text = " ".join(gen_words)

    m = near_verbatim_metrics(book_text, gen_text)
    assert m.matched == len(excerpt)
    assert m.additional == 8
    assert len(m.blocks) == 1
    assert m.blocks[0][2] == len(excerpt)
    assert m.nv_recall == pytest.approx(len(excerpt) / len(book_words))
