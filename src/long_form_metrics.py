"""
Long-form near-verbatim similarity
(this is different from the short form of Phase 1)

Aims to implement the 3-step procedure described in the original paper
1) Identify all verbatim matching blocks between a reference text and
   a LLM-generated text (difflib.SequenceMatcher, greedy matching)
2) Merge adjacent blocks when they are nearby and approx. aligned
3) Filter to keep only sufficiently long near-verbatim blocks.

Two merge and filter passes:
(tau is because it's the symbol used in the paper)
- Pass 1: (tau_gap=2, tau_align=1, min_len=20)
- Pass 2: (tau_gap=10, tau_align=3, min_len=100)
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import argparse
import re
from typing import (
    Iterable,
    List,
    Sequence,
    Tuple
)


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """collapse whitespace and strip."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text.strip())
    return text


def text_to_words(text: str, *, lower: bool = False) -> List[str]:
    """Split text on whitespace into words"""
    text = normalize_text(text)
    if not text:
        return []
    if lower:
        text = text.lower()
    return text.split(" ")


@dataclass(frozen=True)
class Block:

    i: int
    j: int
    m: int
    i_end: int
    j_end: int

    @staticmethod
    def from_verbatim(i: int, j: int, m: int) -> "Block":
        return Block(i=i, j=j, m=m, i_end=i + m, j_end=j + m)

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.i, self.j, self.m)


def identify_verbatim_blocks(
    book_words: Sequence[str],
    gen_words: Sequence[str],
    *,
    autojunk: bool = False,
) -> List[Block]:
    """Identify an ordered set of matching blocks."""
    if not book_words or not gen_words:
        return []

    sm = SequenceMatcher(None, book_words, gen_words, autojunk=autojunk)
    blocks: List[Block] = []
    for match in sm.get_matching_blocks():
        if match.size <= 0:
            continue
        blocks.append(Block.from_verbatim(match.a, match.b, match.size))
    return blocks


def merge_blocks(blocks: Sequence[Block], *, tau_gap: int, tau_align: int) -> List[Block]:
    """Iteratively merge consecutive blocks if merge constraints are met."""
    if not blocks:
        return []

    merged: List[Block] = []
    k = 0
    while k < len(blocks):
        cur = blocks[k]
        while k + 1 < len(blocks):
            nxt = blocks[k + 1]
            delta_b = nxt.i - cur.i_end
            delta_g = nxt.j - cur.j_end

            if max(delta_b, delta_g) <= tau_gap and (delta_b - delta_g) <= tau_align:
                cur = Block(
                    i=cur.i,
                    j=cur.j,
                    m=cur.m + nxt.m,
                    i_end=nxt.i_end,
                    j_end=nxt.j_end,
                )
                k += 1
                continue
            break

        merged.append(cur)
        k += 1
    return merged


def filter_blocks(blocks: Iterable[Block], *, min_len: int) -> List[Block]:
    """Filter blocks to thoose with sufficient matched length"""
    return [b for b in blocks if b.m >= min_len]


def near_verbatim_blocks(
    book_text: str,
    gen_text: str,
    *,
    tau_gap_1: int = 2,
    tau_align_1: int = 1,
    min_len_1: int = 20,
    tau_gap_2: int = 10,
    tau_align_2: int = 3,
    min_len_2: int = 100,
    lower: bool = False,
    autojunk: bool = False,
) -> List[Tuple[int, int, int]]:
    """Return the final ordered set of blocks"""
    book_words = text_to_words(book_text, lower=lower)
    gen_words = text_to_words(gen_text, lower=lower)

    blocks = identify_verbatim_blocks(book_words, gen_words, autojunk=autojunk)

    blocks = merge_blocks(blocks, tau_gap=tau_gap_1, tau_align=tau_align_1)
    blocks = filter_blocks(blocks, min_len=min_len_1)

    blocks = merge_blocks(blocks, tau_gap=tau_gap_2, tau_align=tau_align_2)
    blocks = filter_blocks(blocks, min_len=min_len_2)

    return [b.as_tuple() for b in blocks]


@dataclass(frozen=True)
class NearVerbatimMetrics:
    blocks: List[Tuple[int, int, int]]
    matched: int
    nv_recall: float
    missing: int
    additional: int
    book_len_words: int
    gen_len_words: int

    def to_dict(self) -> dict:
        return {
            "blocks": self.blocks,
            "matched": self.matched,
            "nv_recall": self.nv_recall,
            "missing": self.missing,
            "additional": self.additional,
            "book_len_words": self.book_len_words,
            "gen_len_words": self.gen_len_words,
        }


def near_verbatim_metrics(
    book_text: str,
    gen_text: str,
    *,
    tau_gap_1: int = 2,
    tau_align_1: int = 1,
    min_len_1: int = 20,
    tau_gap_2: int = 10,
    tau_align_2: int = 3,
    min_len_2: int = 100,
    lower: bool = False,
    autojunk: bool = False,
) -> NearVerbatimMetrics:
    """Compute matched words, recall, missing, and additional"""
    book_words = text_to_words(book_text, lower=lower)
    gen_words = text_to_words(gen_text, lower=lower)

    blocks = near_verbatim_blocks(
        book_text,
        gen_text,
        tau_gap_1=tau_gap_1,
        tau_align_1=tau_align_1,
        min_len_1=min_len_1,
        tau_gap_2=tau_gap_2,
        tau_align_2=tau_align_2,
        min_len_2=min_len_2,
        lower=lower,
        autojunk=autojunk,
    )

    matched = sum(m for _, _, m in blocks)
    book_len = len(book_words)
    gen_len = len(gen_words)
    nv_recall = (matched / book_len) if book_len > 0 else 0.0
    missing = book_len - matched
    additional = gen_len - matched

    return NearVerbatimMetrics(
        blocks=blocks,
        matched=matched,
        nv_recall=nv_recall,
        missing=missing,
        additional=additional,
        book_len_words=book_len,
        gen_len_words=gen_len,
    )


def _main() -> int:
    p = argparse.ArgumentParser(description="")
    p.add_argument("--ref", required=True, help="Path to file with reference text")
    p.add_argument("--gen", required=True, help="Path to LLM-generated text")
    p.add_argument("--lower", action="store_true", help="Lowercase before tokenizing")
    args = p.parse_args()

    with open(args.ref, "r", encoding="utf-8") as f:
        book_text = f.read()
    with open(args.gen, "r", encoding="utf-8") as f:
        gen_text = f.read()

    m = near_verbatim_metrics(book_text, gen_text, lower=args.lower)
    print(f"matched_words={m.matched}")
    print(f"nv_recall={m.nv_recall:.6f}")
    print(f"missing_words={m.missing}")
    print(f"additional_words={m.additional}")
    print(f"num_blocks={len(m.blocks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
