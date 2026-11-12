"""Reference-aligned sequence windows for simple, VCF-anchored human indels."""
from __future__ import annotations

from dataclasses import dataclass

from .sequence import ReferenceContext, ReferenceGenome, normalize_chrom


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


@dataclass(frozen=True)
class IndelVariant:
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str = ""
    label: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "chrom", normalize_chrom(self.chrom))
        try:
            position = int(self.pos)
        except (TypeError, ValueError):
            raise ValueError("pos must be a positive integer (1-based)") from None
        if position < 1 or str(position) != str(self.pos).strip():
            raise ValueError("pos must be a positive integer (1-based)")
        object.__setattr__(self, "pos", position)
        ref, alt = str(self.ref).upper(), str(self.alt).upper()
        if (
            min(len(ref), len(alt)) != 1 or len(ref) == len(alt)
            or set(ref + alt) - set("ACGT") or ref[0] != alt[0]
        ):
            raise ValueError("Expected a pure insertion or deletion with one shared VCF anchor base")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "alt", alt)
        if self.label not in (None, 0, 1):
            raise ValueError("label must be 0, 1, or None")

    @property
    def id(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

    @property
    def variant_id(self) -> str:
        return self.id


@dataclass(frozen=True)
class IndelContext(ReferenceContext):
    """variant_index and alt_index mark the first aligned downstream tokens."""

    alt_index: int


def _oriented_context(before, after, ref_edit, alt_edit, window, index):
    upstream = before[-index:]
    return IndelContext(
        (upstream + ref_edit + after)[:window],
        (upstream + alt_edit + after)[:window],
        index + len(ref_edit),
        index + len(alt_edit),
    )


def indel_contexts(
    genome: ReferenceGenome, variant: IndelVariant, window: int, variant_index: int,
) -> tuple[IndelContext, IndelContext]:
    """Place the edit after identical upstream context in each strand orientation."""
    if not 1 <= variant_index < window <= 8192:
        raise ValueError("Invalid indel context window or edit offset")
    ref_edit, alt_edit = variant.ref[1:], variant.alt[1:]
    if max(len(ref_edit), len(alt_edit)) >= window - variant_index:
        raise ValueError("Indel leaves no aligned downstream positions in the context window")

    # VCF POS is the 1-based anchor; numerically it is also the zero-based edit boundary.
    edit_start = variant.pos
    edit_end = edit_start + len(ref_edit)
    # Each genomic flank supplies upstream context in one direction and downstream in the other.
    flank = max(variant_index, window - variant_index)
    sequence = genome.reference_segment(variant.chrom, edit_start - flank, edit_end + flank)
    edit_stop = flank + len(ref_edit)
    if sequence[flank - 1:edit_stop] != variant.ref:
        raise ValueError(f"REF mismatch for {variant.id}")
    left, right = sequence[:flank], sequence[edit_stop:]

    forward = _oriented_context(left, right, ref_edit, alt_edit, window, variant_index)
    reverse = _oriented_context(
        reverse_complement(right), reverse_complement(left),
        reverse_complement(ref_edit), reverse_complement(alt_edit), window, variant_index,
    )
    return forward, reverse
