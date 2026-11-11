"""Strict 1-based, GRCh38 autosomal SNV context extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def normalize_chrom(chrom: str) -> str:
    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    if value not in {str(i) for i in range(1, 23)}:
        raise ValueError(f"Only human autosomes 1..22 are supported: {chrom!r}")
    return value


@dataclass(frozen=True)
class Variant:
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
        except (ValueError, TypeError) as exc:
            raise ValueError("pos must be a positive integer") from exc
        if str(position) != str(self.pos).strip() or position < 1:
            raise ValueError("pos must be a positive integer (1-based)")
        object.__setattr__(self, "pos", position)
        object.__setattr__(self, "ref", str(self.ref).upper())
        object.__setattr__(self, "alt", str(self.alt).upper())
        object.__setattr__(self, "gene", str(self.gene or "").strip())
        if self.ref not in "ACGT" or len(self.ref) != 1:
            raise ValueError("REF must be one A/C/G/T base")
        if self.alt not in "ACGT" or len(self.alt) != 1 or self.alt == self.ref:
            raise ValueError("ALT must be one different A/C/G/T base")
        if self.label not in (None, 0, 1):
            raise ValueError("label must be 0, 1, or None")

    @property
    def id(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

    @property
    def variant_id(self) -> str:
        return self.id


@dataclass(frozen=True)
class ReferenceContext:
    ref_sequence: str
    alt_sequence: str
    variant_index: int


class ReferenceGenome:
    """FASTA identity/build is supplied by the user; coordinates are never lifted.

    The file must be an uncompressed or BGZF pyfaidx-readable GRCh38 FASTA.
    Ordinary gzip is not supported. Its index is created beside the FASTA.
    """

    def __init__(self, fasta: str | Path):
        from pyfaidx import Fasta

        self.path = Path(fasta)
        self._fasta = Fasta(str(self.path), as_raw=True, sequence_always_upper=True)
        self._contigs = {}
        for name in self._fasta.keys():
            try:
                chrom = normalize_chrom(name)
            except ValueError:
                match = re.fullmatch(r"NC_0*(\d+)\.\d+", name)
                if not match or not 1 <= int(match.group(1)) <= 22:
                    continue
                chrom = str(int(match.group(1)))
            if chrom in self._contigs:
                raise ValueError(f"Ambiguous FASTA contigs for chromosome {chrom}")
            self._contigs[chrom] = name
        if not self._contigs:
            raise ValueError("FASTA has no recognizable autosomal contigs")

    def context(self, variant: Variant, window: int = 256, *, variant_index: int | None = None) -> ReferenceContext:
        if not isinstance(window, int) or not 3 <= window <= 8192:
            raise ValueError("window must be an integer between 3 and 8192")
        index = window // 2 if variant_index is None else variant_index
        if type(index) is not int or not 0 <= index < window:
            raise ValueError("variant_index must be an integer inside the context window")
        if variant.chrom not in self._contigs:
            raise ValueError(f"FASTA lacks chromosome {variant.chrom}")
        contig = self._fasta[self._contigs[variant.chrom]]
        start = variant.pos - 1 - index
        end = start + window
        if start < 0 or end > len(contig):
            raise ValueError(f"Cannot extract full {window}-bp context for {variant.id}")
        reference = str(contig[start:end]).upper()
        if len(reference) != window or set(reference) - set("ACGT"):
            raise ValueError(f"Context contains ambiguous bases or is incomplete: {variant.id}")
        if reference[index] != variant.ref:
            raise ValueError(f"REF mismatch for {variant.id}: FASTA={reference[index]}")
        alternate = reference[:index] + variant.alt + reference[index + 1:]
        return ReferenceContext(reference, alternate, index)

    def close(self):
        self._fasta.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
