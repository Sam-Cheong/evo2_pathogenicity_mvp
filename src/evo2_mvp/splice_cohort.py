"""Build chromosome-held-out splice cohorts from public human GRCh38 resources."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import gzip
from pathlib import Path
import random
import re
from urllib.parse import unquote

from .sequence import ReferenceGenome, Variant, normalize_chrom
from .utils import sha256_file


SPLIT_CHROMS = {
    "train": set(range(1, 14)),
    "val": {14, 15},
    "test": {16, 17},
    "challenge": set(range(18, 23)),
}
LABELS = {
    "Pathogenic": 1,
    "Likely_pathogenic": 1,
    "Pathogenic/Likely_pathogenic": 1,
    "Benign": 0,
    "Likely_benign": 0,
    "Benign/Likely_benign": 0,
}
REVIEWED = {
    "practice_guideline",
    "reviewed_by_expert_panel",
    "criteria_provided,_multiple_submitters,_no_conflicts",
}
CONSEQUENCES = {"donor": "splice_donor_variant", "acceptor": "splice_acceptor_variant"}


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _autosome(value):
    try:
        return normalize_chrom(value)
    except ValueError:
        return None


def _transcripts(path):
    """Yield (chromosome, strand, gene, 1-based closed exon intervals)."""
    with _open(path) as handle:
        if ".gtf" in path.suffixes:
            transcripts = defaultdict(set)
            attr_pattern = re.compile(r'(transcript_id|gene_name|gene_id) "([^"]+)"')
            for line in handle:
                if line.startswith("#"):
                    continue
                chrom, _, feature, start, end, _, strand, _, attributes = line.rstrip().split("\t")
                chrom = _autosome(chrom)
                if chrom is None or feature != "exon" or strand not in ("+", "-"):
                    continue
                attrs = dict(attr_pattern.findall(attributes))
                gene = attrs.get("gene_name", attrs.get("gene_id"))
                if gene and "transcript_id" in attrs:
                    transcripts[(chrom, strand, gene, attrs["transcript_id"])].add((int(start), int(end)))
            for (chrom, strand, gene, _), exons in transcripts.items():
                yield chrom, strand, gene, sorted(exons)
        else:
            for line in handle:
                _, _, chrom, strand, _, _, _, _, count, starts, ends, _, gene, *_ = line.rstrip().split("\t")
                chrom = _autosome(chrom)
                if chrom is None or strand not in ("+", "-") or not gene:
                    continue
                left = [int(x) + 1 for x in starts.rstrip(",").split(",")]
                right = [int(x) for x in ends.rstrip(",").split(",")]
                if len(left) != int(count) or len(right) != int(count):
                    raise ValueError("UCSC genePred exon count does not match coordinates")
                yield chrom, strand, gene, list(zip(left, right))


def _sites(path):
    sites = defaultdict(set)
    for chrom, strand, gene, exons in _transcripts(path):
        for (_, left_end), (right_start, _) in zip(exons, exons[1:]):
            if right_start - left_end < 5:
                continue
            left, right = left_end + 1, right_start - 1
            donor, acceptor = (left, right) if strand == "+" else (right, left)
            sites[(chrom, strand, donor)].add((gene, "donor"))
            sites[(chrom, strand, acceptor)].add((gene, "acceptor"))
    if not sites:
        raise ValueError("No autosomal splice sites found in the GRCh38 transcript annotation")
    return sites


def _usable_context(genome, chrom, pos, strand):
    index = 192 if strand == "+" else 63
    start = pos - 1 - index
    try:
        genome.reference_segment(chrom, start, start + 256)
        return True
    except ValueError:
        return False


def _annotation_rows(sites, genome, counts, rng):
    partitions, used_genes = {}, set()
    for partition, count in zip(("train", "val", "test"), counts):
        allowed = SPLIT_CHROMS[partition]
        candidates = [
            (chrom, strand, pos, *next(iter(labels)))
            for (chrom, strand, pos), labels in sites.items()
            if int(chrom) in allowed and len(labels) == 1
        ]
        candidates.sort()
        rng.shuffle(candidates)
        rows = []
        for target in ("donor", "acceptor", "background"):
            initial_size = len(rows)
            for chrom, strand, pos, gene, annotation in candidates:
                if gene in used_genes or (target != "background" and annotation != target):
                    continue
                if target == "background":
                    pos += rng.choice((-41, -29, -19, 19, 29, 41))
                    if any((chrom, strand, pos + offset) in sites for offset in (-1, 0, 1)):
                        continue
                if not _usable_context(genome, chrom, pos, strand):
                    continue
                rows.append(dict(chrom=chrom, pos=pos, strand=strand, gene=gene, annotation=target))
                used_genes.add(gene)
                if len(rows) - initial_size == count:
                    break
            if len(rows) - initial_size != count:
                raise ValueError(f"Insufficient {target} examples for the {partition} split")
        partitions[partition] = rows
    return partitions, used_genes


def _challenge_rows(path, sites, genome, used_genes, count, rng):
    by_gene = defaultdict(list)
    for (chrom, strand, pos), annotations in sites.items():
        if int(chrom) not in SPLIT_CHROMS["challenge"]:
            continue
        for gene, annotation in annotations:
            if gene not in used_genes:
                by_gene[(chrom, gene)].append((pos, strand, annotation))
    for records in by_gene.values():
        records.sort()

    release, candidates, conflicts = None, {}, set()
    with _open(path) as handle:
        for line in handle:
            if line.startswith("##fileDate="):
                release = line.split("=", 1)[1].strip()
            if line.startswith("##reference=") and "GRCh37" in line:
                raise ValueError("ClinVar VCF must use GRCh38")
            if line.startswith("#"):
                continue
            chrom, pos, accession, ref, alt, _, _, raw_info = line.rstrip().split("\t")[:8]
            chrom = _autosome(chrom)
            if chrom is None or int(chrom) not in SPLIT_CHROMS["challenge"]:
                continue
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT" or ref == alt:
                continue
            info = dict(item.split("=", 1) for item in raw_info.split(";") if "=" in item)
            pos = int(pos)
            key = chrom, pos, ref, alt
            label = LABELS.get(unquote(info.get("CLNSIG", "")))
            if "CLNSIGCONF" in info or label is None:
                conflicts.add(key)
                continue
            if unquote(info.get("CLNREVSTAT", "")) not in REVIEWED:
                continue
            genes = {record.split(":", 1)[0] for record in info.get("GENEINFO", "").split("|")}
            consequences = {term.split("|")[-1] for term in info.get("MC", "").split(",")}
            matches = set()
            for gene in genes:
                records = by_gene.get((chrom, gene), ())
                left = bisect_left(records, (pos - 32,))
                right = bisect_right(records, (pos + 32, "~"))
                for anchor, strand, annotation in records[left:right]:
                    relative = (pos - anchor) * (1 if strand == "+" else -1)
                    canonical = relative in ((0, 1) if annotation == "donor" else (-1, 0))
                    if label == 1 and not (canonical and CONSEQUENCES[annotation] in consequences):
                        continue
                    if label == 0 and (canonical or not -32 <= relative <= 1):
                        continue
                    matches.add((anchor, strand, annotation, gene))
            if len(matches) != 1:
                continue
            anchor, strand, annotation, gene = next(iter(matches))
            if not _usable_context(genome, chrom, anchor, strand):
                continue
            if genome.reference_segment(chrom, pos - 1, pos) != ref:
                continue
            variant = Variant(chrom, pos, ref, alt, gene, label)
            row = dict(chrom=chrom, pos=pos, ref=ref, alt=alt, gene=gene, strand=strand,
                       anchor_pos=anchor, annotation=annotation, label=label,
                       variant_id=variant.id, clinvar_variation_id=accession,
                       consequence=", ".join(sorted(consequences - {""})),
                       hgvs=unquote(info.get("CLNHGVS", "")))
            if key in candidates and candidates[key] != row:
                conflicts.add(key)
            candidates[key] = row

    rows = [row for key, row in sorted(candidates.items()) if key not in conflicts]
    rng.shuffle(rows)
    selected, seen_genes = [], set()
    for label in (0, 1):
        for annotation, size in (("donor", count // 2), ("acceptor", count - count // 2)):
            group = []
            for row in rows:
                if row["label"] != label or row["annotation"] != annotation or row["gene"] in seen_genes:
                    continue
                group.append(row)
                seen_genes.add(row["gene"])
                if len(group) == size:
                    break
            if len(group) != size:
                raise ValueError(f"Only {len(group)} usable ClinVar challenge samples for label={label}, {annotation}; need {size}")
            selected.extend(group)
    return selected, release


def prepare_cohort(gtf_path, vcf_path, fasta_path, *, sites_per_kind=(16, 8, 8),
                   variants_per_kind=12, seed=42):
    """Return three REF splits and a separate ClinVar challenge, using 1-based coordinates."""
    if len(sites_per_kind) != 3 or any(type(n) is not int or n < 1 for n in sites_per_kind):
        raise ValueError("sites_per_kind requires positive train, val, and test counts")
    if type(variants_per_kind) is not int or variants_per_kind < 2:
        raise ValueError("variants_per_kind requires at least two samples per clinical label")
    annotation_path, vcf_path, fasta_path = map(Path, (gtf_path, vcf_path, fasta_path))
    rng = random.Random(seed)
    sites = _sites(annotation_path)
    with ReferenceGenome(fasta_path) as genome:
        annotation, used_genes = _annotation_rows(sites, genome, sites_per_kind, rng)
        variants, release = _challenge_rows(vcf_path, sites, genome, used_genes, variants_per_kind, rng)
    return {
        "schema_version": 1,
        "genome_build": "GRCh38",
        "seed": seed,
        "annotation_source": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "vcf_source": str(vcf_path),
        "vcf_sha256": sha256_file(vcf_path),
        "clinvar_release": release,
        "fasta_sha256": sha256_file(fasta_path),
        "chromosome_partitions": {key: sorted(values) for key, values in SPLIT_CHROMS.items()},
        "site_anchor": "first intronic base for donor and last intronic base for acceptor in transcript direction",
        "challenge_sampling": "canonical splice annotated pathogenic SNVs versus nearby noncanonical benign controls",
        "annotation": annotation,
        "variants": variants,
    }
