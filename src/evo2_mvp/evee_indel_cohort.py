"""Select a deterministic, gene-held-out ClinVar indel evaluation cohort."""
from __future__ import annotations

import csv
import gzip
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

from .evee_indel import IndelVariant, indel_contexts
from .sequence import ReferenceGenome
from .utils import sha256_file


CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
_AUTOSOMES = {str(i) for i in range(1, 23)}
_BASES = frozenset("ACGT")
_LABELS = {
    "Benign": 0,
    "Likely_benign": 0,
    "Benign/Likely_benign": 0,
    "Pathogenic": 1,
    "Likely_pathogenic": 1,
    "Pathogenic/Likely_pathogenic": 1,
}
_STARS = {
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "reviewed_by_expert_panel": 3,
    "practice_guideline": 4,
}
_STRATA = [(kind, label) for kind in ("deletion", "insertion") for label in (0, 1)]


def _gene_names(value: str) -> set[str]:
    return {name.strip().upper() for name in re.split(r"[;,|]", value)
            if name.strip().upper() not in {"", ".", "-", "NA", "N/A", "NONE", "UNKNOWN"}}


def _held_out_genes(source_variants: Path, training_cohort: dict) -> set[str]:
    genes = set()
    with source_variants.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "gene" not in (reader.fieldnames or []):
            raise ValueError("The original SNV CSV must include gene annotations")
        for row in reader:
            genes.update(_gene_names(row["gene"]))
    for partition in training_cohort["partitions"].values():
        for row in partition:
            genes.update(_gene_names(row["gene"]))
    if not genes:
        raise ValueError("No SNV cohort genes could be identified for exclusion")
    return genes


def _read_candidates(vcf: Path, max_indel_bases: int) -> tuple[dict, dict, Counter]:
    candidates = {}
    blocked = set()
    header = {}
    counts = Counter()
    opener = gzip.open if vcf.suffix in {".gz", ".bgz"} else open
    with opener(vcf, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                key, _, value = line.rstrip().partition("=")
                if key in {"##fileDate", "##reference", "##assembly"}:
                    header[key[2:]] = value
                continue
            counts["vcf_records"] += 1
            fields = line.rstrip("\n").split("\t", 7)
            if len(fields) != 8:
                continue
            chrom, pos, accession, ref, alt, _, _, raw_info = fields
            chrom = chrom.removeprefix("chr")
            length = abs(len(ref) - len(alt))
            if (chrom not in _AUTOSOMES or min(len(ref), len(alt)) != 1
                    or not 1 <= length <= max_indel_bases or not pos.isdecimal()):
                continue
            shorter, longer = (ref, alt) if len(ref) < len(alt) else (alt, ref)
            pos = int(pos)
            if pos < 1 or not longer.startswith(shorter) or set(longer) - _BASES:
                continue
            counts["simple_indel_rows"] += 1
            identifier = f"{chrom}-{pos}-{ref}-{alt}"
            if identifier in blocked:
                continue
            info = {key: value for key, _, value in (item.partition("=") for item in raw_info.split(";"))}
            clinical_significance = info.get("CLNSIG", "")
            label = _LABELS.get(clinical_significance)
            if label is None or "CLNSIGCONF" in info:
                blocked.add(identifier)
                candidates.pop(identifier, None)
                counts["excluded_classification"] += 1
                continue
            genes = _gene_names("|".join(gene.split(":", 1)[0]
                                         for gene in info.get("GENEINFO", "").split("|")))
            review_status = info.get("CLNREVSTAT", "")
            stars = _STARS.get(review_status, 0)
            old = candidates.get(identifier)
            if old is not None:
                counts["duplicate_rows"] += 1
                if old["label"] != label:
                    blocked.add(identifier)
                    candidates.pop(identifier)
                    counts["conflicting_duplicates"] += 1
                    continue
                genes.update(_gene_names(old["gene"]))
            if old is None or stars > old["review_stars"]:
                candidates[identifier] = {
                    "variant_id": identifier, "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                    "label": label, "indel_type": "deletion" if len(ref) > len(alt) else "insertion",
                    "indel_bases": length, "source_accession": accession, "review_stars": stars,
                    "review_status": review_status, "clinical_significance": clinical_significance,
                    "clinvar_variant_type": info.get("CLNVC", ""),
                }
            candidates[identifier]["gene"] = ";".join(sorted(genes))
    return candidates, header, counts


def prepare_indel_cohort(vcf, fasta, source_variants, training_cohort, *,
                         per_stratum=8, seed=42, max_indel_bases=8, window=256) -> dict:
    """Balance insertion/deletion and label on genes absent from the original SNV data."""
    if type(per_stratum) is not int or per_stratum < 1:
        raise ValueError("per_stratum must be a positive integer")
    if type(window) is not int or not 3 <= window <= 8192:
        raise ValueError("window must be an integer between 3 and 8192")
    effect_region = window - 3 * window // 4
    if type(max_indel_bases) is not int or not 1 <= max_indel_bases < effect_region:
        raise ValueError("The indel length limit must fit in the short-window effect region")
    vcf, fasta, source_variants, training_cohort = map(Path, (vcf, fasta, source_variants, training_cohort))
    reference_cohort = json.loads(training_cohort.read_text(encoding="utf-8"))
    reference_sha256 = sha256_file(fasta)
    if reference_cohort["genome_build"] != "GRCh38" or reference_sha256 != reference_cohort["reference_sha256"]:
        raise ValueError("Indel evaluation requires the SNV probe's GRCh38 reference FASTA")
    excluded_genes = _held_out_genes(source_variants, reference_cohort)
    candidates, header, counts = _read_candidates(vcf, max_indel_bases)
    if "GRCh38" not in header.get("reference", header.get("assembly", "")):
        raise ValueError("The ClinVar VCF header must identify GRCh38")
    release = header.get("fileDate", "")
    if len(release) == 8 and release.isdecimal():
        release = f"{release[:4]}-{release[4:6]}-{release[6:]}"
    release = date.fromisoformat(release).isoformat()
    strata = {key: [] for key in _STRATA}
    for record in candidates.values():
        genes = _gene_names(record["gene"])
        if record["review_stars"] < 2:
            counts["excluded_review"] += 1
        elif not genes or genes & excluded_genes:
            counts["excluded_snv_or_unknown_gene"] += 1
        elif not record["source_accession"].isdecimal():
            counts["excluded_missing_clinvar_accession"] += 1
        else:
            strata[(record["indel_type"], record["label"])].append((record, genes))

    rng = random.Random(seed)
    selected, selected_counts, used_genes = [], Counter(), set()
    with ReferenceGenome(fasta) as genome:
        for key in sorted(_STRATA, key=lambda stratum: len(strata[stratum])):
            eligible = sorted(strata[key], key=lambda row: row[0]["variant_id"])
            rng.shuffle(eligible)
            for record, genes in eligible:
                if genes & used_genes:
                    continue
                variant = IndelVariant(record["chrom"], record["pos"], record["ref"], record["alt"],
                                       record["gene"], record["label"])
                try:
                    indel_contexts(genome, variant, window, 3 * window // 4)
                except ValueError:
                    counts["excluded_invalid_context"] += 1
                    continue
                selected.append(record)
                used_genes.update(genes)
                selected_counts[key] += 1
                if selected_counts[key] == per_stratum:
                    break
            if selected_counts[key] != per_stratum:
                raise ValueError(f"Only {selected_counts[key]}/{per_stratum} valid gene-held-out records in {key}")

    selected.sort(key=lambda row: (int(row["chrom"]), row["pos"], row["ref"], row["alt"]))
    return {
        "schema_version": 1,
        "genome_build": "GRCh38",
        "source": "ClinVar",
        "source_url": CLINVAR_URL,
        "source_sha256": sha256_file(vcf),
        "clinvar_release": release,
        "reference_sha256": reference_sha256,
        "source_variants_sha256": sha256_file(source_variants),
        "training_cohort_sha256": sha256_file(training_cohort),
        "selection": "balanced_by_indel_type_and_label_one_per_gene_disjoint_from_source_snv_cohorts",
        "minimum_review_stars": 2,
        "max_indel_bases": max_indel_bases,
        "per_stratum": per_stratum,
        "seed": seed,
        "window": window,
        "excluded_snv_genes": len(excluded_genes),
        "eligible_counts": {f"{kind}_{label}": len(strata[(kind, label)]) for kind, label in _STRATA},
        "filter_counts": dict(counts),
        "records": selected,
    }
