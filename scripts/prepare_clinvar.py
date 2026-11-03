"""Filter an official ClinVar GRCh38 VCF into traceable high-quality SNVs.

Every VCF row is examined before sampling, so late duplicate conflicts cannot
escape detection. Disk-backed SQLite bounds memory during full ClinVar scans.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import unquote

from evo2_mvp.sequence import ReferenceGenome, Variant


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clinical_label(value: str):
    value = unquote(value).replace("_", " ").strip().lower()
    allowed = {"pathogenic": 1, "likely pathogenic": 1, "pathogenic/likely pathogenic": 1,
               "benign": 0, "likely benign": 0, "benign/likely benign": 0}
    return allowed.get(value)


def review_stars(value: str) -> int:
    value = re.sub(r"\s+", " ", unquote(value).replace("_", " ").strip().lower())
    value = re.sub(r"\s*,\s*", ", ", value)
    return {"practice guideline": 4, "reviewed by expert panel": 3,
            "criteria provided, multiple submitters, no conflicts": 2}.get(value, 0)


def prepare(vcf, fasta, out, release_date, window=256, target=2000, seed=42,
            source_url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"):
    date.fromisoformat(release_date)
    if target < 1:
        raise ValueError("target must be positive")
    vcf, fasta, out = Path(vcf), Path(fasta), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(vcf)
    counters = Counter()
    header = []
    opener = gzip.open if vcf.suffix == ".gz" else open
    with tempfile.TemporaryDirectory(prefix="clinvar_", dir=out.parent) as tempdir:
        connection = sqlite3.connect(str(Path(tempdir) / "dedup.sqlite"))
        connection.execute("CREATE TABLE records (id TEXT PRIMARY KEY, label INTEGER, payload TEXT, blocked INTEGER)")
        with opener(vcf, "rt", encoding="utf-8") as handle, ReferenceGenome(fasta) as genome:
            for line in handle:
                if line.startswith("##"):
                    header.append(line.rstrip())
                    if line.lower().startswith(("##reference=", "##assembly=")) and "grch37" in line.lower():
                        raise ValueError("VCF header specifies GRCh37; GRCh38 is required")
                    if line.startswith("##fileDate="):
                        file_date = line.split("=", 1)[1].strip()
                        if len(file_date) == 8 and file_date.isdigit():
                            file_date = f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:]}"
                        if file_date != release_date:
                            raise ValueError(f"Requested release date {release_date} differs from VCF fileDate {file_date}")
                    continue
                if line.startswith("#"):
                    continue
                counters["rows_examined"] += 1
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    counters["malformed"] += 1
                    continue
                chrom, pos, accession, ref, alt, _, _, raw_info = fields[:8]
                try:
                    variant = Variant(chrom, pos, ref, alt)
                except ValueError:
                    counters["not_autosomal_snv"] += 1
                    continue
                info = dict(item.split("=", 1) if "=" in item else (item, "")
                            for item in raw_info.split(";") if item)
                label = clinical_label(info.get("CLNSIG", ""))
                conflicting = "CLNSIGCONF" in info or "conflict" in info.get("CLNSIG", "").lower()
                stars = review_stars(info.get("CLNREVSTAT", ""))
                payload = None
                if label is None or conflicting:
                    counters["excluded_classification"] += 1
                elif stars < 2:
                    counters["excluded_review"] += 1
                else:
                    gene = ";".join(sorted({item.split(":", 1)[0] for item in info.get("GENEINFO", "").split("|")
                                            if item and item != "."}))
                    try:
                        genome.context(variant, window)
                    except (ValueError, KeyError) as exc:
                        counters["invalid_context_or_ref"] += 1
                    else:
                        payload = {"chrom": variant.chrom, "pos": variant.pos, "ref": variant.ref,
                                   "alt": variant.alt, "gene": gene, "label": label,
                                   "variant_id": variant.id, "genome_build": "GRCh38",
                                   "source": "ClinVar", "source_url": source_url,
                                   "source_accession": accession, "review_status": info.get("CLNREVSTAT", ""),
                                   "review_stars": stars, "clinical_significance": info.get("CLNSIG", ""),
                                   "data_release_date": release_date, "source_sha256": source_sha256,
                                   "synthetic": "false"}
                previous = connection.execute("SELECT label,payload,blocked FROM records WHERE id=?", (variant.id,)).fetchone()
                blocked = bool(conflicting or label is None)
                if previous is None:
                    connection.execute("INSERT INTO records VALUES (?,?,?,?)",
                                       (variant.id, label, json.dumps(payload) if payload else None, int(blocked)))
                else:
                    counters["duplicate_rows"] += 1
                    old_label, old_payload, old_blocked = previous
                    blocked = blocked or bool(old_blocked) or old_label != label
                    if old_payload:
                        old = json.loads(old_payload)
                        combined_genes = set(old["gene"].split(";"))
                        combined_accessions = set(old["source_accession"].split(";"))
                        if payload:
                            combined_genes.update(payload["gene"].split(";"))
                            combined_accessions.update(payload["source_accession"].split(";"))
                        if payload is None or old["review_stars"] >= stars:
                            payload = old
                        # Preserve annotations even if the retained review has more stars.
                        payload["gene"] = ";".join(sorted(combined_genes - {""}))
                        payload["source_accession"] = ";".join(sorted(combined_accessions - {""}))
                    connection.execute("UPDATE records SET payload=?,blocked=? WHERE id=?",
                                       (json.dumps(payload) if payload and not blocked else None, int(blocked), variant.id))
                if counters["rows_examined"] % 50000 == 0:
                    connection.commit()
            connection.commit()
        eligible_count = connection.execute("SELECT COUNT(*) FROM records WHERE blocked=0 AND payload IS NOT NULL").fetchone()[0]
        # Deterministic reservoir sample over all eligible records, no first-chromosome truncation.
        rng, selected = random.Random(seed), []
        for i, (payload,) in enumerate(connection.execute("SELECT payload FROM records WHERE blocked=0 AND payload IS NOT NULL ORDER BY id")):
            record = json.loads(payload)
            if i < target:
                selected.append(record)
            else:
                index = rng.randrange(i + 1)
                if index < target:
                    selected[index] = record
        connection.close()
    selected.sort(key=lambda r: (int(r["chrom"]), r["pos"], r["ref"], r["alt"]))
    columns = ["chrom", "pos", "ref", "alt", "gene", "label", "variant_id", "genome_build", "source",
               "source_url", "source_accession", "review_status", "review_stars", "clinical_significance",
               "data_release_date", "source_sha256", "synthetic"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)
    report = {"source": "ClinVar", "source_url": source_url, "source_path": str(vcf.resolve()),
              "source_sha256": source_sha256, "data_release_date": release_date,
              "genome_build": "GRCh38", "fasta_path": str(fasta.resolve()),
              "fasta_sha256": sha256_file(fasta), "window": window, "seed": seed,
              "requested": target, "eligible": eligible_count, "selected": len(selected),
              "target_reached": len(selected) >= target, "label_counts": dict(Counter(str(r["label"]) for r in selected)),
              "missing_gene": sum(not r["gene"] for r in selected), "filter_counts": dict(counters),
              "vcf_header": header, "clinical_use": "research_only"}
    out.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True, help="Official ClinVar GRCh38 VCF or VCF.gz")
    parser.add_argument("--fasta", required=True, help="GRCh38 uncompressed/BGZF reference FASTA")
    parser.add_argument("--out", required=True)
    parser.add_argument("--release-date", required=True, help="Source release date, YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-url", default="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz")
    args = parser.parse_args()
    print(json.dumps(prepare(args.vcf, args.fasta, args.out, args.release_date,
                             args.window, args.target, args.seed, args.source_url), indent=2))


if __name__ == "__main__":
    main()
