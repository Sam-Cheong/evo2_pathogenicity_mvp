"""Build a local exact-allele index from the public GRCh38 ClinVar VCF."""
import gzip
from pathlib import Path
import sqlite3


def build(source, destination):
    temporary = destination.with_suffix(".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    autosomes = {str(n) for n in range(1, 23)}
    try:
        with sqlite3.connect(temporary) as db, gzip.open(source, "rt", encoding="utf-8") as vcf:
            db.execute("CREATE TABLE snvs (key TEXT PRIMARY KEY, accession TEXT) WITHOUT ROWID")
            db.execute("CREATE TABLE source (size INTEGER, mtime_ns INTEGER, release TEXT)")
            batch = []
            release = None
            for line in vcf:
                if line.startswith("##fileDate="):
                    release = line.strip().split("=", 1)[1]
                if line.startswith("#"):
                    continue
                chrom, position, accession, ref, alt, _ = line.split("\t", 5)
                if chrom in autosomes and len(ref) == len(alt) == 1 and ref in "ACGT" and alt in "ACGT" and ref != alt:
                    batch.append((f"{chrom}-{position}-{ref}-{alt}", accession))
                if len(batch) >= 10000:
                    db.executemany("INSERT OR IGNORE INTO snvs VALUES (?, ?)", batch)
                    batch.clear()
            db.executemany("INSERT OR IGNORE INTO snvs VALUES (?, ?)", batch)
            info = source.stat()
            db.execute("INSERT INTO source VALUES (?, ?, ?)", (info.st_size, info.st_mtime_ns, release))
            print(f"Indexed {db.execute('SELECT COUNT(*) FROM snvs').fetchone()[0]} ClinVar SNVs ({release})")
        db.close()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    build(Path("data/clinvar.vcf.gz"), Path("work/splice_web/clinvar_snvs.sqlite"))
