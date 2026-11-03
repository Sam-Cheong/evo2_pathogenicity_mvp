"""Create deterministic SYNTHETIC pipeline fixtures; never a performance dataset."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()
    if args.n < 6:
        parser.error("--n must be >=6 for pipeline splits")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sequence = "ACGT" * ((args.n * 300 + 1000) // 4)
    with (out / "toy.fa").open("w", encoding="ascii") as handle:
        handle.write(">1 SYNTHETIC_NOT_GRCh38\n")
        for i in range(0, len(sequence), 80):
            handle.write(sequence[i:i+80] + "\n")
    with (out / "variants.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chrom", "pos", "ref", "alt", "gene", "label", "synthetic", "source"])
        writer.writeheader()
        for i in range(args.n):
            pos = 400 + 300*i
            ref = sequence[pos-1]
            alt = {"A": "C", "C": "G", "G": "T", "T": "A"}[ref]
            writer.writerow({"chrom": "1", "pos": pos, "ref": ref, "alt": alt,
                             "gene": f"TOY_GENE_{i//2}", "label": i % 2, "synthetic": "true", "source": "synthetic"})
    (out / "metadata.json").write_text(json.dumps({"synthetic": True, "n": args.n,
        "warning": "Synthetic sequence and arbitrary labels. Pipeline verification only; no pathogenicity performance claims."}, indent=2))
    print(json.dumps({"fasta": str(out / "toy.fa"), "variants": str(out / "variants.csv"),
                      "synthetic": True, "n": args.n}))


if __name__ == "__main__":
    main()
