"""Prepare label-isolated two-stage student data from the existing Evo 2 cache."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evo2_mvp.two_stage_data import prepare_two_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", default="outputs/sh2_two_stage_20261104/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=256)
    args = parser.parse_args()
    metadata = prepare_two_stage(args.variants, args.splits, args.cache, args.fasta,
                                 args.output, args.seed, args.window)
    print(json.dumps({"status": metadata["status"], "output": str(Path(args.output).resolve()),
        "counts": metadata["counts"], "representation_partition": metadata["representation_partition"],
        "validation": metadata["validation"], "files_sha256": metadata["files_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
