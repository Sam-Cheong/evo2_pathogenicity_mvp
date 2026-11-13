"""Fit human splice-site probes and turn public ClinVar SNV scores into evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import local

from filelock import FileLock
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .evee_features import FEATURE_DIM, reverse_complement
from .evee_short_window import ENDPOINT, LAYER, MODEL
from .metrics import binary_metrics
from .nvidia_encoder import NvidiaEvo2Encoder
from .sequence import ReferenceGenome, Variant
from .two_stage_data import _atomic_npz
from .utils import fingerprint, save_json, sha256_file


WINDOW = 256
TOKEN = 192
CLASSES = ("acceptor", "background", "donor")
PROTOCOL = {
    "model": MODEL, "endpoint": ENDPOINT, "layer": LAYER,
    "window": WINDOW, "scored_token": TOKEN, "anchor_offset_in_transcript_direction": 1,
    "genome_build": "GRCh38", "revision": "nvidia-hosted-splice-block27-v1",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def context(genome, row, variant=None):
    """Use an identical transcript-oriented window for both alleles."""
    direction = {"+": 1, "-": -1}[row["strand"]]
    anchor = int(row.get("anchor_pos", row["pos"]))
    scored = anchor + direction
    genomic_index = TOKEN if direction == 1 else WINDOW - 1 - TOKEN
    start = scored - 1 - genomic_index
    ref = genome.reference_segment(row["chrom"], start, start + WINDOW)
    alt = None
    if variant is not None:
        index = variant.pos - 1 - start
        if not 0 <= index < WINDOW or ref[index] != variant.ref:
            raise ValueError(f"SNV outside splice window or REF mismatch for {variant.id}")
        if direction * (scored - variant.pos) < 0:
            raise ValueError(f"SNV is beyond the causal scoring position for {variant.id}")
        alt = ref[:index] + variant.alt + ref[index + 1:]
    if direction == -1:
        ref = reverse_complement(ref)
        alt = reverse_complement(alt) if alt is not None else None
    return ref, alt


class TokenCache:
    """Persist only the selected token vector for each input sequence."""

    def __init__(self, directory, workers):
        self.directory = directory
        self.workers = workers
        self.state = local()
        directory.mkdir(parents=True, exist_ok=True)

    def get(self, sequence):
        key = fingerprint({"protocol": PROTOCOL, "sequence": sequence})
        path = self.directory / f"{key}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                vector = archive["vector"]
        else:
            if not hasattr(self.state, "encoder"):
                self.state.encoder = NvidiaEvo2Encoder(
                    model=MODEL, endpoint=ENDPOINT, layer=LAYER,
                    service_revision=PROTOCOL["revision"], timeout_seconds=45,
                    poll_timeout_seconds=60, max_retries=3,
                )
            hidden = np.asarray(self.state.encoder.token_activations(sequence))
            if hidden.shape != (WINDOW, FEATURE_DIM):
                raise ValueError(f"Unexpected aligned activation shape {hidden.shape}")
            vector = hidden[TOKEN].astype(np.float32, copy=True)
            if not np.isfinite(vector).all():
                raise ValueError("Nonfinite activation")
            _atomic_npz(path, {"vector": vector})
        if vector.shape != (FEATURE_DIM,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
            raise ValueError("Invalid cached splice activation")
        return vector

    def collect(self, pairs):
        sequences = list(dict.fromkeys(sequence for pair in pairs for sequence in pair if sequence is not None))
        print(f"Preparing {len(sequences)} unique allele windows", flush=True)
        result = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for i, (seq, vector) in enumerate(zip(sequences, pool.map(self.get, sequences)), 1):
                result[seq] = vector
                if i % 8 == 0 or i == len(sequences):
                    print(f"Features {i}/{len(sequences)} ready", flush=True)
        return result


def probabilities(artifact, x):
    x = (np.asarray(x, dtype=np.float64) - artifact["mean"]) / artifact["scale"]
    logits = x @ np.asarray(artifact["coef"]).T + artifact["intercept"]
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def evidence_for(row, artifact, vectors, *, validated):
    variant = Variant(row["chrom"], row["pos"], row["ref"], row["alt"], row["gene"])
    metadata = {"variant_id": variant.id, "gene": variant.gene}
    for key in ("clinvar_variation_id", "consequence", "hgvs_c", "hgvs_p"):
        if row.get(key):
            metadata[key] = row[key]
    if row.get("hgvs"):
        metadata["hgvs_g"] = row["hgvs"]
    ref, alt = probabilities(artifact, vectors)
    sites = []
    for annotation in ("donor", "acceptor"):
        index = artifact["classes"].index(annotation)
        sites.append({
            "annotation": annotation, "position": int(row["anchor_pos"]), "strand": row["strand"],
            "ref_score": float(ref[index]), "alt_score": float(alt[index]),
            "delta": float(alt[index] - ref[index]),
        })
    sites.sort(key=lambda item: -abs(item["delta"]))
    return {"variant": metadata, "disruptions": sites[:10],
            "evaluation": {"validated": validated}, "protocol": PROTOCOL}


def annotation_metrics(rows, artifact, vectors):
    p = probabilities(artifact, vectors)
    return {cls: binary_metrics([int(r["annotation"] == cls) for r in rows], p[:, i])
            for i, cls in enumerate(artifact["classes"]) if cls != "background"}


def fit(cohort_path, fasta, output, workers):
    cohort = read_json(cohort_path)
    if cohort.get("schema_version") != 1 or cohort.get("genome_build") != "GRCh38":
        raise ValueError("Expected a prepared GRCh38 splice cohort")
    partitions = cohort["annotation"]
    training = partitions["train"]
    testing = partitions["test"]
    for name, rows in partitions.items():
        if set(r["annotation"] for r in rows) != set(CLASSES):
            raise ValueError(f"All three annotation labels required in {name}")
    groups = [{r["gene"] for r in rows} for rows in partitions.values()]
    if any(left & right for i, left in enumerate(groups) for right in groups[i+1:]):
        raise ValueError("Annotation gene overlap across splits")
    variants = cohort["variants"]
    if any(set(r["gene"].split(";")) & set.union(*groups) for r in variants):
        raise ValueError("ClinVar challenge genes overlap annotation probe genes")
    if {row["label"] for row in variants} != {0, 1}:
        raise ValueError("The ClinVar challenge requires both classes")
    fasta_hash = sha256_file(fasta)
    if cohort.get("fasta_sha256", fasta_hash) != fasta_hash:
        raise ValueError("FASTA differs from prepared cohort")
    with ReferenceGenome(fasta) as genome:
        ref_pairs = {name: [context(genome, row) for row in rows] for name, rows in partitions.items()}
        variant_pairs = [
            context(genome, r, Variant(r["chrom"], r["pos"], r["ref"], r["alt"], r["gene"]))
            for r in variants
        ]
    cache = TokenCache(output / "tokens", workers)
    vectors = cache.collect([pair for pairs in ref_pairs.values() for pair in pairs] + variant_pairs)
    x = {name: np.stack([vectors[seq] for seq, _ in pairs]) for name, pairs in ref_pairs.items()}
    scaler = StandardScaler().fit(x["train"])
    classifier = LogisticRegression(max_iter=1500, random_state=42).fit(
        scaler.transform(x["train"]), [r["annotation"] for r in training],
    )
    if classifier.n_iter_.max() >= classifier.max_iter:
        raise RuntimeError("Splice-site logistic probe did not converge")
    artifact = {
        "protocol": PROTOCOL, "fasta_sha256": fasta_hash, "cohort_sha256": sha256_file(cohort_path),
        "classes": classifier.classes_.tolist(), "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(), "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    save_json(output / "splice_probe.json", artifact)
    metrics = {name: annotation_metrics(rows, artifact, x[name])
               for name, rows in partitions.items() if name != "train"}
    validated = False  # This pilot supplies held-out metrics, with clinical validation outside its scope.
    items = [evidence_for(r, artifact, [vectors[ref], vectors[alt]], validated=validated)
             for r, (ref, alt) in zip(variants, variant_pairs)]
    scores = [(1 - next(d["delta"] for d in item["disruptions"] if d["annotation"] == row["annotation"])) / 2
              for row, item in zip(variants, items)]
    result = {"annotation": metrics, "annotation_counts": {k: len(v) for k, v in partitions.items()},
              "clinvar_challenge": binary_metrics([int(r["label"]) for r in variants], scores),
              "clinvar_score": "decrease in the corresponding annotation score at the specified anchor",
              "research_only": True}
    save_json(output / "metrics.json", result)
    for item in items:
        save_json(output / "evidence" / (item["variant"]["variant_id"] + ".json"), item)
    print(json.dumps(result, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    prepare = phases.add_parser("prepare")
    prepare.add_argument("--gtf", type=Path, required=True, help="GENCODE GTF or UCSC genePred table")
    prepare.add_argument("--vcf", type=Path, default=Path("data/clinvar.vcf.gz"))
    prepare.add_argument("--fasta", type=Path, default=Path("data/GRCh38.fa"))
    prepare.add_argument("--cohort", type=Path, required=True)
    prepare.add_argument("--sites-per-kind", nargs=3, type=int, default=(16, 8, 8))
    prepare.add_argument("--variants-per-kind", type=int, default=12)
    train = phases.add_parser("fit")
    train.add_argument("--cohort", type=Path, required=True)
    train.add_argument("--fasta", type=Path, default=Path("data/GRCh38.fa"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    interpret = phases.add_parser("explain")
    interpret.add_argument("--evidence", type=Path, required=True)
    interpret.add_argument("--model", required=True)
    score = phases.add_parser("score")
    score.add_argument("--cohort", type=Path, required=True)
    score.add_argument("--fasta", type=Path, default=Path("data/GRCh38.fa"))
    score.add_argument("--output", type=Path, required=True, help="Directory of the fitted pilot")
    score.add_argument("--variant-id", required=True, help="ID in the prepared public ClinVar challenge")
    score.add_argument("--model", help="Optionally send the scored evidence to an OpenAI Responses model")
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        from .splice_cohort import prepare_cohort

        cohort = prepare_cohort(args.gtf, args.vcf, args.fasta,
                                sites_per_kind=tuple(args.sites_per_kind),
                                variants_per_kind=args.variants_per_kind)
        save_json(args.cohort, cohort)
        print(json.dumps({"annotation": {k: len(v) for k, v in cohort["annotation"].items()},
                          "clinvar": len(cohort["variants"])}))
        return
    if args.phase == "score":
        artifact = read_json(args.output / "splice_probe.json")
        if (artifact["protocol"] != PROTOCOL or artifact["fasta_sha256"] != sha256_file(args.fasta)
                or artifact["cohort_sha256"] != sha256_file(args.cohort)):
            parser.error("The model requires its recorded protocol, reference and cohort")
        matches = [r for r in read_json(args.cohort)["variants"] if r["variant_id"] == args.variant_id]
        if len(matches) != 1:
            parser.error("Choose a unique variant ID from the prepared ClinVar challenge")
        row = matches[0]
        variant = Variant(row["chrom"], row["pos"], row["ref"], row["alt"], row["gene"])
        with ReferenceGenome(args.fasta) as genome:
            ref, alt = context(genome, row, variant)
        with FileLock(str(args.output / ".fit.lock"), timeout=0):
            cache = TokenCache(args.output / "tokens", workers=1)
            item = evidence_for(row, artifact, [cache.get(ref), cache.get(alt)], validated=False)
        evidence_path = args.output / "evidence" / (variant.id + ".json")
        save_json(evidence_path, item)
        if args.model:
            from .splice_explanation import explain

            item["explanation"] = explain(item, args.model)
        print(json.dumps(item, ensure_ascii=False, indent=2))
        return
    if args.phase == "explain":
        from .splice_explanation import explain

        print(explain(read_json(args.evidence), args.model))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".fit.lock"), timeout=0):
        fit(args.cohort, args.fasta, args.output, args.workers)


if __name__ == "__main__":
    main()
