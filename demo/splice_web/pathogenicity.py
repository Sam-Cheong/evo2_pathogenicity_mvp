"""Score a GRCh38 SNV with the saved EVEE covariance head."""
from collections import OrderedDict
import re
import sqlite3
from threading import Lock

import numpy as np
import torch

from evo2_mvp.evee_covariance import EVEEStyleCovarianceProbe
from evo2_mvp.evee_features import (
    EVEEFeatureCache, ORIENTATIONS, _fetch_direction, _metadata, _sequence_hash, variant_contexts,
)
from evo2_mvp.nvidia_encoder import NvidiaEvo2Encoder
from evo2_mvp.sequence import ReferenceGenome, Variant
from evo2_mvp.utils import sha256_file
from local_probe import ChatInputError, RESIDUAL_DNA, VARIANT_ID


LOCATION = re.compile(r"GRCh38\s+chr(\d{1,2}):([1-9]\d{0,9})", re.I)
DNA = re.compile(r"[ACGT\s]+", re.I)
CACHED_LIMIT = 32


class PathogenicityProbe:
    def __init__(self, run, fasta):
        self.fasta = fasta
        self.clinvar_index = fasta.parent.parent / 'work/splice_web/clinvar_snvs.sqlite'
        self.reference_hash = sha256_file(fasta)
        self.cache = EVEEFeatureCache(run / "features")
        self.protocol = self.cache.metadata
        expected = {
            "reference_sha256": self.reference_hash, "genome_build": "GRCh38", "window": 256,
            "top_k": 32, "feature_dim": 4096, "layer": "blocks.27", "model": "arc/evo2-7b",
        }
        if any(self.protocol.get(key) != value for key, value in expected.items()):
            raise ValueError("The GRCh38 reference or feature protocol differs from the training run.")
        state = torch.load(run / "probe" / "evee_covariance.pt", map_location="cpu", weights_only=True)
        if (state["artifact_kind"] != "evee_style_covariance_probe"
                or state["config"]["input_shape"] != [2, 32, 4096]):
            raise ValueError("Expected the trained 7B covariance probe.")
        self.model = EVEEStyleCovarianceProbe(**state["model_kwargs"]).eval()
        self.model.load_state_dict(state["model_state"])
        self.recent = OrderedDict()
        self.lock = Lock()

    def resolve(self, message):
        lines = message.strip().splitlines()
        if not lines or not re.match(r"GRCh38\b", lines[0].strip(), re.I):
            return None
        location = LOCATION.fullmatch(lines[0].strip())
        if not location or len(lines) < 2 or not lines[1].strip().upper().startswith("DNA:"):
            raise ChatInputError("Use GRCh38 chr19:11110700 on the first line and DNA: followed by your sequence on the second.")
        sequence_lines = [lines[1].strip()[4:].strip()]
        end = 2
        while end < len(lines) and DNA.fullmatch(lines[end]):
            sequence_lines.append(lines[end])
            end += 1
        sequence = "".join("".join(sequence_lines).split()).upper()
        if not 20 <= len(sequence) <= 2000 or not re.fullmatch(r"[ACGT]+", sequence):
            raise ChatInputError("Enter 20–2,000 A/C/G/T bases in the forward genomic orientation.")
        question = "\n".join(lines[end:]).strip()
        if RESIDUAL_DNA.search(question) or any(line.lstrip().startswith(">") for line in lines[end:]):
            raise ChatInputError("Put the DNA block and your question on separate lines.")
        chrom, start = location[1], int(location[2])
        with ReferenceGenome(self.fasta) as genome:
            ref = genome.reference_segment(chrom, start - 1, start - 1 + len(sequence))
        changes = [(start + i, a, b) for i, (a, b) in enumerate(zip(ref, sequence)) if a != b]
        if not changes:
            raise ChatInputError("No base differences found at the supplied coordinates.")
        selected = {"differences_found": len(changes)}
        if len(changes) == 1:
            variant = Variant(chrom, *changes[0])
        else:
            if not self.clinvar_index.is_file():
                raise ChatInputError("The local ClinVar SNV index is unavailable.")
            with sqlite3.connect(self.clinvar_index.as_uri() + "?mode=ro", uri=True) as db:
                size, modified, release = db.execute("SELECT * FROM source").fetchone()
                info = (self.fasta.parent / "clinvar.vcf.gz").stat()
                if (size, modified) != (info.st_size, info.st_mtime_ns):
                    raise ChatInputError("The local ClinVar SNV index needs to be rebuilt.")
                for number, change in enumerate(changes, 1):
                    variant = Variant(chrom, *change)
                    hit = db.execute("SELECT accession FROM snvs WHERE key = ?", (variant.id,)).fetchone()
                    if hit:
                        selected.update(matched_difference=number, clinvar_variation_id=hit[0],
                                        clinvar_release=release)
                        break
                else:
                    raise ChatInputError(f"Found {len(changes)} base differences. None exactly matches the local ClinVar release.")
        mentioned = {match.group().upper() for match in VARIANT_ID.finditer(question)}
        if mentioned - {variant.id}:
            raise ChatInputError("The case ID in your question differs from the SNV identified in the sequence.")
        return variant, question or "Explain the pathogenicity research score in simple language.", selected

    def features(self, variant, contexts, allow_nvidia):
        record = self.cache.manifest["records"].get(variant.id, {})
        if all(
            record.get(direction, {}).get("ref_sequence_sha256") == _sequence_hash(ctx.ref_sequence)
            and record.get(direction, {}).get("alt_sequence_sha256") == _sequence_hash(ctx.alt_sequence)
            for direction, ctx in zip(ORIENTATIONS, contexts)
        ):
            try:
                return self.cache.get(variant.id)["delta"], "verified feature cache"
            except (KeyError, ValueError, OSError, EOFError):
                pass
        with self.lock:
            if variant.id in self.recent:
                self.recent.move_to_end(variant.id)
                return self.recent[variant.id], "session feature cache"
            if not allow_nvidia:
                raise ChatInputError("This SNV needs Evo 2 features. Enable the NVIDIA option to send the reconstructed REF/ALT windows.")
            encoder = NvidiaEvo2Encoder(
                model=self.protocol["model"], endpoint=self.protocol["endpoint"],
                api_schema=self.protocol["api_schema"], layer=self.protocol["layer"],
                service_revision=self.protocol["service_revision"], timeout_seconds=35,
                poll_timeout_seconds=40, max_retries=1,
            )
            try:
                if _metadata(encoder, 256, 32, self.reference_hash) != self.protocol:
                    raise RuntimeError("NVIDIA feature configuration differs from the training run.")
                values = np.stack([_fetch_direction(encoder, ctx, 32)[0]["delta"] for ctx in contexts])
            finally:
                encoder.session.close()
            self.recent[variant.id] = values
            if len(self.recent) > CACHED_LIMIT:
                self.recent.popitem(last=False)
            return values, "NVIDIA Evo 2 API"

    def score(self, variant, allow_nvidia=False):
        with ReferenceGenome(self.fasta) as genome:
            contexts = variant_contexts(genome, variant, 256)
        values, source = self.features(variant, contexts, allow_nvidia)
        with torch.inference_mode():
            score = self.model(torch.from_numpy(values).unsqueeze(0)).double().sigmoid().item()
        return {
            "variant": {"variant_id": variant.id},
            "disruptions": [],
            "pathogenicity": {
                "score": score, "score_out_of_100": round(100 * score, 1), "feature_source": source,
            },
        }
