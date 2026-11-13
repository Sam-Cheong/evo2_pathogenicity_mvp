"""Resolve public ClinVar inputs and run the local probe over cached Evo 2 tokens."""
from collections import defaultdict
import re

import numpy as np

from evo2_mvp.sequence import ReferenceGenome, Variant
from evo2_mvp.splice_workflow import PROTOCOL, WINDOW, FEATURE_DIM, context, evidence_for, read_json
from evo2_mvp.utils import fingerprint, sha256_file


VARIANT_ID = re.compile(r"(?<![\w-])(?:[1-9]|1[0-9]|2[0-2])-\d+-[ACGT]-[ACGT](?![\w-])", re.I)
DNA_LINE = re.compile(r"(?:(?:REF|ALT|DNA)\s*[:=]\s*)?([ACGTN\s]+)", re.I)
RESIDUAL_DNA = re.compile(r"(?:[ACGTN][\s,;|\-]*){20,}", re.I)


class ChatInputError(ValueError):
    pass


class LocalProbe:
    def __init__(self, pilot, fasta):
        self.artifact = read_json(pilot / "splice_probe.json")
        cohort = pilot / "cohort.json"
        if (self.artifact["protocol"] != PROTOCOL
                or self.artifact["cohort_sha256"] != sha256_file(cohort)
                or self.artifact["fasta_sha256"] != sha256_file(fasta)):
            raise ValueError("The probe, cohort and GRCh38 reference do not match.")
        self.rows = {}
        self.tokens = {}
        self.sequence_ids = defaultdict(set)
        with ReferenceGenome(fasta) as genome:
            for row in read_json(cohort)["variants"]:
                variant = Variant(row["chrom"], row["pos"], row["ref"], row["alt"], row["gene"])
                vectors = []
                for sequence in context(genome, row, variant):
                    self.sequence_ids[sequence].add(variant.id)
                    key = fingerprint({"protocol": PROTOCOL, "sequence": sequence})
                    with np.load(pilot / "tokens" / f"{key}.npz", allow_pickle=False) as cached:
                        vector = cached["vector"]
                    if vector.shape != (FEATURE_DIM,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                        raise ValueError("Invalid cached splice vector.")
                    vectors.append(vector)
                self.rows[variant.id] = row
                self.tokens[variant.id] = vectors

    def score(self, variant_id):
        return evidence_for(self.rows[variant_id], self.artifact, self.tokens[variant_id], validated=False)

    def resolve(self, message, previous=None):
        if not isinstance(message, str) or not 0 < len(message.strip()) <= 4000:
            raise ChatInputError("Enter a question or a supported sequence, up to 4,000 characters.")
        sequence_blocks, pending, question = [], [], []
        for line in message.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if pending:
                    sequence_blocks.append("".join(pending))
                    pending.clear()
                continue
            match = DNA_LINE.fullmatch(line)
            if match:
                if pending and re.match(r"(?:REF|ALT|DNA)\s*[:=]", line, re.I):
                    sequence_blocks.append("".join(pending))
                    pending.clear()
                pending.append("".join(match[1].split()).upper())
            else:
                if pending:
                    sequence_blocks.append("".join(pending))
                    pending.clear()
                question.append(line)
        if pending:
            sequence_blocks.append("".join(pending))

        question = "\n".join(question)
        if RESIDUAL_DNA.search(question):
            raise ChatInputError("Put each DNA sequence on its own line, separate from your question.")
        requested = {item.group().upper() for item in VARIANT_ID.finditer(question)}
        if len(requested) > 1:
            raise ChatInputError("Use one ClinVar case per message.")
        candidates = set(requested) if requested else None
        for block in sequence_blocks:
            if len(block) % WINDOW or len(block) > 2 * WINDOW:
                raise ChatInputError("Paste the exact 256-base REF or ALT window for a demo case.")
            for start in range(0, len(block), WINDOW):
                hits = self.sequence_ids.get(block[start:start + WINDOW], set())
                if not hits:
                    raise ChatInputError("This sequence does not match a REF or ALT window in the 24-case demo.")
                candidates = set(hits) if candidates is None else candidates & hits
        if candidates is None:
            candidates = {previous} if isinstance(previous, str) else set()
        if len(candidates) != 1 or not candidates <= self.rows.keys():
            raise ChatInputError("Provide a supported case ID or a matching 256-base window. Try 19-11110772-G-A.")
        return candidates.pop(), question.strip() or "Explain the predicted splice changes."
