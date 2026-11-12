import numpy as np
import pytest

from evo2_mvp.evee_features import EVEEFeatureCache, extract_features, variant_contexts
from evo2_mvp.evee_indel import IndelVariant
from evo2_mvp.sequence import ReferenceGenome, Variant


LEFT = "TAGACGATCCAGTGCAGTC"
EDIT = "GA"
RIGHT = "ACGTTTCTGACACGAACGT"
WINDOW = 16
UPSTREAM = 12
COMPLEMENT = str.maketrans("ACGT", "TGCA")


class MockForward:
    metadata = {
        "model": "arc/evo2-7b",
        "endpoint": "https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward",
        "api_schema": "nim_v1",
        "model_revision": "not_exposed_by_provider",
        "service_revision": "fixture",
        "layer": "blocks.27",
        "tokenizer": "character_level_no_bos",
        "adapter_version": "nvidia-forward-v1",
    }

    def __init__(self):
        self.calls = []
        self.last_request_info = None

    def token_activations(self, sequence):
        self.calls.append(sequence)
        self.last_request_info = {"request_id": str(len(self.calls))}
        vectors = np.zeros((len(sequence), 4096), dtype=np.float32)
        vectors[np.arange(len(sequence)), ["ACGT".index(base) for base in sequence]] = 1
        if len(self.calls) % 2 == 0:
            # Identify the ALT offset in every retained delta as well as the base.
            vectors[:, 4] = np.arange(1, len(sequence) + 1)
        return vectors


@pytest.fixture(params=["insertion", "deletion"])
def indel_case(tmp_path, request):
    ref_edit, alt_edit = ("", EDIT) if request.param == "insertion" else (EDIT, "")
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">chr1\n{LEFT}{ref_edit}{RIGHT}\n", encoding="ascii")
    variant = IndelVariant("1", len(LEFT), LEFT[-1] + ref_edit, LEFT[-1] + alt_edit, "GENE", 1)
    return fasta, variant, ref_edit, alt_edit


def test_strand_relative_windows_align_downstream_of_indel(indel_case):
    fasta, variant, ref_edit, alt_edit = indel_case
    with ReferenceGenome(fasta) as genome:
        forward, reverse = variant_contexts(genome, variant, WINDOW)

    assert forward.ref_sequence == (LEFT[-UPSTREAM:] + ref_edit + RIGHT)[:WINDOW]
    assert forward.alt_sequence == (LEFT[-UPSTREAM:] + alt_edit + RIGHT)[:WINDOW]
    reverse_left = RIGHT[:UPSTREAM].translate(COMPLEMENT)[::-1]
    reverse_right = LEFT.translate(COMPLEMENT)[::-1]
    assert reverse.ref_sequence == (reverse_left + ref_edit.translate(COMPLEMENT)[::-1] + reverse_right)[:WINDOW]
    assert reverse.alt_sequence == (reverse_left + alt_edit.translate(COMPLEMENT)[::-1] + reverse_right)[:WINDOW]
    for context in (forward, reverse):
        assert context.variant_index == UPSTREAM + len(ref_edit)
        assert context.alt_index == UPSTREAM + len(alt_edit)
        matched_length = WINDOW - max(context.variant_index, context.alt_index)
        assert context.ref_sequence[context.variant_index:][:matched_length] == (
            context.alt_sequence[context.alt_index:][:matched_length]
        )


def test_feature_extraction_uses_paired_offsets_and_an_indel_cache(tmp_path, indel_case):
    fasta, variant, ref_edit, alt_edit = indel_case
    cache_dir = tmp_path / "features"
    encoder = MockForward()
    stats = extract_features(encoder, [variant], fasta, cache_dir, window=WINDOW, top_k=1, indels=True)

    assert stats == {"processed": 1, "skipped": 0, "failed": 0}
    assert len(encoder.calls) == 4
    features = EVEEFeatureCache(cache_dir).get(variant.id)
    matched_length = WINDOW - UPSTREAM - max(len(ref_edit), len(alt_edit))
    last_ref_offset = UPSTREAM + len(ref_edit) + matched_length - 1
    last_alt_offset = UPSTREAM + len(alt_edit) + matched_length - 1
    np.testing.assert_array_equal(features["positions"], [[last_ref_offset], [last_ref_offset]])
    np.testing.assert_array_equal(features["delta"][:, :, :4], np.zeros((2, 1, 4)))
    np.testing.assert_array_equal(features["delta"][:, :, 4], [[last_alt_offset + 1], [last_alt_offset + 1]])
    np.testing.assert_allclose(features["mean"][:, 4], last_alt_offset + 1 - (matched_length - 1) / 2)

    no_calls = MockForward()
    assert extract_features(no_calls, [variant], fasta, cache_dir, window=WINDOW, top_k=1, indels=True)["skipped"] == 1
    snv = Variant(variant.chrom, variant.pos, LEFT[-1], "A", "GENE", 1)
    with pytest.raises(ValueError, match="identity changed"):
        extract_features(no_calls, [snv], fasta, cache_dir, window=WINDOW, top_k=1)
    assert no_calls.calls == []
