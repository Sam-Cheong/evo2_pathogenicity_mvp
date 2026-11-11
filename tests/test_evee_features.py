import json
from threading import Barrier, Event, get_ident

import numpy as np
import pytest

from evo2_mvp import evee_features
from evo2_mvp.evee_features import EVEEFeatureCache, extract_features, variant_contexts
from evo2_mvp.nvidia_encoder import NvidiaAPIError
from evo2_mvp.sequence import ReferenceGenome, Variant


WINDOW = 12
INDEX = 9
WIDTH = 4096


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

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.last_request_info = None

    def token_activations(self, sequence):
        self.calls.append(sequence)
        self.last_request_info = {"request_id": str(len(self.calls))}
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def clinical_snv(tmp_path):
    sequence = "GACGCTTCTTAGCCAACGATGCTCGATCACTG"
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">chr1\n{sequence}\n", encoding="ascii")
    pos = 14
    variant = Variant("1", pos, sequence[pos - 1], "T", "GENE", 1)
    return fasta, sequence, variant


def activation_pairs():
    ref = np.zeros((WINDOW, WIDTH), dtype=np.float32)
    ref[:, 0] = 1
    forward = ref.copy()
    forward[INDEX - 1, :2] = [-1, 0]
    forward[INDEX:, :2] = [[-1, 0], [0, 1], [0, -1]]
    reverse = ref.copy()
    reverse[INDEX:, :2] = [[2, 0], [-1, 0], [0, -1]]
    return ref, forward, ref, reverse


def test_orientation_windows_and_directional_top_cosine_features(tmp_path, clinical_snv):
    fasta, sequence, variant = clinical_snv
    pos = variant.pos - 1
    reverse_start = pos - (WINDOW - 1 - INDEX)
    complement = str.maketrans("ACGT", "TGCA")
    with ReferenceGenome(fasta) as genome:
        forward, reverse = variant_contexts(genome, variant, WINDOW)
        assert genome.context(variant, WINDOW).variant_index == WINDOW // 2
    assert forward.variant_index == reverse.variant_index == INDEX
    assert forward.ref_sequence == sequence[pos - INDEX:pos - INDEX + WINDOW]
    assert reverse.ref_sequence == sequence[reverse_start:reverse_start + WINDOW].translate(complement)[::-1]
    assert forward.ref_sequence != reverse.ref_sequence.translate(complement)[::-1]
    for context, base in ((forward, variant.alt), (reverse, variant.alt.translate(complement))):
        assert context.alt_sequence[INDEX] == base
        assert context.alt_sequence[:INDEX] == context.ref_sequence[:INDEX]
        assert context.alt_sequence[INDEX + 1:] == context.ref_sequence[INDEX + 1:]

    output = tmp_path / "cache"
    encoder = MockForward(activation_pairs())
    progress = []
    counts = extract_features(encoder, [variant], fasta, output, window=WINDOW, top_k=2,
                              progress=lambda key, counts: progress.append((key, counts)))
    assert counts == {"processed": 1, "skipped": 0, "failed": 0}
    assert progress == [(variant.id, counts)]
    assert encoder.calls == [
        forward.ref_sequence, forward.alt_sequence, reverse.ref_sequence, reverse.alt_sequence,
    ]
    cache = EVEEFeatureCache(output)
    features = cache.get(variant.id)
    np.testing.assert_array_equal(features["positions"], [[INDEX, INDEX + 1], [INDEX + 1, INDEX + 2]])
    assert features["delta"].dtype == features["mean"].dtype == np.float32
    assert features["delta"].shape == (2, 2, WIDTH)
    np.testing.assert_array_equal(features["delta"][:, :, :2], [[[-2, 0], [-1, 1]], [[-2, 0], [-1, -1]]])
    np.testing.assert_allclose(features["mean"][:, :2], [[-4 / 3, 0], [-2 / 3, -1 / 3]])
    assert cache.metadata["model"] == "arc/evo2-7b"
    assert cache.metadata["layer"] == "blocks.27"
    assert cache.metadata["orientations"] == ["forward", "reverse_complement"]
    assert cache.metadata["reference_sha256"]
    assert forward.ref_sequence not in cache.manifest_path.read_text(encoding="utf-8")


def test_resume_committed_direction_and_repair_corrupt_record(tmp_path, clinical_snv):
    fasta, _, variant = clinical_snv
    ref, forward, _, reverse = activation_pairs()
    output = tmp_path / "cache"
    interrupted = MockForward([ref, forward, NvidiaAPIError("quota")])
    with pytest.raises(NvidiaAPIError, match="quota"):
        extract_features(interrupted, [variant], fasta, output, window=WINDOW, top_k=2)
    status = json.loads((output / "extraction_status.json").read_text(encoding="utf-8"))
    assert status["failed_variant"] == variant.id
    assert len(list(output.glob("*.npz"))) == 1
    with pytest.raises(KeyError):
        EVEEFeatureCache(output).get(variant.id)

    resumed = MockForward([ref, reverse])
    assert extract_features(resumed, [variant], fasta, output, window=WINDOW, top_k=2)["processed"] == 1
    assert len(resumed.calls) == 2
    read_only = MockForward([])
    assert extract_features(read_only, [variant], fasta, output, window=WINDOW, top_k=2)["skipped"] == 1
    assert read_only.calls == []

    next(output.glob("*.forward.npz")).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        EVEEFeatureCache(output).get(variant.id)
    repaired = MockForward([ref, forward])
    assert extract_features(repaired, [variant], fasta, output, window=WINDOW, top_k=2)["processed"] == 1
    assert len(repaired.calls) == 2

    with pytest.raises(ValueError, match="identity changed"):
        extract_features(read_only, [variant], fasta, output, window=WINDOW, top_k=1)
    assert read_only.calls == []


def test_invalid_7b_shape_stops_without_committing(tmp_path, clinical_snv):
    fasta, _, variant = clinical_snv
    encoder = MockForward([np.ones((WINDOW, 8192), dtype=np.float32)])
    output = tmp_path / "cache"
    with pytest.raises(NvidiaAPIError, match="width 4096"):
        extract_features(encoder, [variant], fasta, output, window=WINDOW, top_k=2)
    assert len(encoder.calls) == 1
    assert EVEEFeatureCache(output).manifest["records"] == {}


def test_parallel_uses_independent_encoders_and_commits_on_caller_thread(tmp_path, clinical_snv, monkeypatch):
    fasta, _, variant = clinical_snv
    ref, alt, _, _ = activation_pairs()
    both_started = Barrier(2, timeout=5)
    commits = []
    workers = []
    caller = get_ident()

    class Session:
        closed = False

        def close(self):
            self.closed = True

    class Worker(MockForward):
        def token_activations(self, sequence):
            if not self.calls:
                both_started.wait()
            return super().token_activations(sequence)

    def factory():
        worker = Worker([ref, alt])
        worker.session = Session()
        workers.append(worker)
        return worker

    put = EVEEFeatureCache._put

    def record_commit(cache, *args):
        commits.append(get_ident())
        return put(cache, *args)

    monkeypatch.setattr(EVEEFeatureCache, "_put", record_commit)
    output = tmp_path / "cache"
    master = MockForward([])
    counts = extract_features(master, [variant], fasta, output, window=WINDOW, top_k=2,
                              workers=2, encoder_factory=factory)
    assert counts == {"processed": 1, "skipped": 0, "failed": 0}
    assert len(workers) == 2
    assert all(len(worker.calls) == 2 and worker.session.closed for worker in workers)
    assert commits == [caller, caller]
    assert master.calls == []
    assert EVEEFeatureCache(output).get(variant.id)["delta"].shape == (2, 2, WIDTH)
    assert extract_features(master, [variant], fasta, output, window=WINDOW, top_k=2,
                            workers=2, encoder_factory=factory)["skipped"] == 1
    assert len(workers) == 2


def test_parallel_hard_failure_drains_started_direction_and_stops_queue(tmp_path, clinical_snv, monkeypatch):
    fasta, sequence, first = clinical_snv
    second = Variant("1", 16, sequence[15], "T", "GENE2", 0)
    ref, alt, _, _ = activation_pairs()
    with ReferenceGenome(fasta) as genome:
        forward, reverse = variant_contexts(genome, first, WINDOW)
    forward_started = Event()
    allow_forward_finish = Event()
    issued = []

    class Worker(MockForward):
        def token_activations(self, sequence):
            issued.append(sequence)
            if sequence == forward.ref_sequence:
                forward_started.set()
                assert allow_forward_finish.wait(5)
                return ref
            if sequence == reverse.ref_sequence:
                assert forward_started.wait(5)
                raise NvidiaAPIError("quota")
            assert sequence == forward.alt_sequence
            return alt

    wait = evee_features.wait

    def wait_for_failure(*args, **kwargs):
        result = wait(*args, **kwargs)
        allow_forward_finish.set()
        return result

    monkeypatch.setattr(evee_features, "wait", wait_for_failure)
    output = tmp_path / "cache"
    with pytest.raises(NvidiaAPIError, match="quota"):
        extract_features(MockForward([]), [first, second], fasta, output, window=WINDOW, top_k=2,
                         workers=2, encoder_factory=lambda: Worker([]))
    assert len(issued) == 3
    assert set(issued[:2]) == {forward.ref_sequence, reverse.ref_sequence}
    assert issued[-1] == forward.alt_sequence
    assert set(EVEEFeatureCache(output).manifest["records"][first.id]) == {"forward"}
    assert second.id not in EVEEFeatureCache(output).manifest["records"]
    status = json.loads((output / "extraction_status.json").read_text(encoding="utf-8"))
    assert status["failed_variant"] == first.id
