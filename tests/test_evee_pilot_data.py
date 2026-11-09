"""Fixtures exercise the EVEE input boundary without training any model."""

from io import BytesIO
import csv
import json

import numpy as np
import pytest
import torch

from evo2_mvp import evee_pilot_data as pilot
from evo2_mvp.metrics import file_sha256, fingerprint
from evo2_mvp.two_stage_student import model_kwargs, state_sha256


def archive_bytes(rows):
    reference = np.zeros((len(rows), 4, 256), dtype=np.float32)
    reference[:, 0, :] = 1.0
    alternate = reference.copy()
    alternate[:, 0, 128], alternate[:, 2, 128] = 0.0, 1.0
    handle = BytesIO()
    np.savez(handle, ref=reference, alt=alternate,
             labels=np.asarray([row[2] for row in rows], dtype=np.float32),
             variant_ids=np.asarray([f"1-{row[0]}-A-G" for row in rows]))
    return handle.getvalue()


@pytest.fixture
def prepared(tmp_path):
    records = {
        "train": [(300, "A", 0), (600, "A;B", 1), (900, "B|C", 0),
                  (1200, "C", 1), (1500, "D", 0), (1800, "E", 1)],
        "val": [(2100, "F", 0), (2400, "G", 1)],
        "test": [(2700, "H", 0), (3000, "I", 1)],
    }
    csv_path = tmp_path / "variants.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["chrom", "pos", "ref", "alt", "gene", "label", "variant_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, rows in records.items():
            for position, gene, label in rows:
                writer.writerow({"chrom": "1", "pos": position, "ref": "A", "alt": "G",
                                 "gene": gene, "label": label if name != "test" else "sealed",
                                 "variant_id": f"1-{position}-A-G"})
    splits = {name: [f"1-{row[0]}-A-G" for row in rows] for name, rows in records.items()}
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(splits), encoding="utf-8")
    data_dir, encoders = tmp_path / "data", tmp_path / "encoders"
    data_dir.mkdir()
    encoders.mkdir()
    payloads = {name: archive_bytes(rows) for name, rows in records.items()}
    # The historical test archive is physically absent in the development fixture.
    for name in ("train", "val"):
        (data_dir / f"probe_{name}.npz").write_bytes(payloads[name])
    import hashlib
    hashes = {f"probe_{name}.npz": hashlib.sha256(body).hexdigest() for name, body in payloads.items()}
    metadata = {
        "status": "complete",
        "synthetic": False,
        "data_provenance": {"synthetic": False, "n_variants": 10, "variants_sha256": file_sha256(csv_path)},
        "source_hashes": {"variants_sha256": file_sha256(csv_path), "splits_sha256": file_sha256(split_path)},
        "split_ids": splits,
        "split_fingerprint": fingerprint(splits),
        "files_sha256": hashes,
        "cache_metadata": {"window": 256},
    }
    meta_path = data_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_sha = file_sha256(meta_path)
    config = {}
    protocol = {"prepared_metadata_sha256": metadata_sha, "configuration_sha256": fingerprint(config)}
    initial, best = {"weight": torch.ones(3)}, {"weight": torch.full((3,), 2.0)}
    initial_sha, best_sha = state_sha256(initial), state_sha256(best)
    torch.save({"protocol": protocol, "model_kwargs": model_kwargs(config, "linear"),
                "encoder_state": initial, "encoder_sha256": initial_sha}, encoders / "initial_encoder.pt")
    best_path = encoders / "best_encoder.pt"
    torch.save({"protocol": protocol, "model_kwargs": model_kwargs(config, "linear"),
                "encoder_state": best, "encoder_sha256": best_sha,
                "initial_encoder_sha256": initial_sha}, best_path)
    status = {"status": "completed", "initial_encoder_sha256": initial_sha, "best_encoder_sha256": best_sha,
              "best_encoder_file_sha256": file_sha256(best_path)}
    (encoders / "distillation_status.json").write_text(json.dumps(status), encoding="utf-8")
    return data_dir, encoders, csv_path, config, metadata_sha, payloads["test"]


def test_preflight_loads_development_only_and_connects_multigene_rows(prepared):
    data_dir, encoders, csv_path, config, metadata_sha, _ = prepared
    result = pilot.load_pilot_inputs(data_dir, encoders, csv_path, config)
    assert result.train["groups"].tolist() == ["A|B|C"] * 4 + ["D", "E"]
    assert result.val["groups"].tolist() == ["F", "G"]
    assert set(result.encoder_artifacts) == {"random", "distilled"}
    assert result.provenance["prepared_metadata_sha256"] == metadata_sha
    assert set(result.provenance["development_probe_sha256s"]) == {"train", "val"}
    assert not (data_dir / "probe_test.npz").exists()


def test_historical_archive_requires_same_preparation_lock(prepared):
    data_dir, _, csv_path, _, metadata_sha, test_payload = prepared
    with pytest.raises(ValueError, match="locked validation decision"):
        pilot.load_historical_test(data_dir, csv_path, None)
    lock = {"status": "locked_for_historical_test", "checkpoint_hashes": {"model": "1" * 64},
            "prepared_metadata_sha256": "2" * 64}
    with pytest.raises(ValueError, match="different prepared data"):
        pilot.load_historical_test(data_dir, csv_path, lock)
    (data_dir / "probe_test.npz").write_bytes(test_payload)
    lock["prepared_metadata_sha256"] = metadata_sha
    partition = pilot.load_historical_test(data_dir, csv_path, lock)
    assert partition["variant_ids"] == ["1-2700-A-G", "1-3000-A-G"]
    assert partition["groups"].tolist() == ["H", "I"]
    assert partition["labels"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("modified", ["variants", "splits", "archive", "encoder"])
def test_preflight_rejects_modified_inputs(prepared, modified):
    data_dir, encoders, csv_path, config, _, _ = prepared
    if modified == "variants":
        path, message = csv_path, "variant CSV checksum"
    elif modified == "splits":
        path, message = csv_path.with_name("splits.json"), "clinical split checksum"
    elif modified == "archive":
        path, message = data_dir / "probe_train.npz", "partition checksum mismatch"
    else:
        path, message = encoders / "best_encoder.pt", "differ from the completed"
    if modified == "encoder":
        value = torch.load(path, map_location="cpu", weights_only=True)
        value["encoder_state"]["weight"][0] = 3.0
        torch.save(value, path)
    else:
        with path.open("ab") as handle:
            handle.write(b"\n")
    with pytest.raises(ValueError, match=message):
        pilot.load_pilot_inputs(data_dir, encoders, csv_path, config)


def test_grouped_folds_are_reproducible_and_cover_every_variant_once():
    partition = {"labels": torch.tensor([0, 1] * 10), "groups": np.repeat(np.arange(10), 2).astype(str)}
    first = pilot.grouped_folds(partition, n_splits=5, seed=42)
    second = pilot.grouped_folds(partition, n_splits=5, seed=42)
    for (fit, validation), (same_fit, same_validation) in zip(first, second):
        assert np.array_equal(fit, same_fit) and np.array_equal(validation, same_validation)
        assert not set(partition["groups"][fit]) & set(partition["groups"][validation])
        assert set(partition["labels"][validation].tolist()) == {0, 1}
    assert sorted(np.concatenate([validation for _, validation in first])) == list(range(20))
