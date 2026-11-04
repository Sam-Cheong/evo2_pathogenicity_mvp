"""Create the independently executable cache-only two-stage SH2 pilot notebook."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def main():
    cells = []

    def add(kind, identifier, source):
        cell = (nbf.v4.new_code_cell if kind == "code" else nbf.v4.new_markdown_cell)(dedent(source).strip())
        cell["id"] = identifier
        cells.append(cell)

    add("markdown", "protocol", r'''
    # StripedHyena 2 小型参考变体：先表示蒸馏，再冻结编码器训练判别头

    本轮范围：自定义小型 SH2 reference variant，随机初始化；SE/MR/LI/causal rotary
    attention 四块，group_dim=1，192维中心表示。不是官方预训练checkpoint，也不是HyenaDNA。
    线性头/MLP完整推理参数分别1,505,281/1,582,721，均<2M；训练用投影单独计数，推理时丢弃。

    真实教师为已有 NVIDIA Evo 2 40B `blocks.20.mlp.l3`，8192维REF/ALT中心向量。
    仅复用既有缓存，不调用任何Evo 2 API、不使用Teacher分类头软标签。
    Stage 1：原train中的960条无标签配对用于训练，240条按基因/重叠上下文分组的无标签配对用于选模。
    Stage 2：编码器冻结且eval，ClinVar 1200/400用于训练/验证linear和小MLP头。
    两个编码器臂是相同初始encoder和表示蒸馏后encoder；同类型头保持相同随机初值。
    四个头全部完成后才执行独立evaluate阶段读取400条test，不能根据test调参。
    测试集已在历史实验查看过；本轮是小试，不是新的独立临床验证。

    项目根目录PowerShell：
    ```powershell
    .\.venv\Scripts\python.exe -m pip install -r requirements-sh2.txt -r requirements-notebook.txt
    .\.venv\Scripts\python.exe scripts/run_hyenadna_notebook.py --notebook StripedHyena2_two_stage_experiment.ipynb --output StripedHyena2_two_stage_experiment.executed.ipynb --status outputs/sh2_notebook_execution.json
    ```
    ''' )
    add("code", "setup-and-identity", r'''
    import os, sys, json, hashlib
    from pathlib import Path
    from datetime import datetime, timezone
    ROOT = Path.cwd().resolve()
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch, numpy as np
    from evo2_mvp.utils import save_json, sha256_file
    from hyenadna_notebook_support import run_stage
    torch.set_num_threads(4)
    RUN = Path("outputs/sh2_two_stage_20261104")
    DATA = RUN / "data"
    TRAINING = RUN / "training"
    RUN.mkdir(parents=True, exist_ok=True)
    paths = ["configs/stripedhyena2_two_stage.yaml", "data/variants.csv", "data/splits.json",
             "data/GRCh38.fa", "outputs/nvidia_cache/manifest.json", "outputs/teacher/best_teacher.pt",
             "scripts/prepare_sh2_reference.py", "scripts/prepare_two_stage.py", "scripts/test_stripedhyena2.py"]
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src/evo2_mvp").rglob("*.py"))]
    hashes = {path: sha256_file(ROOT / path) for path in paths}
    IDENTITY = {"identity": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                "files_sha256": hashes, "device": "cpu", "python": sys.version, "torch": torch.__version__}
    save_json(RUN / "notebook_input_identity.json", IDENTITY)
    print("Experiment:", IDENTITY["identity"], "CPU threads:", torch.get_num_threads())
    ''' )
    add("code", "prepare-real-cached-data", r'''
    # Initial preparation may create/check the sealed test archive, but it never fits a model.
    # Existing prepared data are reused here without opening the sealed test file.
    if not (DATA / "metadata.json").is_file():
        run_stage(ROOT, RUN, "data", ["scripts/prepare_two_stage.py", "--variants", "data/variants.csv",
            "--splits", "data/splits.json", "--cache", "outputs/nvidia_cache", "--fasta", "data/GRCh38.fa",
            "--output", str(DATA), "--seed", "42", "--window", "256"], IDENTITY,
            [str(DATA / name) for name in ("metadata.json", "representation_train.npz", "representation_val.npz",
                                          "probe_train.npz", "probe_val.npz")])
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["synthetic"] is False and metadata["status"] == "complete"
    for field, path in {"variants_sha256": "data/variants.csv", "splits_sha256": "data/splits.json",
                        "fasta_sha256": "data/GRCh38.fa", "cache_manifest_sha256": "outputs/nvidia_cache/manifest.json"}.items():
        assert metadata["source_hashes"][field] == IDENTITY["files_sha256"][path]
    for name, digest in metadata["files_sha256"].items():
        if name == "probe_test.npz":
            continue  # Its actual content/hash is first checked by evaluate after all four probes.
        assert sha256_file(DATA / name) == digest
    for name in ("representation_train.npz", "representation_val.npz"):
        with np.load(DATA / name, allow_pickle=False) as archive:
            assert set(archive.files) == {"ref", "alt", "teacher_ref", "teacher_alt", "variant_ids"}
            assert archive["teacher_ref"].shape[1] == 8192
    print(json.dumps(metadata["counts"], indent=2))
    ''' )
    add("code", "official-reference-and-structural-test", r'''
    # Public official source only. All files already downloaded and fixed SHA verified; no model API.
    run_stage(ROOT, RUN, "reference", ["scripts/prepare_sh2_reference.py", "--self-check"], IDENTITY,
        ["work/sh2_reference/manifest.json", "work/sh2_reference/reference_checks.json"])
    run_stage(ROOT, RUN, "structural", ["scripts/test_stripedhyena2.py", "--output", str(RUN / "phase0")],
        IDENTITY, [str(RUN / "phase0" / n) for n in ("result.json", "embedding_ref.npy", "embedding_alt.npy", "features.npy")])
    print(json.loads((RUN / "phase0/result.json").read_text(encoding="utf-8")))
    ''' )
    add("code", "automated-tests", r'''
    os.environ["RUN_HYENADNA_INTEGRATION"] = "1"
    os.environ["RUN_EVO2_INTEGRATION"] = "0"
    # Real local Hyena checkpoint and official SH2 equation parity run; live Evo2 API test stays opt-out.
    run_stage(ROOT, RUN, "tests", ["-m", "pytest", "-q", "--basetemp", "work/pytest_sh2_full",
        "--junitxml", str(RUN / "junit.xml")], IDENTITY, [str(RUN / "junit.xml")])
    ''' )
    add("markdown", "stage1-explanation", r'''
    ## Stage 1：纯表示蒸馏

    只读取representation_train/val，禁止读取probe文件、ClinVar标签或Teacher分类头。
    REF/ALT共享train-only标准化，delta使用独立train-only尺度；教师目标固定，训练学生与两个投影。
    无BCE、无输出分布KD。仅用representation-val MSE选encoder并早停。
    单独记录ref/alt/delta损失、均值预测基线、学生表示方差及训练/推理参数量。
    这只有960条配对，验证的是方法通路，不等于复现基础模型预训练。
    ''' )
    add("code", "stage1-representation-distillation", r'''
    args = ["-m", "evo2_mvp.two_stage_student", "--stage", "distill", "--data-dir", str(DATA),
            "--config", "configs/stripedhyena2_two_stage.yaml", "--output", str(TRAINING), "--device", "cpu", "--resume"]
    # --resume initializes a fresh run if no saved state exists; otherwise validates and continues it.
    run_stage(ROOT, RUN, "distill", args, IDENTITY, [str(TRAINING / name) for name in
        ("initial_encoder.pt", "best_encoder.pt", "last_distillation.pt", "distillation_curve.csv", "distillation_status.json")])
    print(json.loads((TRAINING / "distillation_status.json").read_text(encoding="utf-8")))
    ''' )
    add("markdown", "stage2-explanation", r'''
    ## Stage 2：冻结编码器的判别探针

    分别使用蒸馏前/后的冻结编码器提取`concat(ref,alt,alt-ref)`。
    每种编码器上分别训练linear和MLP，共四个模型；同类型头同初始权重、同训练参数。
    feature scaler只由probe train拟合，按clinical val BCE选择头；编码器全程eval且无梯度。
    记录前后encoder SHA，防止判别训练悄悄微调encoder。
    ''' )
    add("code", "stage2-frozen-probes", r'''
    ARMS = ("random_linear", "distilled_linear", "random_mlp", "distilled_mlp")
    run_stage(ROOT, RUN, "probes", ["-m", "evo2_mvp.two_stage_student", "--stage", "probes", "--data-dir", str(DATA),
        "--config", "configs/stripedhyena2_two_stage.yaml", "--output", str(TRAINING), "--device", "cpu", "--resume"], IDENTITY,
        [str(TRAINING / "probes_status.json")] + [str(TRAINING / "probes" / arm / "best_student.pt") for arm in ARMS])
    print(json.loads((TRAINING / "probes_status.json").read_text(encoding="utf-8")))
    ''' )
    add("code", "heldout-evaluation", r'''
    # All four heads must exist and be marked complete before this command can open probe_test.npz.
    run_stage(ROOT, RUN, "evaluate", ["-m", "evo2_mvp.two_stage_student", "--stage", "evaluate", "--data-dir", str(DATA),
        "--config", "configs/stripedhyena2_two_stage.yaml", "--output", str(TRAINING), "--device", "cpu"], IDENTITY,
        [str(TRAINING / "two_stage_test_comparison.json"), str(TRAINING / "two_stage_test_predictions.csv")]
        + [str(TRAINING / "probes" / arm / "test_metrics.json") for arm in ARMS])
    import pandas as pd
    from IPython.display import display
    rows = []
    for arm in ARMS:
        metrics = json.loads((TRAINING / "probes" / arm / "test_metrics.json").read_text(encoding="utf-8"))
        rows.append({"model": arm, **{key: metrics[key] for key in ("auroc", "auprc", "f1", "recall", "brier_score", "n", "positives")}})
    table = pd.DataFrame(rows)
    table.to_csv(RUN / "model_comparison.csv", index=False)
    display(table)
    import matplotlib.pyplot as plt
    history = pd.read_csv(TRAINING / "distillation_curve.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.epoch, history.train_total, label="representation train")
    ax.plot(history.epoch, history.val_total, label="representation validation")
    ax.set(xlabel="Epoch", ylabel="Standardized representation MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "representation_training_curve.png", dpi=150)
    display(fig)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for arm, ax in zip(ARMS, axes.flat):
        curve = pd.read_csv(TRAINING / "probes" / arm / "training_curve.csv")
        ax.plot(curve.epoch, curve.train_loss, label="train")
        ax.plot(curve.epoch, curve.val_loss, label="validation")
        ax.set(title=arm, xlabel="Epoch", ylabel="BCE")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "probe_training_curves.png", dpi=150)
    display(fig)
    plt.close(fig)
    ''' )
    add("code", "independent-frozen-student-inference", r'''
    # Predetermined demonstration model: distilled_linear. Never select the example model by test performance.
    selected = TRAINING / "probes/distilled_linear/best_student.pt"
    for name in ("prediction", "prediction_reloaded"):
        run_stage(ROOT, RUN, name, ["-m", "evo2_mvp.inference", "--fasta", "data/GRCh38.fa", "--chrom", "1",
            "--pos", "926029", "--ref", "C", "--alt", "T", "--student", str(selected), "--device", "cpu", "--cpu-threads", "4",
            "--output", str(RUN / f"{name}.json")], IDENTITY, [str(RUN / f"{name}.json")])
    prediction = json.loads((RUN / "prediction.json").read_text(encoding="utf-8"))
    assert prediction == json.loads((RUN / "prediction_reloaded.json").read_text(encoding="utf-8"))
    assert prediction["teacher_score"] is None and prediction["clinical_use"] == "research_only"
    print(json.dumps(prediction, indent=2))
    ''' )
    add("code", "run-report", r'''
    from evo2_mvp.models import build_student, load_checkpoint
    from evo2_mvp.distill_student import model_state_sha256
    summaries = {}
    for arm in ARMS:
        checkpoint = load_checkpoint(TRAINING / "probes" / arm / "best_student.pt")
        model = build_student(checkpoint)
        assert sum(p.numel() for p in model.parameters()) < 2_000_000
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        summaries[arm] = {"checkpoint_sha256": sha256_file(TRAINING / "probes" / arm / "best_student.pt"),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "encoder_sha256": model_state_sha256(model.encoder)}
    assert summaries["random_linear"]["encoder_sha256"] == summaries["random_mlp"]["encoder_sha256"]
    assert summaries["distilled_linear"]["encoder_sha256"] == summaries["distilled_mlp"]["encoder_sha256"]
    assert summaries["random_linear"]["encoder_sha256"] != summaries["distilled_linear"]["encoder_sha256"]
    assert sha256_file("outputs/teacher/best_teacher.pt") == IDENTITY["files_sha256"]["outputs/teacher/best_teacher.pt"]
    report = {"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_identity": IDENTITY["identity"], "architecture": "custom_small_SH2_reference_variant",
        "initialization": "random_then_real_representation_distillation", "counts": metadata["counts"],
        "models": summaries, "test_metrics": rows, "standalone_prediction": prediction,
        "independent_reload_predictions_identical": True, "teacher_unchanged": True, "new_evo2_api_calls": 0,
        "limitations": ["single seed", "only 960 representation-training pairs", "center-only teacher targets",
                        "historically examined test cohort", "custom SH2 scale/schedule, not official pretrained weights"],
        "clinical_use": "research_only"}
    save_json(RUN / "run_report.json", report)
    print("Completed:", RUN / "run_report.json")
    ''' )
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Project .venv Python 3.11", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.9"}})
    path = Path(__file__).resolve().parents[1] / "StripedHyena2_two_stage_experiment.ipynb"
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
