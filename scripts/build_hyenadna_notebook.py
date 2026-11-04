"""Write the reviewable, executable experiment notebook into the project root."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def main():
    root = Path(__file__).resolve().parents[1]
    cells = []

    def markdown(identifier, source):
        cell = nbf.v4.new_markdown_cell(dedent(source).strip())
        cell["id"] = identifier
        cells.append(cell)

    def code(identifier, source):
        cell = nbf.v4.new_code_cell(dedent(source).strip())
        cell["id"] = identifier
        cells.append(cell)

    markdown("protocol", r'''
    # 预训练 HyenaDNA d256＋变异中心表示：真实监督与 Evo 2 蒸馏实验

    使用官方固定版本 `LongSafari/hyenadna-tiny-1k-seqlen-d256-hf`，共享 REF/ALT encoder，
    提取变异索引 128 的最后隐藏层，拼接 `ref, alt, alt-ref`（768 维）。
    完整推理模型 1,854,465 参数，编码器与分类头均微调；因果中心表示只使用当前位置及左侧上下文。

    预先固定本轮配置 `configs/hyenadna_d256.yaml`，相同真实预训练权重、随机分类头初值和数据划分，
    对比仅监督与监督＋特征/预测蒸馏。复用既有真实 NVIDIA Evo 2 **40B** 特征和已训练 Teacher；
    此 Notebook 不调用 NVIDIA API、不重训 Teacher、不读取 API key、不使用合成标签报告性能。

    数据为现有 2,000 条 ClinVar GRCh38 常染色体 SNV；训练/验证/测试为 1,200/400/400。
    只按验证 BCE 早停，测试集不参与训练或选超参数。该测试集已在历史 CNN 实验中查看过，
    本轮结果属于同一研究队列的后续比较，不能充当全新外部验证。

    在项目根目录 PowerShell 首次安装并执行：
    ```powershell
    .\.venv\Scripts\python.exe -m pip install -r requirements-hyenadna.txt -r requirements-notebook.txt
    .\.venv\Scripts\python.exe scripts/run_hyenadna_notebook.py
    ```
    执行后的含输出版本保存为项目根目录 `HyenaDNA_d256_experiment.executed.ipynb`。
    如需交互查看：`.\.venv\Scripts\python.exe -m notebook HyenaDNA_d256_experiment.executed.ipynb`。
    所有 cell 的 CLI 均由当前项目 .venv 启动，stdout 同步写入阶段日志。
    发生错误时保存部分 Notebook 并抛出异常；修复后重跑将严格校验并复用已完成阶段。
    中断中的训练阶段会从固定预训练初始化重新开始，未实现 epoch 中间优化器状态恢复。
    ''')
    code("setup", r'''
    import os, sys, json, csv
    from pathlib import Path
    from datetime import datetime, timezone
    ROOT = Path.cwd().resolve()
    assert (ROOT / "src/evo2_mvp").is_dir(), "请从项目根目录执行"
    assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve(), sys.executable
    sys.path.insert(0, str(ROOT / "scripts"))
    os.environ["HF_HOME"] = str(ROOT / "work/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch, transformers, safetensors, numpy as np
    from hyenadna_notebook_support import input_identity, run_stage, validate_training
    from evo2_mvp.utils import save_json, sha256_file
    from evo2_mvp.train_teacher_head import experiment_data, check_cache
    from evo2_mvp.cache import FeatureCache
    from evo2_mvp.distill_student import check_phase0
    RUN_ROOT = Path("outputs/hyenadna_notebook_run_20261104")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    SUPERVISED = RUN_ROOT / "supervised"
    DISTILLED = RUN_ROOT / "distilled"
    torch.set_num_threads(4)
    print(json.dumps({"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                      "transformers": transformers.__version__, "device": "cpu", "run_root": str(RUN_ROOT)}, indent=2))
    ''')
    code("real-data-preflight", r'''
    variants, splits, provenance = experiment_data("data/variants.csv", "data/splits.json", False)
    cache = FeatureCache("outputs/nvidia_cache")
    check_cache(cache, provenance)
    check_phase0("outputs/phase0_api_verified/result.json", cache.metadata)
    # This hashes the actual ~3.3GB FASTA, not merely its filename.
    IDENTITY = input_identity(ROOT)
    assert cache.metadata["reference_sha256"] == IDENTITY["files_sha256"]["data/GRCh38.fa"]
    assert len(variants) == 2000 and provenance["synthetic"] is False
    dataset = {"provenance": provenance, "splits": {s: {"n": len(splits[s]),
        "positives": sum(v.label for v in variants if v.id in set(splits[s]))} for s in ("train", "val", "test")},
        "experiment_identity": IDENTITY["identity"], "new_api_calls_required": 0}
    save_json(RUN_ROOT / "input_identity.json", IDENTITY)
    save_json(RUN_ROOT / "data_check.json", dataset)
    print(json.dumps(dataset, indent=2))
    ''')
    markdown("pretrained-instructions", r'''
    ## 固定真实预训练权重与前向验证

    以下脚本只接受固定 revision 和 SHA256；已有正确文件时离线复核，不重新下载。
    结构测试使用固定合成 DNA，仅检查真实 embedding、梯度与保存重载，不报告随机分类头分数。
    ```powershell
    .\.venv\Scripts\python.exe scripts/prepare_hyenadna.py --output work/hyenadna_d256_pretrained
    .\.venv\Scripts\python.exe scripts/test_hyenadna.py --pretrained-dir work/hyenadna_d256_pretrained --window 256 --threads 4 --compare-official --output outputs/hyenadna_notebook_run_20261104/phase0
    ```
    ''')
    code("prepare-checkpoint", r'''
    from evo2_mvp.hyenadna import PINNED_FILES
    run_stage(ROOT, RUN_ROOT, "prepare", ["scripts/prepare_hyenadna.py", "--output", "work/hyenadna_d256_pretrained"],
              IDENTITY, [str(Path("work/hyenadna_d256_pretrained") / name) for name in PINNED_FILES])
    ''')
    code("real-forward-backward", r'''
    def validate_smoke():
        report = json.loads((RUN_ROOT / "phase0/result.json").read_text(encoding="utf-8"))
        assert report["status"] == "passed" and report["pretrained_encoder_loaded"] is True
        assert report["variant_features_shape"] == [768] and report["finite"] is True
        assert report["official_implementation_comparison"]["status"] == "passed"
        assert report["save_reload_features_bitwise_equal"] and report["encoder_backward_finite_nonzero"]
    run_stage(ROOT, RUN_ROOT, "phase0", ["scripts/test_hyenadna.py", "--pretrained-dir", "work/hyenadna_d256_pretrained",
        "--window", "256", "--threads", "4", "--compare-official", "--output", str(RUN_ROOT / "phase0")], IDENTITY,
        [str(RUN_ROOT / "phase0" / name) for name in ["result.json", "embedding_ref.npy", "embedding_alt.npy",
        "embedding_delta.npy", "variant_features.npy"]], validate_smoke)
    ''')
    code("automated-validation", r'''
    os.environ["RUN_HYENADNA_INTEGRATION"] = "1"
    # Real local checkpoint integration is enabled; a fresh NVIDIA API integration is deliberately not requested.
    run_stage(ROOT, RUN_ROOT, "tests", ["-m", "pytest", "-q", "--basetemp", "work/pytest_notebook_run",
        "--junitxml", str(RUN_ROOT / "junit.xml")], IDENTITY, [str(RUN_ROOT / "junit.xml")])
    ''')
    markdown("training-instructions", r'''
    ## 两个同初始化训练实验

    编码器学习率 2e-5，分类头 1e-3，batch 16，最多 20 epochs，patience 5，梯度裁剪 1.0。
    蒸馏初值 alpha/beta/gamma=1.0/0.1/0.5，T=2；Teacher 投影仅用训练统计并固定。
    两个实验沿用同一配置，全部参数及随机初值 SHA 写入 checkpoint，比较阶段强制核对。
    每轮 `student_epoch` JSON 包含训练损失、验证 BCE、梯度范数和耗时。
    日志 `outputs/hyenadna_notebook_run_20261104/supervised.log` 与 `distilled.log` 可实时查看。
    ''')
    code("supervised-training", r'''
    COMMON = ["--variants", "data/variants.csv", "--splits", "data/splits.json", "--cache", "outputs/nvidia_cache",
        "--fasta", "data/GRCh38.fa", "--teacher", "outputs/teacher/best_teacher.pt",
        "--phase0-report", "outputs/phase0_api_verified/result.json", "--config", "configs/hyenadna_d256.yaml", "--device", "cpu"]
    artifacts = [str(SUPERVISED / name) for name in ["best_student.pt", "metrics.json", "predictions.csv",
        "prediction_metadata.json", "benchmark.json", "training_curve.csv", "training_curve.png", "config.json"]]
    run_stage(ROOT, RUN_ROOT, "supervised", ["-m", "evo2_mvp.distill_student", "--mode", "supervised",
        *COMMON, "--output", str(SUPERVISED)], IDENTITY, artifacts,
        lambda: validate_training(SUPERVISED, "supervised", IDENTITY))
    ''')
    code("distillation-training", r'''
    artifacts = [str(DISTILLED / name) for name in ["best_student.pt", "metrics.json", "predictions.csv",
        "prediction_metadata.json", "benchmark.json", "training_curve.csv", "training_curve.png", "config.json",
        "distillation_training_state.pt"]]
    run_stage(ROOT, RUN_ROOT, "distilled", ["-m", "evo2_mvp.distill_student", "--mode", "distill",
        *COMMON, "--output", str(DISTILLED)], IDENTITY, artifacts,
        lambda: validate_training(DISTILLED, "distill", IDENTITY))
    ''')
    code("controlled-comparison", r'''
    run_stage(ROOT, RUN_ROOT, "comparison", ["scripts/compare_models.py", "--teacher", "outputs/teacher",
        "--supervised", str(SUPERVISED), "--distilled", str(DISTILLED), "--output", str(RUN_ROOT / "comparison.json")],
        IDENTITY, [str(RUN_ROOT / "comparison.json")])
    comparison = json.loads((RUN_ROOT / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["controlled_distillation_comparison"] is True
    assert comparison["same_independent_test_set"] is True
    print("注意：缓存 Teacher 分类头耗时不包含远程 Evo 2，不能拿它计算 Student 整体加速倍数。")
    ''')
    markdown("standalone-instructions", r'''
    ## 独立学生推理与重载检查

    使用完整训练后 checkpoint，不使用原始预训练目录、Teacher 或 API。
    这里再次移除子进程 NVIDIA key 环境变量以验证独立运行；不修改用户 .env。
    ```powershell
    .\.venv\Scripts\python.exe -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --student outputs/hyenadna_notebook_run_20261104/distilled/best_student.pt --device cpu --cpu-threads 4 --output outputs/hyenadna_notebook_run_20261104/prediction.json
    ```
    新模型尚未校准，不能复用旧 CNN 温度。熵单位为 nats，不等于临床置信度。
    ''')
    code("standalone-prediction", r'''
    for name in ("NVIDIA_API_KEY", "NVIDIA_APIKEY", "NVAPI_KEY"):
        os.environ.pop(name, None)
    for suffix in ("prediction", "prediction_reloaded"):
        run_stage(ROOT, RUN_ROOT, suffix, ["-m", "evo2_mvp.inference", "--fasta", "data/GRCh38.fa",
            "--chrom", "1", "--pos", "926029", "--ref", "C", "--alt", "T", "--student", str(DISTILLED / "best_student.pt"),
            "--device", "cpu", "--cpu-threads", "4", "--output", str(RUN_ROOT / f"{suffix}.json")],
            IDENTITY, [str(RUN_ROOT / f"{suffix}.json")])
    prediction = json.loads((RUN_ROOT / "prediction.json").read_text(encoding="utf-8"))
    assert prediction == json.loads((RUN_ROOT / "prediction_reloaded.json").read_text(encoding="utf-8"))
    assert 0 <= prediction["student_score"] <= 1 and np.isfinite(prediction["student_predictive_entropy"])
    assert prediction["teacher_score"] is None and prediction["clinical_use"] == "research_only"
    print(json.dumps(prediction, indent=2))
    ''')
    code("final-report", r'''
    import pandas as pd
    from IPython.display import display
    from evo2_mvp.models import load_checkpoint
    records = []
    for name, directory in [("Evo 2 + MLP", Path("outputs/teacher")),
                            ("历史 CNN 监督", Path("outputs/student_supervised")),
                            ("历史 CNN 蒸馏", Path("outputs/student_distilled")),
                            ("HyenaDNA 监督", SUPERVISED), ("HyenaDNA 蒸馏", DISTILLED)]:
        m = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["test"]
        records.append({"model": name, **{k: m[k] for k in ("auroc", "auprc", "f1", "recall", "brier_score", "n", "positives")}})
    table = pd.DataFrame(records)
    table.to_csv(RUN_ROOT / "model_comparison.csv", index=False)
    display(table)
    epochs = {}
    for name, directory in [("supervised", SUPERVISED), ("distilled", DISTILLED)]:
        c = load_checkpoint(directory / "best_student.pt")
        epochs[name] = {"epochs_run": c["n_epochs_run"], "best_epoch": c["best_epoch"],
                        "checkpoint_sha256": sha256_file(directory / "best_student.pt")}
    report = {"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_identity": IDENTITY["identity"], "data": dataset, "epochs": epochs,
        "student_parameter_count": 1854465, "embedding_dim": 256, "fused_feature_dim": 768,
        "test_comparison": records, "same_initialization_verified": True, "standalone_prediction": prediction,
        "standalone_reload_prediction_identical": True, "new_nvidia_api_calls": 0,
        "limitations": ["single seed", "only 1200 labeled training variants", "previously examined internal test cohort",
                        "causal center does not see downstream context", "research_only; not clinical validation"],
        "clinical_use": "research_only"}
    save_json(RUN_ROOT / "run_report.json", report)
    print("实验已真实完成；性能以表中结果为准，不保证蒸馏改善。报告:", RUN_ROOT / "run_report.json")
    ''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Project .venv Python 3.11", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.9"}})
    path = root / "HyenaDNA_d256_experiment.ipynb"
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
