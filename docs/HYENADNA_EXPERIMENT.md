# 预训练 HyenaDNA d256 + 变异中心表示

本实验保留 `outputs/teacher`、`outputs/student_supervised`、`outputs/student_distilled`。
新监督/蒸馏模型分别写入 `outputs/hyenadna_supervised` 与
`outputs/hyenadna_distilled`，不重新调用 NVIDIA API 提取特征。

## 模型与对照

- 真实预训练模型：`LongSafari/hyenadna-tiny-1k-seqlen-d256-hf`。
- 固定 revision：`6036d4e144922b44470294f510cbfd2539ba3b7e`。
- 原始权重 SHA256：`81c15c4042d5fae646f00b429b76c65644cdbe88aa7e98b61aa781c84ff09031`。
- checkpoint 模型卡声明 BSD-3-Clause，未附独立 LICENSE；上游代码仓库另附
  Apache-2.0 LICENSE。源码、模型卡与来源说明保存在 `src/evo2_mvp/vendor/hyenadna`。
- 官方实现已固定并随项目提供，初始化严格加载全部 backbone 张量；只移除语言模型
  输出头，恢复 safetensors 去重的同一共享频率参数别名，不允许随机漏载。
- A/C/G/T 映射为 token 7/8/9/10；不加特殊 token、不 padding、不截断。
  256 bp 对应 256 tokens，变异索引为 128。编码器最后隐藏层为 `[B,256,256]`，
  中心 REF/ALT 各 256 维，`concat(ref,alt,alt-ref)` 为 768 维。
- 共享编码器 + `768→256→64→1` 头共 **1,854,465** 个参数，编码器
  **1,641,088** 个参数。训练用蒸馏投影不进入独立推理 checkpoint。
- HyenaDNA 是因果模型：中心表示能看到左侧及当前位置，不能看到右侧 127 bp。
  这和当前 Evo 2 中心表示方向一致；本实验没有双向或反向互补增强。
- 监督与蒸馏模型使用同一真实预训练权重、随机分类头初值、优化设置和固定划分。
  编码器学习率 2e-5、头学习率 1e-3，最多 20 epochs、验证 BCE 早停，具体见配置。
  这些是本轮实验初值，没有通过测试集选择。
- 蒸馏保留监督 BCE、训练集标准化后的固定 Teacher 投影特征 MSE，以及 T=2 的
  二元预测蒸馏；alpha/beta/gamma=1.0/0.1/0.5。该方法尚未被证明能改善此队列。

数据：官方 ClinVar GRCh38 SNV，release 2026-11-01；共 2,000 条，
1,695 B/LB、305 P/LP。沿用现有 gene-held-out split：train 1,200（183 阳性）、
val 400（74 阳性）、test 400（48 阳性）。训练/选模不读取测试指标。
测试集已用于历史 CNN 比较，因此它不是新的前瞻性验证集；下一轮模型选择完成后
仍需另取独立数据验证。

## PowerShell 实际命令

以下从项目根目录执行，不依赖激活环境。新实验依赖另外安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-hyenadna.txt
.\.venv\Scripts\python.exe scripts/prepare_hyenadna.py --output work/hyenadna_d256_pretrained
.\.venv\Scripts\python.exe scripts/test_hyenadna.py --pretrained-dir work/hyenadna_d256_pretrained --window 256 --output outputs/hyenadna_phase0 --compare-official
```

预训练 smoke test 只验证真实编码器与数据通路，不报告随机分类头的致病性分数。
确认 smoke 通过后，在相同 Teacher、cache 和 splits 上训练两种学生：

```powershell
.\.venv\Scripts\python.exe -m evo2_mvp.distill_student --mode supervised --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --fasta data/GRCh38.fa --teacher outputs/teacher/best_teacher.pt --phase0-report outputs/phase0_api_verified/result.json --config configs/hyenadna_d256.yaml --output outputs/hyenadna_supervised --device cpu
.\.venv\Scripts\python.exe -m evo2_mvp.distill_student --mode distill --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --fasta data/GRCh38.fa --teacher outputs/teacher/best_teacher.pt --phase0-report outputs/phase0_api_verified/result.json --config configs/hyenadna_d256.yaml --output outputs/hyenadna_distilled --device cpu
.\.venv\Scripts\python.exe scripts/compare_models.py --teacher outputs/teacher --supervised outputs/hyenadna_supervised --distilled outputs/hyenadna_distilled --output outputs/hyenadna_comparison.json
```

训练会保存最佳 checkpoint、曲线、全部 split 的指标与预测 CSV、谱系 sidecar、
初始化权重信息、配置、参数量及实测组件耗时。验证集 BCE 负责选模；最终 test 只报告。
比较脚本拒绝架构或预训练初始化不同的监督/蒸馏模型被当成蒸馏增益对照。

独立重新评估已训练学生（不依赖预训练目录；Teacher/cache 仅用于实验谱系核对）：

```powershell
.\.venv\Scripts\python.exe -m evo2_mvp.distill_student --mode distill --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --fasta data/GRCh38.fa --teacher outputs/teacher/best_teacher.pt --phase0-report outputs/phase0_api_verified/result.json --config configs/hyenadna_d256.yaml --evaluate outputs/hyenadna_distilled/best_student.pt --output outputs/hyenadna_distilled_eval --device cpu
```

单条 SNV 独立预测：

```powershell
.\.venv\Scripts\python.exe -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --student outputs/hyenadna_distilled/best_student.pt --device cpu --cpu-threads 4 --output outputs/hyenadna_prediction.json
```

此路径只加载完整学生 checkpoint 与 GRCh38 FASTA，不需 NVIDIA key、网络、
Evo 2、Teacher checkpoint 或原始预训练权重目录；HyenaDNA 的 Python 运行依赖仍需安装。
旧 CNN 校准参数与新模型不兼容，不能混用。新模型默认 `not_calibrated`，熵为 nats，
研究分数不代表 ACMG 分级或临床风险。

测试（项目内临时目录可避免本机系统临时目录的沙箱写入限制）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp work/pytest_hyenadna
$env:RUN_HYENADNA_INTEGRATION = '1'
.\.venv\Scripts\python.exe -m pytest -q -m hyenadna_integration --basetemp work/pytest_hyenadna_real
```

## 本轮运行记录

真实权重 CPU 加载和 256 bp REF/ALT 前向已通过。早期 CLI 训练产物位于
`outputs/hyenadna_supervised` / `outputs/hyenadna_distilled`。
2026-11-04 的完整 Notebook 独立运行已经完成，正式报告路径见下节；不依赖早期报告文件。

### Jupyter Notebook 独立运行

项目根目录 `HyenaDNA_d256_experiment.ipynb` 是新的完整实验入口，含数据谱系核对、
真实模型结构验证、完整 pytest、同初始化监督/蒸馏、比较及两次独立进程预测。
本次运行另存到 `outputs/hyenadna_notebook_run_20261104`，不覆盖上面的早期实验。
Notebook 的最终完成凭据是该目录 `run_report.json` 以及
`outputs/hyenadna_notebook_execution.json`，不是早期实验目录是否存在。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe scripts/run_hyenadna_notebook.py
Get-Content outputs/hyenadna_notebook_run_20261104/supervised.log -Tail 3
Get-Content outputs/hyenadna_notebook_run_20261104/distilled.log -Tail 3
```

执行器将每个单元格的输出落盘到根目录 `HyenaDNA_d256_experiment.executed.ipynb`，
出错后保留部分结果并以非零状态退出。修复后重新执行同一命令会按内容校验复用成功阶段。
训练中途没有保存优化器/数据迭代位置，因此失败阶段从固定预训练初值重新训练。
环境或代码变更会拒绝复用旧完成阶段，应明确设置新的 `RUN_ROOT`，避免混合结果。

### 2026-11-04 实测结果

根目录 `HyenaDNA_d256_experiment.executed.ipynb` 的 10 个代码单元格均已执行成功。
Notebook 全流程运行时间约 15 分钟，CPU；143 项 pytest 通过，1 项新 NVIDIA API
集成测试未运行。真实本地 HyenaDNA 权重集成测试已经执行通过。
监督与蒸馏各训练 6 epochs，按验证 BCE 早停，均恢复第 1 epoch 的最佳状态。
从相同初始模型开始，两者预训练编码器均实际更新；完整学生参数量均为 1,854,465。

同一测试集 400 条 SNV（48 P/LP、352 B/LB）：

| 模型 | AUROC | AUPRC | F1 @ 0.5 | Brier |
|---|---:|---:|---:|---:|
| Evo 2 40B API + MLP（既有 Teacher） | 0.9371 | 0.7798 | 0.6804 | 0.06389 |
| 原 CNN 监督（历史） | 0.5615 | 0.1970 | 0 | 0.10710 |
| 原 CNN 蒸馏（历史） | 0.5606 | 0.1738 | 0 | 0.10744 |
| HyenaDNA d256 监督 | 0.5676 | 0.1718 | 0 | 0.10809 |
| HyenaDNA d256 蒸馏 | 0.5815 | 0.1684 | 0 | 0.10502 |

0.5 仅用于统计计算。两个新 Student 在该统计阈值下均未预测阳性，召回率为 0。
蒸馏模型 AUROC 略高、AUPRC 更低；本轮未证实有效或稳定的蒸馏收益。
模型与实验管线真实运行成功，不代表已取得可用的致病性判别性能。
训练损失下降、验证损失上升，有明显过拟合迹象；仅一个随机种子、训练标签仅 1,200 条。
该测试队列已经在历史实验中被查看，后续结论仍需新的独立队列验证。

单条 `1-926029-C-T` 的新蒸馏学生原始分数为 `0.11535679548978806`，
预测熵 `0.35757049022074805` nats，未校准。两次全新 Python 进程加载学生得到完全一致结果，
不调用 Evo 2 或 NVIDIA API。这里不为该变异作临床分类。

真实运行产物：

- `outputs/hyenadna_notebook_run_20261104/supervised/best_student.pt`
- `outputs/hyenadna_notebook_run_20261104/distilled/best_student.pt`
- `outputs/hyenadna_notebook_run_20261104/run_report.json`
- `outputs/hyenadna_notebook_run_20261104/artifact_audit.json`
- `outputs/hyenadna_notebook_run_20261104/comparison.json`、`model_comparison.csv`
- 同目录两个训练日志、每个模型的曲线/指标/预测 CSV/谱系，以及 `requirements.lock.txt`。

独立产物审计已经通过，重新计算 train/val/test 全部指标、核对初始 SHA、验证权重有限、
验证预训练编码器实际更新、确认 Teacher 文件未变化。可重复执行：

```powershell
.\.venv\Scripts\python.exe scripts/audit_hyenadna_run.py
```

本次 Codex 受限环境启动 Jupyter 时曾在 Windows Proactor `socketpair` 回环建连处阻塞，
尚未进入训练；改为允许本机内核通信的执行方式后成功完成，未修改模型以绕过训练错误。
Notebook 训练与测试阶段没有报错，本轮新增 NVIDIA API 请求数为 0。

参考：[官方模型卡](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-d256-hf)、
[HyenaDNA 源码](https://github.com/HazyResearch/hyena-dna)、
[HyenaDNA 论文](https://arxiv.org/abs/2306.15794)。
