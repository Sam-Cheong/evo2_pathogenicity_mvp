# 两阶段 StripedHyena 2 小型参考变体实验

本轮是用户于 2026-11-04 确认的缓存小试：先无标签表示蒸馏，再冻结学生编码器训练判别头。
只复用现有真实 NVIDIA Evo 2 40B 特征，不新增 Evo 2 API 请求。
根目录入口为 `StripedHyena2_two_stage_experiment.ipynb`，含输出版本为
`StripedHyena2_two_stage_experiment.executed.ipynb`，独立产物目录为
`outputs/sh2_two_stage_20261104`。原 Teacher、CNN、HyenaDNA 实验均保留。

## 模型与证据

学生是自定义的小型 **SH2 reference variant**，从固定随机种子初始化，不具有官方预训练权重。
由 SE、MR、LI 和因果 RoPE attention 四个残差块组成，包含输入短卷积、门控 FFN 与 RMSNorm。
多尺度算子公式和前向/梯度已与固定 Savanna 原始函数对照；具体差异、来源和容差见
[架构审计](SH2_ARCHITECTURE_AUDIT.md)。不将 HyenaDNA 改名为 SH2，不声称完整模型等同 Evo 2。

中心表示 192 维，REF/ALT/delta 融合为 576 维。完整推理模型：

- 线性头：1,505,281 参数；MLP：1,582,721 参数。
- 训练用 REF/ALT 共享投影与 delta 独立投影共 3,162,112 参数，只在 Stage 1 使用。
- Stage 1 总训练参数为 4,666,816；投影不进入独立推理模型。

Teacher 是已验证的 `arc/evo2-40b`，层 `blocks.20.mlp.l3`，`pooling=variant`，
每条 REF/ALT 8192 维。Teacher 与 Student 均取当前位置的因果表示，只看到左侧和当前位置。
现有缓存没有完整 token 张量，不能由这些中心向量恢复其他位置或其他隐藏层。

## 数据隔离

真实数据沿用 ClinVar GRCh38 的 2,000 条常染色体 SNV，外层 train/val/test 为 1200/400/400。
第一阶段仅使用原 train 的序列和缓存：按基因及重叠 256bp 上下文连通分组，划为
960 representation-train / 240 representation-val，不按致病标签分层或选择。
连通组 717 个；本轮与 clinical val/test 上下文重叠而排除的 train 数量实际为 0。

`representation_train.npz`、`representation_val.npz` 严格只含：
`ref, alt, teacher_ref, teacher_alt, variant_ids`，没有标签、基因或审核结论字段。
只有原 train 对应的 Teacher NPZ 被读取并校验 REF/ALT 上下文 hash。

Probe 数据独立保存 `ref, alt, labels, variant_ids`，完整保留原 1200/400/400。
数据准备可以生成并校验封存测试文件；任何模型训练阶段均不打开测试归档。
Notebook 复用已准备数据时也不预读测试归档；四个头全部完成后独立 evaluate 阶段才打开它。

## Stage 1：表示蒸馏

不加载 ClinVar CSV、probe 文件、Teacher 分类头或软标签。损失为：

`0.5 * MSE_REF + 0.5 * MSE_ALT + 1.0 * MSE_delta`。

REF/ALT 使用 representation-train 共同统计量；delta 单独使用 train-only 均值和尺度。
教师目标固定，只训练学生编码器及两个投影；避免共同学习的 Teacher 投影导致目标塌缩。
记录每项验证误差、常数训练均值预测基线、学生表示方差、梯度范数与单轮时间。
只按 representation-val 总损失选择编码器，默认最多 20 epochs、patience=5。
保存随机初始 encoder、最佳 encoder 和每轮 optimizer 续训状态。

这些 960 个配对只适合作为方法小试，不能称复现或压缩整个 Evo 2 基础模型。

## Stage 2：冻结表示探针

预先定义四组，均使用真实标签训练头：

| 内部名字 | 编码器 | 判别头 |
|---|---|---|
| random_linear | 蒸馏前的同一个随机初始 encoder，冻结 | 线性头 |
| distilled_linear | 表示蒸馏后的 encoder，冻结 | 线性头 |
| random_mlp | 蒸馏前的同一个随机初始 encoder，冻结 | 小型 MLP |
| distilled_mlp | 表示蒸馏后的 encoder，冻结 | 小型 MLP |

`random` 指 encoder 的初始化来源，不是随机生成预测结果。四个分类头均实际监督训练。
相同类型头在两个编码器臂上使用相同初始参数、shuffle 和 dropout 随机种子。
编码器全程 `requires_grad=False`、`eval()`，并校验权重 SHA 不变。
融合特征标准化仅拟合 probe-train，头按 clinical-val BCE 早停。
所有四组完成且谱系检查通过后才允许评估测试集，不按测试结果选模型或阈值。
独立预测示例预先指定 `distilled_linear`，不因测试成绩更改。

## PowerShell 命令

从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-sh2.txt -r requirements-notebook.txt
.\.venv\Scripts\python.exe scripts/prepare_sh2_reference.py --download --self-check
.\.venv\Scripts\python.exe scripts/prepare_two_stage.py --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --fasta data/GRCh38.fa --output outputs/sh2_two_stage_20261104/data --seed 42 --window 256
.\.venv\Scripts\python.exe scripts/test_stripedhyena2.py
.\.venv\Scripts\python.exe scripts/run_hyenadna_notebook.py --notebook StripedHyena2_two_stage_experiment.ipynb --output StripedHyena2_two_stage_experiment.executed.ipynb --status outputs/sh2_notebook_execution.json
```

已有正确参考文件时，`prepare_sh2_reference.py --self-check` 可离线运行。
Notebook 显式将 `RUN_EVO2_INTEGRATION=0`，不受用户旧环境开关影响。
执行器保存每个单元格的输出，异常停止并保留部分 Notebook，不将报错记为成功。
默认配置为 `configs/stripedhyena2_two_stage.yaml`。

三个训练/评估阶段也可单独执行：

```powershell
.\.venv\Scripts\python.exe -m evo2_mvp.two_stage_student --stage distill --data-dir outputs/sh2_two_stage_20261104/data --config configs/stripedhyena2_two_stage.yaml --output outputs/sh2_two_stage_20261104/training --device cpu --resume
.\.venv\Scripts\python.exe -m evo2_mvp.two_stage_student --stage probes --data-dir outputs/sh2_two_stage_20261104/data --config configs/stripedhyena2_two_stage.yaml --output outputs/sh2_two_stage_20261104/training --device cpu --resume
.\.venv\Scripts\python.exe -m evo2_mvp.two_stage_student --stage evaluate --data-dir outputs/sh2_two_stage_20261104/data --config configs/stripedhyena2_two_stage.yaml --output outputs/sh2_two_stage_20261104/training --device cpu
```

`--resume` 在首次运行时正常初始化；有状态时强制核对数据、配置、初始和最佳模型谱系。
表示训练在 epoch 边界恢复 optimizer；probe 只复用完整训练结束的臂，半途臂从相同 head 初值重训。

独立学生预测不需要 Teacher、API key、原始参考源码目录或网络：

```powershell
.\.venv\Scripts\python.exe -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --student outputs/sh2_two_stage_20261104/training/probes/distilled_linear/best_student.pt --device cpu --cpu-threads 4 --output outputs/sh2_two_stage_20261104/prediction.json
```

检查进度：

```powershell
Get-Content outputs/sh2_two_stage_20261104/distill.log -Tail 2
Get-Content outputs/sh2_two_stage_20261104/training/distillation_status.json
Get-Content outputs/sh2_notebook_execution.json
```

## 结果与限制

本文件创建时训练尚在执行。只有 `outputs/sh2_notebook_execution.json` 和
`outputs/sh2_two_stage_20261104/run_report.json` 均为 completed，才表示本轮全流程完成。
最终指标来自真实预测 CSV，不以 embedding MSE 下降代替下游分类性能。
本轮仅一个种子，沿用已被历史实验查看过的测试队列；所有分数仅供研究。
