# evo2_pathogenicity_mvp

新增用户确认的两阶段小试：自定义小型 StripedHyena 2 reference 先蒸馏真实 Evo 2 中心表示，
再冻结 encoder 训练线性/MLP 判别头。入口为根目录 `StripedHyena2_two_stage_experiment.ipynb`，
独立输出 `outputs/sh2_two_stage_20261104`，不新增 Evo 2 API 请求。
协议、CLI、真实执行状态与限制见 [SH2 两阶段实验](docs/SH2_TWO_STAGE_EXPERIMENT.md)。

当前使用 **NVIDIA hosted Evo 2 40B API 提取 DNA 特征，在本机训练 Frozen Encoder + MLP**。新增学生实验为**真实预训练 HyenaDNA d256 + 变异中心表示**，分别进行仅监督训练与知识蒸馏；原 CNN 实验保留。Student 推理独立运行，不调用 NVIDIA API。

项目根目录的 `HyenaDNA_d256_experiment.ipynb` 包含本轮完整实验指令和可执行单元格，
执行输出保存到 `HyenaDNA_d256_experiment.executed.ipynb`。从根目录 PowerShell 运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-hyenadna.txt -r requirements-notebook.txt
.\.venv\Scripts\python.exe scripts/run_hyenadna_notebook.py
```

本轮 Notebook 的独立产物目录为 `outputs/hyenadna_notebook_run_20261104`。逐轮日志位于
`supervised.log` / `distilled.log`，执行状态为 `outputs/hyenadna_notebook_execution.json`。
出错会停止并保存部分 Notebook；修复后重跑，完成阶段必须通过输入/代码/产物 SHA256
复核才能复用。中断中的模型训练会从固定初始化重新开始，不声称可恢复中途优化器状态。
`run_report.json` 和执行状态均为 `completed` 才表示整个 Notebook 实验完成。
交互查看：`.\.venv\Scripts\python.exe -m notebook HyenaDNA_d256_experiment.executed.ipynb`。

**真实 NVIDIA Evo 2 40B REF/ALT Phase 0 已通过。** 实测 token embedding 为 `[1,256,8192]`，REF/ALT 各为 `[8192]`，拼接特征为 `[24576]`。现有 2,000 个真实 ClinVar 变异已完成缓存、Teacher 和旧 CNN 训练；Teacher 测试 AUROC 0.9371，旧蒸馏 CNN 为 0.5606。新 HyenaDNA Notebook 于 2026-11-04 真实执行完成：仅监督/蒸馏测试 AUROC 为 0.5676/0.5815，AUPRC 为 0.1718/0.1684；学生性能仍弱，未证实有效蒸馏收益。实际结果见 [HyenaDNA 实验说明](docs/HYENADNA_EXPERIMENT.md)；原 `docs/DELIVERY_REPORT.md` 是早期阶段的历史记录。

此方案不需要本地下载 Evo 2 权重、安装 Transformer Engine/Flash Attention 或配备 Hopper GPU。远程服务模型为 **40B，不是原先的 1B**；服务未暴露可固定的权重 SHA/revision。原本地 1B 的历史核查保存在 `docs/official_model_audit.md`，不是当前默认安装流程。API、模型身份和新旧协议差异见 `docs/nvidia_api_audit.md`。

## 1. 创建 Python 环境

推荐 Python 3.11。本次真实 API 验收使用 Windows 上的 Python **3.11.9 / PyTorch 2.7.1+cpu** 虚拟环境，无需本机 CUDA。已有该环境时直接激活即可；新建环境可执行以下命令。以下均从项目根目录运行。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 2. 安装依赖并配置 NVIDIA key

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

基础依赖包括 PyTorch、requests、python-dotenv 和数据/评估工具。当前 API 方案**无需安装 `requirements-evo2.txt`**，无需 Evo 2 本地权重。

已有 `.env` 时保留现有配置；下载 ZIP 不包含 `.env` 或任何密钥，新安装需复制模板。PowerShell：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

在编辑器内填入：

```dotenv
NVIDIA_API_KEY=你的实际NVIDIA密钥
```

也可在 **PowerShell 7** 临时设置环境变量，输入会隐藏：

```powershell
$env:NVIDIA_API_KEY = Read-Host -MaskInput 'NVIDIA API key'
```

Linux/macOS Bash：

```bash
read -rsp 'NVIDIA API key: ' NVIDIA_API_KEY
export NVIDIA_API_KEY
```

进程环境变量优先于项目根目录 `.env`。不要把 key 写入 YAML、Python、命令行参数或发到聊天中；不要提交 `.env`。项目只报告 key 是否存在，不打印密钥。

## 3. 检查环境与 CUDA

```bash
python scripts/check_environment.py --config configs/mvp.yaml --check-network --require-evo2 --output outputs/environment_api.json
```

此命令检查当前 API backend 的 key 配置和基础运行条件；本机 CUDA 状态仍会列出，但不作为远程 Evo 2 的门槛。key 存在或网络可达不等于已完成认证/模型推理。MLP/CNN 可用本机 CPU 训练。

## 4. 调用真实 Evo 2：固定 REF/ALT 单序列 Phase 0

```bash
python scripts/test_evo2.py --config configs/mvp.yaml --window 256 --output outputs/phase0_api_verified
```

命令使用固定 256bp DNA，在 index 128 做 A→G，分别请求 REF/ALT embedding，计算 delta 和拼接特征，检查 shape、有限值、重复推理及保存重载。只有真实成功才生成 `ref.npy`、`alt.npy`、`delta.npy`、`features.npy` 与成功报告。没有 key、认证失败或接口错误会非零退出，保存明确状态，不产生模拟结果。

本次已实际通过：`status=passed`、token shape `[1,256,8192]`、REF/ALT/delta shape `[8192]`、拼接 `[24576]`。测得一对 REF/ALT 请求 wall time **8.53885 秒**（包括网络和服务端时间），重复特征最大绝对差 **0**；finite 和 save/reload 检查均为 true。该时间是本次测量，不是分类器耗时或跨账户性能保证。输出文件保存在 `outputs/phase0_api_verified/`。

输出记录远程模型、endpoint、profile、实际层和形状、HTTP wall time、可用的服务端耗时。远程 GPU 型号/峰值显存和本地权重冻结状态无法由客户端测量，相应字段保持 null。成功 API 调用也不等于本地加载过权重。

默认 `configs/mvp.yaml` 为 `arc/evo2-40b`、`nim_v1`、`blocks.20.mlp.l3`，**这是本次真实验证成功的托管协议**。`configs/nvidia_v2.yaml` 请求 decoder 层，本次在相同端点返回 HTTP 422：实际 StripedHyena 模型没有 `decoder` 属性。因此当前托管方案使用 `mvp.yaml`，不建议切到 v2；v2 只保留供其他明确支持该协议的部署使用。官网新层示例不能覆盖实测结果。

若遇到 400/422，应检查脱敏验证错误和实际部署，不按状态码自动换协议。切换 profile、层、窗口或服务实验标识后使用**新缓存目录**。同一模型名的远程权重可能更新，`service_revision` 是本地实验标识，不是提供商保证的权重 revision。

256bp 成功后可单独试 512bp：

```bash
python scripts/test_evo2.py --config configs/mvp.yaml --window 512 --output outputs/phase0_api_512
```

可选 `--include-sequence-scores` 另算 REF/ALT 平均序列 log-likelihood；它不是致病概率，也不替代 embedding baseline。API 调用会使用账户配额；批量特征提取前先通过这一小测试。

## 5. 准备 ClinVar GRCh38 真实数据与独立划分

提供官方 GRCh38 ClinVar VCF/VCF.gz 和匹配的完整 `data/GRCh38.fa`。FASTA 需未压缩或 BGZF，支持 `1`、`chr1`、RefSeq 染色体名。坐标为 **1-based**，不进行 liftOver。`--release-date` 必须与 VCF `##fileDate` 一致。

```bash
python scripts/prepare_clinvar.py --vcf data/clinvar.vcf.gz --fasta data/GRCh38.fa --out data/variants.csv --release-date 2026-11-01 --target 2000 --window 256 --seed 42
python -m evo2_mvp.data --variants data/variants.csv --out data/splits.json --seed 42
```

完整 VCF 可从 [NCBI 官方 ClinVar](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/) 获取；未来新版本应使用实际发布日期。脚本完整扫描后抽样：仅常染色体 SNV、P/LP=1、B/LB=0、至少两星审核；排除 VUS/冲突/其他分类、重复冲突、REF 不符、N/越界上下文。保存来源、审核、日期、VCF/FASTA SHA256；不足 2,000 时如实报告，不放宽标准。

CSV 最少字段为 `chrom,pos,ref,alt,gene,label`。共享基因注释按连通组隔离，检查所有 variant ID 与 gene 不跨 train/val/test；缺失 gene 或不足三个独立组时拒绝 gene-held-out 划分。划分固定 seed，不按模型测试表现选择。基因很少、阳性组很少或相邻序列重叠仍限制独立性和可信度。标签与审核信息只用于监督/追踪，不进入 DNA 模型输入。

附带 `data/real_demo/` 是本次实际取得的官方 ClinVar 2026-11-01 前 5,000 条与 GRCh38 chr1 局部参考，严格保留 **473 条真实 SNV：466 B/LB、7 P/LP**。它有严重区域/类别偏倚，测试分区没有阳性；不适合报告泛化或蒸馏效果。来源、范围及校验在该目录 README 与 provenance JSON。可离线复核：

```bash
python scripts/prepare_clinvar.py --vcf data/real_demo/clinvar_official_prefix.vcf --fasta data/real_demo/grch38_chr1_prefix.fa --out outputs/real_demo_rechecked.csv --release-date 2026-11-01 --target 2000 --window 256
```

## 6. 批量提取真实 embedding 并缓存

Phase 0 真正成功后，使用同一配置：

```bash
python -m evo2_mvp.cache --variants data/variants.csv --fasta data/GRCh38.fa --config configs/mvp.yaml --output outputs/nvidia_cache
```

每个 SNV 的 REF/ALT 共用参考坐标和上下文，特征为 `concat(ref,alt,alt-ref)`。缓存按 variant ID 原子保存，验证 shape/有限值/校验值，支持断点续跑和跳过已有记录。CLI 在读取 manifest 或调用 API 前取得 `.writer.lock` 独占写锁，整个提取过程保持锁；第二个写入同目录的 CLI 会立即明确拒绝。JSON/NPZ 使用独立临时文件，避免固定 `.tmp` 名称冲突。不要同时在两个 PowerShell 窗口运行同一缓存命令，也不要在进程运行时删除锁文件。进程退出后操作系统自动释放锁；锁文件存在本身不表示仍有任务运行。低层 `FeatureCache` 编程接口仍要求调用方保证单写者。manifest 记录 endpoint/model/profile/layer/pooling、实验身份、GRCh38、FASTA SHA256、窗口和实际维度。训练 MLP 时只读缓存，不再调用 API。

默认位点 token 表示是因果模型的表示，不能看到该 token 右侧全部上下文；可在配置改为 `local_mean` 并新建缓存。特征协议选择只能用训练/验证集，不能用测试表现挑选。

## 7. 训练与评估 Teacher：Frozen Encoder + MLP

```bash
python -m evo2_mvp.train_teacher_head --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --config configs/mvp.yaml --output outputs/teacher --device cpu
python -m evo2_mvp.train_teacher_head --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --config configs/mvp.yaml --output outputs/teacher_eval --evaluate outputs/teacher/best_teacher.pt --device cpu
```

默认 AdamW、lr=1e-3、20 epochs、batch 32、MLP 256→64→1、GELU/dropout 0.2、BCE。仅训练集拟合特征标准化，验证集 BCE 早停选最佳 checkpoint。远程 encoder 没有本地优化器参数；只训练 MLP，不能声称已检查服务端 `requires_grad`。

输出权重、配置/seed/谱系、训练曲线 CSV/PNG、分开的 train/val/test 指标、预测 CSV 及模型来源 sidecar、计时/内存报告。实现 AUROC、AUPRC、Accuracy、Precision、Recall、F1、confusion matrix、Brier；单类别分区 AUROC/AUPRC 为 null 并解释。0.5 只用于统计指标，不构成临床判定。

## 8. Teacher 验收后训练两个 Student

以下保留原 CNN 训练命令，现有训练结果位于 `outputs/teacher`、`outputs/student_supervised`、`outputs/student_distilled`。新的 HyenaDNA 命令与独立输出目录见 [HyenaDNA 实验说明](docs/HYENADNA_EXPERIMENT.md)。

```bash
python -m evo2_mvp.distill_student --mode supervised --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --fasta data/GRCh38.fa --teacher outputs/teacher/best_teacher.pt --phase0-report outputs/phase0_api_verified/result.json --config configs/mvp.yaml --output outputs/student_supervised --device cpu
python -m evo2_mvp.distill_student --mode distill --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --fasta data/GRCh38.fa --teacher outputs/teacher/best_teacher.pt --phase0-report outputs/phase0_api_verified/result.json --config configs/mvp.yaml --output outputs/student_distilled --device cpu
python -m evo2_mvp.distill_student --mode distill --variants data/variants.csv --splits data/splits.json --cache outputs/nvidia_cache --fasta data/GRCh38.fa --teacher outputs/teacher/best_teacher.pt --phase0-report outputs/phase0_api_verified/result.json --config configs/mvp.yaml --output outputs/student_distilled_eval --evaluate outputs/student_distilled/best_student.pt --device cpu
```

监督 Student 的评估使用第三条，改为 `--mode supervised` 及其权重/目录。评估 CLI 的 Teacher/cache 用于实验谱系核对；独立预测不需这些文件。

Student 为共享 1D CNN + mean/max pooling + REF/ALT/delta 融合 + MLP，默认推理参数 **83,265（<2M）**。蒸馏损失为监督 BCE + 0.1×feature MSE + 0.5×binary KD，T=2，KD 乘 T²；初值未经真实实验优化。固定 Teacher 投影、训练集标准化与 L2 归一化避免自由目标塌缩，训练用 projection 不进入 Student 推理。

## 9. 单个 GRCh38 SNV 预测 JSON

以下命令使用真实训练得到的 `.pt`。Teacher 使用 API 特征加本机 CPU MLP，key 仍需存在：

```bash
python -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --gene SAMD11 --teacher outputs/teacher/best_teacher.pt --student outputs/student_distilled/best_student.pt --device cpu --output outputs/prediction.json
```

Student 独立 CPU 预测，**不需 NVIDIA key、网络、Evo 2 包或权重**：

```bash
python -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --student outputs/student_distilled/best_student.pt --device cpu --output outputs/student_prediction.json
```

输出含 `variant_id,genome_build,teacher_score,student_score,student_predictive_entropy,ensemble_std,calibrated_score,calibration_status,clinical_use`，未请求的模型值为 null，`clinical_use=research_only`。熵单位 nats，不能称临床置信度；当前未训练三种子 ensemble，标准差保持 null。模型分数不代表 ACMG 等级、诊断或真实患病风险。

## 10. 可选验证集温度校准

```bash
python -m evo2_mvp.calibration --predictions outputs/student_distilled/predictions.csv --checkpoint outputs/student_distilled/best_student.pt --splits data/splits.json --output outputs/student_distilled/calibration.json
python -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --student outputs/student_distilled/best_student.pt --calibration outputs/student_distilled/calibration.json --device cpu --output outputs/student_prediction_calibrated.json
```

只用 val 拟合温度，在未参与拟合的 test 报告前后指标；CSV/sidecar、checkpoint、标签与划分必须匹配。旧 CNN 的校准文件不能用于新 HyenaDNA。

## 11. 三模型比较与实测耗时

```bash
python scripts/compare_models.py --teacher outputs/teacher --supervised outputs/student_supervised --distilled outputs/student_distilled --output outputs/comparison.json
```

脚本核对模型类型、监督/蒸馏模式、共同 Teacher 和相同独立测试集。Teacher 的缓存分类头计时不含 API，不能直接与完整 Student 比较。分别测同一 SNV 的完整 warm 路径：

```bash
python -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --teacher outputs/teacher/best_teacher.pt --device cpu --benchmark-output outputs/teacher_e2e.json
python -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --student outputs/student_supervised/best_student.pt --device cpu --benchmark-output outputs/supervised_e2e.json
python -m evo2_mvp.inference --fasta data/GRCh38.fa --chrom 1 --pos 926029 --ref C --alt T --student outputs/student_distilled/best_student.pt --device cpu --benchmark-output outputs/distilled_e2e.json
python scripts/compare_models.py --teacher outputs/teacher --supervised outputs/student_supervised --distilled outputs/student_distilled --teacher-e2e outputs/teacher_e2e.json --supervised-e2e outputs/supervised_e2e.json --distilled-e2e outputs/distilled_e2e.json --output outputs/comparison.json
```

远程 Teacher 耗时包含网络和服务排队，Student 是本机模型；必须连同执行位置一起解释，不能称同 GPU 算子提速。进程 RSS 为调用间采样，非瞬态峰值；远程 GPU 显存未知。重复 benchmark 也会消耗 API 配额。

## 12. 自动化验证与目录

```bash
python scripts/create_toy_dataset.py --out-dir work/toy --n 24
python -m pytest -q --junitxml=outputs/validation/junit.xml
```

真实 API 集成测试需 key 并显式启用。PowerShell：

```powershell
$env:RUN_EVO2_INTEGRATION = '1'
python -m pytest -q -m integration
```

Bash：`RUN_EVO2_INTEGRATION=1 python -m pytest -q -m integration`。默认 skip 不等于通过。单元测试使用 Fake Encoder/mock HTTP 验证坐标、REF/ALT、冻结/反传、缓存、去重/基因划分、响应解析、key缺失、NaN/错误处理、checkpoint重载和Student独立推理；这些不证明真实 API 成功。合成标签仅作明确标记的管线测试，正式预测拒绝合成 checkpoint。

主要文件：`configs/` 为默认/备选协议；`src/evo2_mvp/nvidia_encoder.py` 为 API 适配器；`sequence.py/data.py/cache.py` 为数据与缓存；`models.py/train_teacher_head.py/distill_student.py` 为训练；`inference.py/calibration.py` 为预测和研究性不确定性；`scripts/` 为实际 CLI；`data/real_demo/` 为公开真实数据演示；`outputs/` 为验证及后续实验结果。

真实 API Phase 0、2,000 变异特征缓存、Teacher 和旧 CNN 实验已完成。新增 HyenaDNA 实验复用 Teacher，不覆盖旧输出。后续仍需更大且更有代表性的队列及全新独立评估；当前模型仅供研究。

研究依据：[Evo 2](https://github.com/ArcInstitute/evo2)、[Hinton et al.](https://arxiv.org/abs/1503.02531)、[GPN](https://pubmed.ncbi.nlm.nih.gov/37883436/)、[ACMG/AMP](https://pubmed.ncbi.nlm.nih.gov/25741868/)。本项目为研究性二分类，不实现临床 ACMG 判定。
