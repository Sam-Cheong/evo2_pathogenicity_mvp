# 当前交付与验证报告

日期：2026-11-03（Asia/Shanghai）。**真实 NVIDIA Evo 2 40B REF/ALT Phase 0 已通过。** 当前默认方案为 NVIDIA hosted Evo 2 API + 本机 MLP/CNN；Teacher 和 Student 分类训练仍未完成。

## 当前方案与实际完成

已创建完整 Python 项目、基础依赖、默认/备选 API 配置、环境与 key 检查、NVIDIA forward adapter、Phase 0 CLI、SNV/FASTA 管线、ClinVar 严格筛选与基因分组、可恢复 embedding cache、Teacher MLP 训练/评估、监督 CNN、知识蒸馏、独立 Student 推理、熵/验证集校准和同测试集模型比较。README 包含可执行命令。

默认配置为 `configs/mvp.yaml`：`arc/evo2-40b`、`https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward`、`nim_v1`、`blocks.20.mlp.l3`，已通过本次真实请求验收。当前托管部署为 StripedHyena，`configs/nvidia_v2.yaml` 的 decoder 层实测返回 422；不建议用于当前服务。官方新旧文档存在命名不一致，以实际验证的 v1 配置为准；不能静默切换或混用缓存，详见 `nvidia_api_audit.md`。

**这是远程 40B 方案，不是本地 1B 已运行成功。** 服务没有公开本次可固定的 checkpoint SHA/revision；本地 `service_revision` 只是实验身份。基础安装不需要下载 Evo 2 权重或安装 Transformer Engine/Flash Attention，CPU 可训练分类头和 Student。

本次在用户授权下使用其本机 `.env` 完成真实调用，没有输出或交付密钥。进程环境变量覆盖 `.env`；ZIP 不包含 `.env` 和任何密钥，新安装按 README 从 `.env.example` 复制。已有用户配置应保留。

## 真实模型状态

| 项目 | 本次实际状态 |
|---|---|
| 当前 backend | NVIDIA hosted API，默认模型 40B |
| API 端点访问与认证 | 真实 HTTP 200 已验证，不保存密钥 |
| 真实 API REF/ALT 调用 | **Phase 0 passed**；固定 256bp、index128 A→G |
| 本地 Evo 2 权重下载/加载 | 未执行，当前 API 方案不需要 |
| 实际 token embedding shape | **`[1,256,8192]`** |
| 实际 REF/ALT/delta 向量 shape | **各 `[8192]`** |
| 实际拼接 variant features shape | **`[24576]`** |
| 实际 REF/ALT 请求耗时 | **8.538849599994137秒**，包含网络和服务端时间 |
| 重复结果最大绝对差 | **0.0** |
| 有限值 / 保存重载 | **均通过** |
| 远程 GPU 型号与峰值显存 | 提供商未暴露 / null |
| 服务端参数冻结状态 | 客户端无法直接检查；仅本地 MLP 被优化 |
| 真实 Phase 0 特征 | 已保存 `.npy`；批量训练 cache 尚未生成 |
| 真实 Teacher 训练 | 未完成 |
| Student 监督训练/蒸馏实验 | 未开始；保留 Teacher 优先顺序 |
| Teacher/Student 真实测试指标 | 不存在 |
| 实际致病性预测 JSON | 没有生成，缺少真实已训练权重 |

[真实 Phase 0 完整报告](../outputs/phase0_api_verified/result.json) 位于 `outputs/phase0_api_verified/`，同目录保存真实 `ref.npy`、`alt.npy`、`delta.npy`、`features.npy`。报告含 `real_model_verified=true`、`finite=true`、`save_reload_consistent=true`、`repeated_run_consistent=true`。这证明真实远程特征推理与文件验收通过，不表示存在已训练分类器。

早期 `outputs/phase0_api/` 中的缺 key/失败状态和 `outputs/phase0/` 的本地 1B 受阻记录保留作历史追踪，不能覆盖此次成功结果。`docs/official_model_audit.md` 也属于本地 1B 历史核查。

## 验证边界

本次完整真实 Phase 0 使用用户现有的 **Windows / Python 3.11.9 / PyTorch 2.7.1+cpu** 虚拟环境，`cuda_available=false`，已证实当前 API 路径不依赖本机 CUDA。早期无权重测试使用 Python 3.10.6 / PyTorch 2.5.1+cu121。开发机 RTX 3050 Ti Laptop 4GiB 不是 NVIDIA 服务端硬件；服务端显存与冻结参数状态仍无法由客户端测量。

无权重验证包括严格 REF/坐标/ALT/上下文检查、ClinVar 标签/审核和重复冲突处理、基因分组隔离、Fake embedding cache、MLP 反传与两 epoch 合成管线训练/保存重载、Student 结构与独立进程推理、KD detach、熵/校准谱系、HTTP JSON/Base64-NPZ/ZIP和错误响应解析。合成标签和 mock 响应不用于真实性能。

完整回归测试为 **109 passed、1 skipped**，记录为 `outputs/validation/junit_api.xml`；另外在用户 Python 3.11.9 / PyTorch 2.7.1+cpu 环境对 API、诊断、缓存和谱系执行了 **80 项测试，全部通过**，记录为 `outputs/validation/junit_python311_api.xml`。真实 REF/ALT 集成已通过单独执行的 Phase 0 CLI 验收；pytest 的需显式启用网络测试仍为 skip，二者分开记录。未运行任何真实分类训练实验。

## 实际真实数据

本次取得官方 ClinVar GRCh38 **2026-11-01** VCF 前 **5,000** 条及 NCBI RefSeq **NC_000001.11:1–1,212,000** 的真实参考序列。原始前缀、FASTA、URL、下载范围和 SHA256 保存在 `data/real_demo/`。

严格常染色体 SNV、P/LP与B/LB、至少两星审核、去重以及 REF/完整256bp核验后得到 **473 条：466 B/LB、7 P/LP**。目标 2,000 未达到，没有放宽要求。该数据为 chr1 前缀演示，不能代表完整 ClinVar 队列。

固定 seed42 连通基因分组为 train270（6阳性，2组）、val27（1阳性，3组）、test176（0阳性，1组）。它的单类别测试集无法计算有效 AUROC/AUPRC，也不能回答蒸馏是否改善泛化。本次未将这批标签用于真实模型训练。正式研究仍需完整官方队列与完整 GRCh38 FASTA，预先固定有效划分，不按测试表现选择。

## 下一步

真实 Phase 0 已完成。下一步应准备代表性完整数据与有效独立划分，使用已验证的 `configs/mvp.yaml` 批量提取新缓存，先训练/验收 Teacher，再执行监督/蒸馏 Student 对比。本次没有启动这些真实训练阶段，没有真实性能指标或临床分类结论。

Student 独立推理不需要 key、API、Evo 2 包或权重。所有最终分数、熵及校准结果仅用于研究，不构成 ACMG 分级、临床置信度或诊断。

