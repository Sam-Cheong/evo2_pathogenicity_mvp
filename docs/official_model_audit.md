# 历史记录：本地 Evo 2 1B 官方接口审计

核查日期：2026-11-03（Asia/Shanghai）。这是切换到 NVIDIA hosted API 之前的本地 1B 核查记录，保留供追踪；当前默认方案是远程 40B API，安装和运行以项目 README 为准，无需按本文准备本地 Hopper/FP8 环境。本文件中的 1B 预期形状和本地受阻状态不能替代当前 API 验证结果。

## 权重身份和下载条件

官方名称为 `evo2_1b_base`，HF 仓库为 `arcinstitute/evo2_1b_base`，文件为 `evo2_1b_base.pt`。固定 revision 为 `2279e1df422c991037470302360edd40d0d2ea1e`。仓库公开、非 gated，模型卡标注 Apache-2.0。原始 LFS 指针记录 2,698,327,083 字节，SHA256 为 `8ffba7d0e6445a8f2c92d9ff1c4e772c7f73ca9179f5e0c699b8b0ca1b966f64`。[官方权重及文件](https://huggingface.co/arcinstitute/evo2_1b_base/tree/2279e1df422c991037470302360edd40d0d2ea1e)、[原始指针](https://huggingface.co/arcinstitute/evo2_1b_base/raw/2279e1df422c991037470302360edd40d0d2ea1e/evo2_1b_base.pt)。

本次以 HEAD 请求检查固定 revision 的 resolve 链接，最终 CDN 返回 HTTP 200。未下载 2.7 GB 权重；HEAD 成功只能验证公开资源当时可访问，不能证明完整权重下载、校验或模型加载成功。普通沙箱 Python DNS 失败；获准的公共只读网络请求可读取 HF/GitHub。精确元数据见 `model_provenance.json`。

## 支持的软件和硬件

固定官方源码 commit 为 `53f195997257c56c00e5ef8d33a54f5baad143a6`，其包版本为 0.6.0，要求 Python `>=3.11,<3.13`，依赖 `vtx>=1.1.0`，未锁定确切 Vortex 版本。可将本项目的运行环境锁到 vtx 1.1.0，并记录完整依赖快照。[官方 pyproject](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/pyproject.toml)。

官方 README 要求 Linux（WSL2 支持有限）、CUDA 12.1+、cuDNN 9.3+、Python 3.11/3.12，并建议 PyTorch 2.6/2.7。1B 模型为保障数值准确性要求 Transformer Engine FP8 和 Hopper GPU。本机原生 Windows / Python 3.10 / RTX 3050 Ti 4 GB 不满足这一官方运行组合。[官方要求](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/README.md#requirements)。

审阅的官方页面未给出“1B、256 bp 推理的实测最低显存”。权重文件大小不能当作峰值显存；模型加载峰值、FP8 工作空间、激活和 CUDA 开销需要在支持的 GPU 上实际测量。此处不编造所需显存或推理耗时，也不关闭 FP8 来宣称官方 1B 已验证。

## 输入、隐藏层与评分 API

1B 配置有 25 层、隐藏宽度 1920、训练上下文及 `max_seqlen` 为 8192、配置最大 batch size 为 1，并启用 FP8 input projections。初始实现采用 256 bp，512 bp 应作为另一次真实集成测试。[固定官方配置](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/evo2/configs/evo2-1b-8k.yml)。

已核实的构造签名为 `Evo2(model_name, local_path=None, use_kernels=False)`，`local_path` 可加载已固定和校验的本地官方权重。官方前向接口支持 `return_embeddings=True` 和 `layer_names`；它通过 `model.get_submodule(name)` 注册 hook，对 tuple 输出取首个张量，并在 finally 中移除 hooks。`blocks.12` 是 25 层中的合法候选中间块；运行时仍需核对模块存在、返回张量维数及有限值。官方 7B 示例的 `blocks.28.mlp.l3` 不能直接用于 25 层 1B。[官方模型实现](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/evo2/models.py)。

Vortex 的 `CharLevelTokenizer(512)` 将 ASCII DNA 字符编码为各自字节，不自动添加 BOS。对于纯 A/C/G/T，长度 L 的序列对应 L 个 token，因此零基序列位置 k 对应隐藏状态 `[:, k, :]`。整块返回 `(hidden, inference_state)`；隐藏宽度保持配置值。由源码可推得候选 hidden 形状为 `[1, 256, 1920]`，位点向量为 `[1920]`，REF/ALT/delta 拼接为 `[5760]`。这些均为**待实测预期值**；本机实际 embedding shape 为 null。[Vortex tokenizer](https://github.com/zymrael/vortex/blob/main/vortex/model/tokenizer.py)、[Vortex 模型](https://github.com/zymrael/vortex/blob/main/vortex/model/model.py)。

Evo 2 是因果模型，位点 k 的隐藏状态已经包含输入碱基 k，可比较 REF/ALT；它没有看到该位置右侧的全部上下文。局部 pooling 可纳入下游隐藏状态的响应。若研究需要双向信息，应另行明确规定正向和反向互补提取，并在整个数据集一致应用，不能静默改变特征定义。

官方 `score_sequences(seqs, batch_size=1, prepend_bos=False, reduce_method='mean', average_reverse_complement=False)` 支持序列 log-likelihood，内部将 logits 与下一 token 对齐。REF/ALT 的分数差可作为标明身份的 scoring baseline；它本身不是致病概率。当前项目优先 embedding baseline，不以 scoring 掩盖 embedding 集成测试未执行。[官方评分实现](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/evo2/scoring.py)。

## 设计复核约束

- 数据缓存身份必须包括固定权重 revision/SHA、模型源码/依赖版本、FASTA 身份、GRCh38、窗口、层、pooling；逐变异保存真实序列或其摘要，并严格校验缓存 ID 和特征维数，拒绝混用缓存。
- 多基因注释用共享基因的连通分量分组，避免 A|B 记录使 A 与 B 分散到不同集合。缺失基因不能冒充独立 held-out 基因。重复冲突标签应先排除再划分。
- 均值/标准差、降维或 teacher projection 只能由训练集拟合。Teacher 的最佳 checkpoint 和校准温度仅使用验证集选择。Student 实验应固定相同 split，并验证 teacher/cache/data 的身份对应。
- 特征蒸馏的 teacher target 需要冻结；同时训练自由 teacher projection 与 student projection 可能塌缩。可直接用 student projection 对齐固定标准化 teacher 特征，训练用 projection 不进入 Student 推理依赖。
- 三个模型使用同一未参与训练/选择的测试集，区分模型初始化种子、split 种子及真实数据版本。阈值 0.5 的研究指标不构成临床判定。

本次审计没有模型输出、训练指标或临床有效性结论。模型加载和 Phase 0 真实推理状态仍为**未运行/受环境阻碍**。
