# NVIDIA 托管 Evo 2 API 审计

核查日期：2026-11-03（Asia/Shanghai）。先进行公开文档审计，随后在用户授权下完成协议探测和完整真实 REF/ALT Phase 0。**当前已确认托管 40B forward 接受 `nim_v1` / `blocks.20.mlp.l3`，完整 Phase 0 已通过；`nim_v2` 的 decoder 层被当前部署拒绝。** 以下保留先前文档不确定性的历史记录，并将实测结论单列。没有在本文或审计材料中保存密钥。

## 本次真实调用结果

两次请求使用同一个 `https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward` 和固定的 256bp DNA，仅改变层名：

| 请求层 | HTTP | 实际返回 |
|---|---|---|
| `decoder.layers.20.mlp.linear_fc2` | 422 | `StripedHyena has no attribute decoder`（服务错误中的属性名为 `decoder`） |
| `blocks.20.mlp.l3` | 200 | JSON 包含 `data` 和 `elapsed_ms`；Base64 解码为有效 NPZ |

成功 NPZ 的实际 key 为 **`blocks.20.mlp.l3.output`**，实际形状为 **`[1, 256, 8192]`**，源数组 dtype 为 **float64**。这次单次 HTTP forward 的 wall time 为 **5.424154 秒**，不是完整 REF/ALT、分类头或训练的计时。API 适配器可显式转为 float32 后池化和缓存。

对应请求 id：422 为 `7008f85e-d701-4329-8938-548e1ba8771f`；200 为 `b384b747-3502-4a91-8620-b2cd697393b3`。脱敏探测记录保存在本次工作材料 `work/user_nvidia_probe.json`；只含端点、层、状态、请求 id、耗时和数组元数据，不含 Authorization 或 key。

因此，本次 422 的已确认原因是 **层命名与实际 StripedHyena 部署不匹配**，不是本次 256bp 输入超过长度上限，也不是 `output_layers` 必须改为字符串。成功的同端点请求已经验证本次凭据可访问该端点；单独看到 422 本来不足以证明认证成功。

当前应使用 `configs/mvp.yaml` 的 `nim_v1` 配置；`configs/nvidia_v2.yaml` 只保留为另一部署协议的显式备选，不建议用于本次托管服务。两套协议不得混入同一缓存。上述两次探测本身只证明单次 forward；随后另行完成的完整 Phase 0 结果如下，仍不代表 Teacher 已训练完成。

## 完整真实 Phase 0 验收

[成功报告](../outputs/phase0_api_verified/result.json) 记录了固定 256bp 上下文、index128 的 A→G REF/ALT 测试，运行环境为 Windows、Python **3.11.9**、PyTorch **2.7.1+cpu**，本机 CUDA 不可用。

- `status=passed`、`real_model_verified=true`。
- 实测 token embedding 为 `[1,256,8192]`，REF/ALT/delta 各 `[8192]`，拼接特征 `[24576]`。
- 实测一对 REF/ALT 请求 wall time 为 **8.538849599994137 秒**，含网络与服务端时间；不是单独服务端算子耗时或分类器耗时。
- 重复特征最大绝对差为 **0.0**，`finite=true`、`repeated_run_consistent=true`、`save_reload_consistent=true`。
- 实际 `ref.npy`、`alt.npy`、`delta.npy`、`features.npy` 保存在 `outputs/phase0_api_verified/`。

报告中的远程 GPU 峰值显存和 `weights_frozen` 为 null，服务端硬件与参数状态没有被客户端直接检查。托管模型权重 revision 仍为 `not_exposed_by_provider`。这些验收证据支持真实远程特征推理已完成；尚未完成真实 Teacher 分类训练、Student 蒸馏或致病性预测性能评价。

## 模型身份

NVIDIA 目录中的 `arc/evo2-40b` 是 **40B**，不能标记为原项目的 `evo2_1b_base`。托管服务没有向本次公开资料提供可固定的权重 SHA/revision；即使推理成功，也应记录为 hosted、revision unknown/unversioned，不能沿用本地 1B 权重身份。NVIDIA 试用服务与模型/容器存在各自条款。[官方模型卡](https://build.nvidia.com/arc/evo2-40b/modelcard)。

## 公开文档与当前部署的冲突（含历史核查）

`https://docs.api.nvidia.com/nim/reference/arc-evo2-40b-forward` 本次直接请求返回 404。40B API reference 的 forward 导航实际指向 **7B forward** 页面。其内嵌 OpenAPI 明确给出的服务地址是 `https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward`，不能把该页面当成当前 40B forward 部署的直接证明。[官方 40B 页](https://docs.api.nvidia.com/nim/reference/arc-evo2-40b)、[实际 forward reference](https://docs.api.nvidia.com/nim/reference/arc-evo2-7b-infer)。

历史核查：`https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward` 出现在官方项目仓库的公开问题中；在真实调用之前，这只能支持历史端点身份，不能证明当时仍能推理。[ArcInstitute issue 95](https://github.com/ArcInstitute/evo2/issues/95)、[issue 93](https://github.com/ArcInstitute/evo2/issues/93)。本次随后取得上述 HTTP 200，已补足当前可调用性证据。

本次重新读取的 [官方 40B-forward build markdown](https://build.nvidia.com/arc/evo2-40b-forward.md) 的 Prototype 仍使用同一 40B forward URL，但其示例请求 `output_layer` 和 `decoder.layers.20.mlp`。页面标题为 40B，模型描述还出现 7B 字样，API reference 导航也指向 7B 页面。**不能据这些文档假定托管实例已经迁移到 Megatron decoder 命名。** 实测当前返回 StripedHyena 的 decoder 属性错误，使用 Vortex blocks 层成功。端点与 profile 必须显式记录，不能静默更换模型大小。

## 可实现的请求和响应 contract

官方 forward OpenAPI 的请求对象只允许两个字段；以下使用本次实际成功的层名：

```json
{"sequence": "ACGT", "output_layers": ["blocks.20.mlp.l3"]}
```

`sequence` 是非空字符串；`output_layers` 为 1–100 个非空字符串，`additionalProperties=false`。不要添加未经 schema 支持的 `model`、`temperature` 或 generation 参数。[官方 forward API](https://docs.api.nvidia.com/nim/reference/arc-evo2-7b-infer)。

成功 JSON 对象的必要字段是 `data`（Base64 字符串，解码后为 NPZ）和 `elapsed_ms`（服务端毫秒整数）。API 文档的官方 Python 示例也处理 `application/zip` 大响应；它只把响应保存为 ZIP，没有保证其中成员的文件名或目录布局。解析器应在内存中检查归档内容，拒绝 pickle/object dtype，要求所请求层名精确对应，记录实际 NPZ keys/shape/dtype，并检查有限值。[官方 Endpoints](https://docs.nvidia.com/nim/bionemo/evo2/endpoints)。

`data` 是 NPZ 本体，不是 embedding 的 JSON 数值数组。不要将 Base64 文本当作特征，也不要对 JSON 外层做任意 flatten。一个请求只发一条 DNA，REF 和 ALT 各调用一次；两次必须使用同一个 endpoint/profile/layer。

## 两套明确分开的层命名

| Profile | 官方文档支持的候选层 | 文档中的 logits 排布 | 适配策略 |
|---|---|---|---|
| `nim_v1` / Vortex | `blocks.20.mlp.l3`，最终 `unembed` | `[batch, length, 512]` | 本次中间层实测 `[1,256,8192]`、key 为 `blocks.20.mlp.l3.output`；最终 logits 尚需单独调用验证 |
| `nim_v2` / Megatron | `decoder.layers.20.mlp.linear_fc2`，最终 `output_layer` | `[length, batch, 512]` | 当前托管部署实测 HTTP 422；保留为不同部署的备选协议，不标记为本次通过 |

旧版还支持 `embedding_layer`，但它是静态碱基查表，不能用来宣称得到上下文 embedding。[NIM 1.0.2 官方层清单](https://docs.nvidia.com/nim/bionemo/evo2/1.0.2/endpoints.html)。新版 2.2.0 文档采用 decoder 层命名，40B 共 50 层，7B 共 32 层。[新版官方清单](https://docs.nvidia.com/nim/bionemo/evo2/endpoints)。

文档明确描述的是最终 logits 轴顺序；中间层 tensor 的轴、`.output` 后缀、tuple 的保存方式应靠实际返回验证，不能单凭 profile 假定成功。本次已实际观察到隐藏宽度 8192；此前仅由架构推断的状态已更新。其他层、窗口或接口变更仍须重新验证。

当前默认层依据本次成功调用选择 `nim_v1`。公开文档和托管部署仍存在不一致；出现 422 时应保存脱敏后的明确验证错误，显式选择协议并使用新缓存，不能把多个 profile 的结果混入同一缓存。缓存必须包含 endpoint、model identity、profile、层名、pooling、窗口、服务响应格式及真实观察的 embedding shape。

## 异步、大结果和超时

NVIDIA 官方 NVCF 结果查询端点为：

`GET https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{requestId}`

它支持认证请求；202 表示未完成，200 表示完成，302 使用 `Location` 获取大结果；`NVCF-POLL-SECONDS` 最大 300。[官方 polling reference](https://docs.api.nvidia.com/cloud-functions/reference/getfunctioninvocationresult)。

本次没有找到官方资料证明 health 网关的 `/v1/status/{requestId}` 对当前 Evo2 托管实例必然可用，不能将其描述为已经核实。可将 poll URL 模板设为配置项；若初始响应提供明确的可信轮询位置，优先保留并验证它。请求头中的 `nvcf-reqid` 可作为 request id；应记录 id、重试次数和实际 HTTP 状态，不记录 Authorization。

Base64 NPZ 与直接 ZIP 是不同的外层传输形式。302 的签名下载链接不应携带 NVIDIA Authorization 到第三方 CDN；按可信响应处理下载，同时限制大小、超时与归档解码。256 bp 请求无需预上传 assets；官方 forward schema 要求直接提供 `sequence`，不能擅自改用 asset id。没有文档依据时，不实现猜测的 `responseReference` 字段。

`elapsed_ms` 是服务端时间，HTTP wall time 另测。托管 GPU 型号、峰值显存与冻结状态不是客户端能直接检查的本地指标；未知值保持 null。成功返回隐藏状态能证明真实远程推理，不能证明本地成功加载权重。

## 可复核材料

`work/nvidia_evidence/` 保存官方 build markdown、文档 HTML、抽取的 `api7b_openapi.json` 以及下载 SHA 清单；`forward40b_current.md` 保存本次重新读取的 40B-forward 页面。内嵌 OpenAPI 的 `x-nvai-meta` 提供 JSON/Base64-NPZ 与 ZIP 处理示例。`work/user_nvidia_probe.json` 为本次真实调用的脱敏状态和张量元数据。上述审计材料没有保存 API key；真实模型特征与完整 Phase 0 报告由相应实验脚本另存，不能与文档示例混淆。
