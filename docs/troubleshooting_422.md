# 本次 HTTP 422 的已验证修复

2026-11-03，在用户 Python 3.11.9 / PyTorch 2.7.1+cpu 环境中进行了真实 API 检查。

当前 `https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward` 的服务实际使用 StripedHyena/Vortex 层命名。请求 `decoder.layers.20.mlp.linear_fc2` 时，服务端返回 HTTP 422，错误为 `StripedHyena has no attribute decoder`。当前网站的新 BioNeMo 示例与这个实际部署不一致。

使用项目中已验证的配置：

```powershell
python scripts/test_evo2.py --config configs/mvp.yaml --window 256 --output outputs/phase0_api
```

其实际层为 `blocks.20.mlp.l3`，profile 为 `nim_v1`。`configs/nvidia_v2.yaml` 仅保留作未来支持新协议服务的参考，不适用于本次确认的托管 40B 服务。没有自动回退或悄悄更换模型。

完整 REF/ALT Phase 0 已通过：token embedding `[1,256,8192]`，各序列向量 `[8192]`，拼接特征 `[24576]`；REF/ALT 请求合计 8.5388496 秒（含网络与服务端时间）。重复推理最大绝对差为0，数值有限，四个 `.npy` 文件保存及重载一致。完整报告在 `outputs/phase0_api_verified/result.json`。

代码同时补充安全的服务端 JSON 校验信息、request id 和明确修复提示；过滤输入回显、Authorization、key/token 等内容。Base64 解码兼容 ASCII 换行/空白，同时严格拒绝其他无效字符。API 模式的环境报告不再混入本地1B的CUDA/Transformer Engine阻碍。

用户 D 盘项目的已修改源文件有逐文件备份，目录为 `work/api_fix_20261103_backup`。`.env` 没有修改，API key 没有写入日志、报告或 ZIP。

本次通过的是远程 Evo 2 40B 的特征推理验证；Teacher 分类头和 Student 尚未完成真实训练，embedding 不能当作致病性分数。

