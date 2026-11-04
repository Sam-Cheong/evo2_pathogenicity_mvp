# 小型 StripedHyena 2 reference 架构审计

审计日期：2026-11-04。当前项目的 `stripedhyena2_tiny` 是**按官方算子公式实现的自定义小型 SH2 reference 变体**，使用随机初始化。它既不是官方预训练 SH2 checkpoint，也不是完整 Evo 2 的缩小 checkpoint；不使用 HyenaDNA 权重，更不能把 HyenaDNA 改名为 SH2。本次审计和数值检查均未调用 Evo 2 API、未训练模型、未下载模型权重。

## 固定来源及复核方法

官方来源是 [Zymrael/savanna](https://github.com/Zymrael/savanna)，固定 commit `7b819b8609ce0b29deff79358a294b59d8299d07`（通过 `git ls-remote ... refs/heads/main` 实际获取）。官方代码许可证为 Apache-2.0；下载副本保留原 LICENSE 和源文件。这里固定的是**参考源码**，不代表 NVIDIA 托管服务的内部源码或权重版本。

| 官方参考文件 | 本项目核查内容 |
|---|---|
| [hyena.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/hyena/hyena.py) | SE 门控、短卷积、MR/LI 门控及对角 skip、group 语义 |
| [explicit.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/hyena/parametrization/explicit.py) | MR 显式滤波器及固定指数衰减 |
| [implicit_modal.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/hyena/parametrization/implicit_modal.py) | LI 的 `ImplicitModalFilter.compute_filter` |
| [fftconv.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/ops/fftconv.py) | 可在 CPU 执行并反向传播的原始 `fftconv_func` |
| [block.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/block.py) | 三路投影、输入短卷积、interleaved split、残差及 prenorm |
| [norms.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/local/norms.py) | 原生 PyTorch RMSNorm 公式 |
| [base.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/local/base.py) | `ParallelGLU.forward_func` 的门控 FFN |
| [positional_embeddings.py](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/positional_embeddings.py) | RoPE `rotate_half` 和原始 apply 函数 |
| [40b_test_config.yml](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/configs/model/evo2/40b_test_config.yml) | 官方示例中可配置的算子排列、filter、norm 与长度；不据此声称等同实际 hosted 配置 |

`scripts/prepare_sh2_reference.py` 只下载这些小体积公开文件。每个文件都有固定 SHA256，下载后及每次加载前均校验；`work/sh2_reference/manifest.json` 保存 URL、commit、哈希、字节数和核验时间。已存在而哈希不符的文件会报错，不自动覆盖。

```powershell
python scripts/prepare_sh2_reference.py --download --self-check
python -m pytest tests/test_sh2_official_parity.py -q
```

有本地参考文件后，后续两条检查均可离线执行。第一条不带 `--download` 时强制来源齐全，缺失即失败。第二条单元测试允许未下载可选参考源码时 skip；哈希不符仍失败，不能把 skip 当作已验证。

## 算子与可执行参考

以下均采用因果卷积 `C_h(u)[t] = Σ_{τ=0}^{min(t,K-1)} h[τ] u[t-τ]`。`h[:,0]` 是 lag 0；PyTorch `F.conv1d` 执行互相关，因此比较时需 `left_pad(K-1)` 并翻转 `h`。FFT 实现使用至少 `2L` 的零填充，避免循环卷积混入未来位置。[官方 FFT](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/ops/fftconv.py) 的 `rfft(h)/size` 配合 `irfft(...,norm='forward')` 与本项目的默认 FFT 归一化等价；实际调用 `gelu=False`，不额外施加激活。

**输入与输出投影。** 每个 Hyena mixer 先对 hidden 执行无 bias `Linear(d,3d)`，再对全部 `3d` 通道执行因果 depthwise 短卷积，随后以 `(B,L,G,DG,3)` 的 interleaved 排列拆出 `x1,x2,v`。mixer 后接无 bias `Linear(d,d)`。自定义实现选择 `G=d,DG=1`；因此不涉及 `use_slow_heads` 分支的组内外积/收缩，也不冒充完整 40B 的分组设定。[官方 block](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/block.py)

**Hyena-SE。** 采用可学习短 FIR：`u=x2⊙v; y=x1⊙C_h(u)`。官方可分别开关 pregate/postgate，本项目固定双 gate。SE 的原始短卷积分支没有 MR/LI 的对角 `D*u` skip；本项目也不添加该项。输入 featurizer 短卷积和 SE mixer 的短卷积是两次独立操作。[官方 SE 与 depthwise forward](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/hyena/hyena.py)

**Hyena-MR。** `h_c[τ]=a_c[τ]exp(-r_c t_τ)`，其中 `a` 可训练，`t=linspace(0,1,K)`，`r=logspace(log_r_min,log_r_max,G)` 是固定 buffer。输出为 `y=x1⊙[C_h(x2⊙v)+D⊙x2⊙v]`，其中 `D` 可训练。strong/normal/weak preset 分别对应 `(0,2)/(-1,2)/(-2,2)`。固定衰减不是另一个任意可学习衰减网络。[官方 ExplicitSingleDecayFilter](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/hyena/parametrization/explicit.py)

**Hyena-LI。** 本项目选择官方 `implicit_modal` 分支，不能与同文件的 `implicit_real_modal`、`implicit_complex_modal` 或早期 HyenaDNA 的 sine-MLP 滤波器混称。`p,gamma,R` 形状均为 `(G,order)`，`t=arange(L,dtype=float32)[None,None,:]`，原始方法严格为：

```python
logp = -torch.exp(p.float())
glogp = logp * torch.exp(gamma.float())
h = torch.exp(glogp[..., None] * t)
h = torch.einsum('do,dot->dt', R.float(), h)[None]
```

即 `h_c[t]=Σ_j R_cj exp[-exp(p_cj)exp(gamma_cj)t]`。原始方法返回 `(h,None)`，`h` 为 `(1,G,L)`；项目移除首个 singleton 维度供 depthwise FFT 使用。等价极点为 `a_cj=exp[-exp(p_cj)exp(gamma_cj)]∈(0,1)`，所以也可用状态递推验证。LI 使用与 MR 相同的双 gate 和对角 skip。[官方 ImplicitModalFilter](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/hyena/parametrization/implicit_modal.py)

**Attention、FFN 与 norm。** Attention 使用 `softmax(causal_mask(Q_rope K_rope^T / sqrt(d_head))) V`，以 PyTorch SDPA 替代 CUDA FlashAttention；RoPE 使用 half-rotation、base `1,000,000`。FFN 为 `W3(SiLU(W1x)⊙W2x)`，这是官方可配置 GLU 家族中的 SiLU 选择。每块为 `x←x+mixer(RMSNorm(x))`，再 `x←x+FFN(RMSNorm(x))`，按本项目配置加 dropout。原生 Savanna PyTorch RMSNorm 为 `scale*x/(sqrt(mean(x²))+eps)`，eps 在平方根外；这与许多标准 RMSNorm 的 eps 位置不同，本项目显式保留前者用于数值对照。[官方 norm](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/local/norms.py)、[GLU](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/local/base.py)、[RoPE](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/savanna/model/operators/positional_embeddings.py)

## 相对官方大模型的明确差异

本项目选择 `d=192,FFN=384,6 heads`，四块 `[SE,MR,LI,causal_attention]`。完整 inference encoder 为 1,504,704 参数，加线性分类头共 1,505,281 参数；训练用投影头必须单独统计，不能随最终 Student 交付为必要推理依赖。

官方 `operator-config` 显式控制各层排列，不存在要求所有 SH2 都必须重复这四块的硬性顺序。官方 40B 测试配置有 50 层、宽度 8192、不同 attention 插入频率与不同 Hyena 分组；本项目的四块排列只是一个小型结构验证。[官方测试配置](https://github.com/Zymrael/savanna/blob/7b819b8609ce0b29deff79358a294b59d8299d07/configs/model/evo2/40b_test_config.yml)

当前 MR 使用 `K=64,strong` 衰减；上述官方 40B 测试配置使用 `K=128,weak`。本项目 MR/SE 使用常规随机滤波器初始化；官方 `ExplicitSingleDecayFilter` 默认 `small_init=True` 额外乘 `1e-5`。本项目 SE 默认 mixer 长度 3，官方部分示例为 7。它们都必须记录为自定义实验设置，不能写成逐参数复刻。

DNA 输入仅为四碱基 token，无 ASCII byte tokenizer、语言模型 LM head、special token 或官方预训练权重。因果中心 token 只能感知该位置及其左侧；256 bp 上下文的中心表示不利用右半侧碱基。REF/ALT 必须保持同一窗口和中心坐标，不能以未来突变改变中心表示。该选择的局限需在变异效应实验中保留。

## 实际验证范围

完整 Savanna 依赖 CUDA 初始化、TransformerEngine、模型/序列并行与分布式 runtime，不能把直接 `import savanna` 当作 Windows CPU 路径。这里跳过整包导入及硬件初始化，仅从 SHA256 已验证的源码 AST 提取原函数/方法；只去掉编译/JIT decorator，数学 body 不修改，传入单进程 CPU 状态。需要 `torch` 和 `einops`，不需要 CUDA、Mamba、Triton 或 Evo 2 权重。

已在 Windows、Python 3.11、PyTorch `2.7.1+cpu` 执行：

- `prepare_sh2_reference.py --self-check`：7 项原始函数对独立公式的数值/梯度检查通过；保存在 `work/sh2_reference/reference_checks.json`。包含短/长 FFT、SE 原门控+原短卷积、MR、LI、双 gate+skip、RMSNorm。LI 原 float32 方法对独立极点递推的输出最大差约 `1.19e-7`，梯度最大差约 `4.77e-7`。
- `pytest tests/test_sh2_official_parity.py -q`：**12 passed**。实际本项目模块对固定原始方法逐项比较，包括 FFT、SE、MR/LI 门控、MR、LI、RMSNorm、GLU、RoPE；检查前向数值及所有对应输入/参数的梯度。float32 值容差 `atol=1e-5,rtol=1e-4`，梯度 `atol=5e-5,rtol=3e-4`。

这些检查证明所测算子的数学及 autograd 一致性；不等于已复现官方完整 SH2、匹配其 checkpoint、验证训练效果或得到致病性预测性能。Attention 的 CUDA FlashAttention 完整模块未在 CPU 直接执行；已对 RoPE 原函数做数值/梯度对照，SDPA 使用明确因果数学定义。

## 两阶段表示蒸馏必须保持的边界

阶段一只允许既有 cache 的临床训练分区参与表示学习；随机种子固定后再拆内部 representation train/holdout，禁止从临床 val/test 读取特征、序列或标签选择 encoder。任何标准化/PCA/投影的拟合也仅用阶段一训练子集。阶段一不使用 ClinVar 标签或 Teacher 分类概率。当前实际 Teacher cache 元数据为 `layer=blocks.20.mlp.l3,pooling=variant,feature_mode=embedding,tokenizer=character_level_no_bos`，REF/ALT 各 8192 维；本轮是中心表示蒸馏，不是整段 mean pooling，也不能声称多层或全序列逐 token 蒸馏。Teacher 与 Student 均为因果中心表示，因此没有因 mean pooling 额外引入右半窗信息的不对称。

阶段二冻结 encoder 并设为 eval，训练分类头；冻结状态须通过 encoder state hash 前后相同与 `requires_grad=False` 验证。阶段二训练分区可使用真实标签，临床验证分区仅用于分类头选择/校准，最终测试分区只评价一次。随机初始化对照与表示蒸馏对照必须使用相同结构、划分、种子、分类头训练过程和评估集；Student 推理不调用 Teacher/API。当前源码来源审计不宣称任何训练阶段已经完成，实际结果以正式产物和独立训练审计为准。

只读复核已落盘的 `two_stage_data.py` 和 `two_stage_student.py`：表示数据由原始 1200 个 train 变异内部拆为 960/240，按 gene 与重叠上下文连通组划分；表示归档没有标签字段，Teacher NPZ 读取只遍历 outer-train。临床 400/400 val/test 保留于后续 probe。Stage 1 trainer 仅打开两个 representation 归档，标准化只拟合 representation-train；Stage 2 仅优化 head 并逐 epoch 检查 encoder frozen/eval/state hash，四个预定义 probe 完成且产物校验后才打开 clinical test。Linear/MLP 各自对照臂共享相同 head 初值和训练随机种子。以上是静态审查，不代替正式训练产物验收。`probes --resume` 会复用已完成臂，对中断臂从相同初值重训；不代表对中断 probe 优化器逐步续训。
