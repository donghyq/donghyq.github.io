---
layout: post
title: "主流大模型架构解析：Attention、GDN State、MLA、MoE 与推理缓存"
date: 2026-10-23 12:00:00 +0800
categories: [LLM Infra, Architecture]
tags: [llm, attention, gdn, mla, moe, kv-cache, qwen, glm, deepseek]
description: "从长上下文请求出发，拆解 Qwen、GLM、DeepSeek 的模型结构、GDN 循环状态、MLA 压缩、MoE 通信和推理缓存设计。"
math: true
toc: true
---

# 主流大模型架构解析：Attention、GDN State、MLA、MoE 与推理缓存

> 从 Qwen3.8、GLM-5.3、DeepSeek-V4.1 到 GPT-6、Claude 5.5，沿一条长上下文请求拆解模型内部的数据流、状态布局和推理成本。
>
>
> 2026 年 10 月 · 架构 / 推理系统 · 约 35–45 分钟阅读

## 目录

- [1. 一条 128K Agent 请求的计算路径](#s1)
- [2. Transformer 中三个独立的设计维度](#s2)
- [3. Attention：从 GQA 到 MLA](#s3)
- [4. GDN：固定大小的历史状态](#s4)
- [5. 稀疏注意力：Indexer 先于 Attention](#s5)
- [6. MoE：路由与 GPU 间通信](#s6)
- [7. Qwen3.8：GDN / Full Attention 混合层](#s7)
- [8. GLM-5.3：DSA、KDA 与 IndexPool](#s8)
- [9. DeepSeek-V4.1：CED、CSA2 与 Bounded Replay](#s9)
- [10. GPT-6 与 Claude 5.5：公开能力及结构参照](#s10)
- [11. 推理缓存演进：从 KV Pages 到多类型 State](#s11)
- [12. 未来架构与性能验证](#s12)
- [参考资料](#references)

---

<a id="s1"></a>
## 1. 一条 128K Agent 请求的计算路径

假设一个 Research Agent 已经与模型交互了多轮，历史上下文累计 128K Token。这次工具返回约 2K Token 的检索结果，模型需要继续生成 800 Token。一次典型执行可以拆成四段：恢复可复用的历史状态，Prefill 新输入，Decode 新输出，提交这次运行产生的可复用状态。

![图 1｜128K Agent 请求：从历史上下文到下一个 Token](/assets/img/posts/llm-architecture-2026/fig01-request.svg)

对于传统 Decoder-only Transformer，每一层都为已经处理过的 Token 保存 K/V。历史越长，Decode 每步需要读取的 KV 越多；如果上下文没有命中缓存，新一轮 Prefill 还要重新计算历史输入。128K 长上下文下，模型结构的变化直接改变了四个成本项：**算多少、读多少、存多少、搬多少**。

最新公开模型在不同位置做减法。GQA、MLA 减少单个 Token 的缓存；GDN/KDA 将一部分历史压缩成固定大小的循环状态；DSA/QSA/CSA2 缩小参与注意力计算的历史集合；DeepSeek 的 CED 进一步让多层共享历史 Global KV，减少 Prefill 必须穿越的计算层数。

这些方案并行存在。一款模型可以同时使用 MoE、压缩 KV、稀疏索引和跨层共享。理解模型架构，需要把它们放回同一条请求路径里。

<a id="s2"></a>
## 2. Transformer 中三个独立的设计维度

Transformer Block 的基础结构仍很稳定：归一化、Token Mixer（词元信息混合层）、残差连接、FFN（Feed-Forward Network，前馈网络）。现在最活跃的创新分布在三层：

1. **Token Mixer**：如何读取历史。Full/GQA、MLA、Sparse Attention、GDN/KDA 决定注意力计算和状态形态。
2. **Channel Mixer**：怎样分配计算。Dense FFN 或 MoE 决定每个 Token 激活哪些参数，以及是否需要跨 GPU 派发 Token。
3. **Residual / Auxiliary Memory**：信息沿层传播和外部参数查找的方式。mHC、Gated Residual、Engram/N-gram Embedding 改变残差通路或引入条件记忆。

![图 2｜一层 Transformer 的三个架构扩展点](/assets/img/posts/llm-architecture-2026/fig02-transformer.svg)

图中 Token Mixer 和 FFN 是每个 Block 的主要计算路径；Residual 负责把上下层的 Hidden States（隐藏状态）连接起来。这里的 *Hidden State* 指当前前向计算的激活；GDN 的 *Recurrent State* 则是跨 Token 保留的历史状态；Engram/N-gram Embedding 是持久的模型权重。这三种“状态/记忆”对应完全不同的内存生命周期。

| 机制 | 优化对象 | 保存形式 | 随上下文长度的增长 |
| --- | --- | --- | --- |
| GQA（分组查询注意力） | KV Head 数量 | 每 Token K/V | 线性 |
| MLA（多头潜在注意力） | 单 Token 表示维度 | 每 Token Latent KV | 线性、常数更小 |
| Sparse Attention（稀疏注意力） | 实际访问的历史位置 | KV + 索引/选择信息 | 依模型具体实现 |
| GDN / KDA（循环式 Delta 注意力） | 历史表示方式 | 矩阵 State + Conv 窗口 | 单个续算点近似固定 |
| MoE（混合专家） | 每 Token 激活的权重 | Expert Weights + Router | 权重不随请求长度增长 |
| Engram / N-gram | 高频模式的参数化查找 | 条件访问的权重表 | 参数存储与请求历史不同 |

下文先从数据量最容易计算的 KV Cache 开始，再逐步进入循环状态和跨层共享。

<a id="s3"></a>
## 3. Attention：从 GQA 到 MLA

### 3.1 KV Cache 的数据量

在标准 Causal Self-Attention（因果自注意力）中，当前 Query 要与历史每个 Key 计算相关性，再加权累加 Value：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

`M` 是因果掩码。自回归 Decode 时，一次只生成一个新 Token，已经处理过的 K/V 可以缓存，下一步读取它们即可。

对于普通多头或分组注意力，最基础的逻辑 KV 容量公式是：

$$B_{KV}=L\cdot N_{\mathrm{layers}}\cdot 2\cdot H_{KV}\cdot d_h\cdot b.$$

其中 `L` 为上下文长度，`H_KV` 为 KV Head 数，`d_h` 为 Head Dimension，`b` 为每元素字节数。这里尚未计入分页碎片、分配器元数据和计算工作区。

以一个**说明计算方法的配置**为例：32 层、8 个 KV Head、Head Dim 128、BF16、128K Token，逻辑 KV 占用为 **8 GiB**。改用 2 个 KV Head，在其余参数相同时容量为 2 GiB。模型参数及 Attention Head 划分必须一起看；仅凭总参数量无法推断 KV 大小。

![图 3｜MHA / GQA / MLA 的历史缓存布局](/assets/img/posts/llm-architecture-2026/fig03-kv-mla.svg)

### 3.2 GQA：减少 K/V 的组数

Multi-Head Attention（多头注意力，MHA）通常为各个 Query Head 分别生成 K/V。Grouped-Query Attention（分组查询注意力，GQA）让多个 Query Head 共享一组 K/V。Qwen3.8 的 Full Attention 层采用 64 个 Query Head、4 个 KV Head，因此每 16 个 Query Head 共享一组 K/V。每层每 Token 的逻辑缓存比对应的 64 KV Head 方案少 16 倍。

GQA 对现有 Paged Attention 和 KV Page 管理比较友好：分页单位仍对应一段 Token，变化主要落在每 Page 的张量维度。

### 3.3 MLA：保存低维 Latent 而非完整 K/V

Multi-head Latent Attention（多头潜在注意力，MLA）把 Hidden State 投影到低维的联合 KV Latent：

$$c_t^{KV}=W_{DKV}h_t,\qquad K_t=W_{UK}c_t^{KV},\quad V_t=W_{UV}c_t^{KV}.$$

缓存只需保留 `c_t^{KV}` 及位置编码所需的分量。DeepSeek-V3 的经典 MLA 配置中，KV Latent 为 512 维，额外的 RoPE Key 为 64 维，合计每 Token 每层 576 个元素；这也为后续 DeepSeek 和 GLM 的压缩注意力设计奠定了基础。

MLA 的执行性能还依赖 **Weight Absorption（权重吸收）**：

$$q_t^\top (W_{UK}c_j^{KV})=(W_{UK}^\top q_t)^\top c_j^{KV}.$$

推理时可预先合并部分线性变换，避免对所有历史 Token 逐一展开完整 Key 和 Value。RoPE 分支单独保留，保证位置相关的内积计算正确。它降低缓存读写字节数，也改变 Attention Kernel 的 Q/K 布局与矩阵乘法方式。

### 3.4 MLA 的状态组织

Page Table 中一条历史记录可能不再是 `[KV Heads, Head Dim] × K/V`，而变成 `[Latent Dim] + [RoPE Dim]`。对跨实例 KV 迁移系统，传输协议需要明确布局版本、量化格式、RoPE 变体、并行切分方式和模型版本。仅凭 Token Prefix 相同，无法判定两份 KV Tensor 的二进制内容可互换。

<a id="s4"></a>
## 4. GDN：固定大小的历史状态

### 4.1 从逐 Token 读取到 State 更新

传统 Softmax Attention 在 Decode 第 `t` 步要读取 `t` 个历史位置。Linear Attention（线性注意力）采用另一条路径：把历史 K/V 关联压进一个状态矩阵 `S`，当前 Query 通过矩阵乘法读取它。

给定单个 Head：`q_t, k_t ∈ R^{d_k}`，`v_t ∈ R^{d_v}`，定义：

$$S_t\in\mathbb{R}^{d_k\times d_v},\qquad S_t=S_{t-1}+k_tv_t^\top,\qquad y_t=S_t^\top q_t.$$

这就是最基础的关联状态。固定的 `S` 可总结任意长度前缀，但简单累加会让相似 Key 的历史关系互相叠加，缺乏精确覆盖和遗忘机制。

### 4.2 Delta Rule：用当前 Key 修正旧关联

DeltaNet 首先读取当前 Key 在旧状态上的预测 `Sᵀk`，再计算与当前 Value 的误差：

$$e_t=v_t-S_{t-1}^\top k_t,\qquad S_t=S_{t-1}+\beta_t k_t e_t^\top.$$

这里的 `β_t` 是写入强度。对同一个 Key 的更新由预测误差驱动，相当于让 State 按需修正已有映射。

Gated DeltaNet（门控 Delta 网络，GDN）再引入衰减门 `α_t`：

$$\begin{aligned}
\widetilde S_{t-1}&=\alpha_t S_{t-1},\\
e_t&=v_t-\widetilde S_{t-1}^{\top}k_t,\\
S_t&=\widetilde S_{t-1}+\beta_t k_t e_t^{\top},\\
y_t&=S_t^{\top}q_t.
\end{aligned}$$

`α_t` 控制旧状态保留比例；`β_t` 控制新关联写入强度。Qwen 的实现还对 Q/K/V 加入短因果卷积（Short Causal Conv）、Q/K L2 Normalization、Sigmoid 输出门和最终线性投影。GDN 的数学公式与 Qwen3.8-Flash-Next 报告第 2.1.1 节一致。

![图 4｜GDN 的 Read → Delta Update → Output 数据流](/assets/img/posts/llm-architecture-2026/fig04-gdn-flow.svg)

用简化伪代码表示单步 Decode：

```python
# 单个 GDN head；省略输入投影、短卷积、归一化和输出门
# S: [d_k, d_v]，沿请求生命周期持续保留

def gdn_step(q, k, v, S, alpha, beta):
    old_state = alpha * S
    old_value = old_state.T @ k
    error = v - old_value
    new_state = old_state + beta * outer(k, error)
    output = new_state.T @ q
    return output, new_state
```

这里有两层时间尺度。Decode 时，State 随每个 Token 串行演化；Prefill 时，框架会使用 Chunkwise / Fused Kernel（分块计算与融合内核），在块内批量处理 Token 并传递块间状态，避免纯 Python 循环逐 Token 调用 CUDA Kernel。Qwen 论文中的 FlashQLA 还专门优化了 GDN 的分块前后向计算。

### 4.3 实际 State 由两部分组成

GDN 的短因果卷积需要记录最近 `kernel_size − 1` 个输入；循环计算需要记录矩阵 `S`。这两块一起构成一次请求在该层的可续算状态。

![图 5｜GDN State 的张量排布：Recurrent Matrix + Conv Ring](/assets/img/posts/llm-architecture-2026/fig05-gdn-layout.svg)

使用 Qwen3.8-2.4T 的公开部署配置来计算一层状态（vLLM，BF16 State）：

- `SSM State`：`128 value heads × 128 value dim × 128 key dim × 2 B = 4 MiB`；
- `Conv State`：`(2×16×128 + 128×128) × (4−1) × 2 B = 120 KiB`；
- 每个 GDN 层约 **4 MiB + 120 KiB**，69 层合计约 **284 MiB / 请求**的逻辑状态。

它的主要特点是**单个续算点的状态大小不随历史 Token 数线性扩展**。不过它仍按请求、层和必要的恢复边界占用空间。如果 Prefix Cache 希望支持多个中间边界，每个边界可能需要自己的 State Checkpoint，容量管理会再次与缓存命中粒度联系起来。

这一点解释了混合模型为何需要两套容量预算：Full Attention KV 的消耗跟 Token 数相关，GDN State 的消耗则更接近 *请求数 × GDN 层数 × 每层状态字节数*。

### 4.4 GDN / KDA 的共享难点

传统 RadixAttention 可以按 Token Page 对公共前缀做复用。GDN 的状态代表“读完这个前缀后的整体历史摘要”。若在第 8192 Token 保存了 State，而下一次请求命中前 10000 Token，那么从 8192 恢复并重放后续 1808 Token 才能得到 10000 的准确状态。更长的前缀命中深度，也不保证存在对应的 State Checkpoint。

在分布式恢复中，至少需要 `(模型/权重版本, State ABI, Layer, 前缀边界, State Tensor, Conv Tensor)` 的一致性。投机解码还会引入已接受/已拒绝 Token 的回滚问题：当前请求只能把最终接受边界的状态提交到缓存。

<a id="s5"></a>
## 5. 稀疏注意力：Indexer 先于 Attention

MLA 在每个 Token 上节省空间，GDN 用有限状态吸收历史。Sparse Attention（稀疏注意力）则给查询选择一个历史子集：

$$\mathcal I_t=\operatorname{TopK}(\operatorname{Indexer}(q_t,K_{\mathrm{index}})),\qquad
 o_t=\operatorname{Attention}(q_t,K[\mathcal I_t],V[\mathcal I_t]).$$

实际开销必须计入 Indexer；减少核心 Attention FLOPs 后，Index Score、TopK、Gather 和非连续内存访问常常变成主要成本。

![图 6｜全量历史 → 轻量 Indexer → 稀疏选择 → Core Attention](/assets/img/posts/llm-architecture-2026/fig06-sparse.svg)

### 5.1 DSA 与 IndexShare

DeepSeek Sparse Attention（DeepSeek 稀疏注意力，DSA）引入专门的轻量 Indexer，在较低计算精度下挑选历史位置，再让主注意力读取被选中的 Key/Value。GLM-5.3 延续 MLA+DSA 结构，并通过 IndexShare 让一部分相邻层复用选中索引，减少重复计算的 Indexer 成本。

Indexer 的计算发生在另一个张量表示上；Indexer KV 与主 Attention KV 未必相同。对 Cache Manager 而言，这些索引信息有不同的规模、寿命和精度要求。

### 5.2 QSA：从 Token 级选择变成 Micro-block 级选择

Qwen3.8-Flash-Next 提出的 QSA（Qwen Sparse Attention，千问稀疏注意力）将相邻 Token 按 `r` 个划成一个 Micro-block。Indexer 先对该块的轻量 Key 做 Avg Pool，再选择 Top-K Blocks，最后展开对应 Token 做主注意力：

$$\hat k_b=\mathrm{RMSNorm}\left(\operatorname{AvgPool}(k_{br:\,(b+1)r})\right).$$

QSA 在计算块表示之后再施加位置编码，这避免了把不同旋转相位的 Token 表示直接混在一起。论文用 Dense Distillation（密集注意力蒸馏）初始化 Indexer，再切换到 Sparse Training（稀疏训练），让主干适应新的可见历史模式。

### 5.3 三种不同的共享方向

- **序列维压缩**：QSA 的 Micro-block、DeepSeek CSA2 的位置压缩，缩小待检索的历史项数。
- **层维共享**：GLM IndexShare、DeepSeek CSA2 的 Reuse/Reindex，让多个层复用索引或 Global KV。
- **维度与精度压缩**：MLA / FP8 / FP4，缩小每条历史项的字节数。

将这三维分别计入后，缓存成本可近似写为：

$$B_{global}\approx N_{\mathrm{stored\,tokens}}\cdot N_{\mathrm{KV\,producers}}\cdot d_{\mathrm{stored}}\cdot b\; +\; B_{index}+B_{metadata}.$$

这条公式能直接解释 DeepSeek 后续的压缩路线：历史项数、生产 KV 的层数、每项表示维度、量化字节数都在变化。

<a id="s6"></a>
## 6. MoE：路由与 GPU 间通信

Mixture of Experts（混合专家，MoE）让模型有很大的总参数量，但只让每个 Token 通过少数几个 Expert。一个通用的 Routed FFN 可以写成：

$$y_t=\sum_{e\in\operatorname{TopK}(r(h_t))}p_{t,e}\,\mathrm{FFN}_e(h_t)+\mathrm{FFN}_{shared}(h_t),$$

是否有 Shared Expert（共享专家）、Router 分数如何归一化，取决于具体模型。

![图 7｜MoE：Router、All-to-All Dispatch、Expert GEMM 与 Combine](/assets/img/posts/llm-architecture-2026/fig07-moe.svg)

在 Expert Parallelism（专家并行，EP）部署下，一层的执行包含五个阶段：Router 得分；Top-K 选择；按照 Expert 排列 Token；跨 GPU Dispatch；专家 GEMM；再 Combine 回原序。真正困难的场景发生在小 Batch 或负载不均时：专家收到的 Token 数不同，某些设备在排队等待；跨 GPU All-to-All 使端到端时间受到网络拓扑限制。

因此，`95B active`、`18B active` 描述的是典型每 Token 激活参数规模，并不能单独预测延迟。吞吐依赖 Token 分布、Expert Placement、EP/TP 划分、Network Bandwidth、Dispatch Kernel 和模型使用的权重量化格式。

<a id="s7"></a>
## 7. Qwen3.8：GDN / Full Attention 混合层

Qwen3.8-2.4T-A95B 的语言主干有 92 层，其层级结构是 `23 × [3 × (GDN → MoE) + 1 × (Gated Full Attention → MoE)]`。也就是 69 层 GDN、23 层全注意力。Full Attention 保留长程 Token 级随机访问，GDN 以固定循环状态承担大量连续上下文建模。

![图 8｜Qwen3.8：92 层、23 组 3:1 Hybrid Attention](/assets/img/posts/llm-architecture-2026/fig08-qwen.svg)

| 组件 | Qwen3.8-2.4T-A95B |
| --- | --- |
| 总参数 / 激活参数 | 2.4T / 95B |
| 主干 | 92 层，69 GDN + 23 Gated Attention |
| Full Attention | 64 Q Heads / 4 KV Heads / Head Dim 256 |
| GDN | 16 QK Heads / 128 V Heads / Head Dim 128 |
| MoE | 每层 512 Routed Experts，Top-10 + 1 Shared |
| 上下文 | 原生 262,144 Token，可扩展约 1.01M |
| 其他 | Multi-Token Prediction（多 Token 预测，MTP） |

一个 Qwen3.8 的 KV 容量可以分解为：

$$B_{request}(L)=B_{GDN\,checkpoint}+L\cdot (23\cdot B_{KV/token/layer}).$$

在 vLLM 报告的一套配置中，Full Attention 的 KV 使用 FP8，单 Token、单层为 `4 KV Heads × 256 Dim × 2(K/V) × 1 byte = 2 KiB`；而 GDN State 是 BF16。此时 69 层 GDN 的固定状态约 284 MiB，23 层 Full Attention 每增加 1K Token 增加约 46 MiB 逻辑 KV。不同精度和多 GPU 分片会改变每张卡的实际占用。

在以吞吐为目标的 PD（Prefill/Decode，预填充与解码）分离中，调度器需要同时估算每请求的 GDN State 和随 Token 增长的 Full Attention KV。尤其在短上下文、高并发场景，固定 State 的容量约束可能先于 KV Token Pool 触发。

### 7.1 Qwen3.8-Flash-Next：从 3:1 Hybrid 继续演进

Qwen3.8-Flash-Next 的研究原型沿用了 GDN 混合骨架，但让部分全注意力层改为 QSA，并增加四分支 Gated Residual（门控残差）。Gated Residual 为每个残差分支引入读取门，稳定深层网络训练；研究中还引入额外约 51B 参数的 N-gram Embedding，将高频词组关联查表从通用矩阵计算中分离。该嵌入表可放在 Host Memory，通过异步加载供计算使用。

![图 9｜Qwen3.8-Flash-Next：QSA、四路残差与 Host Embedding](/assets/img/posts/llm-architecture-2026/fig09-qwen-next.svg)

从成本上看，QSA 进一步减少全局注意力的 Indexer 负担，Gated Residual 改善优化稳定性，而大型 N-gram 表引入了另一种需要管理的非活跃权重访问方式。它们分别落在 Attention、Residual 和参数存储三条路径上。

<a id="s8"></a>
## 8. GLM-5.3：DSA、KDA 与 IndexPool

GLM-5.3 和 GLM-5.3-Flash 采用两条不同的架构路径。GLM-5.3 继续沿用 GLM-5.2 的大型 MLA+DSA 稀疏注意力基座，模型共有 744B 总参数、约 40B 激活参数、78 层 Decoder，最新提升主要来自后训练。GLM-5.3-Flash 则以 320B 总参数、18B 激活参数重新设计主干。

### 8.1 GLM-5.3：IndexShare 的层间索引复用

GLM-5.3 仍需要在大量层上执行长上下文全局相关性选择。IndexShare 允许部分层复用之前算好的选择位置：只有一部分 Indexer 产生新 Top-K，其余层读取共享结果。MLA 控制每个 Token 的缓存表示，DSA 控制访问哪些位置，IndexShare 控制需要计算几次索引。

### 8.2 GLM-5.3-Flash：34 KDA + 11 Sparse MLA

GLM-5.3-Flash 共 45 层：前 44 层采用 `[KDA,KDA,KDA,Sparse MLA] × 11`，末尾再接一层 KDA。第一到第三层是 Dense FFN，后 42 层采用 MoE。KDA（Kimi Delta Attention，Kimi 增量注意力）属于 Delta Rule 循环注意力家族，允许更细粒度的状态遗忘机制。Sparse MLA 的 Indexer 则借助 IndexPool 将四个历史索引 Key 压为一个汇总项，减轻 1M 上下文的检索成本。

![图 10｜GLM-5.3-Flash：KDA / Sparse MLA、IndexPool 与两套 Pool](/assets/img/posts/llm-architecture-2026/fig10-glm.svg)

| 组件 | GLM-5.3-Flash |
| --- | --- |
| 主干 / 多模态 | 45 文本层 + 24 层 Vision Encoder |
| Attention | 34 KDA + 11 Sparse MLA / DSA |
| KDA | 64 Heads × 128 Dim，Conv Kernel 4 |
| Sparse MLA | KV Latent 512、NoPE、Top-2048 |
| IndexPool | 4 个 Indexer Key 聚合为 1 个 |
| Experts | 288 Routed，Top-8 + 1 Shared；前 3 层 Dense |
| Context | 1,048,576 Token |

GLM-5.3-Flash 还应用 mHC（Manifold-Constrained Hyper-Connections，流形约束超连接），把传统单路残差拓展到多路可学习连接，并约束混合矩阵以提升训练稳定性。

### 8.3 部署中的容量约束

SGLang 官方部署文档将 GLM-5.3-Flash 的缓存拆为 **Paged KV Pool** 和 **KDA State Pool**。如果 KDA State Pool 不够，即使 KV Pool 还有空闲，也会降低 `max_running_requests`。KDA State 与 Sparse KV 的精度、生命周期和复用粒度不同，因此容量调优应分别观察两个池的使用率。

GLM 团队还公开了 ReplaySSM（SSM 状态重放）、混合精度缓存、Layer Split（按层拆分部署）和 EPD（Encode–Prefill–Decode，编码/预填充/解码分离）等部署手段。对有图片、视频和文本混合输入的场景，Vision Encoder 的显存峰值和调度开销可以单独隔离，避免干扰 Decode。

<a id="s9"></a>
## 9. DeepSeek-V4.1：CED、CSA2 与 Bounded Replay

DeepSeek 的设计路线从 DeepSeek-V3 的 MLA、DeepSeek-V3.2 的 DSA，进一步演进到 V4 系列的压缩稀疏注意力。2026 年 9 月的 DeepSeek-V4.1-Flash 将优化点推进到了**跨层共享和 Prefill/Decode 非对称计算**。

### 9.1 CED：上半段不再完整 Prefill

DeepSeek-V4.1-Flash 的语言主干共有 40 层：20 层 Causal Encoder（因果编码器）和 20 层 Decoder。对于 Decoder 的 Global KV，投影的输入来自 Encoder 最后一层 Hidden State，而非 Decoder 每层前向计算时的 Hidden State。

$$C_l=H_{20}W^{KV}_l,\qquad Z_l=H_{20}W^Z_l,\qquad l>20.$$

因此，对于长输入 Prefill，可以让计算主要经过 Encoder 20 层，然后从其最终表示生成上层 Global KV；Decoder 不必为了长提示词逐 Token 完整计算 20 层。Decoder 自己的 Sliding-Window Attention（滑动窗口注意力，SWA）仍然需要准备状态，后文的 Bounded Replay 负责这部分工作。

![图 11｜DeepSeek-V4.1 CED：20 层 Encoder / 20 层 Decoder](/assets/img/posts/llm-architecture-2026/fig11-ced.svg)

按照公开技术报告的模型激活口径，Prefill 激活约 8B 参数，Decode 约 16B，总主干参数 552B。这种输入与输出计算量的不对称适合多工具返回、长输入和相对短输出的 Agent 工作负载。

### 9.2 CSA2：Full、Reindex、Reuse 三种层模式

CSA2（Compressed Sparse Attention 2，压缩稀疏注意力第二版）把 Global KV、Indexer Key 和 Top-K Selection 视作可以分别复用的对象：

- **Full**：本层新生成 Main KV、Indexer K，并计算新 Top-K。
- **Reindex**：复用 Main KV / Indexer K；本层重新计算 Indexer Q 和 Top-K。
- **Reuse**：复用 Main KV、Indexer K 和上一层可用的 Top-K；直接进行 Sparse Attention。

三种模式都会生成自己的 Query 和本层 SWA KV。它们在 Global Cache 和索引路径上的共享程度不同。

![图 12｜CSA2：Full / Reindex / Reuse 的跨层共享](/assets/img/posts/llm-architecture-2026/fig12-csa2.svg)

DeepSeek 还加入 Hierarchical Sparse Indexer（分层稀疏索引器）。Decoder 中较早的 Full 层先为当前查询建立一个候选池，后续 Reindex 层只在该池内重新评分。例如先选 2048 个 8-Token Block，就形成 16,384 个候选位置，再由各层挑选最终 Top-512。这样后续层的 Indexer 不需要每次扫描整个 1M 上下文。

### 9.3 FP4 Cache：进一步压缩每条历史项

DeepSeek-V4.1-Flash 在训练中加入 FP4 Main KV 的量化感知优化。Main KV 采用 E2M1 加按组 Scale 的格式，而 SWA KV 仍使用 FP8。不同缓存选择不同精度，是在存储密度与模型对误差敏感度之间做出的分工。

其 **Global KV Cache 约为 890 Bytes / Token**，约为 DeepSeek-V4-Flash 的四分之一；公开报告同时给出 HBM Global KV 与持久化 KV 两个不同的容量口径。对 128K Token，`890 × 131072 ≈ 111 MiB` 只是 Global KV 理论字节量，不包含 SWA、索引工作区、Page Padding、并发请求等其他内存。

### 9.4 Bounded Replay：将低频状态改为按需重放

DeepSeek-V4.1-Flash 将 Global KV 与 SWA KV 分层管理：Global KV 在持久化缓存中保留较长时间，短生命周期的 SWA KV 放在 Host DRAM Pool；SWA 命中失败时，只重放最近一个有界窗口。Encoder SWA Replay 读取缓存的 Global KV，恢复缺失的局部状态；Decoder SWA Replay 为即将开始的 Decode 准备局部窗口。

![图 13｜Global KV 长驻、SWA 短驻与 Bounded Replay](/assets/img/posts/llm-architecture-2026/fig13-replay.svg)

完整重建深层 SWA 状态可能涉及多层窗口递归展开。Bounded Replay 用最近 `n_win` 个 Token 的有限重放截断这种依赖，属于模型验证过的近似恢复策略。由于可以不在长时间持久缓存中保存大量 SWA KV，V4.1 的持久化缓存占用约为 V4 的 **1/8**。这是模型结构、精度与缓存策略共同产生的效果。

### 9.5 辅助结构：Engram、mHC、DSpark

DeepSeek-V4.1-Flash 同时采用 Engram 条件记忆（约 196B 查表参数）、Single-Pass mHC，以及 DSpark Speculative Decoding（投机解码）。Engram 的参数在请求间共享，与每条请求独立的 KV/SSM State 生命周期不同；mHC 改变残差信息路由；DSpark 通过半自回归草稿与验证提升多 Token 生成效率。三者分别作用在参数读、Hidden State 流和输出步进上。

<a id="s10"></a>
## 10. GPT-6 与 Claude 5.5：公开能力及结构参照

GPT-6 与 Claude 5.5 是当前主流前沿闭源模型系列，但官方材料侧重系统能力、推理训练、Agent 使用、评测和成本，并未提供可用于重建网络的层数、Expert Layout、具体 Attention 变种或 KV State ABI。这一部分使用已经开放的同厂模型和推理接口作为技术参照。

### 10.1 OpenAI：GPT-6 与 gpt-oss 的结构窗口

GPT-6 已在 ChatGPT 产品中引入更强的推理和动态界面生成能力。OpenAI 对公开结构讲解较充分的模型是 `gpt-oss-120b`：**36 层**、约 **116.8B 总参数 / 5.1B 激活参数**，每层 MoE 有 128 个 Experts、每 Token Top-4，Attention 交替采用 Full 和宽度 128 的 Local Window，并采用 GQA、RoPE 与 MXFP4 专家权重量化。

![图 14｜公开结构与闭源模型：gpt-oss / GPT-6 / Claude 5.5](/assets/img/posts/llm-architecture-2026/fig14-closed.svg)

这一公开模型很适合作为 MoE/GQA 的可复现实验参照。对 GPT-6 的具体内部 Attention/Expert 机制，当前更有价值的研究材料是它的 System Card、推理能力与工具/界面协同方式；这些信息主要反映模型输出和系统接口。

### 10.2 Anthropic：Claude 5.5 的产品执行路径

Claude Opus 5.5、Sonnet 5.5、Haiku 5.5 在成本、速度和任务深度上分别定位不同工作负载。相关公开资料更适合从长程 Coding Agent、Prompt Cache（提示词缓存）、工具调用、Subagent（子代理）调度及上下文管理切入。

从推理 Infra 的角度，可以分析其对请求形状的影响：反复出现的系统提示词、长历史工具结果、短增量回合、长输出代码生成与大量并发子任务都会改变 Prefill / Decode 比例和缓存命中率。内部网络尚无与 Qwen/GLM/DeepSeek 开放权重配置同等级的公开描述，因此不将这些系统行为写成特定 Attention 架构。

<a id="s11"></a>
## 11. 推理缓存演进：从 KV Pages 到多类型 State

到这里，把三家开源模型放回最开始的 128K Agent 请求会发现：Runtime（推理执行环境）管理的可复用状态已经远远超出一组 KV Tensor。

![图 15｜多类型推理 State：生命周期、存储层与 P/D Transfer](/assets/img/posts/llm-architecture-2026/fig15-runtime.svg)

建议从数据结构上把状态类型区分开：

| State 类型 | 生产阶段 | 常用访问粒度 | 典型复用策略 |
| --- | --- | --- | --- |
| Full/GQA KV Pages | 每个 Transformer Attention 层 | Token Page / Prefix | 复用、分层迁移 |
| MLA Latent KV | MLA / Sparse Layer | 压缩 Token Page | 量化、跨节点搬运 |
| GDN/KDA Recurrent State | 线性注意力层 | 特定 Token 边界 / Request | Checkpoint、重放 |
| Conv State | GDN/KDA 短卷积 | 最近窗口 / Request | 随 Recurrent State 一起提交 |
| Sparse Indexer KV / Top-K | Indexer / Sparse Attention | Token/Block/跨层 | 索引复用、按需重算 |
| SWA State | 局部注意力层 | 最近窗口 | 短驻、Bounded Replay |
| Engram / N-gram 表 | 模型初始化 | 权重行 / Hash Key | Host / HBM 参数缓存 |

### 11.1 一份可复用的状态记录应包含什么

对混合架构，可将 Manifest（状态清单）抽象为：

```python
@dataclass
class StateManifest:
    model_revision: str
    tokenizer_revision: str
    kv_abi: str                 # dtype / layout / head sharding / quant scales
    request_generation: int     # 防止旧写覆盖新代状态
    prefix_token_hash: str
    committed_token_pos: int    # 此边界对应的全部 State 都已提交
    tensor_refs: dict[str, list[TensorRef]]
    # keys: full_kv / latent_kv / gdn_ssm / conv / sparse_index / swa
```

一次跨节点恢复至少经历：匹配前缀、选取可用的 State 边界、检查模型与 State ABI、预留目标内存池、并行传输、跨层与多 Rank 校验、原子提交或回退重算。前缀命中率高不代表恢复一定快；当 State 较小或重放窗口有限时，Recompute（重算）可能更合算。

### 11.2 Restore vs Recompute 的成本分解

将一次恢复的关键路径写为：

$$T_{restore}\approx T_{lookup}+T_{queue}+T_{network}+T_{H2D}+T_{install}+T_{sync}.$$

重算路径则为：

$$T_{recompute}\approx T_{schedule}+T_{prefill}(L_{missing},\mathrm{architecture}).$$

最后一项对不同模型的含义发生明显变化。Full Attention 的重新 Prefill 可能读取更多历史；GDN 可以从最近 State Checkpoint 重放一段增量；DeepSeek 的 Bounded Replay 有意把局部状态重建限制在小窗口。恢复价值需要按照具体 State 类型和业务 SLO 测量。

### 11.3 三个最值得测量的系统指标

一是 **State Bytes / Request**，按 KV、SSM、Conv、Indexer、SWA 分类，而非只看“可缓存 Token 数”。二是 **TTFT（Time to First Token，首 Token 延迟）**，分解 Queue、Miss Recompute、Transfer、Install。三是 **Goodput（满足 SLO 的有效吞吐）**，在指定输入/输出长度与并发下评估实际收益。

对于 Agent 工作负载，还应加入多轮前缀命中率、Tool Gap（工具等待间隔）分布、State Checkpoint 间距、恢复失败回退率与一致性测试。只有一起观察，才能判断更大的缓存或更激进的预取是否真正改善端到端效果。

<a id="s12"></a>
## 12. 未来架构与性能验证

这几代模型有一个明确趋势：**Attention 的存储优化从 Head 维度进入序列维、层维和状态生命周期。**

![图 16｜缓存压缩的四个轴：Head × Token × Layer × Precision](/assets/img/posts/llm-architecture-2026/fig16-axes.svg)

在模型侧，Qwen 将固定 GDN State 与稀疏全局访问混合；GLM 将 KDA 与 IndexPool 结合，并把部署推进到 EPD；DeepSeek 将缓存重用扩展到 Decoder 层之间，同时用 Bounded Replay 缩小必须持久保存的局部状态。这些架构分别推动不同的计算/内存边界。

在 Runtime 侧，未来更可能出现按照状态类型组织的缓存管理器：Full KV、Compressed KV、Recurrent State、Sparse Index 和 Short-window State 拥有不同的访问成本与生命周期。Cache Manager 需要理解模型如何从一次正确的 Checkpoint 继续执行，而不仅管理统一尺寸的 KV Page。

真正比较几种架构时，最好固定业务输入/输出画像与服务质量约束，至少分成三组实验：`短输入高并发`、`长输入短输出`、`多轮 Agent + 工具间隔`。每组同时比较 TTFT、ITL（Inter-Token Latency，Token 间延迟）、Goodput、HBM State、Host/SSD Byte 和 GPU Kernel 级消耗。对 GDN 增加 State/Conv Pool 的占用曲线，对 Sparse Attention 增加 Indexer + Gather 时延，对 MoE 增加 Dispatch / Combine 和 Expert Load 分布。

模型架构与推理 Infra 的界面正在变得更紧密。理解一个架构，不妨沿着这条链路检查：**它怎样表示历史、哪些层生产状态、谁消费状态、状态在哪个设备上、怎样迁移、失效后怎样恢复。** 这些问题直接决定它能否高效支撑百万上下文和长程 Agent。

---

<a id="references"></a>
## 参考资料

1. [Qwen3.8 官方 Model Card 与结构配置](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)；[Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8)。
2. Qiu et al., [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320), 2026。特别参考 Figure 2、Figure 3、§2.2、§2.3。
3. vLLM Team, [PD Serving of Qwen3.8-2.4T](https://vllm-project.github.io/2026/09/21/qwen38-pd-serving.html), 2026。
4. Z.ai, [GLM-5.3: Frontier Coding with Emergent Cyber Capabilities](https://z.ai/blog/glm-5.3), 2026；[GLM-5.3-Flash: Frontier Intelligence, Flash Cost](https://z.ai/blog/glm-5.3-flash), 2026。
5. [GLM-5.3-Flash 官方 config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)；[SGLang 官方部署说明](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)。
6. DeepSeek-AI, [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969), 2026。重点为 §2.2 CED、§2.3 CSA2、§3.2 Cache Management / Bounded Replay。
7. [DeepSeek-V4.1-Flash 官方 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。
8. OpenAI, [gpt-oss Model Card](https://deploymentsafety.openai.com/gpt-oss)；[GPT-6 and Intelligent UI for everyone](https://openai.com/index/gpt-6-for-everyone/), 2026。
9. Anthropic, [Introducing Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5)；[Claude Models](https://www.anthropic.com/news/models), 2026。
10. [SGLang HiCache 解析](https://donghyq.github.io/posts/sglang-hicache/)：本文技术博客组织方式的参考。

*全部结构图根据上述技术报告和公开配置重新绘制；除特别标注外，公式中的容量推导为逻辑张量规模，部署口径以引用报告说明为准。*