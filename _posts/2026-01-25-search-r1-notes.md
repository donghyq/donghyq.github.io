---
title: "Search-R1：让 LLM 学会边思考边搜索"
date: 2026-01-25 10:00:00 +0800
categories: [LLM Infra, RL]
tags: [search-r1, grpo, ppo, verl, rag, rl-infra, state-masking, fsdp]
---

## 一、项目概述

Search-R1 是一个强化学习训练框架，目标是让 LLM 在推理过程中**自主决定何时搜索、搜什么**。

与传统 RAG 的本质区别在于：传统 RAG 的检索逻辑由系统硬编码，模型是被动接收方；Search-R1 里，**模型通过 RL（PPO / GRPO / REINFORCE）自主涌现出搜索策略**——何时搜、搜什么、要不要再搜一轮，全部由模型在试错中习得，无需人工标注搜索轨迹。这个主动性是 RL 训练出来的，不是 Prompt 工程堆出来的。

完整推理链路：模型收到问题后开始推理，遇到知识盲区时吐出 `<search>query</search>`；系统捕获标签、调用搜索引擎、将结果注入上下文；模型继续推理，可多轮搜索，最终输出 `<answer>最终答案</answer>`。RL 用规则奖励（EM / SubEM）对整条轨迹打分，反向优化模型的搜索时机和搜索质量。

## 二、整体架构

整体架构由四个核心模块构成，通过统一的 HTTP 接口（`POST /retrieve`）解耦连接：LLM 推理引擎负责生成轨迹，搜索引擎后端提供检索服务，RL 训练引擎驱动参数更新，Ray WorkerGroup 提供分布式 GPU 调度。

各模块职责：

- **LLM 推理引擎（Actor）**：负责生成推理轨迹和搜索调用
- **搜索引擎后端**（FastAPI `/retrieve`）：支持 BM25、Dense Retriever（FAISS+E5/DPR）和在线搜索（Google/Bing）三种后端
- **RL 训练引擎**（veRL + Ray）：负责 PPO/GRPO 训练、奖励计算和 State Masking
- **Ray WorkerGroup**：提供分布式 GPU 支持，包含 ActorRollout、Critic、RefPolicy 三类 WorkerGroup

## 三、RAG 演进脉络与 Search-R1 的定位

从 Naive RAG 到 Search-R1，RAG 技术经历了四代演进，核心矛盾始终是"检索策略由谁决定"。

四代演进路径：

- **Naive RAG**（2020-2021）：单次检索，无法处理需要多步推理的复杂问题
- **Multi-hop RAG**（2021-2022）：支持多次检索，但检索逻辑由人工规则硬编码，灵活性差
- **Agentic RAG**（2023-2024）：用 Prompt/SFT 教模型调工具，摆脱了硬编码，但能力上限受限于 Prompt 工程
- **RL-based RAG**（2025）：用 RL 训练模型自主习得搜索策略，能力从奖励信号中涌现，Search-R1 属于这一代

这里有两个容易混淆的概念需要区分。**多跳检索（Multi-hop Retrieval）**是信息检索领域的术语，指回答一个问题需要跨多个文档"跳转"才能拼出完整答案，HotpotQA、2WikiMultihopQA 等数据集的核心挑战就在于此。**多轮搜索（Multi-turn Search）**是训练框架的概念，指 LLM 在一次回答过程中可以多次调用搜索引擎（对应代码里的 `max_turns=2`）。前者是问题的性质，后者是解决问题的手段——Search-R1 让模型自主决定要不要再搜一轮，而不是由系统硬编码检索次数。

## 四、PPO 训练完整链路

每个训练 Step 由 Driver 进程（CPU）协调，通过 Ray 分发到多个 GPU Worker 并行执行，共 7 个步骤：

1. **ActorRollout_WG 多轮搜索生成**：问题 batch 分发给多张 GPU（例如默认配置 512 个问题 / 8 张 GPU），每张 GPU 跑 vLLM 推理，遇到 `</search>` 截断后调检索服务，拼接结果继续生成。训练阶段截断通过 vLLM 的 `stop_token_ids` 参数在 sampling 层面实现，而非 HuggingFace 的 `StoppingCriteria` 接口
2. **重算 log prob**：对完整轨迹重算 log prob（多轮分段生成后需重算）
3. **RefPolicy_WG 计算参考 log prob**：计算冻结初始策略的 log prob，用于 KL 惩罚
4. **Critic_WG 估值**：估计每个 token 的状态价值 V(s)
5. **Driver 本地计算奖励与优势**：计算 EM 奖励、KL 惩罚和 GAE 优势函数
6. **Critic_WG 更新价值网络**：更新价值网络参数
7. **ActorRollout_WG 更新策略网络**：用 PPO clip 目标函数更新策略网络参数，State Masking 屏蔽搜索结果 token 的梯度

其中 Step 3（RefPolicy）和 Step 4（Critic）可以并行执行，因为它们是不同的 WorkerGroup，互不依赖。Step 6（Critic 更新）和 Step 7（Actor 更新）在 colocated 部署下是串行的，因为两者共享同一批 GPU，需要交替占用显存。FSDP 在反向传播阶段使用 Reduce-Scatter 同步梯度（先 Reduce 求和，再 Scatter 切片分发），而非 DDP 的 AllReduce。

## 五、核心代码解析

### 5.1 多轮搜索 Agent（generation.py）

`LLMGenerationManager` 是整个项目最重要的新增代码，控制多轮交互循环。核心方法 `run_llm_loop` 最多跑 `max_turns` 轮，每轮依次完成：

- 调用 vLLM 生成（遇到 `</search>` 或 `</answer>` 截断）
- 解析标签，决定调搜索还是结束
- 批量调检索服务（`batch_search`）
- 拼接结果回上下文，继续下一轮
- 同步维护 `info_mask`（搜索结果部分标记为 0）
- 所有请求完成后调用 `_compose_final_output()` 生成 `info_mask` tensor

### 5.2 State Masking（info_mask）

Search-R1 的训练轨迹里混杂着两种来源的 token：模型自己生成的（思考过程、搜索 query、最终答案），以及外部检索服务返回的（搜索结果文档片段）。如果对整条轨迹不加区分地计算 loss，模型会发现"把搜索结果原文复制进 `<answer>` 里"是得高分的捷径——检索结果本身就包含正确答案，直接抄比真正推理容易得多。这种行为叫 **reward hacking**，模型学到的是复制能力，不是推理能力。

标准 RLHF（如 InstructGPT）的轨迹全部是模型自己生成的，不存在"外部注入内容"的问题（prompt 部分通常用 attention mask 屏蔽，不计入 loss），所以不需要额外的 mask 机制。Search-R1 引入了工具调用，外部返回的内容不应被当作模型的"决策"来训练，这是 State Masking 存在的根本原因。

`info_mask` 是一个与完整轨迹 token 序列等长的 0/1 tensor：模型自己生成的 token 置 1（参与梯度），外部检索结果的 token 置 0（屏蔽梯度）。一条典型的多轮搜索轨迹对应的 mask 如下：

```
[问题 token] [think token × N] [search query token] [搜索结果 token × M] [think token × N] [answer token]
[    1  ×  ] [     1    × N ] [       1      ×   ] [      0      × M ] [     1    × N ] [    1   ×  ]
```

`info_mask` 的构造在 `generation.py` 的 `_compose_final_output()` 中完成。每轮搜索循环里，`run_llm_loop` 维护一个 `info_mask` 列表：LLM 生成阶段追加全 1 的 mask，拼接搜索结果时追加全 0 的 mask（长度等于检索文档的 token 数）。循环结束后，`concat_with_padding` 把所有轮次的 token 和 mask 拼成完整序列：

```python
# tensor_helper.py（简化）
info_mask_list.append(torch.ones(gen_len))   # LLM 生成部分：置 1
info_mask_list.append(torch.zeros(ret_len))  # 检索结果部分：置 0
info_mask = torch.cat(info_mask_list)        # 拼成完整 mask
```

训练侧的应用在 `verl/trainer/ppo/ray_trainer.py` 的 `update_actor` 调用前。当配置项 `actor_rollout_ref.actor.state_masking=true` 时，用 `info_mask` 覆盖默认的全 1 loss mask：

```python
# ray_trainer.py（简化）
if self.config.actor_rollout_ref.actor.state_masking:
    loss_mask = batch.batch['info_mask'][:, -response_length:]
else:
    loss_mask = batch.batch['attention_mask'][:, -response_length:]

actor_output = self.actor_rollout_wg.update_actor(batch)
```

这样，PPO 的 policy gradient 只会在 `loss_mask=1` 的位置（模型自己生成的 token）上计算，检索文档的 token 完全不参与参数更新。模型被迫学会"如何利用搜索结果推理"，而不是"如何复制搜索结果"。

### 5.3 动作截断与多轮上下文拼接

`_postprocess_responses` 在每轮生成后对 token 序列做截断和标记，其结果直接决定 `run_llm_loop` 下一轮的走向，分三种情况：

- 出现 `</search>` token：截断到该 token 之后（保留搜索标签），标记为**"需要搜索"**状态，下一轮调用检索服务
- 出现 `</answer>` token：截断到该 token 之后，标记为**"已完成"**状态，直接进入最终输出阶段
- 两者都未出现（达到 max_response_length）：保留完整序列，标记为**"超长截断"**状态，不再继续搜索，直接进入最终输出阶段

多轮上下文拼接由 `tensor_helper.py` 中的 `concat_with_padding` 完成。由于不同请求的生成长度不同，每轮结束后需要对 batch 内所有序列做右对齐 padding，再拼接搜索结果 token，才能送入下一轮 vLLM 推理。`info_mask` 在拼接时同步更新：LLM 生成部分置 1，搜索结果部分置 0，最终形成与完整轨迹等长的 mask tensor，供 PPO 训练时的 State Masking 使用。

### 5.4 奖励函数（qa_em.py）

奖励函数采用规则打分，无需单独的奖励模型。信号清晰、无歧义，是 RL 训练能稳定收敛的关键。实现上分四个层次：

- `normalize_answer(s)`：去冠词、去标点、小写化
- `extract_solution(solution_str)`：从 `<answer>...</answer>` 提取答案，要求至少出现 2 次，取最后一次出现的内容。格式要求模型先在 think 过程中写出初步答案，再在最终输出中给出经过完整推理的最终答案
- `compute_score_em(solution_str, ground_truth)`：精确匹配，归一化后完全相同得 1 分，否则 0 分
- 奖励只打在最后一个 token 上：`reward_tensor[i, valid_response_length - 1] = score`

### 5.5 Ray 分布式架构

Ray 是整个分布式训练的基础设施，对用户完全透明。WorkerGroup 是一组同类 Ray Actor 的管理器，对外暴露统一接口，内部自动把调用分发到每个 worker。调用 `actor_rollout_wg.generate_sequences(batch)` 时，WorkerGroup 内部会把 batch 按 GPU 数切分，同时发给所有 GPU worker，等所有 GPU 完成后把结果拼回来返回给 Driver。

四种 WorkerGroup 各司其职：

- **ActorRollout WG**：负责生成轨迹（vLLM + 多轮搜索）
- **Critic WG**：负责估值 V(s)
- **RefPolicy WG**：负责计算 KL 惩罚（冻结参数，不更新）
- **RewardModel WG**：本项目用规则 EM 奖励，不需要此 WG

Ray 里 Task（Remote Function）是无状态的，调用一次执行完销毁；Actor（Remote Class）是有状态的，进程长期存活，状态在调用之间保留。Search-R1 几乎全用 Actor，因为模型权重很大（3B~7B），不能每次调用都重新加载，且训练过程中模型参数在不断更新（`update_actor()` 修改 Actor 内部的 `self.model`），这种跨调用的状态变更只有 Actor 能做。

WorkerGroup Colocated 设计：ActorRollout 和 Critic 默认部署在同一批 GPU 上（`colocate_actor_critic=true`），两者在时间上交替使用显存——rollout 阶段 Critic 空闲，update 阶段 vLLM 引擎卸载权重。这样 8 张 GPU 既能跑 rollout 又能跑 critic，避免了为 Critic 单独申请一批 GPU 的资源浪费。代价是需要精细管理显存：vLLM 的 KV Cache 在 update 阶段必须释放，否则 Critic forward pass 会 OOM。

### 5.6 Search-R1 对 veRL 的改动

veRL 是字节跳动开源的通用 RL 训练框架（volcengine/verl），Search-R1 将其代码直接 vendor 进来并做了扩展。新增的自研模块：

- `search_r1/llm_agent/generation.py`：多轮搜索 Agent 核心
- `search_r1/llm_agent/tensor_helper.py`：tensor 拼接工具
- `search_r1/search/retrieval_server.py`：检索服务 FastAPI
- `search_r1/search/retrieval.py`：BM25/Dense 检索实现
- `search_r1/search/google_search_server.py`：Google 搜索后端
- `search_r1/search/rerank_server.py`：重排序服务

在 veRL 基础上修改的文件：

- `main_ppo.py`：新增 RewardManager（QA EM 奖励）
- `ray_trainer.py`：新增搜索分支、state masking、metrics
- `fsdp_workers.py`：新增 `compute_log_prob`（重算 log prob）
- `qa_em.py`：新增 QA 评分函数
- `ppo_trainer.yaml`：新增 `max_turns`、`retriever` 等配置项

### 5.7 项目中的高级 Python 语法

`@dataclass + field`：`GenerationConfig` 用 `@dataclass` 声明配置类，`DataProto` 用 `field(default_factory=dict)` 解决可变默认值问题（不能写 `= {}`，每个实例需各自创建新 dict）。

`@contextmanager + yield`：`_timer` 上下文管理器在 `yield` 前启动计时器，`yield` 把控制权交给 `with` 块内部，`with` 块结束后自动把耗时写入 `timing_raw` 字典。

自定义装饰器（WorkerGroup 分发机制的核心）：`@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` 把元信息"贴"到函数对象上，`WorkerGroup._bind_worker_method()` 在初始化时扫描所有被 `@register` 标记的方法，读取元信息，自动生成"分发→并行执行→收集"的包装函数并替换原方法。这是 Python 元编程的经典用法：用声明式的方式（装饰器）表达意图，框架在运行时读取意图并自动生成实现。

`functools.partial` 闭包晚绑定陷阱：在 `_bind_worker_method` 里用循环批量生成包装函数时，若直接写 `lambda: call(method_name)`，所有 lambda 共享同一个 `method_name` 变量，循环结束后全部指向最后一个方法名（经典的 Python 闭包晚绑定问题）。正确做法是用 `functools.partial(call, method_name=name)` 在创建时立即绑定当前值，每个包装函数持有独立的 `method_name` 副本，互不干扰。

魔术方法组合（`DataProto` 的序列化）：`__post_init__` 在 dataclass 初始化完成后自动调用做合法性校验；`__len__` 支持 `len(data_proto)`；`__getitem__` 支持切片；`__getstate__` / `__setstate__` 控制序列化行为——`DataProto` 要在 Ray 的不同进程间传输，`TensorDict` 不能直接 pickle，所以自定义序列化逻辑，先用 `torch.save` 转成 bytes。

## 六、FSDP 与分布式训练深度解析

### 6.1 FSDP 核心机制

FSDP（Fully Sharded Data Parallel，全切片数据并行）是 PyTorch 原生的大模型分布式训练方案。与 DDP（数据并行，每张 GPU 保存完整模型副本）不同，FSDP 把模型参数、梯度、优化器状态全部切片分散到所有 GPU 上，每张 GPU 只保存 1/N 的参数。对于参数、梯度和优化器状态这部分显存，占用降低接近 N 倍（激活值显存不在此列，见下文）。

All-Gather（前向传播）：每个 FSDP Unit（通常是一个 Transformer Layer）在前向计算前，触发 All-Gather 操作，从所有 GPU 收集完整参数，临时在本地重建完整层权重，完成前向计算后立即释放（只保留切片）。这个"用时收集、用完释放"的机制是 FSDP 节省显存的核心。

Reduce-Scatter（反向传播）：反向传播计算出完整梯度后，立即执行 Reduce-Scatter——先对所有 GPU 的梯度求和（Reduce），再把结果切片分发回各 GPU（Scatter），每张 GPU 只保留自己负责的那段梯度切片，用于更新本地参数切片。

激活值显存：FSDP 切片的是参数和梯度，但激活值（前向传播的中间结果）仍然按 batch 分配在各 GPU 上，不做切片。对于长序列训练（如 Search-R1 的多轮搜索轨迹，单条序列可达 4096+ token），激活值显存可能超过参数显存，需要配合 Gradient Checkpointing（重计算激活值，以时间换空间）使用。

### 6.2 TP vs FSDP：两种并行策略的本质区别

Tensor Parallelism（张量并行，TP）是另一种大模型并行策略，代表实现是 Megatron-LM。TP 把单个矩阵乘法切开：对于 `Y = XW`，把权重矩阵 W 按列切成 N 份，每张 GPU 计算 `Y_i = X * W_i`，最后 All-Reduce 合并结果。TP 的通信发生在每个矩阵乘法内部，延迟极低（需要高速 NVLink），适合单机多卡。

FSDP 与 TP 的核心区别在于切分粒度和通信时机：FSDP 按层切分参数（每层 All-Gather 一次），通信量与参数量成正比，适合跨节点（InfiniBand）；TP 按矩阵切分计算，通信嵌入在每次矩阵乘法中，延迟敏感，必须用 NVLink。

Search-R1 选择 FSDP 而非 TP，主要有两点原因：

- 训练集群通常是多节点（8 卡 × N 节点），节点间只有 InfiniBand，TP 的高频通信在 InfiniBand 上延迟过高
- FSDP 的 All-Gather 可以与计算 overlap（prefetch 下一层参数），有效隐藏通信延迟

FSDP 与 vLLM 共存的挑战：训练阶段用 FSDP（参数切片），推理阶段用 vLLM（需要完整参数）。Search-R1 的解决方案是在 rollout 前调用 `sync_fsdp_params_to_vllm()`，触发 All-Gather 把切片参数重建为完整权重，同步到 vLLM 的 GPU 显存；rollout 结束后再释放完整权重，恢复 FSDP 切片状态。这个同步操作是每个 PPO step 的开销之一，约占单步时间的 5-10%。

## 七、PPO 算法细节

### 7.1 KL 惩罚 vs Clip 机制

PPO 有两种防止策略更新过大的机制，Search-R1 同时使用两者。KL 惩罚（KL Penalty）：在奖励中减去当前策略与参考策略（初始模型）的 KL 散度，`r_adjusted = r - kl_coef * KL(pi_theta || pi_ref)`，`kl_coef=0.001` 是一个很小的系数，主要防止模型在训练后期完全偏离初始分布（"忘记"原有语言能力）。

Clip 机制（PPO-Clip）：限制单步策略更新幅度，`L_clip = min(r_t * A_t, clip(r_t, 1-e, 1+e) * A_t)`，其中 `r_t = pi_theta(a|s) / pi_theta_old(a|s)` 是新旧策略的概率比，`e=0.2` 是 clip 范围。当概率比超出 [0.8, 1.2] 时，梯度被截断，防止单步更新过大导致训练不稳定。KL 惩罚是软约束（通过奖励函数），Clip 是硬约束（直接截断梯度），两者互补。

### 7.2 By-Token 奖励与 Teacher Forcing

Search-R1 的奖励是稀疏的 By-Token 奖励：只在最后一个有效 token 上打分（EM 得 1 或 0），其余 token 奖励为 0。实现上把分数放在最后一个 token 位置，GAE 负责把这个信号沿时间步反向摊薄到整条轨迹——越靠近答案的 token 获得越高的优势估计。这与 By-Sequence 奖励（整条序列一个分数）的区别在于：稀疏奖励需要 GAE 做信号传播，方差更大但信号更精确。

Teacher Forcing 在 PPO 的 log prob 重算阶段（Step 2）体现：用完整的已生成轨迹作为输入，让模型重新计算每个 token 的生成概率。这与训练语言模型时的 Teacher Forcing 一致——用真实的历史 token（而非模型自己生成的 token）作为上下文，避免误差累积。在多轮搜索场景下，"真实历史"包含了搜索结果 token，但 State Masking 确保这些 token 不参与梯度计算，只作为上下文条件。

### 7.3 GAE 优势函数估计

GAE（Generalized Advantage Estimation）是 PPO 中估计优势函数 A(s,a) 的标准方法，平衡偏差和方差。`A_t = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...`，其中 `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)` 是 TD 误差，`gamma` 是折扣因子，`lambda` 是 GAE 参数。`lambda=1` 退化为 Monte Carlo 估计（高方差低偏差），`lambda=0` 退化为单步 TD（低方差高偏差）。Search-R1 默认 `gamma=1.0`（无折扣，因为序列不长）、`lambda=0.95`。由于奖励只在最后一个 token 非零，GAE 实际上是把最终奖励沿时间步反向"摊薄"到每个 token，越靠近答案的 token 获得越高的优势估计。

## 八、搜索后端详细对比

Search-R1 支持三类搜索后端，通过统一的 `POST /retrieve` 接口暴露，训练代码无需感知后端类型。三类后端在召回质量、延迟、成本和离线可用性上各有取舍。

### 8.1 BM25（稀疏检索）

BM25 是基于词频统计的经典稀疏检索算法，是 Elasticsearch/Lucene 的默认排序算法。实现上用 `pyserini` 库，索引预先构建在本地磁盘，检索时完全离线。优点是速度极快（毫秒级）、无 GPU 依赖、可复现；缺点是词汇鸿沟问题——查询词和文档词必须完全匹配，无法处理同义词（如查"automobile"找不到含"car"的文档）。适合训练阶段的大规模 rollout（需要高吞吐、低延迟），是 Search-R1 论文实验的默认后端。

### 8.2 Dense Retrieval（稠密检索）

FAISS Flat（精确最近邻）：用 E5/DPR 等双塔模型把文档和查询编码为稠密向量，FAISS IndexFlatIP 做精确内积搜索。召回质量显著优于 BM25（能处理语义相似但词汇不同的情况），但索引构建慢（需要对全量语料跑 embedding），检索时需要 GPU 加速，延迟比 BM25 高 5-10 倍。适合对召回质量要求高的评估场景。

FAISS HNSW（近似最近邻）：Hierarchical Navigable Small World 图索引，用图结构加速近似最近邻搜索。相比 Flat 索引，HNSW 牺牲少量召回率（通常 <1%）换取 10-100 倍的检索速度提升，且检索时不需要 GPU（纯 CPU 即可）。适合生产部署场景，是稠密检索的工程首选。

Cross-Encoder 重排序（rerank_server.py）：在 BM25 或 FAISS 初检后，用 Cross-Encoder 模型（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）对 top-K 候选文档重新打分排序。Cross-Encoder 把查询和文档拼接后一起编码（而非双塔分别编码），能捕捉查询-文档的细粒度交互，召回质量接近 GPT-4 级别的 reranker，但延迟较高（每次重排需要跑 K 次 forward pass）。Search-R1 的 rerank_server.py 实现了可选的两阶段检索：先 BM25 召回 top-50，再 Cross-Encoder 重排取 top-3。

### 8.3 在线搜索（SerpAPI / Google）

SerpAPI 和 Google Custom Search API 提供实时网络搜索能力，适合需要最新信息的场景（如新闻问答）。优点是信息最新、覆盖面广；缺点是有 API 调用费用、延迟高（网络 RTT + 搜索引擎处理，通常 500ms-2s）、训练时不可复现（同一查询不同时间返回不同结果）。Search-R1 的 `google_search_server.py` 封装了 SerpAPI，通过环境变量 `SERPAPI_KEY` 配置密钥，训练时一般不用（用 BM25 保证可复现性），评估时可切换到在线搜索测试真实场景性能。

## 九、训练配置与关键超参

Search-R1 在 `verl/trainer/config/ppo_trainer.yaml` 中新增了以下关键配置项，控制搜索行为、序列长度和训练策略：

```yaml
max_turns: 2                    # 最多 2 轮搜索
do_search: true                 # 是否启用搜索
retriever.url: "http://127.0.0.1:8000/retrieve"
retriever.topk: 3               # 每次检索 top-3 文档
data.max_start_length: 2048     # 初始问题长度上限
data.max_response_length: 500   # 每轮 LLM 生成最多 500 token
data.max_obs_length: 500        # 搜索结果最多 500 token
actor_rollout_ref.actor.state_masking: true
algorithm.kl_coef: 0.001        # KL 散度约束
algorithm.no_think_rl: false    # 去掉 <think> 的消融实验开关
```

## 十、数据格式

训练样本为 JSON 格式，每条样本包含三个字段：`prompt`（用户问题，chat 格式）、`reward_model`（奖励配置，`style: rule` 表示规则奖励，`ground_truth` 为标准答案）、`ability`（任务类型，如 `fact-reasoning`）。语料库（检索文档）格式为 `{"id": "0", "contents": "\"标题\"\n正文内容"}`。

支持的训练数据集：NQ（Natural Questions）、TriviaQA、PopQA、HotpotQA（多跳）、2WikiMultihopQA（多跳）、MuSiQue（多跳）、BambooGLE。

## 十一、推理框架选择：为什么不用 vLLM/SGLang

`infer.py` 用的是原生 HuggingFace Transformers 的 `model.generate()`，而不是 vLLM/SGLang。推理脚本是单请求、交互式的——每轮生成后要停下来调用搜索 API，拿到结果再继续，这个"暂停-插入-继续"的多轮循环用 vLLM/SGLang 反而更复杂。Transformers 的 `StoppingCriteria` 机制天然支持"遇到 `</search>` 就停"的逻辑，用在这里刚好合适。

vLLM 在这个项目里只用于训练阶段的 rollout 生成（`verl/workers/rollout/vllm_rollout/`）——每个 step 对大批量问题同时批量采样，高吞吐场景才值得用。vLLM 相比 HuggingFace 的核心优势是 PagedAttention（类似操作系统虚拟内存管理 KV Cache，显存利用率大幅提升）和 Continuous Batching（某个请求生成完就立刻移出，新请求插入，GPU 几乎不空转），在批量推理场景下吞吐量显著高于 HF。

---

参考资料：
- 项目地址 [PeterJinGo/Search-R1](https://github.com/PeterJinGo/Search-R1)，
- 论文 [Search-R1: Training LLMs to Reason and Search with Reinforcement Learning](https://arxiv.org/abs/2503.09516)。
