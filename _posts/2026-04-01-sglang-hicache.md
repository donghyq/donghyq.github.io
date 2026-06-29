---
title: SGLang HiCache 解析
date: 2026-04-01 10:00:00 +0800
categories: [LLM Infra, SGLang]
tags: [kv-cache, inference, sglang, rdma, pd-disaggregation]
description: 从 Radix Tree 的节点状态机到 GPU/Host/Storage 三层异步搬运引擎，拆解 SGLang 如何用分层缓存将 KV Cache 容量扩展数个数量级，并在 PD 分离架构下演化为跨节点 RDMA 数据总线。
---

## 目录

**Part I — 背景与全链路概览**

1. [为什么需要 HiCache](#ch1)
2. [三层存储架构：L1 GPU / L2 Host / L3 Storage](#ch2)
3. [一次请求经历的全链路](#ch3)

**Part II — 数据结构与内存管理**

4. [Radix Tree 与 TreeNode 的多级状态](#ch4)
5. [内存池：GPU Pool、Host Pool 与 Storage 的寻址方式](#ch5)
6. [TreeNode 状态机：insert → backup → evict → prefetch](#ch6)

**Part III — 异步搬运引擎**

7. [load_back 与 compute 的 overlap](#ch7)
8. [Storage 层：prefetch_thread 与 backup_thread](#ch8)

**Part IV — 上层集成与架构扩展**

9. [PD 分离下的 RDMA 传输](#ch9)
10. [MLA 的 latent KV 与 HiCache 适配](#ch10)

---

<a id="ch1"></a>
## 1. 为什么需要 HiCache

### 1.1 KV Cache 的显存困境

在 Transformer 架构的自回归生成中，每一层的 Attention 都需要读取之前所有 token 的 Key 和 Value 向量。Attention 的计算本质是两次矩阵乘法：

![Attention 矩阵乘法流程](/assets/img/posts/sglang-hicache/fig-attention-matmul.svg)

Q 是当前 token 的 Query（生成阶段只有 1 行），K 和 V 是历史所有 token 的 Key/Value 向量。随着生成推进，K 和 V 不断增长——这就是 KV Cache。它避免了每次生成都重新计算所有历史 token 的 K/V，但代价是显存占用随序列长度线性增长。

对于一个 70B 参数的模型、128K 上下文窗口，单个请求的 KV Cache 可能超过 40GB——直接超过一张 A100 的显存。

SGLang 在 HiCache 之前已经通过 Radix Tree 实现了多请求之间的前缀复用：如果两个请求的 system prompt 完全相同，它们共享同一棵子树上的 KV Cache，避免重复存储。但即使有前缀复用，当并发请求多、上下文长时，显存仍然会被打满，导致触发 eviction（驱逐）——此时 KV 数据被直接丢弃，下次使用时必须重新计算。

### 1.2 HiCache 的核心思路

HiCache 的解法是：不丢弃，降级存储。当 GPU 显存不够时，把 KV Cache 搬到 Host 内存（CPU RAM）；Host 满了再搬到外存（SSD 或分布式内存池）。需要时再异步搬回来——利用排队等待和逐层计算的时间窗口隐藏搬运延迟。

关键在于：搬运是异步的、流水线化的。HiCache 用独立的 CUDA Stream 做 GPU-Host DMA 拷贝，用独立的 Python 线程做 Host-Storage IO，并且和前向计算在时间上重叠。

---

<a id="ch2"></a>
## 2. 三层存储架构：L1 GPU / L2 Host / L3 Storage

![HiCache 三层存储概览](/assets/img/posts/sglang-hicache/fig-1-1-three-tier.svg)

从上到下，三层存储的带宽递减、容量递增：

| 层级 | 物理介质 | 带宽 | 容量 | 管理方式 |
|------|----------|------|------|----------|
| L1 | GPU HBM | 3.35 TB/s | 80 GB (H100) | `MHATokenToKVPool`，连续 Tensor + page slot 索引 |
| L2 | Host Pinned Memory | ~200 GB/s (内存带宽) | 256-1024 GB | `MHATokenToKVPoolHost`，结构与 L1 对齐 |
| L3 | SSD / RDMA 内存池 | ~7 GB/s (NVMe) 或 ~100 GB/s (RDMA) | 近乎无限 | Content-Addressable，SHA256 hash 寻址 |

层间搬运全部异步：L1↔L2 通过 `cudaMemcpyAsync` 走 PCIe（~32 GB/s），L2↔L3 通过独立 Python 线程做文件或 RDMA IO。两者都不阻塞 GPU 的前向计算。

---

<a id="ch3"></a>
## 3. 一次请求经历的全链路

一个请求从进入 Scheduler 到生成完毕，经历四个阶段，每个阶段对应一组 HiCache 操作：

![请求全生命周期数据流](/assets/img/posts/sglang-hicache/fig-3-1-request-lifecycle.svg)

### 阶段一：排队期 —— L3 预取

请求刚进入 `waiting_queue` 时，HiCache 调用 `prefetch_from_storage()`。排队时间被用来做最慢的 Storage IO：后台 prefetch_thread 去 L3 查询前缀 hash 命中情况，命中则异步读取到 Host 内存。

### 阶段二：调度期 —— 前缀匹配与 L2→L1 回填

Scheduler 准备运行请求时，调用 `match_prefix()` 遍历 Radix Tree。函数返回两个断点：`last_device_node`（GPU 命中深度）和 `last_host_node`（Host 备份深度）。如果 Host 侧有额外命中，触发 `load_back()` 将数据从 L2 搬到 L1，请求进入等待。

### 阶段三：计算期 —— 逐层 Overlap

模型做 Forward 时，第 i 层的 DMA 与第 i-1 层的 Attention 并行。LayerDoneCounter 管理逐层同步——算第 i 层之前 `wait(i)` 确认该层 DMA 完成即可，后续层的传输持续进行。

### 阶段四：完成期 —— 插入与备份

新 token 生成完毕后，`insert()` 把 KV Cache 挂入 Radix Tree。如果 Write-Through Selective 策略下该节点 `hit_count >= 2`（热数据），异步备份到 L2；然后 backup_thread 继续写入 L3。

> 搬运操作始终藏在排队和计算的背后。只要 IO 延迟 ≤ 等待+计算延迟，用户感知为零。

### 单体 vs PD 分离：物理执行路径转移

PD 分离架构复用了上述四个逻辑阶段，但改变了数据处理所在的物理节点：

| 阶段 | 单体模式 | PD 分离模式 |
|------|----------|-------------|
| 排队期 prefetch | L3 = 本机 SSD | L3 = 远端 RDMA 内存池（Mooncake/Nixl） |
| 调度期 load_back | L2→L1 本机 DMA | 同上（Decode 节点本地） |
| 计算期 overlap | 同 | 同 |
| 完成期 backup | L1→L2→L3 本机写入 | Prefill 端写入全局 L3，Decode 端通过 prefetch 读取 |

在这两种架构中，系统没有为 PD 分离重写全新的传输系统。Prefill 端的 `write_backup → backup_thread` 承担了向远端发送数据的作用，而 Decode 端的 `prefetch_from_storage` 负责从远端接收。三层架构在这里演变成了跨节点的数据总线。至于 RDMA 零拷贝细节和 ZMQ 事件广播机制，将在第 9 章展开。

---

<a id="ch4"></a>
## 4. Radix Tree 与 TreeNode 的多级状态

### 4.1 Radix Tree 在哪里

Radix Tree 是 HiRadixCache 类内部维护的一棵多叉树，完全存在于 Python 进程的堆内存中。它不占用 GPU 显存，也不涉及任何 DMA 操作——它是纯粹的"目录索引"，记录每份 KV 数据在哪一层的哪些 slot 里。

Scheduler 线程是唯一操作这棵树的线程。所有的 `insert()`、`match_prefix()`、`evict()` 都在 Scheduler 的主循环中同步执行。后台线程（prefetch_thread、backup_thread）只通过队列和字典与主线程通信，不直接操作树结构。

### 4.2 Tree 结构

Radix Tree 以 token_id 序列为 key，按公共前缀合并。每个节点的 key 长度强制为 `page_size`（通常 16 tokens）的整数倍——这是 `RadixKey.page_aligned()` 的作用。不满一整 page 的尾部 token 留在请求内部，不参与树的共享。

![Radix Tree 示例](/assets/img/posts/sglang-hicache/fig-2-1-radix-tree.svg)

### 4.3 TreeNode 里存了什么

TreeNode 不存储真实的 KV Tensor 数据。它存储的是三组"指针"和一组状态标记：

| 属性 | 类型 | 含义 |
|------|------|------|
| `value` | List[int] 或 None | GPU Pool 中的 page slot indices。为 None 表示已被驱逐出 GPU |
| `host_value` | List[int] 或 None | Host Pool 中的 page slot indices。不为 None 表示在 CPU 有备份 |
| `hash_value` | List[bytes] 或 None | L3 Storage 的内容寻址 hash。用于读写外存 |
| `evicted` | property | `self.value is None` —— 数据已不在 GPU |
| `backuped` | property | `self.host_value is not None` —— 在 Host 有备份 |
| `lock_ref` | int | 引用计数。>0 时该节点正被请求使用或 DMA 中，禁止从 GPU 驱逐 |
| `host_ref_counter` | int | Host 引用计数。>0 时后台线程正在读写该 Host 数据，禁止释放 |
| `hit_count` | int | 被命中次数。Write-Through Selective 策略下达到阈值（通常 2）触发备份 |

树的操作（遍历、分裂、插入）是纯 Python 指针操作，极其轻量；真正的重活（数据搬运）全部委托给 HiCacheController 异步执行。

> 相关联系：TreeNode 作为索引，三个字典（`ongoing_write_through` / `ongoing_load_back` / `ongoing_prefetch`）用于追踪异步任务的生命周期。

---

<a id="ch5"></a>
## 5. 内存池：GPU Pool、Host Pool 与 Storage 的寻址方式

### 5.1 GPU Pool（L1）

GPU 层的物理存储由 `MHATokenToKVPool` 管理。核心是两个预分配的 4D Tensor：

```python
# k_buffer 和 v_buffer 的形状：
# [num_layers, pool_capacity, num_kv_heads, head_dim]
k_buffer = torch.zeros(num_layers, capacity, num_kv_heads, head_dim, dtype=dtype, device="cuda")
v_buffer = torch.zeros(num_layers, capacity, num_kv_heads, head_dim, dtype=dtype, device="cuda")
```

当 TreeNode 的 `value = [5, 6, 7]` 时，意味着节点的 3 个 page 位于 Pool 的第 5、6、7 行（每行存放 16 个 token 的 KV 数据）。模型在做 Attention 计算时，直接通过 `k_buffer[:, indices, :, :]` 获取连续的 KV 张量。

### 5.2 Host Pool（L2）

Host 层由 `MHATokenToKVPoolHost` 管理，结构与 GPU Pool 完全对齐——相同的 shape、相同的 page_size，区别在于内存类型是 **Pinned Memory**（锁页内存）。锁页内存保证 CUDA DMA 拷贝时物理地址不变（不会被 OS 换出到 swap）。需要强调的是：**`cudaMemcpyAsync` 只有在源和目的都是 Pinned Memory 或 Device Memory 时才真正异步**，如果使用普通的 CPU 内存，拷贝操作会被强行退化为同步阻塞，从而彻底破坏 Overlap 流水线。

```python
# Host Pool 同样是按 page slot 索引的
host_k_buffer = torch.zeros(num_layers, host_capacity, num_kv_heads, head_dim,
                            dtype=dtype, device="cpu").pin_memory()
host_v_buffer = torch.zeros(num_layers, host_capacity, num_kv_heads, head_dim,
                            dtype=dtype, device="cpu").pin_memory()
```

当 TreeNode 的 `host_value = [12, 13, 14]` 时，它表明该节点的 3 个 page 保存在 Host Pool 的第 12、13、14 个 slot 中。

### 5.3 Storage（L3）

Storage 层完全不同于 L1/L2 的"连续 Tensor + slot 索引"方式。它采用 **Content-Addressable Storage**（内容寻址存储）：每个 page 的 KV 数据被序列化为一个二进制 blob，以该 blob 内容的 SHA256 hash 为 key 存入 backend。

```python
# 存储格式（概念模型）：
# key:   SHA256(kv_data_bytes)
# value: 序列化的 [num_layers, page_size, num_kv_heads, head_dim] 张量

# TreeNode.hash_value = [hash_page_0, hash_page_1, hash_page_2]
# 每个 hash 对应 Storage 中的一个 blob
```

这种方法自然实现了数据去重。如果两个请求产生了内容完全相同的 page（例如具有相同的前置 prompt），它们会生成同样的 hash，Storage 中只保留一份拷贝。这既适用于本地 SSD，也适用于分布式共享内存池（如 Mooncake、Nixl）。

### 5.4 TP 下 Pool 的变化

Tensor Parallelism 将模型的 Attention Heads 切分到多个 GPU（rank）上。不同的 Attention 架构在 HiCache 中有不同的行为：

| 架构 | 每个 rank 存什么 | Page 大小 | backup_skip | config_suffix |
|------|-----------------|-----------|-------------|---------------|
| MHA / GQA | 只存自己负责的 head 切片 | `page_size * (num_kv_heads // tp_size) * head_dim` | False（每个 rank 各备份各的） | `_tp{rank}` |
| MLA (DeepSeek) | 所有 rank 的 latent KV 完全相同 | `page_size * kv_lora_rank` | True（只有 rank 0 写 L3） | 无后缀 |

在 TP 环境下，所有会改变 Tree 状态的操作都通过 `all_reduce(MIN)` 保持跨 rank 一致性。比如 `writing_check()` 中，每个 rank 各自查询自己的 CUDA Event 完成数，取最小值作为全局完成数——确保没有任何 rank 提前解锁某个节点导致不一致。

---

<a id="ch6"></a>
## 6. TreeNode 状态机：insert → backup → evict → prefetch

### 6.1 Insert

前向计算每完成一个 page（通常 16 tokens），`insert()` 被调用。它沿着 Radix Tree 向下匹配 token 序列：如果节点已存在，增加 hit_count；如果不存在，创建新节点并分配 GPU indices。

```python
def insert(self, params: InsertParams) -> InsertResult:
    # 沿树匹配，找到最长公共前缀
    # 如果匹配到已有节点：
    node.hit_count += 1
    if self.write_policy == "write_through_selective" and node.hit_count >= 2:
        self.write_backup(node)  # 异步备份到 Host

    # 如果需要创建新节点：
    new_node = TreeNode(key=remaining_tokens, value=new_gpu_indices)
    parent.children[edge_key] = new_node
    
    # 为 Storage 层计算内容哈希
    if self.enable_storage or self.enable_kv_cache_events:
        new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
```

这种部分数据的淘汰方式提供了缓冲空间。第一次命中的数据通常使用率较低，不占用内存备份；当相同前缀再次命中时，系统会将其判定为高频数据并执行 DMA 备份。

### 6.2 Evict

当 GPU 显存不足时触发 `evict(num_tokens)`。从 LRU 队列中弹出节点，根据节点状态选择不同路径：

```python
def evict(self, num_tokens):
    while evicted_count < num_tokens:
        node = self.eviction_heap.pop()  # LRU 弹出最冷节点

        if node.lock_ref > 0:
            continue  # 正在被使用，跳过

        if node.backuped:
            # Case 1: 已有 Host 备份 → 直接清 GPU
            self._evict_backuped(node)
            # node.value = None, evicted = True
        
        elif self.write_policy == "write_back":
            # Case 2: 没备份 + write_back → 紧急搬到 Host 再清 GPU
            self.write_backup(node, write_back=True)
            # 必须阻塞等 DMA 完成
        
        else:
            # Case 3: 没备份 + write_through → 彻底丢弃
            self._evict_regular(node)
```

在 Case 1 中，如果之前 Write-Through 备份已经完成，程序只需清空元数据即可。Case 2 则需要在关键路径上等待 DMA 完成。在 Case 3 中，对于命中次数不足 2 的数据，直接丢弃避免了额外的拷贝开销。

### 6.3 状态流转全图

![TreeNode 状态机](/assets/img/posts/sglang-hicache/fig-4-1-state-machine.svg)

节点从 State A 逐步降级到 State D。回升时通过 prefetch (D→C) 和 load_back (C→B) 逆向攀升。

---

<a id="ch7"></a>
## 7. load_back 与 compute 的 overlap

> HiCacheController 维护独立的 `load_stream` 和 `write_stream`，与主计算流完全分离。LayerDoneCounter 让 DMA 和 Forward 逐层交错执行。

### 7.1 双 Stream 设计

CUDA 中同一个 Stream 内的操作是串行的，但不同 Stream 间可以并行。HiCacheController 利用这一点：

```python
class HiCacheController:
    def __init__(self):
        self.write_stream = torch.cuda.Stream()  # GPU→Host DMA
        self.load_stream  = torch.cuda.Stream()  # Host→GPU DMA
        # 主计算使用默认 stream (stream 0)
        # 三者物理并行执行
```

`write_stream` 负责将 GPU 数据拷贝到 Host（write_backup 时使用）。`load_stream` 负责将 Host 数据拷贝回 GPU（load_back 时使用）。两者互不干扰，也不阻塞默认计算流上的前向传播。

### 7.2 整体流程：逐层流水线如何运转

假设一个请求命中了 L2（Host 内存）中的 KV Cache，需要搬回 GPU 并同时做 Attention 计算。三条 Stream 对应三个硬件单元（SM 计算核心 + 两个 Copy Engine），以一个 32 层模型为例，时序如下：

```
时间 →

load_stream (Copy Engine):  [搬 layer 0][搬 layer 1][搬 layer 2][搬 layer 3] ...
                                ↓ event     ↓ event     ↓ event     ↓ event
compute stream (SM):         wait(0)→[算 L0] wait(1)→[算 L1] wait(2)→[算 L2] ...
```

具体步骤：

1. **Scheduler 发起 load_back**：检测到请求所需的 KV Cache 在 Host 上（TreeNode 状态 C：有 `host_value`，无 `gpu_value`），调用 `start_loading(host_indices, device_indices)`。
2. **load_stream 逐层搬运**：在 `load_stream` 上循环 32 层，每层执行一次 `cudaMemcpyAsync(HostToDevice)`。搬完一层立即 `record` 一个 CUDA Event 标记"第 i 层到位"。整个过程由 Copy Engine 执行，不占用 SM。
3. **Forward 逐层计算**：同时在 default stream 上启动前向传播。算到第 i 层时，先 `wait(layer_i)` 查询对应的 CUDA Event——若 DMA 已完成则瞬间通过，否则短暂等待（通常只需几十微秒，因为 DMA 已提前启动并领先若干层）。
4. **流水线效果**：由于 load_stream 比 compute 先启动，正常情况下 DMA 总是领先于计算。理想状态下 compute 侧的 `wait()` 几乎不产生等待，总延迟 ≈ max(DMA 总时间, Compute 总时间)，远优于"先搬完再算"的串行模式（延迟 = DMA + Compute）。

### 7.3 LayerDoneCounter 逐层同步

逐层 Overlap 的核心在于 `LayerDoneCounter`。它管理一组 `LayerLoadingEvent`，每个 Event 包含每一层的 CUDA Event：

```python
class LayerDoneCounter:
    # 三缓冲：最多同时有 3 组 load-back 请求在进行中
    def __init__(self, num_layers):
        self.events = [LayerLoadingEvent(num_layers) for _ in range(3)]
    
    # Producer 侧（load_stream）：每搬完一层就 record 一个 event
    def start_loading(self, host_indices, device_indices):
        with torch.cuda.stream(self.load_stream):
            for layer_i in range(self.num_layers):
                # DMA: host_pool[layer_i, host_indices] → gpu_pool[layer_i, device_indices]
                gpu_pool[layer_i, device_indices].copy_(host_pool[layer_i, host_indices])
                # 标记第 layer_i 层完成
                current_event.complete(layer_i)  # record CUDA event
    
    # Consumer 侧（compute stream）：算第 i 层之前等第 i 层的 DMA 完成
    def wait(self, layer_i):
        # stream.wait_event() 不阻塞 CPU 线程！
        # 它只让 GPU 上的主计算流等待 load_stream 的 event
        torch.cuda.current_stream().wait_event(self.load_events[layer_i])
        # 随后 launch 的 Attention Kernel 会在 GPU 侧自动挂起，直到 DMA 拷贝到达
```

![DMA 与计算的逐层流水线](/assets/img/posts/sglang-hicache/fig-5-1-layer-pipeline.svg)

### 7.4 writing_check 与 DMA 确认

Write-Through 的 DMA（GPU→Host）也运行在独立的 `write_stream` 上。每个 Step 结束时，Scheduler 调用 `writing_check()` 非阻塞地查询已完成的 DMA：

```python
def writing_check(self, write_back=False):
    if write_back:
        # 阻塞模式：evict 时必须等所有 DMA 完成
        while len(self.ongoing_write_through) > 0:
            for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                finish_event.synchronize()
                for ack_id in ack_list:
                    node = self.ongoing_write_through.pop(ack_id)
                    self.dec_lock_ref(node)  # 解锁，允许驱逐
                    if self.enable_storage:
                        self.write_backup_storage(node)  # 继续写 L3
        return

    # 非阻塞模式：只处理已完成的 event
    for _, finish_event, _ in self.cache_controller.ack_write_queue:
        if not finish_event.query():  # query() 不阻塞
            break
        finish_count += 1
    
    # TP 同步：取最小值确保所有 rank 状态一致
    finish_count = all_reduce_min(finish_count)
```

正常运行时，系统通过查询判断完成情况，不阻碍 GPU 计算；只有在使用 write_back 进行 eviction 的情况下才会发生阻塞等待。

---

<a id="ch8"></a>
## 8. Storage 层：prefetch_thread 与 backup_thread

> Storage IO 主要通过独立的 Python 后台线程执行，利用 Queue 进行解耦。主线程处理排队和响应操作，自身不进行 IO 阻塞。

### 8.1 线程架构

![Storage 层线程模型](/assets/img/posts/sglang-hicache/fig-6-1-storage-threads.svg)

### 8.2 Prefetch 完整流程

Prefetch 分为四步：主线程入队 → prefetch_thread 查命中 → prefetch_io_aux_thread 做 IO → 主线程检查进度并挂载到树。

```python
# Step 1: 主线程入队
def prefetch_from_storage(self, req_id, last_host_node, new_input_tokens):
    host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
    operation = self.cache_controller.prefetch(req_id, host_indices, prefetch_key)
    self.ongoing_prefetch[req_id] = (last_host_node, prefetch_key, host_indices, operation)

# Step 2: prefetch_thread 查询命中
def prefetch_thread_func(self):
    operation = self.prefetch_queue.get()
    hash_value, hit_count = self._storage_hit_query(operation)
    hit_count = all_reduce_min(hit_count)  # TP 同步
    if hit_count < threshold:
        self.prefetch_revoke_queue.put(operation.request_id)  # 撤销
    else:
        self.prefetch_buffer.put(operation)  # 交给 IO 线程

# Step 3: IO 线程实际读取
def prefetch_io_aux_func(self):
    operation = self.prefetch_buffer.get()
    self._page_transfer(operation)  # 批量从 Storage 读到 Host memory

# Step 4: 主线程检查并挂载
def check_prefetch_progress(self, req_id):
    completed_tokens = self.cache_controller.terminate_prefetch(operation)
    self._insert_helper_host(last_host_node, fetched_key, written_indices, hash_value)
```

Step 2 中的 `all_reduce_min(hit_count)`：TP 环境下，每个 rank 各自查询 Storage（MHA 模型各 rank 的 hash 可能不同），取最小命中数保证所有 rank 的决策一致——要么全部预取，要么全部撤销。

### 8.3 Backup 流程

```python
def backup_thread_func(self):
    operation = self.backup_queue.get()
    if not self.backup_skip:  # MLA 模型只 rank 0 备份
        self._page_backup(operation)
    self.ack_backup_queue.put(operation)  # 通知主线程

def _page_backup(self, operation):
    for i in range(0, len(operation.hash_value), self.storage_batch_size):
        batch_hashes = operation.hash_value[i:i+batch_size]
        batch_host_indices = operation.host_indices[...]
        success = self.page_set_func(batch_hashes, batch_host_indices)
        if not success:
            break
```

Backup 相对简单：从 Host Pool 读出 KV 数据，按 batch 写入 Storage。`backup_skip` 标志位在 MLA 模型下被设为 True（非 rank 0 跳过），避免多 rank 写同一份数据造成浪费。

---

<a id="ch9"></a>
## 9. PD 分离下的 RDMA 传输

### 9.1 通信通道的重构

在单节点部署中，L3 作为冷备存储运行。但在 PD 分离架构中，它变成了一个跨节点的数据传输通道。Prefill 节点计算完成的 KV Cache 会写入 L3（如 Mooncake 提供的 RDMA 内存池），然后再传输给 Decode 节点。

SGLang 的聪明之处在于：它没有为 PD 分离重写一套传输系统，而是直接复用了 HiCache 的三层架构——Prefill 端的 "backup to L3" 就是发送，Decode 端的 "prefetch from L3" 就是接收。

### 9.2 调度队列的物理隔离

```python
def _add_request_to_queue(self, req: Req):
    if self.disaggregation_mode == DisaggregationMode.NULL:
        # 单体模式：普通 waiting_queue
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)

    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        # Prefill 节点：只做前向计算，把 KV Cache "bootstrap" 给全局
        self._prefetch_kvcache(req)
        self.disagg_prefill_bootstrap_queue.add(req, self.model_config.num_key_value_heads)

    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        # Decode 节点：先从全局 L3 预分配和拉取 KV Cache
        self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
```

这些策略使得 Prefill 和 Decode 节点能分别处理各自的任务。Prefill 节点在计算结束后尽快写入数据，而 Decode 节点则在数据就绪后直接开始生成过程。

### 9.3 ZMQ 事件广播

Prefill 节点算完一个 page 后，除了走正常的 insert → write_backup → backup_thread 流程写入 L3，还会通过 ZeroMQ 广播一个 `BlockStored` 事件，让整个集群实时感知"这段 KV Cache 现在在 L3 可用了"：

```python
# kv_events.py
class ZmqEventPublisher(EventPublisher):
    def _publisher_thread(self):
        while self._running:
            event = self._event_queue.get(timeout=0.1)
            payload = self._pack.encode(event)  # msgpack 序列化
            seq_bytes = seq.to_bytes(8, "big")
            # ZMQ PUB/SUB 广播给所有订阅者（Router/Decode 节点）
            self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))
```

使用 msgspec.msgpack 进行极致的二进制序列化，ZMQ 的 PUB/SUB 模式保证了低延迟的多播能力。Router 收到事件后，就知道可以把对应的 Decode 任务调度出去了。

### 9.4 Zero-Copy RDMA 直写

当 Storage Backend 是支持 RDMA 的分布式内存池（Mooncake、Nixl、EIC）时，HiCache 替换默认的 IO 函数为零拷贝版本：

```python
def attach_storage_backend(self, ...):
    if self.storage_backend_type in ["hf3fs", "mooncake", "eic", "nixl", "simm"]:
        self.page_get_func = self._page_get_zero_copy
        self.page_set_func = self._page_set_zero_copy

def _page_get_zero_copy(self, operation, hash_values, host_indices, extra_info=None):
    # RDMA 网卡绕过 CPU，直接把远端数据写入 host_indices 对应的 Pinned Memory
    results = self.storage_backend.batch_get_v1(hash_values, host_indices, extra_info)
```

对比普通模式下的 `_generic_page_get`（需要在 Python 层创建临时 buffer、反序列化、再拷贝），零拷贝模式让 RDMA 网卡直接写入预分配的锁页内存物理地址。这是 PD 分离能做到低延迟的核心保障。

### 9.5 跨机前缀复用

因为 L3 中的 KV Cache 基于内容 hash 进行存储，如果两个请求有相同的 system prompt，它们生成的 KV Cache hash 值也将一致。当 Decode 节点尝试读取时，能够直接获取已存在的数据，无需再次计算，将前缀复用的范围扩大到了跨节点层级。

### 9.6 多模态 Embedding 直传

在 PD 分离的变体中，多模态 Encoder 和 LLM 可能分布在不同机器上。视觉 Embedding 体积巨大（一张高分辨率图片可能产生数万 token 的 Embedding），走常规网络序列化开销不可接受。SGLang 的做法是向 RDMA 引擎预注册一块内存，让远端 Encoder 直接写入：

```python
# encode_receiver.py
class MMReceiverBase:
    async def allocate_embedding_buffer(self, req_id, total_bytes):
        embeddings = torch.empty(total_bytes, dtype=torch.uint8)
        # 向 Mooncake 引擎注册内存，暴露物理指针给远端写入
        self.embeddings_engine.register(embeddings.data_ptr(), embeddings.nbytes)
        self.embeddings_buffer[req_id] = embeddings
        return embeddings.data_ptr()

    async def _recv_mm_data(self, req_id, ...):
        if self.encoder_transfer_backend == "mooncake":
            raw_buffer = self.embeddings_buffer.pop(req_id)
            self.embeddings_engine.deregister(raw_buffer.data_ptr())
            # view + reshape：零拷贝转换为 PyTorch Tensor
            embedding = raw_buffer[offset:offset+size].view(self.dtype).reshape(shape)
```

Encoder 完成计算后，RDMA 网卡直接将 Embedding 数据写入 Decoder 侧预注册的物理内存。本地只需一次 `.view(dtype).reshape(shape)`（纯元数据操作，零数据拷贝）就得到可用的 PyTorch Tensor。传统的 ZMQ 接收需要将数据从 Socket Buffer 复制到 Python Buffer，而 Mooncake 后端通过 `register(embeddings.data_ptr())` 直接将一块 Pinned Memory 交给远端网卡，使得大体积的视觉特征传输实现了 CPU 零干预（Zero-Copy）。

![PD 分离下的数据流全景](/assets/img/posts/sglang-hicache/fig-7-1-pd-separation.svg)

---

<a id="ch10"></a>
## 10. MLA 的 latent KV 与 HiCache 适配

> DeepSeek 提出的 Multi-head Latent Attention (MLA) 改变了 KV Cache 结构。本节讨论它如何在 HiCache 系统中进行适配。

### 为什么 MLA 需要特殊适配

HiCache 的核心假设是"每个 TP rank 持有各自不同的 KV 切片"——MHA/GQA 下第 i 个 rank 只存第 i 组 head 的 K/V，各 rank 之间数据互不相同，因此必须各自独立备份到 L2/L3。

MLA 打破了这个假设。它不再存 K/V 的 per-head 切片，而是存一个所有 head 共享的低维 latent 向量 `c_kv`。这个 `c_kv` 在投影阶段就已经计算完成，**不随 TP 切分而变化**——每个 rank 的 KV Cache 内容完全相同。

如果 HiCache 不做适配，会导致 N 个 rank 各自写一份相同的数据到 L3，白白浪费 (N-1) 倍 Storage 带宽。同时 `MLATokenToKVPool` 的 buffer shape 也与 MHA 不同（只有 1 个 head 维度），Host Pool 必须同步调整才能保证 `cudaMemcpyAsync` 的逐页拷贝正确对齐。

### 10.1 MLA 的核心思想

传统 MHA（Multi-Head Attention）中，每个 head 独立计算并存储 K 和 V 向量。对于 `num_kv_heads=128`、`head_dim=128` 的模型，每个 token 的 KV Cache 大小为 `2 * 128 * 128 = 32K` 个元素——这是显存消耗的主要来源。

MLA 的做法是：不再存储完整的 K/V 向量，而是将它们投影到一个低秩的 latent 空间。具体来说：

```python
# 传统 MHA：存完整的 K, V
# 每个 token 缓存: 2 * num_kv_heads * head_dim 个元素
k_cache[token] = W_k @ hidden_state   # shape: (num_kv_heads, head_dim)
v_cache[token] = W_v @ hidden_state   # shape: (num_kv_heads, head_dim)

# MLA：只存压缩后的 latent 向量
# 每个 token 缓存: kv_lora_rank 个元素（通常 512）
c_kv[token] = W_compress @ hidden_state  # shape: (kv_lora_rank,)
# 推理时再解压
k = W_uk @ c_kv  # 恢复 K
v = W_uv @ c_kv  # 恢复 V
```

`kv_lora_rank` 通常为 512，远小于 `2 * num_kv_heads * head_dim`（32768）。这意味着 MLA 模型的 KV Cache 体积可能缩小 **60 倍以上**。

### 10.2 对 HiCache 的影响

MLA 的这种根本性变化在 HiCache 链路中引发了四个适配点：

**适配点 1：Page shape 大幅缩小**

```python
# MHA/GQA:
page_size_bytes = page_size * (num_kv_heads // tp_size) * head_dim * dtype_size * 2  # K+V

# MLA:
page_size_bytes = page_size * kv_lora_rank * dtype_size  # 只有一个 latent 向量
```

单 page 体积的缩小增加了 Host Pool 存放 page 的数量，并且按比例减少了 L3 的读写时长。

**适配点 2：所有 TP rank 数据完全相同**

MHA/GQA 中，不同 rank 存各自负责的 head 切片，数据互不相同。但 MLA 的 latent 向量 `c_kv` 是在投影之前计算的——所有 rank 看到的都是同一份压缩结果。这导致：

```python
# srt/managers/cache_controller.py 中的逻辑
self.storage_config = self._generate_storage_config(...)

# 对于 MLA 模型，只有 rank 0 需要去备份 KV Cache 到 L3
# 因为所有 rank 的 latent KV 数据是完全一样的
self.backup_skip = (
    self.storage_config.is_mla_model
    and self.storage_config.tp_rank != 0
)
```

`backup_skip=True` 让只有 rank 0 执行 backup_thread 的写入操作，其余 rank 直接 ACK。这避免了 N 个 rank 重复写同一份数据到 L3 造成 N 倍带宽浪费。

**适配点 3：Hash 去重命中率极高**

由于全部 rank 输出的 latent 向量一致，它们的 content hash 也完全一样。在 L3 中只需保留一份记录，从而提高了存储利用率。

**适配点 4：跨机 Prefetch 简化**

PD 分离场景下，Decode 节点拉取 KV Cache 时：MHA 模型的每个 rank 需要用各自的 hash 去 L3 拉各自的切片（`all_reduce_min` 保证一致性），而 MLA 模型所有 rank 用同一个 hash 拉同一份数据——prefetch_thread 的逻辑更简单，`all_reduce_min` 的结果天然一致。

### 10.3 TP 一致性保证

尽管 MLA 简化了很多逻辑，TP 一致性检查仍然保留：

```python
# writing_check 中的 all_reduce 不因 MLA 而省略
finish_count = all_reduce_min(finish_count)
# 原因：即使数据相同，DMA 完成时间可能因硬件差异而不同
# 必须等所有 rank 都确认完成，才能安全解锁节点
```

这不仅是应对 DMA 速度差异的防御性设计，更是为了**保证控制面状态机在所有 Rank 上的严格同步**。在 Tensor Parallelism 下，所有 Rank 的调度器必须保持 100% 镜像一致。如果某个 Rank 提前释放了节点并将其分配给新请求，而其他 Rank 还没释放，就会导致整个集群的显存池映射出现致命分叉。因此，哪怕部分 Rank 已经提前完成，也要强制等最慢的那个，确保所有的 `TreeNode` 元数据（如 `lock_ref`）在所有卡上严格同频变化。

---

