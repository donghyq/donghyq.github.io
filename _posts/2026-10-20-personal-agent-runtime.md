---
title: "Personal Agent Runtime：Muse 与 OpenAI Dots"
date: 2026-10-20 12:00:00 +0800
categories: [LLM Infra, Agent]
tags: [personal-agent, agent-runtime, cloud-computer, durable-execution, memory, security, kv-cache]
description: Muse 与 OpenAI Dots 的云端执行环境、任务调度、Memory、工具授权、故障恢复与推理基础设施。
---

## 目录

**Part I — Runtime 架构与部署**

1. [Personal Agent Runtime 的设计需求](#ch1)
2. [Muse Secure VM](#ch2)
3. [OpenAI Dots 的云电脑与任务系统](#ch3)

**Part II — 任务与数据**

4. [后台任务的调度与恢复](#ch4)
5. [远程电脑与工具执行](#ch5)
6. [文件、Memory 与 Context](#ch6)

**Part III — 执行安全与推理基础设施**

7. [权限控制与网络出口](#ch7)
8. [外部操作的故障恢复](#ch8)
9. [长任务的推理与 KV Cache](#ch9)
10. [执行轨迹与 Agentic RL](#ch10)
11. [部署成本与后续工程问题](#ch11)

---

2026 年 9 月，Meta 发布了 Muse；随后 OpenAI 发布了 Dots。两者都提供云端计算环境，能够连接外部应用，在用户离开对话界面后继续工作。Muse 将工作环境部署在用户专属 VM 中，Dots 提供属于每个 dot 的云电脑，并支持用户授权连接本地电脑。在这类产品中，任务会跨越多次会话持续存在；执行进程、工作空间和模型请求则各有自己的生命周期。后台调度、数据持久化与外部操作恢复也因此成为 Agent Runtime（智能体执行环境）的组成部分。[1][2][3]

<a id="ch1"></a>
## 1. Personal Agent Runtime 的设计需求

### 1.1 从会话请求到持续任务

典型的 C 端会话助手有一条成熟链路：请求进入网关，读取历史消息，组织 Context（模型上下文），执行模型生成与工具调用，然后把消息和结果写回存储。会话记录可以长期保存，代码执行也可以使用独立沙箱。只要用户主要在当前会话内等待结果，执行环境通常可以按请求或任务创建和回收。

Personal Agent 的任务可能被安排在明天，或在等待用户确认、外部服务返回、定时触发后继续执行。Dots 的 Scheduled、In progress、Completed 任务视图和主动研究能力已经体现了这些要求。Muse 则直接把工作空间、浏览器、工具和后台任务放进用户云电脑。[1][3]

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-01-runtime-patterns.mobile.svg"><img loading="lazy" alt="典型会话助手与 Personal Agent 的部署关系" src="/assets/img/posts/personal-agent-runtime/fig-01-runtime-patterns.svg"></picture>

持续任务需要保存执行状态与唤醒条件。比如一个定时任务凌晨读取文件并生成报告，报告已经写入工作空间，发送前还需要用户审批。任务此时进入等待状态并释放 Worker；审批事件到达后，调度器重新入队，由可用的执行节点继续处理。

调度器持久化当前步骤、下一次唤醒时间、产物引用、待审批动作及版本。浏览器进程和模型请求可按需创建和回收，任务状态与工作文件持续保留。

### 1.2 云电脑的职责

云电脑为 Agent 提供文件系统、浏览器、程序执行、依赖安装和持久工作目录。同一份报告、脚本或工程项目可以被后续任务继续使用；遇到没有专用 API 的操作，也可以借助 CLI（命令行接口）和浏览器完成。

这也增加了环境管理成本：用户数据需要隔离，工作卷需要备份，进程和浏览器需要配额，依赖升级不能破坏已有任务。Muse 和 Dots 都提供云电脑能力，下面分别看其已公开的部署方式。

<a id="ch2"></a>
## 2. Muse Secure VM

### 2.1 用户 VM 与 Runtime Cell

Muse 官方明确采用每用户独立的 Linux VM。VM 内有 Chromium 浏览器、CPU、内存、存储和运行工具所需的环境，允许创建 Skill、运行程序、调度 cron 任务及多个子 Agent。客户端经安全通道连接该 VM。VM 是用户数据的重要存储位置，推理与遥测则通过受约束的代理访问外部基础设施。[2]

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-02-muse-vm.mobile.svg"><img loading="lazy" alt="Muse VM 的内部组件与安全域" src="/assets/img/posts/personal-agent-runtime/fig-02-muse-vm.svg"></picture>

模型驱动的执行框架叫 `Hatch`。Hatch、工作空间和可执行工具运行在 `systemd-nspawn` 创建的 runtime cell（隔离执行单元）里。安全相关服务作为单独的 systemd unit 运行在 cell 外，但仍处于该用户 VM 内。官方给出的主要组件包括：

| 组件 | 位置 | 作用 |
|---|---|---|
| `Hatch` | Runtime cell | Agent Loop、工具与程序执行 |
| `privsep` | Cell 外 | 以受限权限执行内置连接器 |
| `hatch-authd` | Cell 外 | 保存 OAuth 等凭据，提供凭据替身 |
| Sentinel | Cell 外 | 决定连接器动作和网络外发是否允许 |
| `hatch-safety` | Cell 外 | 对模型输入、输出做独立安全检查 |
| PostgreSQL | Cell 外 | 保存持久的应用状态 |
| Inference / Telemetry Proxy | Cell 外 | 向外部推理与观测系统提供受约束的通道 |

模型驱动的工具代码在 cell 内运行，真实凭据与授权策略由 cell 外的服务管理。cell 内部的 root 通过 User Namespace（用户命名空间）映射到外层普通 UID，同时限制 capabilities（内核能力位）与部分系统调用，例如 `CAP_SYS_PTRACE`、`CAP_NET_ADMIN` 和 `io_uring`。[2]

在 Linux 中，可以通过 `/proc/<pid>/uid_map` 查看 UID 映射：

```bash
cat /proc/self/uid_map
cat /proc/self/gid_map
readlink /proc/self/ns/user
```

例如 cell 内的 UID 0 映射到 VM 外层的普通用户，它能够管理 cell 中有权限访问的文件，但不会获得 VM 外层 root 的全部能力。这里还要叠加文件系统、cgroup、网络和系统调用限制。隔离单元处理不可信网页时，后续连接器或网络操作仍需跨越受控接口。

### 2.2 VM 与推理服务的分工

Muse 的用户 VM 通过推理代理访问外部模型服务。工作空间和工具留在 VM 内，Prefill、Decode 与 GPU 调度由推理设施处理；后台等待期间，任务数据保留在工作空间，后续推理仍通过代理发起。

用户 VM 与 GPU 资源独立管理，长时间等待的任务不会持续占用一次模型生成请求。

<a id="ch3"></a>
## 3. OpenAI Dots 的云电脑与任务系统

### 3.1 Cloud Computer 与本地电脑

Dots 使用 GPT-6 Astra，每个 dot 有自己的 Cloud Computer（云电脑）和浏览器，能使用用户连接的插件完成任务。用户可进入电脑查看操作过程，也可以 Take over（接管）后再交还控制。对云端网站的登录状态独立于用户本机浏览器；打开用户本地电脑访问需要在桌面端另外授权。[3][4]

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-03-dots-computer.mobile.svg"><img loading="lazy" alt="Dots 云电脑、连接器与本地设备" src="/assets/img/posts/personal-agent-runtime/fig-03-dots-computer.svg"></picture>

Dots 可以使用云端浏览器、Shell 和插件，也可以在桌面端授权后连接本地电脑，创建使用本地文件、浏览器或 Skills 的独立任务。云电脑、本地电脑、云浏览器、云网络和密码管理器分别受对应权限控制。[4][5]

Dots 对外提供可持续使用的云电脑；实现上可以采用常驻实例、休眠唤醒或实例池与持久工作卷结合。几种方案在冷启动、隔离粒度和空闲成本上存在明显差异，Dots 尚未披露其实际调度与存储布局。

### 3.2 多渠道交互与任务状态

Dots 有持续的工作身份，用户可以从 ChatGPT、部分消息平台等入口给同一个 dot 交付目标；任务工作情况通过 In progress、Scheduled、Completed 等视图查询。任务能够在用户没有发送新消息时进行计划中的检查或主动研究。[3][4]

任务入口和任务执行需要分开：移动端给出指令后，后台任务继续运行；Slack 中的后续消息也可能改变现有任务方向。系统将新消息关联到任务或工作项，再决定恢复、取消、追加输入或发起新任务。

Dots 对 Proactive Research（主动研究）有一个特别明确的限制：后台主动研究使用只读的连接信息，不能直接发消息、修改应用内容或控制浏览器及电脑。用户明确交办的任务可以执行更广泛的操作，但仍受对应权限和审批控制。[3][6]

### 3.3 Muse 与 Dots 的 Runtime 能力对照

| 部分 | Muse | Dots |
|---|---|---|
| 远程环境 | 每用户独立 Linux VM | 每个 dot 拥有云电脑 |
| Agent Runtime | `Hatch` 与 runtime cell | 任务执行、委派和后台工作 |
| 浏览器 | Chromium、CDP Broker、受限操作接口 | 云端浏览器，可查看或接管 |
| 外部连接 | 内置连接器、CLI、可自建工具 | 插件、浏览器、可选本地电脑及 Codex 任务 |
| 持久状态 | PostgreSQL 与工作空间文件位于用户 VM | Task、Memory 与文件持久使用 |
| 授权 | Sentinel、`authd`、`privsep`、出口控制 | 插件权限、Custom Rules、Auto-review、安全监控 |
| 主动工作 | 后台任务、目标跟踪、定时触发 | Scheduled Tasks、Proactive Research |

Muse 公布了 VM 内部的隔离与受控调用路径；Dots 公布了云电脑、任务和用户操作接口。下面采用 Temporal 和 LangGraph 的公开机制分析持久任务的调度与恢复。

<a id="ch4"></a>
## 4. 后台任务的调度与恢复

### 4.1 Task 与 Run 的数据模型

设想一个用户安排了每天上午检查项目动态、必要时修改文档的任务。调度器首先保存计划；时间到达后创建本次 Run（执行实例）；Worker 获得租约，恢复上下文并调用 Agent Runtime。任务运行到外部审批时，Worker 释放执行资源。收到审批事件后，调度器把 Run 从 `WAITING_INPUT` 转回 `READY`，再由 Worker 接手执行。

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-04-task-state.mobile.svg"><img loading="lazy" alt="后台任务调度与恢复时序" src="/assets/img/posts/personal-agent-runtime/fig-04-task-state.svg"></picture>

任务调度器可以使用这样的 Run 记录：

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class AgentRun:
    run_id: str
    task_id: str                 # 同一计划任务可触发多个 Run
    owner_id: str
    status: Literal["READY", "RUNNING", "WAITING_INPUT", "COMPLETED", "FAILED"]
    generation: int             # 执行者代次
    state_ref: str              # 外置状态/产物引用
    lease_owner: str | None
    lease_until_ms: int | None
    next_wakeup_ms: int | None
```

`task_id` 标识用户安排的长期目标或定时计划，`run_id` 标识其中一次执行。多个 Run 可以共享工作空间文件，但各自记录运行结果、执行代次与外部操作。

下面用 PostgreSQL 的条件更新实现一次租约领取：

```sql
UPDATE agent_runs
SET lease_owner = :worker_id,
    lease_until = now() + interval '60 seconds',
    generation = generation + 1,
    status = 'RUNNING'
WHERE run_id = :run_id
  AND status IN ('READY', 'RUNNING')
  AND (lease_until IS NULL OR lease_until < now())
RETURNING generation, state_ref;
```

`generation` 递增后，旧 Worker 即使因为网络暂停而晚到，也不能直接覆盖新一代状态。所有结果提交应携带当前 generation，由权威存储做条件更新。只在 Worker 的内存里设置 `cancelled=True`，无法阻止已经发出的请求晚到写回。

### 4.2 定时器、回调与执行队列

Temporal 的 Workflow（工作流）维护持久历史。Timer 到期、Activity 完成或外部 Signal/Update 到达后，系统生成 Workflow Task，Worker 从历史记录重放执行状态。任务可以在没有常驻 Python/Java 进程的情况下持续等待。[7]

Personal Agent 的调度层通常还要处理每用户并发限制、子 Agent 扇出、工具侧 QPS 配额和后台主动任务的优先级。一个用户下发十个任务，不能直接无限创建浏览器、终端进程和推理请求；调度器需要按当前就绪任务和资源容量安排执行。

这里的状态不只包含模型对话。Run 的等待条件可能来自一个消息、某个定时器、一个网页操作结果，或用户批准的外部动作。将这些事件记录在执行日志中，可以重建运行状态，也能定位重复执行从何处发生。

### 4.3 子 Agent 与任务依赖

Muse 支持并发子 Agent 与自建工具；Dots 支持将任务委派给 Codex 或本地电脑。这些任务涉及父子依赖、并发控制及结果汇聚。[2][4]

子任务独立检索资料、运行脚本或使用浏览器。父任务保留目标和依赖关系，子任务领取所需的数据与工具权限。结束后返回 Artifact 引用及必要的结果摘要，避免把全部网页内容、Shell 输出和工具调用日志逐轮灌回父 Agent 的 Context。

对于存在真实依赖的工作，可以维护 `parent_run_id`、`node_id`、`depends_on`、`artifact_ref` 和 `generation`。调度器在前驱完成后将节点改为 Ready；相互独立的节点允许并发运行。Worker 超时后，父任务可以重试受影响的节点或调整计划，但旧 Worker 必须通过代次校验才能提交结果。与其始终维持一张庞大的静态 DAG，个人助手更可能需要把短期任务计划和长生命周期任务事件结合起来：用户更改目标、外部工具返回失败或审批延期，都会影响后续节点。

并发限制应在父任务和所有子任务之间累计。一个父任务允许 4 路检索，但派生出的每个 Worker 又各自发起 4 路请求时，实际工具扇出会达到 16；模型侧的 token 预算、浏览器并发和外部 API 限流都应按整条任务树计量。

<a id="ch5"></a>
## 5. 远程电脑与工具执行

### 5.1 工具调用和电脑操作

Personal Agent 能通过连接器调用 API，也能通过浏览器和 Shell 完成没有合适 API 的任务。API 和电脑操作具有不同的失败恢复方式。

连接器一般拥有明确的操作类型、请求参数和服务端返回值，适合做权限校验、重试、幂等。浏览器操作依赖页面状态，输入框、登录态和导航过程可能被页面变更影响；“点击成功”和“业务动作已成功提交”也可能不是同一个时刻。Shell 提供更广的执行能力，适合编译、转换文件和运行自定义工具，代价是必须限制可访问的文件、进程与网络。

Muse 支持内置连接器与自建 CLI。内置连接器的代理代码运行在 cell 中，由受限的 `privsep` worker 执行实际的有凭据调用。Dots 通过插件、云端浏览器和可选本地电脑操作外部环境，用户可以观察或接管执行过程。[2][4]

### 5.2 环境持久化与资源复用

远程工作环境里有文件、工具依赖和浏览器 Session，长任务可能依赖先前写下的脚本与配置。持久磁盘能保存很多内容，但操作系统进程、浏览器连接、内存对象不能直接通过“磁盘还在”保证恢复。

部署上可以将用户工作卷与执行实例分开：启动时挂载持久工作卷，任务结束后回收实例；需要持续交互的任务再保留长寿命执行环境。Muse 采用专属 VM，兼顾用户隔离和持久工作目录。

文件产物最好保留独立引用与内容摘要。下面是一个应用侧的 Artifact Manifest（产物清单）示例：

```json
{
  "artifact_id": "report-2026-10",
  "revision": 3,
  "uri": "workspace://reports/october.pdf",
  "sha256": "<sha256-of-file-bytes>",
  "producer_run_id": "run-045",
  "created_at": "2026-10-20T10:30:00Z"
}
```

模型引用报告时，Context 保存 `artifact_id/revision` 和摘要。执行层按照 Manifest 读取指定版本的文件进行处理或发送。文件修改后生成新 revision，已有审批和依赖关系仍指向旧版本，便于恢复和审计。

<a id="ch6"></a>
## 6. 文件、Memory 与 Context

### 6.1 个人记忆的读写

Muse 允许用户查看、编辑和下载工作空间文件，包括自己的记忆。Dots 从 ChatGPT Memory 获取初始上下文，也可以从连接的应用和日常工作中积累记忆；断开插件连接不会自动删除此前形成的记忆。[2][4]

长期任务使用的 Memory（记忆）包括稳定的用户偏好，以及项目或任务进度。用户稳定偏好来自对话和应用信息，需要经过抽取、合并、时效判断；任务记忆则是某个项目已完成什么、有哪些文件、哪些决策已被用户确认。前者有跨任务价值，后者更依赖任务身份与版本。两者可以存储在同一个后端，但应以不同生命周期管理。

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-05-state-memory.mobile.svg"><img loading="lazy" alt="持久状态、记忆与模型上下文的关系" src="/assets/img/posts/personal-agent-runtime/fig-05-state-memory.svg"></picture>

原始事件完整保存用户输入、工具调用和外部观察；本轮 Context 从这些记录、文件和 Memory 中抽取所需内容。摘要和检索索引需要记录来源、生成版本及可见权限。长任务累计的工具输出可能相当庞大，在线生成通过摘要与按需回读控制 Prompt 长度。

### 6.2 检索方式按数据形态选择

比如用户问“上次生成的会议摘要放在哪里”，稳定的任务或文件 ID 查询通常比向量检索更直接；问“之前关于差旅预算的决定是什么”，则可能需要关键词、时间线与语义检索；在大量邮件和文档中查找相关内容，适合 Hybrid Retrieval（混合检索），再回读原始文件片段。

个人记忆检索通常先按 `owner_id`、`memory_type` 和时间过滤，再做向量 Top-K。过滤后候选只有几十条时，CPU 检索就可能满足时延要求；共享文档库、多项目并发和多向量检索则更容易形成 GPU ANN（近似最近邻）吞吐需求。后端选型要测过滤后候选规模、并发和访存成本。

### 6.3 更新与失效

一个用户改掉偏好，或撤销某项应用权限时，后续 Context 应使用更新后的事实。原始数据、搜索索引、摘要、已生成的报告和推理缓存可能处在不同存储层，失效时需要维护数据来源关系。

实现上可以先在权威状态中更新可见性，再异步回收派生索引和物理副本；正在执行的任务在关键写入或外发前，重新确认所用数据和权限版本。代次可以帮助屏蔽旧派生结果，但不能替代底层删除或凭据撤销。长期 Memory 的数据清理还涉及备份和第三方服务副本，应单独定义保留期与删除行为。

<a id="ch7"></a>
## 7. 权限控制与网络出口

### 7.1 Muse 的连接器执行路径

Muse 将执行模型指令的 cell 与权限、凭据服务隔开。内置连接器的 cell 内 CLI 负责解析参数、打开调用方有权访问的文件，然后通过 Unix Domain Socket（Unix 域套接字）传给受限 `privsep` worker。服务端可以用 `SO_PEERCRED` 取得内核认证的对端身份，使用 `SCM_RIGHTS` 传递已打开的文件描述符。[2]

```python
# Linux Socket API 片段：服务端取得对端 PID / UID / GID
import socket
import struct

n = struct.calcsize("3i")
raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, n)
pid, uid, gid = struct.unpack("3i", raw)
```

`hatch-authd` 保管真实 OAuth 凭据，cell 中的代码只接触替身 Token。Sentinel 是连接器与网络出口的授权主体，在具体请求获得允许后，经受控代理插入真实凭据。Muse 的网络出口检查 L4/L7 请求细节，还使用 eBPF 与 LSM 钩子追踪进程是否接触过用户数据，用作外发审批信号。浏览器则由 cell 外的 CDP Broker 管理，Browser Sub-agent（浏览器子智能体）只能通过受限操作接口处理无障碍树快照。[2]

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-06-authorization.mobile.svg"><img loading="lazy" alt="Muse 与 Dots 的操作控制路径" src="/assets/img/posts/personal-agent-runtime/fig-06-authorization.svg"></picture>

权限检查覆盖调用方身份、工具能力与具体动作。以邮件连接器为例，服务端先依据进程身份确认调用来源，再检查所用账户和读取、发送等操作的授权范围。

### 7.2 Dots 的审批与 Auto-review

Dots 的权限控制包括插件权限、Custom Rules（自定义规则）、Auto-review（自动操作审查）和安全监控。用户可设置无需再次确认、需事先授权或需接管的操作。Auto-review 在部分动作执行前检查指令、规则和安全要求；发送邮件时会检查收件人和内容。核心安全检查仍独立于用户自定义规则。[3][6]

Dots 的云电脑与本地电脑访问分开授权；支持的网站登录可由用户接管，凭据不会直接暴露给模型。对于主动研究，工具权限更窄，后台主动发现信息本身不能直接触发浏览器操作、修改插件数据或发送消息。[4][6]

Muse 将进程身份、连接器和出口检查串成受控调用链；Dots 在应用权限基础上使用 Custom Rules 和 Auto-review 控制操作。

<a id="ch8"></a>
## 8. 外部操作的故障恢复

### 8.1 外部提交和本地记录之间的窗口

长期任务可能调用第三方 API 发送消息、创建日历事件或提交采购请求。执行进程在调用前后都可能崩溃。即使已经把任务状态持久化到数据库，仍会出现远端完成操作、本地尚未记录成功的窗口。

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-07-action-recovery.mobile.svg"><img loading="lazy" alt="远端副作用提交与本地状态的恢复路径" src="/assets/img/posts/personal-agent-runtime/fig-07-action-recovery.svg"></picture>

这时直接重试可能造成重复外部效果。恢复逻辑取决于目标服务是否提供幂等键（Idempotency Key，服务端去重标识）和可查询的业务操作记录。控制端可以保存下面这份动作记录：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ActionRequest:
    action_id: str          # 首次创建时固定，重试时复用
    run_id: str
    operation: str
    arguments_digest: str   # 请求参数及文件版本的规范化摘要
    approval_ref: str | None
```

如果外部服务支持服务端幂等，应在所有重试中复用同一 `action_id`。如果只支持查询，应先找到能够证明已完成的远端结果再补记本地状态。查询不到可能源于延迟可见，不能自动推断原操作失败。如果两种能力都没有，待恢复动作需要允许停在 `UNKNOWN`，等待人或其他证据介入。

### 8.2 故障注入实验

实验使用两份 SQLite 数据库模拟 Agent 状态与外部服务，在远端提交前后用 `os._exit()` 结束子进程，再启动新进程恢复。普通重试在“远端已提交、本地未记录”场景产生两次外部效果；引入服务端幂等键后只产生一次；无法确认结果时，动作保持 `UNKNOWN`，等待后续对账。

```text
控制端                     外部服务
  |                           |
  |-- send(action_id=A) ----->|
  |                           |-- commit
  X  进程崩溃                |
  |                           |
  |-- retry(action_id=A) ---->|
  |                           |-- 幂等命中/返回既有结果
  |<--------- result ---------|
  |-- 标记 COMPLETED
```

LangGraph 的 `interrupt()` 恢复时会重新执行中断节点，中断前的外部操作因而需要幂等处理。Temporal 通过事件历史恢复 Workflow，Activity 重试同样要考虑外部系统是否支持去重。[7][8]

### 8.3 长期任务的操作记录

用户中途改变任务目标后，旧 Worker 可能继续运行。因此还需要在控制数据库中维护 generation（执行代次），让旧 Worker 的结果写回失败。代次只能保护检查它的接收端；第三方邮件接口不知道你的 `generation`，无法仅凭它消除外部重复发送。

恢复验收需要同时检查 Run 状态和外部系统结果：文档是否重复修改、邮件是否重复发送、实际执行的参数是否仍在用户授权范围内。

<a id="ch9"></a>
## 9. 长任务的推理与 KV Cache

### 9.1 云电脑和推理集群的资源周期

Muse 的用户 VM 和 Dots 的云电脑保存工作空间，模型推理使用 GPU 资源。任务可能持续数天，但 GPU 只在模型调用期间运行；工具 I/O、审批等待、执行环境准备与推理应分别计时。

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-08-kv-lifecycle.mobile.svg"><img loading="lazy" alt="任务生命周期、计算环境和 KV 缓存" src="/assets/img/posts/personal-agent-runtime/fig-08-kv-lifecycle.svg"></picture>

分离式架构中，Agent Runtime 负责组装 Prompt，Inference Server（推理服务）管理权重、调度队列、KV Cache 和批处理。任务结束一轮推理后，模型推理资源可以释放；如果后续仍使用完全相同的前缀，才可能通过 Prefix Cache（前缀缓存）或远端恢复减少 Prefill（预填充）成本。

Muse 官方提到其模型专门训练了 CLI/Skills 工具使用、长上下文、长轨迹指令遵循和多 Agent 协调能力；Dots 官方确认其核心模型为 GPT-6 Astra。这些属于模型和 Agent Runtime 的协同能力。模型能否规划并调用工具，不决定工具进程一定要和推理权重部署在同一台机器上。[2][3]

### 9.2 长上下文与 KV 容量

以 32 层、8 个 KV Head、Head Dim 为 128、BF16 存储的普通注意力模型为例，每个 Token 的 KV 载荷为：

<div class="formula">2 (K + V) × 32 layers × 8 KV heads × 128 dim × 2 bytes = <strong>128 KiB / token</strong></div>

16K Token 的单个前缀约占 2 GiB KV 载荷；实际显存还受分页、混合注意力和 TP 切分影响。如果大量 Agent 在等待审批时继续占用 HBM（高带宽显存），活跃请求的并发容量就会下降。

SGLang HiCache 将 KV Cache 分层保存在 GPU、Host 和 Storage，利用前缀索引与异步搬运完成下沉和恢复。对于长工具等待，选择 Host/L3 恢复还是直接重算，需要综合前缀长度、排队时间、RDMA/H2D、Install、同步成本及恢复命中率。[9]

### 9.3 Context 改写会影响复用

Personal Agent 的输入包括角色与工具契约、短期消息、Memory、文件片段和工具结果。摘要更新或工具定义顺序变化会改变后续 Token 序列，影响 Prefix Cache 命中。vLLM 通过连续块的 Hash 标识前缀，`get_computed_blocks()` 查找已计算块，`allocate_slots()` 保留命中的块并分配新块；Cache Salt 可限制不同共享域之间的缓存复用。[10]

推理系统需要检查前缀 Token、模型计算身份和缓存格式，再由可信网关控制跨用户的缓存共享域。即使恢复同一个任务，Memory 更新也可能改变 Prompt，此时只能命中没有变化的前缀块。

对于持续任务，调度器可以观测工具等待时长、恢复前缀长度和实例路由，再决定保留、下沉、预取或重算。记录时应区分 `queue_wait`、`storage_lookup`、`transfer`、`host_stage`、`H2D`、`install` 和 `scheduler_ready`：远端存储读得快，仍可能因为等待或安装开销拉长 TTFT（首 Token 延迟）。另外要记录 HBM 驻留量对其他请求吞吐的影响，比较本地缓存、远端恢复和直接重算三种路径。

等待结束后，任务从持久状态续接；推理服务根据重新组装的 Prompt 查找缓存，必要时重新计算 Prefill。

<a id="ch10"></a>
## 10. 执行轨迹与 Agentic RL

### 10.1 运行日志怎样转成训练数据

Muse 说明，推理对话和工具/子 Agent 轨迹经个人身份信息清理后可用于模型训练，用户可以选择退出。Dots 的个人计划由模型改进开关控制符合条件的对话与工作数据是否用于训练，主动后台研究及其笔记不被直接用于训练。[2][6]

活动日志记录任务执行过程；Agentic RL 还需要保存模型实际采样的 Token 和 Logprob（对数概率）。模型输出、工具观察、浏览器读到的数据、用户审批和自动生成的摘要，在行为学习中有不同来源。

<picture class="architecture-figure"><source media="(max-width: 680px)" srcset="/assets/img/posts/personal-agent-runtime/fig-09-trajectory.mobile.svg"><img loading="lazy" alt="Agent 执行记录与模型采样轨迹" src="/assets/img/posts/personal-agent-runtime/fig-09-trajectory.svg"></picture>

以模型连续两次调用工具为例，训练数据需要保存每段采样实际可见的 Prompt Tokens、模型生成的 Response Tokens、对应概率与来源标记。工具结果进入下一段 Attention Context，Policy Loss 则只针对模型生成的动作计算。veRL Agent Loop 用 `prompt_ids`、`response_ids` 和 `response_mask` 表达这一关系。[11]

### 10.2 跨时间的策略与环境版本

长任务运行期间可能经历模型发布、Skill 更新、工具 Schema 变化和外部数据更新。训练侧需要核验采样策略版本、算法允许的 Staleness（陈旧度）、Token/Logprob 的对应关系及环境结果的可复现性；推理侧则依据模型身份、前缀和缓存格式判断历史 KV 是否可用。

真实用户数据可能需要脱敏。改写 Prompt 后继续沿用旧 Logprob，会让条件输入与采样概率错配。可选择保留经受控处理的原始采样记录用于校验，或针对脱敏后的新序列重新构造训练目标；这种边界与线上推理缓存兼容性属于不同问题。

### 10.3 评价长期任务

短问答可以检查最终内容，长期执行还要评价任务状态与外部系统状态。例如是否产生重复消息，日历事件是否写入正确账户，用户撤销权限后是否停止后续操作，失败恢复后是否重复执行。τ-bench 使用任务结束后的数据库状态评估工具 Agent，适合作为这类评测的公开起点。[12]

<a id="ch11"></a>
## 11. 部署成本与后续工程问题

Muse 为每位用户分配独立 VM，工作目录和进程隔离比较直观，空闲 VM 的内存、存储和管理开销也随用户数量增长。休眠能力和唤醒时间会直接影响成本。Dots 的实例调度和存储绑定方式尚未披露，公开资料也没有直接可比的单位用户资源数据。

调度层可以把资源需求拆开计量：任务和文件按存储量，定时任务按触发频率，执行节点按活跃并发，浏览器按实例并发与启动成本，推理服务按 Token 吞吐、显存和 KV 缓存命中情况。比较不同部署方案时，应选相同的活跃任务、文件大小和工具负载，而非简单用用户注册数计算单机容量。

Personal Agent Runtime 的容量评估要覆盖实例冷启动与唤醒耗时、长任务接续成功率、动态 Context 下的 KV 命中率，以及空闲计算环境的内存和存储占用。这几项指标直接影响长任务的响应时间与服务成本。

---

<a id="refs"></a>
## 参考资料

1. [Meta — How We Designed Muse](https://introducing.muse.ai/)（2026-09）
2. [Meta — How We Built Safety Into Muse](https://research.meta.ai/blog/security-and-safety-for-ai-agents-our-approach-with-muse)（2026-09-08）
3. [OpenAI — Introducing dots](https://openai.com/index/introducing-dots/)（2026-09-29）
4. [OpenAI Help — Getting started with your dot](https://help.openai.com/en/articles/20001530-getting-started-with-your-dot)（持续更新）
5. [OpenAI Help — Manage dots in ChatGPT workspaces](https://help.openai.com/en/articles/20001554-manage-dots-in-chatgpt-workspaces)（持续更新）
6. [OpenAI Help — Dots privacy, security, and safety FAQs](https://help.openai.com/en/articles/20001529-dots-privacy-security-and-safety-faqs)（持续更新）
7. [Temporal — Tasks / Workflow replay](https://docs.temporal.io/tasks)
8. [LangGraph — Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
9. [SGLang — HiCache Design](https://docs.sglang.io/docs/advanced_features/hicache_design)
10. [vLLM — Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
11. [veRL — Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
12. [τ-bench: A Benchmark for Tool-Agent-User Interaction](https://arxiv.org/abs/2406.12045)