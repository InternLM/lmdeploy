# Mooncake PD / Store 实施记录

## 已固定的约束

- 用户于 2026-11-11 授权 T0–T8 逐任务实现、验证和 commit。
- MTP 按 `vllm_pd_mtp.md` 的 prompt-tail replay：P 在 H=N−1 停止，D 从 S_H 补算最后一个真实 prompt token；P 的 sampled/draft token 不交接。
- PD 与 Store 独立持有资源。2026-11-18 用户确认 PD 采用有界隔离及 300 秒强制回收；取消不等于 DMA 完成，Store 持有不受 PD 超时影响。
- 不引入 MultiConnector，不修改 Store 的内容寻址身份。

## 2026-11-19 P 采样与缓存统计

P 原有逻辑已跳过 draft 采样和后续 proposal forward，本轮进一步跳过纯 PD 交接批次的 target lm-head/采样，保留必要 MTP KV 前向。参考 vLLM #54222，将 P 的真实缓存命中数随 handoff 传给 D，修复冷请求因导入 KV 而报告接近 100% 的问题；流式响应和请求级 Prometheus 统计一并验证。

H200 回归 **873 passed**；六个缓存场景前后 **384/384 输出 IDs 相同**，8K/32K 的 PD/直接推理对照、32 次取消、D 生成阶段 Store 保存并由 P 恢复均通过。充分预热后 TTFT 均值 664.28 → 656.35 ms，TPOT 基本不变，不能声称显著提速。详细证据和冷 graph 排除过程集中在 [验证报告](mooncake_pd_mtp_cache_usage.md)。

## 2026-11-18 Mooncake 0.3.14 升级

Docker 构建和安装阶段的 Mooncake pin 升至 0.3.14。PD 使用 grouped scatter ticket，区分明确 drained 与 unknown；未知 ticket 后台排空，迟到终态提前释放 PD 引用。整个 pool 注销同时检查运行中的 native Future 和未完成 ticket。保留请求级 300 秒回收，不自动重启 worker/TransferEngine。

实机部署还发现并修复迁移通知的空闲循环空转，避免 D 的推理及 health RPC 被饿死；独立提交 `a4dfe9a5`。新旧性能基线都包含此修复。

- 指定 H200 容器最终 GPU 回归 **897 passed**；真实 RDMA 字节校验、源进程退出后 `failed_drained`、同一接收端 TransferEngine 从新源恢复均通过。
- 32 次取消、D 在生成阶段保存新增 KV 并由 P 加载、native 连接拒绝后的请求恢复均通过；源端 grant 真实等待 **301.51 秒**后回收 8 个 block，无需重启。
- 8K/32K 六个逐 token 对照通过；129-token 的三例差异在新旧包中都存在，根因仍未解决。新版 GSM8K **99/100** 为冒烟结果。
- 传输及端到端性能存在波动，不能声称全面提速。完整分阶段数据、暖机口径、版本依据和验证边界集中在 [升级报告](mooncake_0314_pd_upgrade.md)，不在本进展文档重复展开。

## 2026-11-18 请求级失败恢复

本节更新了后文历史测试使用的“unknown 后封锁 engine、必须重启”策略。用户确认优先保持线上可用性：PD 失败只隔离对应事务，完成后提前释放，隔离满 300 秒强制回收；不默认重启 worker、TransferEngine 或整套缓存。

- D 的 native unknown、取消、源端 abort、续租失败及 180 秒传输等待超时进入有界隔离。第一次失败固定期限，重复取消不延期。已失败请求不会因迟到成功结果重新开始生成。
- P 的 grant 每 30 秒由 D 续租；最后一次续租后 300 秒、或首次 revoke 后 300 秒回收。release/abort 有异步重试，P 放弃已领取交接时向 D 传递原因。
- 原生任务可比事务活得更久；超时回收后只消费其迟到结果，不再修改分页资源。排队和 native 提交前检查截止时间，避免过期任务在 block 复用后才开始 I/O。
- block、compact/legacy state 都只解除 PD 的引用；Store 或其他请求的独立持有继续生效。失败数据不发布到 prefix trie。
- 单个 PD 故障不再关闭整个接收引擎或 Mooncake backend。模型提交/取输出 180 秒无进展时 health 报不健康；正常空闲及单个隔离事务不触发该条件。未新增逐 rank 心跳服务，沿用已有 HTTP health 并检查模型执行进展。
- `/distserve/pd_status` 增加最老隔离年龄、累计强制回收次数；分阶段日志区分异步 release 与强制回收。

边界：300 秒强制回收不证明 DMA 已结束，保留极端迟到 DMA 访问已复用地址的风险；transfer ID/epoch 防止的是过期控制消息和重复释放。整个内存池的注销仍要求 native 完成，Store 自身 unknown 策略未改变。远程 executor 的提交截止时间使用 Unix 时间，要求主机时钟同步。

验证：指定 H200 容器两卡组合回归 **994 passed**，覆盖 PD、paging、compact state、sleep、MTP 输入和 Store scheduler；本机健康检查与 PD 组合 **36 passed**。本机扩大检查时无 NVIDIA 驱动的案例已转到上述 GPU 组合执行。补齐两个既有 sleep 测试构造对象缺少的 `engine_conn`，没有修改生产 sleep 行为。

真实 RDMA 检查覆盖 8K/32K 的连续、未合并、碎片六种布局，每种三次传输，完整目标字节和未写区域全部一致；源进程退出后返回 `completion_unknown`，整个 pool 的提前注销仍被拒绝。该检查验证传输及失败分类，不用于宣称超时强制复用不存在迟到写入风险。

GPU 测试使用指定容器中的独立 `/dev/shm/lmdeploy_pd_recovery_20261118` 副本，不替换现有安装或部署。原始输出保存在本机 `.git/pd_recovery_gpu_20261118.log`、`.git/pd_recovery_rdma_20261118.log`；完整 pre-commit 检查通过。本轮作为 T9 提交，未 push。

## T0 环境与基线

2026-11-11 实测，LMDeploy 起点 `cf52797ec6890fb6fe99aee8c4768d209d5a720b`。

- 指定服务器和 `lmdeploy_new_version_1106` 容器可用，8 × H200（每卡 143771 MiB），检查时 GPU 空闲。
- 容器 Python 3.12，PyTorch `2.12.1+cu130`，Mooncake `mooncake-transfer-engine-cuda13==0.3.12.post1`。
- 有 `batch_transfer_sync_read`，没有 drainable scatter ticket。0 返回值可用于成功完成；负返回值/异常不能证明 drain。此版本失败后隔离整个受影响接收事务并保持源端 grant，停止新迁移，须受控重启；不实现基于超时的物理回收。新版 ticket 能力在 transport 边界检测，允许明确的 failed-drained 恢复。
- 现有 disagg + cache-engine 测试：**22 passed**（5.20 s，容器，改前）。容器补装 pytest/pytest-asyncio。
- `benchmark/mooncake_pd_plan.py` 在指定容器测量 plan 构建，60 rows、64-token blocks、每 row/block 4096 bytes，30 次。32k 连续布局 30720 descriptors、120 MiB，p50 **79.91 ms**、max **91.26 ms**；8k 连续布局 7680 descriptors、30 MiB，p50 **11.19 ms**、max **52.93 ms**。32k 碎片布局 p50 **77.36 ms**、max **79.61 ms**。首轮脚本的排他分位数会外插，后续统一使用 inclusive p99。
- GLM 改前受 role guard 阻止 PD；不能声称有可运行的 GLM PD 改前基线。单体对照与实际部署结果见 T8。

Mooncake 完成契约参考 [官方 Python API](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/transfer-engine.md)。实际容器版本不假定拥有主线的新接口。

## 任务状态

| 任务 | 状态                                                                                                | 验证                                                                                                     |
| ---- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| T0   | 已固定环境、协议前提及计划构建基线                                                                  | 22 tests；计划构建基准                                                                                   |
| T1   | 完成交接描述、资源分组、执行 schema 和控制消息                                                      | 2 tests：容量/语义/旧 peer/replay 边界                                                                   |
| T2   | 完成完整 pool 注册、单 worker TE、批量 I/O、取消持有、all-rank 汇总                                 | 容器 26 passed；32k 连续 60 descriptors，p50 0.104 ms，p99 0.536 ms，max 0.696 ms；32k 碎片 p50 69.45 ms |
| T3   | 完成源端 pin、精确 state 输出、all-rank 发布和 grant/revoke/release                                 | 54 tests；真实 scheduler end-session 竞态与迟到 grant                                                    |
| T4   | 完成 KV-only prefix、私有接收、精确 state load、取消和 unknown 隔离                                 | 119 tests；新增 compact 成功恢复测试；失败/未知完成的真实 allocator 复用检查                             |
| T5   | 完成真实 prompt 截断、D replay、MTP 已知 lookahead 及 Store/local 身份                              | 指定容器 402 passed（含 GLM cache/Store 与 GPU 可用的 spec 测试）                                        |
| T6   | 已实现服务层 handoff、取消控制、runtime TP/DP endpoint 与 epoch 连接身份                            |                                                                                                          |
| T7   | 完成 Store native unknown 隔离、PD teardown 屏障、prepared expiry/epoch reset，开放 GLM Mooncake PD |                                                                                                          |
| T8   | 完成实机验证、部署修正和结果归档；未通过项明确保留                                                  | 完整 GSM8K、GPU 字节 oracle、AR/MTP、非对称拓扑、取消及 sleep/wakeup                                     |

## T6 服务与 router

两个 OpenAI endpoint 透传 handoff，支持无 KV/无 sampled token 的交接；MP engine 显式转发 PD 控制和运行时信息。新 router 要求 v2，按真实 DP HTTP 端口注册，连接键包含双方 incarnation，重复握手幂等。取消调用 revoke；已授权源端保持 pin 到 receiver release。

本地服务/PD suite：1611 passed、1 deselected（Python 3.10 下已有 import mock 问题）；未纳入用户未跟踪的 repetition-penalty 测试文件。新增测试覆盖空 replay handoff 和实际 attention TP。router 的 check/test/release build 结果记录于 T8。

## T7 Store 共存与生命周期

- Store native put/get 负返回、异常或非法结果视为 completion unknown：不发终态通知，不释放源 lease/接收目的地，不关闭注册对象；worker 后续 poll/teardown 报错，使实例退出服务。提交前的 lookup/地址验证失败仍可安全回收。
- PD prepared export 300 秒后可撤销；grant 无 TTL 回收。sleep/reset/shutdown 检查 source grant 和 receiver hold，MR 关闭先于 worker cache 释放，正常 reset 切换 epoch 并清除控制连接。
- 只为 Mooncake v2 开放 GLM PD；DCP 和 CPU cache 拒绝，D Store 要求 producer。
- 指定容器组合测试 **1232 passed**（9.38 s）。另有 35 个 guided-decoding 测试因容器未缓存其指定 Qwen tokenizer 无法收集，独立列为环境限制；不计入通过数。
- T6 router **550 passed**，release build 成功；router commit `91113ab`。T6 服务容器 **1612 passed**。

## T8 实机验证

### 实机发现与修正

1. 执行器中残留的 compact PD 禁用和 prefix 关闭条件，现对 Mooncake v2 开放；保留其他 backend 的限制。
2. DP launcher 原来把外部 GPU mask 替换为从 0 开始的编号，导致两端占用同组 GPU；现在从父进程 `CUDA_VISIBLE_DEVICES` 中分配。
3. compact prefill dummy 补齐 host 已知的 prefill metadata；P 的所有 DP ranks 只执行 MTP 前缀 materialization，跳过未分配 KV 空间的后续 proposal。
4. P 角色提前返回导致长请求 chunk history 不推进，32K 请求重复计算首 chunk；调整为先推进 chunk，再跳过 decode 队列。
5. N=1 的空前缀响应补齐请求 metrics。
6. MP worker 使用同步健康入口，因此将 quarantine 检查与资源计数放在该入口；新增只读 `/distserve/pd_status`。控制冲突使用可跨 MP 传递的结果，旧 epoch 返回 HTTP 409。
7. router 不再把任意 SSE body chunk 当作 DMA 完成证明。HTTP 结束/断开统一幂等 revoke，已授予的源资源仍等待 receiver release。router 收尾 commit `80fd9a5`，check、550 tests、release build 均成功。
8. 私有 KV 尾块阻止再次 prefix match 的条件不再依赖 Store lookup 开关。D producer 或 Store off 的 replay 同样保留精确迁入状态；补充 legacy/compact 两种真实 scheduler admission 回归用例。

### GLM 精度与请求性能

主测试使用 GLM-5.3-Flash snapshot `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`。P/D 各占 4 张 H200，各 DP=4、EP=4、attention TP=1，MTP=5，prefix 开启，P Store both / D producer；四个实际 HTTP 端口均以 round-robin 注册。Store embedded segment 每 worker 16 GiB，local buffer 512 MiB。

GSM8K 使用同一 evalscope 4-shot 模板、temperature=1、top_p=0.95、max_tokens=10240、stream、并发 16。直接推理对照使用同一版本 D 的四个端口、同一 Store producer 配置，绕过 P。不是逐 token 确定性实验。

| 配置                       | 准确率             | 样本数    | TTFT p50 / p99 / max (ms)    | TPOT p50 / p99 / max (ms) |
| -------------------------- | ------------------ | --------- | ---------------------------- | ------------------------- |
| PD + Store，MTP=5          | 1287/1319 = 97.57% | 1319      | 620.20 / 1807.04 / 4157.03   | 11.44 / 17.33 / 31.53     |
| 同版本直接推理，MTP=5      | 1282/1319 = 97.19% | 1319      | 208.14 / 524.25 / 706.94     | 9.37 / 12.30 / 14.37      |
| PD + Store，MTP off        | 98%                | 100 smoke | 混有取消压力，不用作性能对比 | 同左                      |
| 改前单体，Store off，MTP=5 | 99%                | 100 smoke | 不与完整集混算               | 同左                      |

PD 完整集共输出 337230 tokens，评估阶段约 395.58 s；直接对照约 298.79 s。该墙钟时间包含评估收尾，不等同于服务饱和吞吐。记录显示当前 PD 请求路径的 TTFT/TPOT 高于直接推理，不能声称端到端加速，也尚未证明 Store 并行对饱和 ITL 没有回归。迁移计划/提交优化的效果由下面的独立测量支持。

MTP on/off 的 8K、32K 首次、重复、部分前缀用例均与直接访问 D 的 16/32-token 输出一致。MTP=5 的 32K 实际传输为：首次 580082688 bytes、重复 148520704 bytes、部分命中 364723968 bytes，均 27 segments；对应 plan 0.290 / 0.094 / 0.173 ms，含排队的 transfer 30.762 / 5.790 / 13.518 ms。单次样本用于证明命中减少真实传输，不能当分布。

短重复 token 输入在部分 63/64/65/66 边界出现贪心输出差异；Qwen3-0.6B 在 32K 首次/重复两例也有差异。已保留为未解决的逐 token 对照差异，未将其直接归因于浮点误差。完整精度通过并不能证明这些输出逐 token 相同。测试脚本的 `first` 仅表示该轮首次请求，不负责清除已有 Store/local cache。

### 真实 GPU RDMA 与资源安全

`benchmark/mooncake_pd_transport.py` 用两个独立进程注册 target/draft/state/auxiliary 完整 GPU pool，源/目的 capacity 不同；校验迁移后全部目标字节及未写区域。连续、同一连续布局但不合并、碎片三种布局均通过，30 次测量：

| token 等效长度 / 布局 | segments | bytes     | plan p50 / p99 / max (ms)  | transfer p50 / p99 / max (ms) |
| --------------------- | -------- | --------- | -------------------------- | ----------------------------- |
| 8K 连续合并           | 68       | 34619392  | 0.109 / 0.208 / 0.211      | 0.990 / 8.176 / 10.998        |
| 8K 连续不合并         | 7942     | 34619392  | 10.533 / 63.396 / 69.478   | 11.492 / 12.793 / 13.133      |
| 8K 碎片               | 7942     | 34619392  | 10.432 / 56.959 / 60.088   | 11.542 / 11.993 / 12.101      |
| 32K 连续合并          | 68       | 132136960 | 0.221 / 0.307 / 0.323      | 3.320 / 3.680 / 3.806         |
| 32K 连续不合并        | 31750    | 132136960 | 84.236 / 95.363 / 95.710   | 44.283 / 48.984 / 50.306      |
| 32K 碎片              | 31750    | 132136960 | 88.936 / 107.330 / 107.970 | 44.750 / 46.515 / 46.875      |

这里是合成物理布局，不能把这些 bytes/层数说成 GLM 的布局。transfer 包含线程调度及 native 调用，未单独拆分队列等待。首轮连接开销包含在分布内。

源进程退出后再次真实 RDMA read 返回 -1，结果为 completion_unknown；接收端 quarantine，并拒绝 unregister。独立测试进程保持 allocation/TE 引用直到进程退出。没有用超时推断 DMA 已排空。Store unknown、一个 rank 未完成、block/state 复用和 Store/PD 双持有的精确交错由实际 scheduler/allocator 单测覆盖。

### 非对称拓扑、分块对照及重连

随机小型 Qwen3Next GDN 模型使用 BF16、hidden=2048、linear/full 各一层、8 experts、top-2、attention heads=8 / KV heads=2，linear K/V heads=2/4、head dim=64，conv width=4。模型配置及输出 IDs 随结果 JSON 保存；实机权重位于容器 `/dev/shm/lmdeploy_pd_tiny/gdn`。该随机模型用于结构与生命周期测试，不能作为真实模型精度指标。

- P 为 DP=2、EP=4、attention TP=2，两个实际 HTTP 端口；D 为 DP=1、EP=2、attention TP=2。GPU block capacity 分别为 65030、65043；prefix 与 Store 均开启，P both / D producer。跨 attention ranks 的迁移和不同 DP/EP 握手成功。
- 27 个逐 token 对照中 14 个一致、13 个不同；私有尾块修正后的 21 个边界/长输入复测仍有 13 个差异。64-token 的 63+1 直接分块、65-token 的 64+1 直接分块均完整复现 PD 的 16-token 输出；未分块直接推理与其不同。这两例差异可在不经过 PD 的分块路径复现。其余 GDN 差异仍未定位，不能据此宣布全矩阵精度通过。
- MHA 随机小模型（Store/prefix off）33/33 对照一致；真实 Qwen3-0.6B GQA（Store off / prefix on）31/33 一致，32K 首次及重复输入存在差异。
- 64 次早取消/收到首 token 后取消全部结束；P 两个 DP 和 D 的 exports、grants、receives、pinned blocks 均归零，无 quarantine。手动 claim 后 revoke 保持 1 个 export、1 个 grant、2 个 pinned blocks；旧 epoch release 被 409 拒绝，合法 release 后归零。
- P 两个 DP 和 D 并行执行 `/sleep?level=1` 与 `/wakeup`，均返回 200，三个 engine epoch 全部更新。router 自动重新握手后，再执行 64 次取消及 grant/revoke/release 测试，结果同样通过。

### 最终检查与交付边界

- T8 容器组合 suite：1243 passed；最后一次私有尾块修正后，完整 paging suite **512 passed**。两组有重叠，不相加。
- MP health/control RPC suite **14 passed**；服务层 suite **1606 passed**。未纳入用户未跟踪的 repetition-penalty 测试文件。
- router check、**550 tests**、release build 成功，已安装本次 release binary。代码已同步到指定容器，执行最终 `bash install.sh`；部署脚本与数值结果随 T8 提交。
- 未通过/未证明项：上述逐 token 差异；PD 请求 TTFT/TPOT 高于直接推理；尚未建立 Store 并行时的饱和 ITL/吞吐无回归证据；当前 Mooncake 无法实测新版 failed-drained ticket 分支。它们不会被 GSM8K 通过或合并段数改善替代。
- 本次启动的 GPU 服务、router 和 Store master 在测试后停止，日志和模型测试文件保留；没有清理用户的其他服务或文件。

| 任务 | LMDeploy commit                              |
| ---- | -------------------------------------------- |
| T0   | `9d52e9bb`                                   |
| T1   | `04691aba`                                   |
| T2   | `ef2e42cc`                                   |
| T3   | `f6ef3cc6`                                   |
| T4   | `8a200650`                                   |
| T5   | `91ec44d4`                                   |
| T6   | `104454ed`                                   |
| T7   | `9099e3bd`                                   |
| T8   | 本文所在的 GPU integration / validation 提交 |

Router 独立仓库对应 T6 `91113ab`、T8 `80fd9a5`。所有提交均仅包含本任务文件，未 push。

### 复现与环境

- `benchmark/mooncake_pd_glm.sh`：通过 MODEL_PATH、CUDA_VISIBLE_DEVICES、LMDEPLOY_DP_MASTER_PORT、MOONCAKE_CONFIG_PATH 配置两端；MTP_TOKENS=0/5，STORE=0/1，PREFIX_CACHE=0/1。
- `benchmark/mooncake_pd_router.sh`：P_URLS、D_URLS 逐一列出实际 DP HTTP 端口；运行多个 router 时传入不同 `--prometheus-port`。
- `benchmark/mooncake_pd_validate.py`：输出 token IDs、TTFT、TPOT、SSE chunk 时间，支持边界与长输入对照。
- `benchmark/mooncake_pd_cancel.py`：重复 `--p-url` / `--d-url` 监测所有 DP 实例；验证取消、grant+revoke 保持 pin、旧 epoch 拒绝及合法 release。
- 完整数值保留于 `benchmark/results/mooncake_pd_h200_20261111.json`；原始 evalscope 输出和日志在本地 `.git/pd_tmp/`，远端服务日志在容器 `/dev/shm/lmdeploy_pd_*.log`。
- 已执行 `bash install.sh`。备用 pip 索引超时后，在本次 shell 中取消其 extra-index 环境变量，使用脚本原主索引完成安装，未修改用户的 install.sh。安装依赖将 NCCL 从改前的 2.32.3 调整到仓库要求的 2.29.7；同版本完整集对照使用相同依赖。
- schema 校验配置与 cache 执行语义，不计算权重文件内容 hash。部署如需显式权重版本，可以用已有 `--hf-overrides '{"pd_weight_revision":"..."}'`，双方一致且进入 schema；仍应部署同一 checkpoint。

## 追加：分阶段诊断与多轮性能对照（2026-11-11）

### “逐 token 差异”的含义与已知范围

它表示相同输入、temperature=0 时，PD 与直接推理的输出 token ID 序列不完全相同，既不是 TTFT 波动，也不是简单的文本后处理差异。测试使用重复 token 构造边界输入，专门覆盖 64-token block 的尾部、部分命中和长上下文。它们不代表真实业务问题的答案错误率，但确实是尚未全部解决的一致性问题。

| 模型 / 输入                    | 第一个不同的生成 token（从 1 开始） | PD 输出开头            | 直接输出开头            |
| ------------------------------ | ----------------------------------- | ---------------------- | ----------------------- |
| GLM，MTP off，N=63，partial    | 2                                   | `[271,418,9970,47465]` | `[271,4418,803,271]`    |
| GLM，MTP off，N=64，首次       | 2                                   | `[271,4418,4398,271]`  | `[271,40,2776,537]`     |
| Qwen3-0.6B，N=32768，首次/重复 | 3                                   | `[100,17992,101,103]`  | `[100,17992,46580,102]` |
| 随机 GDN，N=64，首次           | 1                                   | `[38843,67534,17946]`  | `[148939,55612,39009]`  |

随机 GDN 的 N=64 / 65 用例，在直接推理的 63+1 / 64+1 分块路径中完整复现 PD 的 16-token 输出。因而这两例的分叉可以在无 RDMA 的条件下产生；其他差异仍需逐层 logits/state 对照定位。不能把所有差异统称为浮点误差，也不能由真实 DMA 字节校验或 GSM8K 97.57% 推导出逐 token 等价。

### 本次增加的观测点

启动 LMDeploy 与 router 时设置 `LMDEPLOY_PROFILE_PHASES=1`；LMDeploy 同时设置 `LMDEPLOY_LOG_PID=1`，保留多进程父子关系以关联 engine 与 HTTP session。

| 记录                                                                         | 解释                                                                         |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| router `connect / prefill_http / decode_headers`                             | 连接检查、P 完整 HTTP 请求、D 响应头；响应头不等于首 token                   |
| `http_prepared / http_first_token / http_complete`                           | 请求准备、流式首 token、非流式结束，关联同一 `X-Request-ID`                  |
| `engine_output`                                                              | 首次/结束输出，QUEUED/SCHEDULED、实际完成时间、cached tokens                 |
| `pd_receive`                                                                 | D 等待接收、claim、executor 往返、release、传输终态                          |
| `pd_native`                                                                  | 地址准备、实际 native 读取、段数和字节数                                     |
| `engine_step`                                                                | host prefetch 与等待 executor 的区间，可能和其他工作重叠                     |
| `worker_step`                                                                | CUDA event 测量的 stream 区间，以及 prefill/decode、real/dummy、batch/tokens |
| `store_save_scheduled / store_save_ready / store_save_finished / store_load` | 保存边界、lookup、已有 readiness event 等待、终态、加载边界与 state hash     |

CUDA event 通过 query 回收，未增加 device synchronize；Store 的 readiness synchronize 原本就存在。stream 时间包含 kernels、collectives 和 launch gap，不是 kernel-only 时间；各 rank 并行，不能把这些数相加当成请求延迟。profiling 默认关闭；本节性能数据均开启观测，未单独测量观测开销。

### 多轮工作负载与比较口径

使用用户指定的本机 `run_glm53_shorttask_382_full_kvcache.sh`，wrapper 仅覆盖 URL 和输出目录。固定数据集 `glm53_flash_swe_smith_32k_4k_5turn_n8_seed1124.json`；8 个会话、并发 8、每会话 5 轮、首轮 32768、后续增量 4096、每轮输出 512、seed=1124、temperature=0、ignore_eos=true。每次测量为 40 个业务请求、20480 个生成 token。

本机 evalscope 1.8.1 有定制：后续轮不追加 assistant 输出，只累积 user/tool 输入。因此它适合固定输入性能比较，但不能单独证明生成部分的 KV 被下一轮消费。`rate=8` 在这个实现中是各会话每轮的泊松等待参数，前一轮完成后才发下一轮，并非全局恒定 8 requests/s；本测试不是饱和吞吐曲线。

统一使用 GLM-5.3-Flash 同一 FP8 checkpoint，MTP=5，DP4/EP4/attention TP1，max batch=16，prefill chunk=8192，GPU cache fraction=0.7，session_len=102400，prefix state budget=16。两端均启用 `--prefix-cache-decode-save-first --prefix-cache-decode-state-interval 1024`、64 对齐及 prefill-last-only。Store 使用用户现有配置（embedded、每 worker global segment 160 GiB / local buffer 4 GiB、RDMA），只改变独立 master 地址。

主对照的总硬件相同：PD 为 4 卡 P + 4 卡 D；非 PD 为两组 4 卡 Hybrid（各 DP4/EP4），共 8 卡。两种模式均通过 router，并在均衡对照中采用 round-robin。P/Hybrid 的 Store 为 both，D 为 producer。首次测量前重建服务和 Store，清除该数据集的历史 cache；热测量复用同一服务及数据。这里的冷缓存指整次测试开始时为空，会话后续轮仍允许正常命中，并非每请求前清空。

吞吐统一按数据库中 `sum(completion_tokens) / (max(completed_time)-min(start_time))` 计算，不使用含报表收尾的进程耗时。p99 采用样本内线性插值；单轮只有 40 个请求，尾延迟不能视为稳定的大样本估计。evalscope 的连接探测不在这 40 个请求中，Prometheus 差值另含一次探测。

PD 的 D `cached_tokens` 包含从 P 迁入的全部 H=N−1，即使 P 刚刚冷算完也会显示接近 100%；它不能表示端到端避免了多少 prefill，更不能直接当作 Store 命中率。应观察 P/Hybrid 的 cached tokens 和 Store load 记录。

上次 GSM8K 使用 round-robin、并发 16，直接对照只使用相同 D 的 4 卡；没有保存分阶段数据。本次长上下文多轮测量不能追溯分摊上次每一项开销，也不能把本次新发现的 cache-aware 倾斜归因到上次 round-robin 结果。

### 均衡路由的实测结果

所有行均为 40/40 成功；TPOT 是请求内 `(总耗时−TTFT)/(输出 token−1)`，不是原始 SSE chunk 间隔。

| 配置               | TTFT mean / p50 / p99 / max (ms)     | TPOT mean / p50 / p99 / max (ms/token) | 全程输出 token/s |
| ------------------ | ------------------------------------ | -------------------------------------- | ---------------- |
| PD 8 卡，冷缓存    | 2429.33 / 899.98 / 9662.88 / 9780.59 | 7.80 / 7.77 / 10.09 / 10.33            | 566.81           |
| PD 8 卡，热测 1    | 945.71 / 868.48 / 1373.72 / 1383.48  | 7.81 / 7.69 / 10.68 / 11.12            | 752.43           |
| PD 8 卡，热测 2    | 872.46 / 850.60 / 1180.35 / 1196.81  | 7.80 / 7.73 / 10.18 / 10.75            | 766.45           |
| 非 PD 8 卡，冷缓存 | 1493.58 / 596.87 / 5232.77 / 5288.89 | 7.13 / 6.91 / 9.31 / 9.33              | 722.79           |
| 非 PD 8 卡，热测 1 | 444.17 / 443.32 / 684.92 / 703.67    | 6.27 / 6.26 / 7.94 / 8.07              | 1035.61          |
| 非 PD 8 卡，热测 2 | 416.99 / 413.85 / 562.31 / 582.36    | 6.39 / 6.19 / 8.71 / 9.00              | 993.67           |

同为 8 卡，本负载的非 PD 更快。PD 冷测首次连接检查均值 248.57 ms；热测 2 为 5.76 ms。两个模式冷测的有效 cached token 均值均为 31137.6（PD 看 P），热测均为 40972.8；热测尚需计算每请求约 39 个尾部 token，PD 还需在 D replay。

### 热测 2 的 TTFT 阶段分布

| 阶段                                        | PD mean / p50 / p99 / max (ms)      | 非 PD 8 卡 mean / p50 / p99 / max (ms) |
| ------------------------------------------- | ----------------------------------- | -------------------------------------- |
| 端到端 TTFT                                 | 872.46 / 850.60 / 1180.35 / 1196.81 | 416.99 / 413.85 / 562.31 / 582.36      |
| P / Hybrid HTTP 准备                        | 84.91 / 84.53 / 113.43 / 115.32     | 91.90 / 92.71 / 137.29 / 139.11        |
| P / Hybrid 调度等待                         | 133.11 / 58.85 / 455.24 / 489.14    | 97.69 / 94.93 / 267.83 / 299.36        |
| P / Hybrid 从 scheduled 到首个 forward 完成 | 203.59 / 184.41 / 355.46 / 368.69   | 185.45 / 180.50 / 299.91 / 360.31      |
| P 完整 HTTP / Hybrid 到首 token             | 460.80 / 426.73 / 722.16 / 744.79   | 412.28 / 409.30 / 557.82 / 578.08      |
| D HTTP 准备                                 | 95.69 / 91.57 / 146.53 / 152.87     | —                                      |
| D 等待开始迁移                              | 7.68 / 7.89 / 11.76 / 11.97         | —                                      |
| claim                                       | 0.69 / 0.54 / 1.67 / 1.67           | —                                      |
| 迁移 executor 往返                          | 42.00 / 18.71 / 128.89 / 130.61     | —                                      |
| 其中 native read（已包含在上一行）          | 8.05 / 5.83 / 22.53 / 22.58         | —                                      |
| release                                     | 0.65 / 0.61 / 1.55 / 1.92           | —                                      |
| 迁移完成到 D 调度                           | 36.39 / 19.97 / 163.85 / 164.74     | —                                      |
| D replay 到首个 forward 完成                | 178.24 / 171.34 / 272.12 / 314.63   | —                                      |
| D HTTP 到首 token                           | 396.31 / 372.56 / 567.46 / 580.41   | —                                      |
| 其他路由、传输与 client 开销                | 15.35 / 10.80 / 57.65 / 73.34       | 4.70 / 4.27 / 9.19 / 9.41              |

总量按不重叠边界相加：PD **460.80 + 396.31 + 15.35 = 872.46 ms**，非 PD **412.28 + 4.70 = 416.99 ms**。表中子阶段与 HTTP 总量有包含关系，不能全表相加。P/Hybrid 的调度等待包括 admission、Store lookup/load 和排队，不等于纯 CPU 调度计算时间。

PD 热测 2 的 native read 平均 8.05 ms，仅占端到端 TTFT 约 0.9%；executor 往返为 42.00 ms，包含计划、进程/线程调度和完成汇总。平均传输 205.32 MB；cache-aware 端口集中时复用 D 本地 KV 更多，传输为 148.52 MB，说明均衡与本地 cache 亲和存在权衡。

### TPOT 的测量解释

PD 热测 2 的 D 实际 decode worker stream 均值 32.23 ms；非 PD 两组为 26.41 / 27.22 ms。PD 热测 1 的真实 decode batch 均值 1.76，而非 PD 约 1.01–1.03：PD 将 8 个会话的 decode 集中在 4 个 D DP，非 PD 分配给 8 个 Hybrid DP。相同总卡数不等于相同 decode 计算资源，热命中时 P 侧难以抵消这部分资源成本。

Prometheus 的迭代间隔均值为 PD 36.44 ms、非 PD 30.17 ms；draft token 接受率为 65.45% / 66.89%，平均接受长度（含 target token）为 4.27 / 4.34。接受率差异较小，主要观测差异是每次迭代用时；这些计数差值另含一次连接探测。新请求的 replay、DP/EP 全局 prefill phase、host 等待及保存检查均可能影响正在 decode 的请求，需结合阶段记录分析，不能把 TPOT 差额直接归因于一次性的 RDMA read。

D 冷测有 40 次实际 Store native put：均值 16.67 ms、p50 4.79 ms、p99 42.60 ms、max 42.73 ms，共写 5.94 GB；热测 2 只有 4 次实际 put，均值 4.77 ms，其他保存命中已有 key。此工作负载的高 cache 命中不能代替 save-on/off 消融。

### 绕过 P、直接访问相同 D4 的控制组

P 服务保持空闲，D 仍为原配置的 Store producer。这对应上次直接访问 D 的路径，但不能作为 8 卡 Hybrid 的替代。该长上下文脚本反复运行后仍未完全热命中：producer 不从 Store 加载，实际复用取决于当前端口的本地 checkpoint；本次未逐项追踪本地 eviction 的原因。下列 cached 比例直接来自 engine 记录，并未假定每次复测都已全热。

| D4 测量 | 有效 cached token 比例 | TTFT mean (ms) | TPOT mean (ms/token) | 输出 token/s |
| ------- | ---------------------- | -------------- | -------------------- | ------------ |
| 准备轮  | 34.72%                 | 3864.14        | 17.77                | 302.95       |
| 复测 1  | 60.93%                 | 2339.08        | 14.67                | 397.04       |
| 复测 2  | 87.15%                 | 971.63         | 10.36                | 603.19       |

D4 复测 2 的 TPOT 为 10.36 ms，反而高于 PD 的 7.80 ms；其 decode worker stream 均值 41.86 ms，MTP 接受率 67.86% 比 PD 略高。结合仍需补算的长 prefix，可见 prefill 与 decode 混跑、cache 条件会改变 TPOT 结论，不能说 PD 的 decode 算法必然更慢。这也不是 PD 在相同总卡数下优于非 PD 的证据：前面的 8 卡 Hybrid 对照更快。若要精确定位上次 GSM8K 的 11.57 vs 9.46 ms，需要对同一 GSM8K 工作负载保留此次阶段记录，不能把本节的不同 cache/并发条件直接套过去。

### 已确认的 cache_aware 路由倾斜

初次配置 P、D 均为 `cache_aware`，每轮 40 个业务请求全部落到 D 的 19700 端口，其他 D DP 没有业务请求，但仍参与 EP。代码在 P 计算前选 D，直至向 D 发 HTTP 时才 increment load；收到 D 的响应头立即 decrement，流式生成仍未结束。cache-aware 在低命中且 load 相等时选择第一个 worker，前述计数窗口不能表示真实的 decode 并发，与本次反复选中同一端口的现象一致；cache 亲和及负载阈值也会影响分配，不能断言只修计数就足以完全消除倾斜。

这是从实际端口分布与源码确认的路由问题，不能把其吞吐损失全部记作 PD/RDMA 开销。后续均衡对照通过配置 round-robin 绕开该问题；本次没有修改路由负载生命周期。进一步修复应将 D 的负载占用从选中时保持至完整响应结束/取消，且覆盖 P 失败与断连回收。

cache-aware 热测四轮平均 TTFT 为 1.206 / 0.938 / 1.225 / 1.071 s，TPOT 为 9.71 / 9.43 / 9.55 / 9.75 ms，吞吐 601 / 672 / 619 / 631 token/s。最后一轮平均 TTFT 分解为 P HTTP 609.28 ms + D HTTP 到首 token 390.80 ms + 其余 70.87 ms；P 排队 168.79 ms，P 首次 forward 完成区间 311.60 ms，D replay forward 区间 192.55 ms。D executor 往返 37.41 ms 中 native 读取只有 6.10 ms。热命中后每次仍迁移 148520704 bytes（精确 recurrent/MTP/aux 等状态与尾部数据）；不能仅凭“只补算一个 token”认为交接和恢复没有固定成本。

### replay 为什么比一个稳态 decode step 慢

当前部署 `piecewise_cudagraph_max_tokens=None`。`graph_runner/runner.py` 只有 `global_is_decoding()` 为真才使用 full CUDA graph；未开启 piecewise 时，prefill 走 eager。D replay 虽然只有一个真实 token，仍走 prefill 路径，并重建本地 MTP proposals。某个 DP rank 的 replay 还会让该组的全局 phase 进入 prefill，影响同组的稳态 decode。

这解释了源码上的固定开销来源，也与观测到的 replay worker stream 约 160 ms、稳态 decode stream 约 27–32 ms 相符。stream 记录未进一步分离 target、draft、kernel、通信与 launch gap，不能把这 160 ms 全部归因于某一个 kernel。后续优化应先评估短 replay 的图执行和 D 侧 batching，再决定是否改 kernel；不能直接把 `is_decoding` 标记改成 true，否则会改变状态恢复与 MTP 语义。

### D 保存后由 P 读取的独立验证

指定的 512-token 输出通常覆盖不到 1024 间隔，因此另外通过 PD router 发 8192-token prompt、生成 1088 token，再把**实际生成 IDs**连同少量新输入构造成 9288-token 后续请求。源请求也经过 P/D；P 从未计算过后续生成的 1024-token 前缀。

- D 首次保存边界 8192：save_id=167，完成成功。
- D 周期保存边界 9216：save_id=168，start_block=128、blocks=16，完成成功。
- 周期保存的 `state_hash=43749d7e7565e92a7c0c45b6b77e4815572b222f223aa16af5880db22bf3c848`。
- P 后续请求加载 boundary=9216，state hash 完全相同，success=true；native get 5.836 ms、161420032 bytes，随后完成后续请求。

使用最终复现脚本及新 router binary 再次验证：不同 P DP 成功加载相同 9216 checkpoint（264515584 bytes、8.606 ms）；续接后的 16 个输出 IDs 与绕过 P 的直接推理 **16/16 相同**。两次读取字节数不同，源于后一次 P 没有前一次的本地 prefix，需要从 Store 读更多 KV。

这覆盖了 save-first、周期保存和 D→Store→P 复用，而不只是观察到 save 调用。主多轮测试也记录了从例如 input_end=32805 保存到 boundary=32832 的成功回调，说明真实 workload 的 D 保存跨过了 prompt 末尾。

Store native put 在独立线程执行，但异步接口不能证明 GPU/NIC 争用没有性能代价。热缓存的已有 key 会跳过 put；仅由热测吞吐不能得出 decode save 无回归。本次没有做 save-on/off 饱和消融，保留这个性能边界。

### 复现与交付

```bash
export LMDEPLOY_PROFILE_PHASES=1 LMDEPLOY_LOG_PID=1
# 每个后端服务使用独立 CUDA_VISIBLE_DEVICES、SERVER_PORT、DP master port。
# P / D: bash benchmark/mooncake_pd_glm.sh Prefill / Decode
# 非 PD: 两组 bash benchmark/mooncake_pd_glm.sh Hybrid
# PD 均衡路由：ROUTER_POLICY=round_robin bash benchmark/mooncake_pd_router.sh
# 非 PD：lmdeploy-pd-router --policy round_robin --worker-urls <全部 8 个实际 HTTP URL>

BENCHMARK_SCRIPT=/mnt/shared-storage-user/ailab-sys/caikun/code/accuracy/multi_turn/run_glm53_shorttask_382_full_kvcache.sh \
BENCH_URL=http://10.102.98.154:9200/v1 \
BENCH_OUTPUT=/absolute/path/to/results \
bash benchmark/mooncake_pd_multiturn.sh

python benchmark/mooncake_pd_decode_save.py \
  --pd-url http://10.102.98.154:9200 \
  --model GLM-5.3-Flash-caikun-mmmu-pro
```

`mooncake_pd_decode_save.py` 负责构造真实输出续接请求；是否命中 Store 要核对 scheduled/finished/load 三类记录，不能只看脚本返回成功。原用户部署脚本、性能脚本和数据集均未修改。

### 本次检查与保留事项

- 所有 14 次多轮测量共 **560/560** 业务请求成功；每次 40 个请求、20480 个输出 token。数值与分阶段分布保存在 `benchmark/results/mooncake_pd_multiturn_h200_20261111.json`，包括配置、脚本/数据 SHA256、decode 保存证据和最终资源状态。
- LMDeploy 针对 PD、Store、engine 与服务的组合 suite **1983 passed**；router **550 passed**，`cargo fmt --check`、release build 通过。只增加观测和复现脚本，没有为日志本身增加镜像式单测。
- LMDeploy 改动已同步至指定容器并执行最终 `bash install.sh`；router commit **7f8f49f** 的 release binary 已安装并验证。所有提交均未 push。
- 最终 8 个 endpoint 的 exports / grants / receives / pinned_blocks 全部为 0，quarantined=false；本次 GPU 服务、router、Store master 已停止，8 卡显存均归零。
- 原始数据库、日志和分析脚本保留在本机 `.git/pd_tmp/`；远端日志在容器 `/dev/shm/lmdeploy_pd_phase/` 和 `/dev/shm/lmdeploy_pd_phase_rr/`。phase_pd 为最初 cache-aware 测量，phase_pd_rr 为均衡测量，phase_hybrid8 为 8 卡非 PD；phase_direct4 仅为绕过 P 的 producer-only 控制。
- 仍未解决的是前述边界 token 差异；仍未证明的是 save-on/off 饱和性能，以及上一轮 GSM8K TPOT 差额的精确阶段归因。此次完成的是阶段观测、指定工作负载比较与 D 保存后实际读取验证。

## 追加：充分预热后的 graph 与 PD 前端链路优化（2026-11-11）

### 参考实现与本次取舍

- [vLLM GLM-5.3 PD recipe](https://recipes.vllm.ai/zai-org/GLM-5.3.md?pd_router=smg&kv_offload=kv_store_distributed_mooncake&frontend=python&strategy=pd_cluster) 的 D 显式使用 `FULL_DECODE_ONLY`，P 使用默认 graph 策略。该页面当前生成的是 NIXL PD + Mooncake Store；不能把 Store 的选项理解成 PD 也使用 Mooncake。本任务仍保留 LMDeploy 的 Mooncake PD + Store 分层，不引入 MultiConnector。
- [vLLM graph 设计](https://docs.vllm.ai/en/latest/design/cuda_graphs/) 区分 uniform decode 与 prefill/mixed：`FULL_DECODE_ONLY` 不为后者启用 graph。核对本机 vLLM `e740badc2e`，Mooncake 对 Mamba/hybrid 仍采用 N−1 交接及尾 token 恢复。LMDeploy 的常规 decode 在本次修改前已经使用 full graph；不能将已有 full graph 的收益计作新增优化。
- [SGLang GLM-5.3-Flash recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx) 的 NVIDIA prefill 可显式启用 `--cuda-graph-backend-prefill breakable`。本机 SGLang `a93e89fd4a` 的 graph 配置分别管理 decode/full 与 prefill/breakable。通用 PD metadata 可携带首 token、draft top-k 和 hidden states；但 Flash recipe 当前仍将 PD 标为 preview，并列出 PD+MTP 尚不能启动的限制，不能据此改掉 LMDeploy 已验证的恢复协议。
- 用户链接是 GLM-5.3，实际测试权重是 hybrid attention 的 **GLM-5.3-Flash**。本次保留其 `piecewise_cuda_graph_token_padding=False`，避免为命中 graph 改变 GEMM 行数。

### 已实现的两项优化

**Graph 与观测（commit `70cbd513`）**：prefill graph 原先仅捕获 512 的倍数及上限，1-token D 恢复和 P 的部分 block 尾部无法命中。现在增加 1～64 的精确桶及 128/256 桶；仍受既有最大 token 数限制，只在启动时捕获，未准备的形状保持 eager。`worker_step` 增加实际 target graph 模式和 target、sampling/draft、finish 的 CUDA event 分段；`graph_capture` 标记真实冷捕获。没有新增同步，也没有变更 KV/state 的分配、引用或释放顺序。

**复用 P 的分词结果**：router 仅对非空纯文本 chat 请求让 P 返回完整、未截断的 prompt IDs，并放入 D 的 `migration_request.prompt_token_ids`。D 保留 messages/tools 给 response parser，输入使用 P 的 IDs；HTTP 层消费后清空该 transport 字段，避免 generation config 深拷贝及 engine RPC 重复携带。原有 handoff 的 token 数、digest、schema 校验继续执行。多模态、需要 `prompt_text` 的请求以及已经提供原始 `input_ids` 的请求保持原路径。

P 的响应还直接构造 `JSONResponse`：`model_dump` 已生成可 JSON 编码的数据，无需 FastAPI 再逐个遍历 prompt ID。本机 49,152 IDs、20 次循环的微测中，`jsonable_encoder + JSONResponse` 平均 42.82 ms，直接 `JSONResponse` 为 3.38 ms；这只是响应编码微测，端到端收益单独测量。

### 配置探索与预热约束

固定 8×H200、P4+D4、DP4/EP4/TP1、MTP=5、max batch=16、prefill chunk=8192、cache fraction=0.7、P Store both / D Store producer，以及原 decode save-first / interval=1024。

| 设置                                       | 验证结果                                                                                     |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| P prefill graph cap=8192，D cap=16         | P 在第一个 8192 桶捕获时 OOM：申请 256 MiB 时仅余 107.56 MiB；没有完成启动，不能作为推荐配置 |
| P cap=64，D cap=16                         | 成功启动；P 每 rank 捕获 64 个短桶，D 捕获 16 个短桶；保留原 KV 容量                         |
| D full decode capture batches=`1 2 4 8 16` | target 和 MTP 均建立 full graph；这是对原默认设置的显式固定                                  |

P 的 cap=64 面向本次多轮缓存命中场景：P 的剩余真实输入主要为 2～63 tokens，D 恢复为 1 token。它不把 8192-token 大 prefill 包含进 graph。更大的输入保留 eager，长 prompt 的 kernel/JIT 冷开销需要另外预热；本次没有用降低 KV cache 容量来换取大 graph。

每次新部署先完成 16 组实际 P/D HTTP 端点的连接、真实 KV 传输和 64-token 生成，再用用户原脚本完整跑两轮预热。正式结果必须检查计时窗口内 `graph_capture=0`，不能只根据启动成功或“跑过一次”认定预热完成。另保留 graph-only 与 token 复用的中间测量，区分各项收益。

使用原 `benchmark/mooncake_pd_multiturn.sh`、原数据和 seed=1124：每轮 8 个会话×5 turn、40 个请求、20,480 个输出 tokens。每项正式结果重复两轮；预热不计入正式统计。该脚本的 rate=8 是 worker 内 turn 间的到达间隔设置，结果不是饱和吞吐。分阶段 stream 时间含 kernel、collective 和 launch gap，不能与重叠的 host 时间相加。

### PD 正式结果与 D 首 token 分解

下表每行合并两轮、共 80 个请求；吞吐为总输出 tokens 除以两轮业务窗口时长之和。基线也使用本次分段观测并充分预热，常规 decode 的 full graph 在所有组中均开启。

| 配置                                              | TTFT mean / p50 / p99 / max (ms)    | TPOT mean (ms) | 输出 token/s |
| ------------------------------------------------- | ----------------------------------- | -------------- | ------------ |
| 原 graph 策略、原 router                          | 919.34 / 859.36 / 1319.98 / 1332.16 | 7.71           | 769.33       |
| 短 prefill graph + prompt IDs 复用 + 精简响应编码 | 681.29 / 680.81 / 974.36 / 1004.27  | 7.51           | 817.43       |

本次 workload 的平均 TTFT 降低 **25.9%**，吞吐提高 **6.3%**；TPOT 仅降低约 **2.6%**。不能将 TTFT 的改善描述为同等幅度的稳态 decode 提速。最终两轮 TTFT 为 685.26 / 677.31 ms，TPOT 为 7.52 / 7.50 ms。

| D 阶段                                   | 基线 mean (ms) | 优化后 mean (ms) | 优化后 p50 / p99 / max (ms)  |
| ---------------------------------------- | -------------- | ---------------- | ---------------------------- |
| 请求准备（handler 内）                   | 93.63          | 1.51             | 1.47 / 2.25 / 2.32           |
| engine 排队，包含迁移等待                | 89.20          | 91.77            | 66.08 / 252.53 / 295.96      |
| 首次调度至 engine 首次输出               | 179.91         | 149.89           | 144.19 / 265.71 / 266.86     |
| 其余 handler 内耗时                      | 38.57          | 34.58            | 33.44 / 49.02 / 50.23        |
| **D handler 到首个可见 token**           | **401.31**     | **277.75**       | **258.75 / 473.46 / 487.28** |
| native KV 读取（包含在上面的迁移等待中） | 8.11           | 7.04             | 5.83 / 22.17 / 22.17         |

这些结果回答了原来的“D 为什么需要约 396 ms”：本次等价热基线为 401 ms，其中约 94 ms 是重复请求准备，约 180 ms 是首次调度后的恢复和输出路径，约 89 ms 是排队及迁移等待。native KV 读取区间只占约 8 ms。优化后 D 首 token 降低 **30.8%**；剩余瓶颈仍是恢复计算与调度等待。

“请求准备”和“D 到首 token”从 API handler 进入开始计时，不含此前 FastAPI 的 JSON/Pydantic 处理；P 阶段在 collect 完成处结束，也不含最后的响应编码。最终 TTFT 的加法分解为 **P 382.58 + D 277.75 + 其余 20.96 = 681.29 ms**。基线的其余部分为 11.85 ms；传递完整 prompt IDs 增加了 JSON/网络处理，该代价已计入端到端结果，没有藏进分词收益中。

中间阶段也各测了两轮，避免把全部收益归给一个开关：

| 实现阶段                                 | TTFT mean (ms) | D 到首 token (ms) | D 请求准备 (ms) | TTFT 中其余部分 (ms) |
| ---------------------------------------- | -------------- | ----------------- | --------------- | -------------------- |
| 原 graph / router                        | 919.34         | 401.31            | 93.63           | 11.85                |
| 仅短 prefill graph                       | 755.56         | 355.57            | 90.18           | 13.76                |
| 再复用 prompt IDs，尚未消除重复复制/编码 | 703.04         | 291.18            | 17.62           | 42.17                |
| 最终实现                                 | 681.29         | 277.75            | 1.51            | 20.96                |

中间版本虽省下分词，却因 IDs 在 generation config/RPC 重复复制，以及 P 响应再次经过 FastAPI 遍历而引入代价；最终实现消除了这两处重复工作。各阶段独立跑完整预热，排队和 batching 会变化，表中差额不是逐算子耗时的严格消融。

worker 的 CUDA event 进一步显示，D 恢复的 target 区间由两轮的 141.73 / 141.01 ms 降到 113.18 / 110.61 ms，sampling/draft 仍约 9.4 ms；完整 stream 区间从约 151 ms 降到约 122 ms。它们是上述 forward 区间的内部观测，不能再次相加。两轮 80 次 D 恢复全部使用 piecewise，常规 D decode 约 97% worker steps 使用 full graph；少数混合步骤仍为 eager，不能宣称所有 mixed 形状均已图化。

源码中的剩余限制是：短恢复仍采用 prefill/MTP 恢复语义，DeepEP 的部分 MoE 边界仍为 eager，低延迟路径又受全局 decode phase 约束。这是下一步定位方向；本次未对 kernel/collective/launch gap 做逐算子归因，不能把约 112 ms 的 target 时间全部归给 MoE 或 RDMA。直接将恢复请求标成 decode 会影响 recurrent state 和 MTP，不属于安全的配置调整。

另一个明确的 graph 覆盖缺口在 `glm5_next.py`：MTP 开启时，prefill 的 `return_input_embeds=True`，decode 为 False；piecewise descriptor 将该常量包含在 key 中。启动 prefill warmup 只准备了 True 的 plan，所以增加 6/12-token 桶本身并不足以使 mixed step 中的本地 decode 命中它。若继续优化，需要为正确的调用契约准备 plan，并验证额外显存与 state 更新，不能通过忽略 descriptor 常量强行复用。

### 相同 8 卡、相同短 prefill graph 的非 PD 对照

非 PD 使用两组 Hybrid4（每组 DP4/EP4/TP1），全部 8 个 endpoint 参与 round-robin；每组启用 cap=64 的 piecewise 和相同 full decode 桶、MTP、Store、decode 保存参数。新部署完整跑两轮预热，随后正式重复两轮。第二轮预热及两轮正式窗口均为 **0 次 graph capture**；全部 80 次正式 prefill 使用 piecewise。

| 同卡数配置             | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | 输出 token/s |
| ---------------------- | -------------------- | -------------------- | ------------ |
| 优化 PD：P4 + D4       | 681.29 / 974.36      | 7.51 / 9.57          | 817.43       |
| 优化非 PD：2 × Hybrid4 | 410.41 / 557.76      | 6.17 / 7.71          | 1051.92      |

非 PD 两轮 TTFT 为 414.81 / 406.00 ms。该热缓存、并发 8 的 workload 上，PD 仍比非 PD 高约 **66% TTFT**，吞吐低约 **22%**，不能得出“开启 graph 后 PD 已优于非 PD”的结论。

原因有明确的工作负载和阶段证据：P 与 Hybrid 平均都已缓存 40972.8 个 tokens，剩余 prefill 很短，分离长 prefill 的收益有限；PD 仍需要 P 和 D 两次恢复/执行及交接。PD 的 8 卡中只有 4 卡承担持续 decode，非 PD 的 8 卡都能参与生成。实际 decode worker 的平均 batch 为 1.80 vs 1.02，stream 时间为 31.18 vs 26.00 ms；MTP 接受率为 65.82% vs 66.74%，并非完全一致，但不足以仅凭接受率解释全部阶段差额。是否在更长未命中 prefill、更高并发或不同 P:D 比例下受益，需要另设 workload，不能从本轮低并发热测外推饱和容量。

### 正确性与保存复用复查

- 长度 1/2/3/63/64/65/129/8192、每种重复两次、每次 32 个输出 IDs：graph-only 与最终版本分别和基线 **16/16 个案例、512/512 个 IDs 完全一致**。
- 另用真实 32783 / 40983 / 49242-token prompt，对比旧 router 重新分词和新 router 复用 IDs：完整 prompt IDs **3/3 一致**，各自 32 个输出 IDs **3/3 一致**；messages/tools 保留给原 response parser。
- decode save-first 在 boundary=8192 成功，周期 checkpoint 在 boundary=9216 成功（D pid=254327，save_id=47）；P 后续请求加载相同 `43749d7e...bf3c848` state hash，读取 264515584 bytes、native 8.452 ms，success=true。实际生成 1088 IDs 后续接，输出与直接推理 **16/16 IDs 一致**。
- 并发性能脚本不能用作逐 token 一致性的 oracle：同一基线两次运行，40 个 512-token 响应仅 3 个完整输出相同；最终版本的两轮为 6/40。该基线已有的不稳定性尚未定位，不能据此断言所有差异只是舍入，也不能把上述串行通过推广为所有边界输入精度问题已解决。

最后一条保留了原任务中的精度边界。本次没有通过放宽交接校验、改变 token padding 或提前释放 KV block 来获取性能。

### 推荐部署参数与交付检查

在本次相同模型、显存配额和多轮缓存场景下，在原部署参数上增加：

```bash
# Prefill：缓存命中后的短尾部使用 piecewise graph。
--piecewise-cudagraph-max-tokens 64

# Decode：1-token 恢复使用 piecewise；常规 target/MTP decode 使用 full graph。
--piecewise-cudagraph-max-tokens 16 --cudagraph-capture-batch-sizes 1 2 4 8 16
```

保留 `--prefix-cache-decode-save-first --prefix-cache-decode-state-interval 1024`、原来的 MTP=5 和 KV cache 配额，不设置 `--eager-mode`。LMDeploy 的 full decode 由 graph runner 自动选择，没有需要照抄的 vLLM `--compilation-config`。本次仅新增了短 prefill 的 piecewise 覆盖，并将原 full decode batch 桶显式固定。

复现脚本对应环境变量为：P `PIECEWISE_GRAPH_TOKENS=64`，D `PIECEWISE_GRAPH_TOKENS=16`；D/Hybrid 均可设 `DECODE_GRAPH_BATCHES='1 2 4 8 16'`。启动服务后先做实际连接和生成，再执行两轮原多轮脚本预热，最后检查正式计时窗口没有 `graph_capture`。例如 graph-only 的第二轮预热仍捕获了 4 个 MTP graph，但其后两轮正式测量均为 0；这说明不能仅以启动 warmup 完成为依据。

- LMDeploy graph/观测提交：`70cbd513`；prompt IDs 复用与响应编码提交：`736aa7c0`。
- router 提交：`5cc5202`。正式 PD workload 使用的 binary 与最终 binary 仅差一个空 messages 的排除条件；正式请求全部为非空文本 messages，不受该差异影响。最终 binary 的该分支已通过单测，并重新构建和安装。
- 最终源码的 API/PD/ChatRunner 相关 CPU 测试 **239 passed**，capture bucket 测试 **13 passed**；GPU 组合检查先通过 59 项，修正旧 plan 数量断言后受影响案例单独 **1 passed**，共覆盖 60 个不同案例。GPU 检查包含跨 64/128 桶的输出、checkpoint 及未触碰 state 的一致性。
- router 最终源码 **551 passed**，`cargo fmt --check` 与 release build 通过；LMDeploy pre-commit 通过。
- 最终源码已同步至指定容器并执行 `bash install.sh`；7 个修改模块的源码/安装文件 SHA256 完全一致。router 安装文件与本机构建文件 SHA256 均为 `8f4118589f6d5ceda62bd5888960398d04a3acc9b0050abe135e81e733ebfb6b`。未 push。

本轮 5 种配置各 2 次预热 + 2 次正式测量，共 **800/800 业务请求成功**；其中正式测量 400 个请求，所有窗口均无冷 graph capture。结果与 mean/p50/p99/max、逐 turn 分布、实际 graph 使用次数、MTP counter 增量、保存复用证据保存在 `benchmark/results/mooncake_pd_graph_h200_20261111.json`。

原始数据库在本机 `.git/pd_tmp/graph_*`，分析脚本与日志快照在 `.git/pd_graph_tmp/`；远端日志在容器 `/dev/shm/lmdeploy_pd_graph_{baseline,piecewise,final,hybrid}/`。原用户部署脚本、性能脚本和数据未修改。

收尾时 P/D 及 Hybrid 的全部 endpoint 均为 exports / grants / receives / pinned_blocks = 0，quarantined=false。本次启动的服务、router、Store master 均已停止，相关监听端口和存活进程为空，8 卡显存全部为 0 MiB。
