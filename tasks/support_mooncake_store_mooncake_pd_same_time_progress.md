# Mooncake PD / Store 实施记录

## 已固定的约束

- 用户于 2026-11-11 授权 T0–T8 逐任务实现、验证和 commit。
- MTP 按 `vllm_pd_mtp.md` 的 prompt-tail replay：P 在 H=N−1 停止，D 从 S_H 补算最后一个真实 prompt token；P 的 sampled/draft token 不交接。
- PD 与 Store 独立持有资源。未知 native 完成状态不得释放 block/state/MR；取消不等于完成。
- 不引入 MultiConnector，不修改 Store 的内容寻址身份。

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
