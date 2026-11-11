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
- GLM 改前受 role guard 阻止 PD；不能声称有可运行的 GLM PD 改前基线。模型部署、单体对照、Store 服务和端到端精度/性能在 T8 保留实际日志与结果，当前尚未执行。

Mooncake 完成契约参考 [官方 Python API](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/transfer-engine.md)。实际容器版本不假定拥有主线的新接口。

## 任务状态

| 任务 | 状态                                                                | 验证                                                                                                     |
| ---- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| T0   | 已固定环境、协议前提及计划构建基线                                  | 22 tests；计划构建基准                                                                                   |
| T1   | 完成交接描述、资源分组、执行 schema 和控制消息                      | 2 tests：容量/语义/旧 peer/replay 边界                                                                   |
| T2   | 完成完整 pool 注册、单 worker TE、批量 I/O、取消持有、all-rank 汇总 | 容器 26 passed；32k 连续 60 descriptors，p50 0.104 ms，p99 0.536 ms，max 0.696 ms；32k 碎片 p50 69.45 ms |
| T3   | 完成源端 pin、精确 state 输出、all-rank 发布和 grant/revoke/release | 54 tests；真实 scheduler end-session 竞态与迟到 grant                                                    |
| T4   | 完成 KV-only prefix、私有接收、精确 state load、取消和 unknown 隔离 | 119 tests；新增 compact 成功恢复测试；失败/未知完成的真实 allocator 复用检查                             |
| T5   | 待实现                                                              |                                                                                                          |
| T6   | 待实现                                                              |                                                                                                          |
| T7   | 待实现                                                              |                                                                                                          |
| T8   | 待执行                                                              |                                                                                                          |
