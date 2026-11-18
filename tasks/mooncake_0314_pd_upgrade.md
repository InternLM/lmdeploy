# Mooncake 0.3.14：PD 升级依据与验证

## 范围和结论

从目标容器的 `mooncake-transfer-engine-cuda13==0.3.12.post1` 升到官方 **0.3.14**。沿用 classic RDMA、P/D 独立 Store 以及请求级 300 秒强制回收；不启用 TENT、不自动重启 worker、TransferEngine 或整套通信。

参照 `55d31016`，同时更新 `docker/install.sh`、`docker/prepare_3rdparty_wheel.sh` 的 CUDA 13 和 CUDA 12 pin。实机只验证 CUDA 13/H200；没有把它记作 CUDA 12 的 GPU 验证。

## 升级依据

调研对比官方 `v0.3.12.post1` 与 `v0.3.14` 标签源码，并检查实际安装 wheel 的导出接口。不能只依据 release notes 中的标题判断某项修复是否为本次新增。

| 变化                                                                                                                                                            | 对当前 PD 路径的意义                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [drainable scatter ticket](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/transfer-engine.md)                             | 旧包仅有 batch read 的整数结果；新包提供明确 drained 状态和后续查询，允许失败及迟到完成时提前释放 PD 引用 |
| [QP generation 校验 #3584](https://github.com/kvcache-ai/Mooncake/pull/3584)                                                                                    | 旧握手回复不能配置或重置更新一代的 QP；避免连接恢复中的代次串扰                                           |
| [peer owner 串行化 #3718](https://github.com/kvcache-ai/Mooncake/pull/3718)、[握手时继续处理 CQ #3814](https://github.com/kvcache-ai/Mooncake/pull/3814)        | 降低连接建立/重试对传输完成处理的干扰                                                                     |
| [握手 daemon 防阻塞 #4451](https://github.com/kvcache-ai/Mooncake/pull/4451)、[回复 EPIPE 代替 SIGPIPE #4463](https://github.com/kvcache-ai/Mooncake/pull/4463) | 异常或退出 peer 不应长期卡住共享握手服务或导致整个进程退出                                                |
| [BatchDesc callback 生命周期 #4569](https://github.com/kvcache-ai/Mooncake/pull/4569)                                                                           | 防止完成回调尚未退出时，调用方释放其仍访问的 batch                                                        |
| [grouped scatter #3310](https://github.com/kvcache-ai/Mooncake/pull/3310)                                                                                       | allocation 内多个 fragment 可组织成分组任务，减少碎片传输的 native task 开销；收益以实测为准              |

排除项：[RDMA port 恢复 #3604](https://github.com/kvcache-ai/Mooncake/pull/3604) 修改的是 TENT，不作为当前 classic 路径收益。release notes 列出的 peer failure 不关闭本地 RNIC 的处理，在旧标签中已能看到相关分支，也不计作本次独有改进。

原生实现位于 `mooncake-integration/transfer_engine/transfer_engine_py.cpp` 和 `mooncake-transfer-engine/src/transfer_engine.cpp`。`drain()` 超时只返回 `COMPLETION_UNKNOWN`；只有终态才释放 operation。未完成 ticket 的析构会等待，不能用丢弃 Python 对象实现有界取消。

## LMDeploy 改动

- 按 allocation 构造 scatter ranges，使用 ticket 的明确枚举，省去新路径中未使用的扁平地址数组。
- 持有未完成 ticket，后台每 100 ms 使用线程池执行一次 `drain(0)`。不在 Python 调度线程内等待，不长期占住提交 slot；其他 peer 的迁移可继续。
- 同时跟踪尚未返回 ticket 的 native Future，拒绝在这段时间注销 pool。只检查 ticket 会漏掉首次 native 调用仍在运行的窗口；注销不能靠 `executor.shutdown()` 阻塞等待后直接继续。
- 迟到 terminal 通过原 Future 返回，复用 scheduler 已有的成功/失败/取消处理。强制回收后的结果不再修改分页资源。已 drained ticket 删除后允许正常注销；真正 unknown 仍禁止整个 pool 注销。
- 保留旧版 batch API 的保守兼容路径。0.3.14 没有把 300 秒超时变成 DMA 完成证明，极端迟到 DMA 的既有取舍仍存在。
- 实际部署发现并修复旧迁移循环的事件空转：独立 receive task 完成会置位 event，空闲循环必须清除此通知再等待，否则会挤占模型执行及 health RPC。新旧版本的性能对照均包含这个修复。

## 测试环境与复现

- 主机：`ailab@gpu-l-lg-cmc-h-h200-0382`，容器：`lmdeploy_new_version_1106`，8 × H200。
- 独立目录：`/dev/shm/lmdeploy_mc0314_20261118`，临时端口 P `20500–20503`、D `20700–20703`、router `20200`、Store master `20451`。
- wheel：`mooncake_transfer_engine_cuda13-0.3.14-cp312-cp312-manylinux_2_28_x86_64.whl`，SHA256 `74fa5f0684b143167503e33f85cccb9ca687d3f9843d4b594dbd5342c7df3e17`，与官方 release asset 一致。
- 包下载使用用户指定的两个镜像；内网地址加入 `NO_PROXY`，避免容器代理造成下载超时。
- GLM-5.3-Flash 固定 checkpoint `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`，P4+D4、DP4/EP4/TP1、MTP=5、P piecewise cap=64、D cap=16、full decode batches=1/2/4/8/16，双端 prefix cache、D save-first/interval=1024。
- 每个版本先预热全部 16 组 P/D HTTP 端点，再用原 `run_glm53_shorttask_382_full_kvcache.sh` 完整运行两轮预热及两轮测量。每轮 8 会话 × 5 turn，20,480 输出 tokens。相同机器、配置、数据及预热口径；不与历史非 PD 数字混算。
- 原始日志及测试驱动保存在本机 `.git/mooncake_0314/` 和上述容器目录。

## 验证结果

### GPU 回归与真实 RDMA

最终源码下 GPU 回归 **897 passed**（7.76 s），覆盖 PD、paging、compact state、MTP 输入及 sleep；本机相关组合 **25 passed**，最终 native 取消组合 **6 passed**。迁移循环修复单独提交为 `a4dfe9a5`。

真实 GPU RDMA 每种布局执行 10 次，全部目标字节及未写区域一致；下表排除每种布局第一次，统计其余 9 次，时间包含 Python 地址准备、native 传输及 Future 返回，不是纯链路带宽。

| Prompt / 布局 | 旧版 p50 / p99 / max (ms) | 0.3.14 p50 / p99 / max (ms) |
| ------------- | ------------------------- | --------------------------- |
| 8K 连续       | 1.047 / 1.086 / 1.087     | 1.086 / 1.155 / 1.158       |
| 8K 未合并     | 11.454 / 11.627 / 11.635  | 7.911 / 8.086 / 8.087       |
| 8K 碎片       | 11.477 / 11.655 / 11.660  | 7.801 / 7.859 / 7.860       |
| 32K 连续      | 3.397 / 3.487 / 3.493     | 3.283 / 3.369 / 3.375       |
| 32K 未合并    | 41.634 / 43.640 / 43.641  | 44.505 / 45.526 / 45.576    |
| 32K 碎片      | 41.541 / 43.615 / 43.621  | 46.011 / 47.637 / 47.652    |

8K 碎片有改善；32K 极碎片（31,750 descriptors，132.14 MB）p50 回退 4.47 ms。升级主要依据是更明确的失败完成契约及连接生命周期修复，不能宣称所有布局都加速。

上述性能采样之后补了首次 native Future 的注销保护。最终源码再次执行六种布局各 3 次，字节校验全部通过；32K 碎片两次暖机样本为 38.61/38.59 ms，说明样本存在明显波动，不能把前表差值视作稳定收益或回退。保留两批原始值，未挑选最快一批覆盖原性能结果。

源进程退出后，新版在 **3,535.80 ms** 返回 `failed_drained`；原 receiver TransferEngine 继续从替换源读取，完整字节校验通过，最后正常注销。旧包同类故障只返回 `completion_unknown`。该检查覆盖进程退出与新 peer，不覆盖物理 NIC down/up 或所有网络分区。

最终源码同类故障复测为 **3,658.22 ms**，相同 receiver engine 恢复及正常注销再次通过。

小 tensor 的非页对齐 GPU 地址注销时，上游打印过 `Failed to restore fork state ... deregister already succeeded: Invalid argument`；注销返回成功、字节校验和进程退出通过。保留原始日志，没有屏蔽该 warning。

复现：`CUDA_VISIBLE_DEVICES=0,1 python benchmark/mooncake_pd_transport.py --source-gpu 0 --destination-gpu 1 --repeat 10 --fail-peer`。

### GLM 部署与故障处理

旧版两轮、新版初测两轮；补齐 native Future 注销保护后，最终源码使用容器全局安装的 0.3.14 再测两轮。每批都重新完成前述预热，全部 **40/40 请求成功**。每轮统计窗口内 graph capture 均为 **0**，D 的 Store save 均为 **40/40 成功**。

| 版本 / 轮次         | TTFT mean / p50 / p99 / max (ms)    | TPOT mean (ms) | 输出吞吐 (token/s) |
| ------------------- | ----------------------------------- | -------------- | ------------------ |
| 0.3.12.post1 / 1    | 698.23 / 678.60 / 989.53 / 994.95   | 7.45           | 838.47             |
| 0.3.12.post1 / 2    | 729.64 / 652.07 / 1515.58 / 1618.13 | 7.54           | 804.49             |
| 0.3.14 / 1          | 665.65 / 628.97 / 1082.00 / 1131.71 | 7.67           | 814.67             |
| 0.3.14 / 2          | 653.88 / 619.87 / 953.95 / 965.74   | 7.66           | 800.26             |
| 0.3.14 最终源码 / 1 | 673.36 / 640.35 / 1004.15 / 1036.69 | 7.46           | 821.25             |
| 0.3.14 最终源码 / 2 | 744.18 / 684.38 / 1508.46 / 1519.62 | 7.52           | 801.46             |

以最终源码的两轮为准：平均 TTFT **713.94 → 708.77 ms**，TPOT **7.50 → 7.49 ms**，输出吞吐 **821.48 → 811.35 token/s**，端到端基本持平。新版初测两轮 TTFT 为 659.76 ms，但降幅未在最终复测稳定出现，不能将初测约 54 ms 的改善作为升级收益。这是短任务多轮负载的观测值，不是饱和吞吐或统计显著性结论。

同一批请求的分阶段均值：

| 版本 / 轮次      | P HTTP (ms) | D 到首 token (ms) | 其中 executor (ms) | 其中 native (ms) |
| ---------------- | ----------- | ----------------- | ------------------ | ---------------- |
| 旧版 / 1         | 405.06      | 268.25            | 53.14              | 8.24             |
| 旧版 / 2         | 403.46      | 256.86            | 43.92              | 6.67             |
| 新版 / 1         | 390.67      | 256.15            | 39.59              | 7.21             |
| 新版 / 2         | 366.37      | 263.41            | 46.47              | 6.43             |
| 新版最终源码 / 1 | 374.19      | 279.70            | 50.92              | 7.35             |
| 新版最终源码 / 2 | 389.71      | 288.04            | 51.71              | 6.75             |

executor 包含排队、RPC、计划和 native；上述列有包含关系，不能相加。每请求平均传输量旧版两轮约 187.74 MB、新版最终源码约 184.35 MB；native 均值 **7.46 → 7.05 ms**。最终 D 排队约 91–98 ms、模型调度执行到输出约 153 ms，D 到首 token 均值为 284 ms。native 传输已占较小比例，本次升级没有消除调度、模型执行和返回路径的开销。

功能与故障验证：

- **32 次 HTTP 取消**全部回收，8 个端点的 exports/receives/pinned blocks/quarantined 均归零。已 claim 的 grant 经 revoke 继续持有，旧 epoch release 被拒绝，正确 release 后归零。
- **D 生成阶段保存并由 P 复用**：8192-token 输入生成 1088 tokens，D 在 boundary=8192 完成 save-first，再在 **9216** 保存新增 16 个 block；后续 9288-token 请求的 P 在 9216 加载相同 `state_hash=43749d7e…`，native load 11.22 ms，成功。该证据确实覆盖生成出的 KV，不只是 prompt 保存。
- **原生连接拒绝**：仅把测试 alias 的 native endpoint 指向未监听端口；D 返回 HTTP 500，端到端 **8.80 ms**，日志为 `failed_drained`，P/D 引用归零且没有强制回收。清理测试 alias 后正常请求成功，双方 engine epoch 不变；没有重启 worker/TransferEngine。这里是立即拒绝的故障，不能等同于半开网络连接的等待时长。
- **真实 300 秒兜底**：P grant 已被测试 receiver 领取，revoke 后仍持有 8 个 block；没有 release/续租，实际 **301.51 s** 后 exports/grants/pinned blocks 归零，forced_releases 增加 1。迟到 release 幂等，P/D health 正常。该案例验证源端 grant 的真实定时回收；D 隔离及 Store 独立引用交错由 scheduler/allocator 回归覆盖。
- **生成正确性**：新版 8K/32K 的首次、重复、部分前缀共 **6/6** 案例与直接 D 的 32-token 输出完全一致。129-token 合成短输入三例有逐 token 差异；旧版对照结果见下文。新版 GSM8K 4-shot、temperature=1、top_p=0.95、并发 16 的 **100 题冒烟为 99/100**，不替代完整精度回归，也不证明上述逐 token 差异无害。

129-token 旧版对照使用相同 checkpoint、P/D 拓扑、graph、MTP 和 Store 参数，也先预热 16 对端点并执行 32 次取消。旧版首次、重复、部分前缀三例同样与直接 D 不一致，首次差异的零基索引为 **22/26/4**，新版为 **6/22/4**；部分前缀案例的新旧 PD 输出及新旧直接 D 输出分别完全相同。可确认这类差异不是 0.3.14 独有，尚不能确定根因，也不能声称新旧版本所有输出等价。旧版未重复四轮长输入性能负载，缓存历史与调度并非完全相同。旧版取消测试首次在 router 监听前启动而连接失败，待 router ready 后重跑通过；该启动时序失败未计作模型或传输失败。

最终源码再次通过 32 次取消和 8K/32K 的 **6/6** 逐 token 对照。D 的 8192/9216 保存及 P 对同一 state hash 的加载再次成功（native load 9.03 ms）。故障复测返回 HTTP 500 耗时 **8.45 ms**，随后显式向**同一个 D 端点 20700**交接并成功生成 16 tokens，双方 epoch 不变。

汇总数据及源码 SHA256 保存在 [测试结果 JSON](../benchmark/results/mooncake_pd_0314_h200_20261118.json)。完整多轮脚本沿用用户提供的脚本，通过 `benchmark/mooncake_pd_multiturn.sh` 指向上述独立 router；取消、逐 token 和 decode save 分别使用 `benchmark/mooncake_pd_cancel.py`、`benchmark/mooncake_pd_validate.py`、`benchmark/mooncake_pd_decode_save.py`。

## 部署边界

本次验证新 P、D、Store master 同时使用 0.3.14；没有验证新旧 Store 混部兼容性。CUDA 12、物理 NIC down/up、真实长时间 DMA 无法排空场景尚未做实机覆盖。300 秒依然是用户确认的可用性兜底，不是底层 DMA 安全完成的证明。

测试容器已全局安装上述已验证的 `mooncake-transfer-engine-cuda13==0.3.14` wheel，安装使用用户指定的镜像环境变量和 `--no-deps`，没有连带升级其他 Python 包。LMDeploy 测试代码位于独立目录，关键文件 SHA256 与本机最终源码一致；未覆盖用户原部署代码目录。测试服务按本次记录的 PID 树清理，不保留占用 GPU 的测试部署。
