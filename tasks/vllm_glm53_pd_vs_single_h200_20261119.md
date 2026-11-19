# vLLM GLM-5.3-Flash：PD、非 PD 与路由缓存对照

测试日期：2026-11-19。机器：`ailab@gpu-l-lg-cmc-h-h200-0382`，容器：`vllm_image_process`。结果见[完整数据](../benchmark/results/vllm_glm53_pd_vs_single_h200_20261119.json)。

## 结论

当前部署使用 **NixlConnector + 本地 prefix cache，没有 Mooncake Store**。`round_robin` 会让同一会话访问不同 DP rank，后续轮次无法保证复用已有缓存。实测中，延长业务预热后 RR 仍明显低于固定会话路由的约 99.75% 命中率。这个现象同时影响 PD 和非 PD，不能只调整其中一组。

**本次热缓存、并发 8 的测试中，vLLM 非 PD 更快，与 LMDeploy 周报的结论方向一致。** 固定会话路由下，PD 的 TTFT 为 404.30 ms、TPOT 为 6.967 ms/token、吞吐为 931.93 token/s；非 PD 分别为 178.50 ms、6.107 ms/token、1064.69 token/s。PD 的 TTFT 高 126.5%，TPOT 高 14.1%，吞吐低 12.5%。这是当前配置和路由分布下的结果，不能外推为所有负载下 PD 都更慢。

RR 下存在大量缓存未命中，需要分别观察 TTFT、TPOT 和吞吐。PD 隔离长 prefill 与 decode 的价值在这种情况下可能体现出来，不能直接套用热缓存结论。

## 配置与预热

| 项目                     | PD                                             | 非 PD                       |
| ------------------------ | ---------------------------------------------- | --------------------------- |
| GPU                      | P：0–3；D：4–7                                 | 两组：0–3、4–7              |
| 并行方式                 | P、D 各 DP4/EP4，TP1                           | 每组 DP4/EP4，TP1           |
| 服务端口                 | 8001、8002                                     | 8100、8102                  |
| KV 缓存                  | 本地 APC；P/D 使用 NIXL 交接；无 Store         | 本地 APC；无 Store          |
| Graph                    | P：`FULL_AND_PIECEWISE`；D：`FULL_DECODE_ONLY` | 两组均 `FULL_AND_PIECEWISE` |
| `max-num-batched-tokens` | P 8192；D 256                                  | 8192                        |

公共参数：MTP=5；`max-num-seqs=32`；GPU memory utilization=0.92；max model len=512000；prefix match unit=64；CLI block size=64。

模型使用同一 FP8 checkpoint `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`。环境为 8×H200、驱动 595.71.05、vLLM 0.32.0、PyTorch 2.13.0+cu130、vllm-router 0.1.17、NIXL 1.5.0。容器虽然安装了 Mooncake Transfer Engine，但本次 KV connector 没有使用它。

原 `single/vllm_p0.sh`、`single/vllm_p1.sh` 与 PD 脚本的公共参数不同。本次只修改临时副本：max seqs 从 16 对齐到 32；启用 FlashInfer autotune；GPU memory utilization 从默认值对齐到 0.92；补齐与 P 相同的 compilation config 和网络环境变量；两组使用独立 DP RPC 端口。原脚本保持不变。Graph 模式和最终生效参数均从启动日志核对，未将重复 CLI 参数的前一个值误作生效值。

负载使用用户提供的 `run_glm53_shorttask_382_full_kvcache.sh` 和原始数据文件：8 会话×5 轮，首轮 32768 user tokens，后续每轮增加 4096，输出 512 tokens，parallel=8、rate=8、seed=1124、temperature=0、ignore_eos=true。仅覆盖 model、服务地址和输出目录。客户端为 evalscope 1.8.1。

实际请求体核对发现，这份负载累积的是 **user 历史，不加入模型生成的 assistant 内容**；所有配置和轮次的 40 个请求体完全一致。因此测试的是同一组可重复的多轮输入，不代表不断出现新内容的线上会话。

每个 PD 测试阶段先覆盖全部 16 对 P/D rank，每对实际生成 64 tokens；非 PD 覆盖全部 8 个 rank。初测每种策略运行 2 轮完整业务预热，再测 3 轮；RR 另做 4 轮业务预热及 3 轮复测。每个结果包含 120 个正式请求。执行顺序为 `PD RR → PD hash → 非 PD RR → 非 PD hash → 非 PD RR 复测 → 恢复 PD 后 RR 复测`；切换策略没有清空本地缓存，PD 最后一次复测从重新启动的服务开始。原始结果全部保留，没有挑选最优轮次。

正式窗口内没有 Graph capture 完成事件或服务错误日志；启动、Graph 捕获、端点连接预热均在测量前完成。RR 的缓存命中率在更多业务预热后仍有波动，应与启动冷开销区分。

## 路由与测量结果

测试 router 使用独立端口 30010，`--intra-node-data-parallel-size 4`，关闭重试。RR 对照显式使用 `round_robin`，与 LMDeploy 主实验一致；PD 同时设置 prefill/decode policy。另一组两端均使用 `consistent_hash`。

原客户端 `evalscope/perf/core/http_client.py::AioHttpClient.post` 每次请求执行：

```python
request_headers['X-Session-ID'] = str(uuid.uuid4())
```

当前 router 日志证实它采用 `header:x-session-id` 作为 hash key。因此，**只把 policy 改成 `consistent_hash`，仍不能维持多轮会话的路由亲和性**。本次通过独立测试入口，在实际发送前覆盖该 header：对每个会话固定一个 key。测试数据的首条 user message 可唯一标识 8 个会话，故使用其 SHA-256；核对每个 key 恰好对应 5 个请求，router 日志也确认映射稳定。没有修改全局 evalscope 包或原测试脚本。线上应使用显式、跨轮次保持不变的 conversation ID。

以下 RR 采用延长预热后的复测；hash 采用固定会话键的三轮正式结果。

| 路由          | 部署  | P/本地缓存命中率 | TTFT 均值（ms） | TPOT 均值（ms/token） | 输出吞吐（token/s） |
| ------------- | ----- | ---------------: | --------------: | --------------------: | ------------------: |
| RR 复测       | PD    |           63.48% |         2093.74 |                 5.701 |              679.10 |
| RR 复测       | 非 PD |           64.18% |         1330.16 |                 7.336 |              676.40 |
| 固定会话 hash | PD    |           99.75% |          404.30 |                 6.967 |              931.93 |
| 固定会话 hash | 非 PD |           99.75% |          178.50 |                 6.107 |             1064.69 |

RR 复测中，PD 的 TTFT 高 57.4%，TPOT 低 22.3%；吞吐仅高 0.4%，不能据此认定有吞吐优势。三轮 PD 吞吐范围为 634.02–711.33 token/s，非 PD 为 658.61–705.60 token/s，轮间变化明显大于这个均值差。两者的缓存命中率都约 64%，与 hash 热缓存组属于不同执行负载。

初测 RR 结果也保留在 JSON：PD 为 2056.76 ms / 5.692 ms/token / 679.53 token/s / 63.55% 命中；非 PD 为 1462.54 ms / 7.669 ms/token / 666.35 token/s / 60.11% 命中。非 PD 切回 RR 后追加四轮预热，命中率仍为 53.79%、58.49%、70.72%、58.84%，没有维持 hash 组的高命中。这支持“只多预热几轮不足以消除本次路由复用差异”，但不证明 RR 在所有缓存容量和工作集下必然低命中。

统计口径：TTFT、TPOT 使用客户端逐请求观测，均值按请求数加权；吞吐为三轮输出 tokens 总数除以三轮业务窗口时间之和。缓存命中率为服务端 `prefix_cache_hits_total` 增量除以 `prefix_cache_queries_total` 增量，按 token 加权。**PD 只统计 P，非 PD 合并全部 8 个 rank**，没有将 D 接收的外部 KV 当作 P 的缓存复用。

evalscope 每次运行还会发送一个 `hello` 连通性探测并生成 512 tokens。因此服务端指标窗口包含 41 次请求，客户端正式结果只有 40 次；探测不计入上表延迟和吞吐。没有使用 evalscope 的“Cached Prompt tok/s”推算 APC 命中率。

## 阶段耗时与解释边界

初测 PD 的服务端指标如下，均值包含上述连通性探测。它们用于比较阶段变化，不是与客户端 TTFT 逐请求对齐、可直接相加的分解。

| 指标                            |    RR 初测 | 固定会话 hash |
| ------------------------------- | ---------: | ------------: |
| P 每次实际计算的 prefill tokens |   14583.38 |        101.17 |
| P 排队均值                      |  223.49 ms |       0.04 ms |
| P prefill 均值                  | 1389.93 ms |      70.00 ms |
| D 服务侧 TTFT 均值              |  190.81 ms |     203.52 ms |
| NIXL transfer 指标均值          |   21.01 ms |      22.91 ms |

TTFT 的主要变化发生在 P 的缓存复用、计算与排队，NIXL 传输没有相同量级的变化。缓存充分命中后，PD 仍保留两端处理与交接；P4+D4 中只有 D 侧持续生成，而两组非 PD 都能生成。这与热缓存结果的方向相符，但该实验没有定量分离这些因素的贡献。[vLLM APC 机制说明](https://docs.vllm.ai/en/v0.32.0/features/automatic_prefix_caching/)也指出，前缀复用直接减少的是 prefill 计算。

固定会话 hash 并不保证均匀负载。每轮 40 个业务请求的分布为：P 的 4 个 rank 为 `[10, 5, 15, 10]`，D 为 `[5, 20, 15, 0]`；非 PD 两组分别为 `[5, 5, 0, 15]`、`[15, 0, 0, 0]`。没有直接接收请求的 DP rank 仍可能通过 EP 参与计算，不能据此说对应 GPU 空闲。hash 组 PD 的 TPOT 比 RR 高，可能同时受到负载倾斜和更快 prefill 带来的 D 并发变化影响；本次没有隔离验证这两项因素。

LMDeploy 周报在 Store 开启、RR、同样 8 卡及并发 8 下记录：PD 为 681.29 ms / 7.51 ms/token / 817.43 token/s，非 PD 为 410.41 ms / 6.17 ms/token / 1051.92 token/s。**两套系统在各自热缓存配置下，非 PD 更快的方向一致；本次 RR 缓存未充分命中的结果需要单独解释。** vLLM 与 LMDeploy 的 connector/Store、max batch、显存比例、上下文上限和 Graph 设置并不相同，不能据此做框架绝对速度排名。这也不是饱和吞吐测试。

## 验证与留档

六组正式结果共 720 请求；连同业务预热，共 1360 请求，全部成功且每请求输出 512 tokens。测量窗口内无抢占，NIXL failed transfer 和 KV expired request 增量均为零。这是性能与请求完成检查，没有重跑问答准确率评估。

原始 SQLite、逐轮 Prometheus 前后快照、服务/router 日志、原始与测试启动参数、会话键入口和统计程序位于本机 `/data/code/lmdeploy/.git/vllm_pd_compare_20261119/`。远端留档：

```text
/mnt/nvme1n1/ml_research/caikun/dev/code/v6/lmdeploy/runs/hybrid_attn/glm53_flash/vllm_pd/logs/benchmark_20261119/vllm_pd_compare_20261119.tar.gz
```

JSON 包含原始脚本、临时非 PD 脚本、router 参数、各轮统计、数据文件 SHA-256 和归档 SHA-256。所有用户原始部署脚本、测试脚本、数据集和全局客户端文件均通过校验。P/D 服务已按原脚本恢复，原 30000 端口 router 保持原进程运行，并验证 64-token 请求成功；临时 30010/29010 router 已停止。
