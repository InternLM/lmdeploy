# LMDeploy consistent_hash：PD 与非 PD 性能对照

2026-11-19，在 `gpu-l-lg-cmc-h-h200-0382` 的 `lmdeploy_new_version_1106` 容器完成测试。两组均使用 `consistent_hash`，同一会话跨轮次保持相同 `X-Session-ID`。

**结论与 vLLM 一致：本次热缓存、并发 8 的负载下，非 PD 更快。** LMDeploy 两组的前缀缓存复用率均为 99.904%，PD 的 TTFT 高 111.8%，TPOT 高 13.0%，输出吞吐低 16.4%。

## 三轮正式结果

每组 120 个请求，每请求输出 512 tokens。TTFT、TPOT 按请求数加权；吞吐为三轮输出 tokens 总数除以三轮业务窗口时间之和。

| 部署             | 缓存复用率 | TTFT 均值（ms） | TPOT 均值（ms/token） | 输出吞吐（token/s） |
| ---------------- | ---------: | --------------: | --------------------: | ------------------: |
| PD：P4+D4        |    99.904% |          669.78 |                 7.895 |              790.07 |
| 非 PD：2×Hybrid4 |    99.904% |          316.25 |                 6.986 |              945.14 |

| 部署  | TTFT p50（ms） | TTFT p99（ms） | TTFT 最大值（ms） | TPOT p99（ms/token） |
| ----- | -------------: | -------------: | ----------------: | -------------------: |
| PD    |         611.46 |        1083.27 |           1093.86 |               10.397 |
| 非 PD |         302.28 |         471.16 |            489.65 |                8.785 |

逐轮结果如下，未挑选最优轮次。

| 部署 / 轮次 | TTFT 均值（ms） | 输出吞吐（token/s） |
| ----------- | --------------: | ------------------: |
| PD / 1      |          651.89 |              781.21 |
| PD / 2      |          655.70 |              782.09 |
| PD / 3      |          701.74 |              807.46 |
| 非 PD / 1   |          312.81 |              975.24 |
| 非 PD / 2   |          309.43 |              926.20 |
| 非 PD / 3   |          326.52 |              935.39 |

与[同日 vLLM 固定会话路由测试](vllm_glm53_pd_vs_single_h200_20261119.md)相比，两套系统的变化方向和幅度接近：

| 系统     | PD 相对非 PD：TTFT |   TPOT | 输出吞吐 |
| -------- | -----------------: | -----: | -------: |
| LMDeploy |            +111.8% | +13.0% |   −16.4% |
| vLLM     |            +126.5% | +14.1% |   −12.5% |

此表比较各自框架内的部署差距。LMDeploy 启用 Mooncake Store、max batch=16、显存比例=0.7、context=102400；vLLM 使用 NIXL、未启用 Store、max seqs=32、显存比例=0.92、context=512000。两者的 Graph 和路由到各 rank 的分布也不同，不能据此做框架绝对速度排名。

## 配置、预热与缓存口径

- 同一 8×H200、FP8 checkpoint `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`。P、D 或每个 Hybrid 组均为 DP4/EP4/TP1。
- 两组均开启 MTP=5、本地前缀缓存、Mooncake Store、decode save-first 和 state interval=1024；max batch=16、prefill chunk=8192、显存比例=0.7、session len=102400、state budget=16。
- P/Hybrid piecewise 最大 tokens=64，D=16；保留 full decode graph 的 1/2/4/8/16 桶。
- Store 每 rank global segment=160GB、local buffer=4GB，与周报主实验一致。切换部署时重建本次测试的 Store master，避免非 PD 继承 PD 的 Store 缓存。
- 运行当前提交 `7ca92464` 的独立源码快照，771 个源码/启动文件 SHA-256 与本机一致；Mooncake 0.3.14、PyTorch 2.12.1+cu130。Router 二进制与周报相同，SHA-256 为 `8f4118589f6d5ceda62bd5888960398d04a3acc9b0050abe135e81e733ebfb6b`。
- 沿用用户的原始脚本和数据：8 会话×5 轮，首轮 32768 user tokens，后续每轮增加 4096，输出 512，parallel=8、rate=8、seed=1124、temperature=0。实际历史只累积 user 消息，不加入生成的 assistant 内容。
- 使用与 vLLM 测试完全相同的稳定会话键；8 个会话各对应 5 个请求。Router 日志确认使用 `header:x-session-id`，每个会话映射保持固定。原客户端每请求重建 UUID 的行为只在独立测试入口中覆盖，未修改全局客户端。
- PD 先实际预热全部 16 对 P/D 端点；非 PD 预热全部 8 个端点。每组再运行两轮完整业务预热、三轮正式测量。PD 业务预热的 TTFT 从首轮约 2313 ms 降至第二轮约 645 ms；预热数据不计入正式结果。

缓存复用率使用正式请求的 `cached_tokens` 总和除以 prompt tokens 总和：PD 取 **P 的 engine output**，非 PD 取 Hybrid 的 engine output，包含本地和 Store 复用。两组每轮均复用 1,638,912 / 1,640,487 tokens。正式窗口中，P 和 Hybrid 的 external cache hit 增量均为 0，说明本轮热前缀复用来自本地缓存。D 接收的迁移 KV 没有作为 P 的缓存命中统计。

每轮还有一个 evalscope 连通性探测；客户端正式统计和上述缓存复用率只覆盖 40 个业务请求。服务端 Prometheus 前后快照含该探测，故单独保存在原始数据中。

## 阶段耗时与解释

PD 的请求级平均 TTFT 可分解为：P HTTP 处理 **348.37 ms** + D handler 到首个输出 **295.25 ms** + 其余网络、Router 等 **26.16 ms** = **669.78 ms**。native 传输平均 **5.97 ms**，已包含在交接路径内，不能重复相加。

非 PD 的 handler 到首个输出为 **309.44 ms**，其余约 **6.81 ms**，合计 **316.25 ms**。缓存充分命中后，PD 仍需两端执行与交接；只有 D 侧持续生成，而两组 Hybrid 都可生成。这与观测差距的方向一致，但没有定量隔离各因素的贡献。

`consistent_hash` 不保证小样本均匀分布。PD 每轮 40 个业务请求在 P 各 rank 的分布为 `[0,15,15,10]`，D 为 `[5,20,10,5]`；非 PD 两组分别为 `[0,5,5,10]`、`[0,15,5,0]`。未直接接收请求的 rank 仍可能通过 EP 参与计算，不能据此认为对应 GPU 空闲。结论限于本次固定会话键、端点集合及并发 8 的工作负载，不代表饱和吞吐或最优路由。

历史周报使用较早的代码和 Mooncake 版本。因此，本次与周报的数值差不能直接解释为切换路由策略带来的收益。

## 验证与留档

两组正式请求共 **240/240 成功**，每请求输出 512 tokens；正式窗口内 **Graph capture=0**。每组均记录 **120/120 次 decode Store 保存成功**。没有重跑问答准确率评估。

[结果 JSON](../benchmark/results/lmdeploy_consistent_hash_h200_20261119.json)包含逐轮分布、阶段统计、启动参数和路由映射。原始数据库、日志、Prometheus 快照、源码快照和脚本保存在本机 `.git/lmdeploy_hash_20261119/`，远端归档为：

```text
/mnt/nvme1n1/ml_research/caikun/dev/code/v6/lmdeploy/runs/hybrid_attn/glm53_flash/perf_test/lmdeploy_hash_20261119.tar.gz
```

原部署脚本、原测试客户端和用户正在修改的周报保持不变。本次 LMDeploy 测试进程已清理，原 vLLM P/D 服务恢复后通过健康检查及 64-token 请求验证；原 vLLM router 始终保留。
