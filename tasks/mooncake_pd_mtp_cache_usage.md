# PD Prefill 采样与缓存命中统计修复

2026-11-19；基线 `9aac56be`；验证模型 GLM-5.3-Flash，MTP=5。

## 实现结论

P 的 draft token 不会交接给 D，因此生成这些 token 没有收益。现有 `SpecModelAgent._async_autoregressive_model_forward` 已在 Prefill 角色跳过 draft lm-head/采样及后续 proposal forward。本次进一步让纯 PD 交接批次跳过 **target lm-head 和 target sampling**。保留全部 target hidden states，以及构建历史 MTP KV 所需的一次前向；shifted 输入的末尾使用真实 prompt lookahead。D 仍从真实 prompt 尾部恢复，正常采样并生成自己的 drafts。

`InputsMakerAsync` 按整个批次设置 `ModelInputs.skip_sampling`。混入普通请求，或显式要求 logits、CE loss、logprobs 时使用原路径。DP dummy rank 同样省略无用 lm-head/采样，必要的 MTP 前向次数保持一致。使用已有 dummy token 机制承接内部输出，不改变交接边界、Store key 或资源生命周期。

缓存统计确实存在用户所指的 [vLLM #44607](https://github.com/vllm-project/vllm/issues/44607) 问题：D 的内部历史包含本次从 P 导入的 KV，直接用于响应就会把新计算的前缀算作缓存命中。LMDeploy 使用尾部 replay，通常表现为 `N−1 / N`，长输入四舍五入后接近 100%。

修复参考本地 `/data/code/vllm` 已合入的 [#54222 / 42919b49c5](https://github.com/vllm-project/vllm/commit/42919b49c573aa0b06710616a3b9785674f36ef5)：

- P 发布 `PDHandoff.prefill_cached_tokens`，计入此前本地前缀和 Mooncake Store 的实际复用，不计本次 P→D 搬运。
- D 成功接收后保存为请求级 `remote_prefill_cached_tokens`；`RequestMetrics` 优先使用此值，包括明确的 0。Chat Completions 的流式/非流式响应、请求缓存命中直方图及 `cached_tokens_total` 共用这个来源。
- D 的内部 `cached_tokens` 保留原用途，避免改变 KV block 计数和调度。新一轮请求重置远端统计；旧 v2 handoff 缺少该字段时保守报告 0。协议拒绝超过导出前缀长度的命中数。

## H200 验证

机器 `ailab@gpu-l-lg-cmc-h-h200-0382`，容器 `lmdeploy_new_version_1106`，独立部署目录 `/dev/shm/lmdeploy_pd_mtp_cache_20261119/repo`，Mooncake 0.3.14。P、D 各 4 卡，DP4/EP4/TP1；P piecewise=64，D piecewise=16，并保留 full decode graph 的 1/2/4/8/16 桶及 decode save-first、interval=1024。

| 场景                        | prompt tokens | P 实际命中 | 修复前 D 返回 | 修复后 D 返回 |
| --------------------------- | ------------: | ---------: | ------------: | ------------: |
| 冷请求                      |          8321 |          0 |          8320 |             0 |
| P 本地重复命中              |          8321 |       8256 |          8320 |          8256 |
| 换 P，通过 Store 命中，流式 |          8321 |       8256 |          8320 |          8256 |
| 追加输入，部分命中，流式    |          8833 |       8320 |          8832 |          8320 |
| 短冷请求，流式              |           257 |          0 |           256 |             0 |
| 32K 冷请求                  |         32769 |          0 |         32768 |             0 |

六例的 D 端 Prometheus `cached_tokens_total` 增量分别为 0/8256/8256/8320/0/0，与响应和 handoff 一致；每例请求计数增量为 1。每例生成 64 tokens，修改前后 **384/384 IDs 相同**。

- GPU 回归：PD、paging、model agent、inputs maker、KV connector、spec agent 共 **873 passed**。新增三个短测试函数，覆盖跳过投影/采样而保留 hidden states、混合批次回退、交接缓存计数与请求重置；扩展现有协议边界断言。同步补齐旧 mock 的 runtime 字段。
- PD 对直接推理：8192/32768-token 输入，各自首次、重复、部分修改，共 **6/6**，每例 32 个输出 IDs 相同。
- 32 次取消后，八个 endpoint 的 exports/grants/receives/pinned blocks 均为 0。
- D 在 boundary=8192 保存首个 checkpoint，并在生成阶段成功保存 boundary=9216；P 随后加载同一 state hash，读取 264515584 bytes，native 8.99 ms。恢复后的 16 个输出 IDs 与直接推理相同。
- 此轮没有解决既有短边界输入的所有数值差异，也不将串行一致性结果推广为并发逐 token 完全一致。

## 性能口径

使用用户原脚本 `/mnt/shared-storage-user/ailab-sys/caikun/code/accuracy/multi_turn/run_glm53_shorttask_382_full_kvcache.sh`：8 个并发会话、每会话 5 轮、每请求输出 512 tokens。先完成全部 16 个 P/D 配对的真实传输，再执行业务预热。

首次新版第二轮测量捕获了 4 个 P 端 MTP dummy graph，key 为 `[2, true, false, 1, 8192, true]`，引起约 2.3 秒的 worker 停顿；该轮 TTFT 均值 1006.74 ms，不能作为热态性能。保留原始记录，补充预热后重测，并以测量窗口没有 `graph_capture` 为准。

下表合并基线两轮和新版补充预热后的两轮，每组各 80 个成功请求，所有正式窗口均无 graph capture；延迟单位 ms。

| 指标                          | 基线 mean / p50 / p99 / max       | 修改后 mean / p50 / p99 / max       |
| ----------------------------- | --------------------------------- | ----------------------------------- |
| TTFT                          | 664.28 / 626.03 / 951.68 / 984.37 | 656.35 / 616.14 / 1041.32 / 1050.71 |
| TPOT                          | 7.625 / 7.588 / 9.920 / 10.241    | 7.631 / 7.608 / 9.816 / 10.623      |
| P 真实 batch 的 worker stream | 117.59 / 116.01 / 130.85 / 131.11 | 117.58 / 115.54 / 140.00 / 140.36   |

输出吞吐为 814.39 → 826.63 tokens/s。P HTTP 均值 385.48 → 371.38 ms，D HTTP 到首 token 均值 259.04 → 264.50 ms。P 的 `sampling_and_draft` 均值 8.95 → 9.04 ms：这个阶段仍包含必要的 MTP KV 前向，不能期待省掉采样后变为 0。

新版所有 76 个真实 P batch 和 206 个 P dummy step 均启用了 `skip_sampling`；D 没有启用。两组真实 P batch 数为 80 和 76，体现实际调度合批差异。CUDA event 的 stream 时间包括 kernel、collective 和提交间隙，不等同于纯 kernel 时间。

**本轮没有证据表明显著端到端提速。** TTFT 均值下降约 1.2%，但 p99 上升，TPOT 和 P worker 均值基本不变；两轮测量不足以证明稳定收益。主要的 draft 生成开销在基线中已经消除，本次落实剩余无用投影/采样的跳过，并修正缓存统计。

逐轮分布、冷 graph 记录、缓存用例和 Store 证据见 [结果 JSON](../benchmark/results/mooncake_pd_mtp_cache_usage_h200_20261119.json)。原始数据库、脚本和日志在本机 `.git/pd_mtp_cache_20261119/`；远端目录同上。测试使用独立源码快照，18 个生产代码/测试文件的 SHA256 与本地一致，未改用户部署脚本。

收尾时八个 endpoint 均健康，PD 资源计数为 0；随后停止本次测试服务、router 和 Store master，测试端口全部释放，8 卡显存均为 0 MiB。pre-commit 全部通过。
