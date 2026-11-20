# GLM-5.3-Flash PD 短边界差异定位与全量精度回归

日期：2026-11-20。对应周报第 7 节的第一个验证项。

## 结论

64-token 复现已定位到第一层 KDA 的分块计算：直接推理一次处理 64 tokens，PD 在 P 处理 63 tokens，在 D 从状态续算 1 token。两条路径在上游 FLA 中也产生相同的数值差异。LMDeploy 各自与对应 FLA 路径逐元素零误差，P/D 传输的 target KV、draft KV、卷积状态、递归状态和辅助状态全部逐字节相同。这不是本复现中 PD 地址映射或传输损坏造成的差异。

差异会被后续层与自回归生成放大，不能将它简单描述为“仅个别近似并列候选的无害差异”，也不承诺不同分块方式逐 token 一致。该结论有明确范围：对 64-token 案例做了算子级因果重放；63-token 案例验证了传输与 logits；其余边界做了生成和重复性检查。

## 环境与控制变量

- 机器：`ailab@gpu-l-lg-cmc-h-h200-0382`，8×H200，容器 `lmdeploy_new_version_1106`。
- LMDeploy 基线：`51154532`；Mooncake TransferEngine CUDA13 0.3.14；PyTorch 2.12.1+cu130。
- 模型：GLM-5.3-Flash，checkpoint `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`，MTP=5。
- P 与 D 各 DP4/EP4/TP1，P 使用 GPU 0–3，D 使用 GPU 4–7；max batch 16，cache fraction 0.7，max prefill 8192，block 64。
- P piecewise graph 64；D piecewise graph 16 + full decode graph 1/2/4/8/16。诊断另有 eager 对照，正式精度回归使用上述 graph 配置。
- Store 每 rank 32 GB、local buffer 4 GB，decode save-first、interval 1024、state budget 16。32 GB 是本次独立部署的容量，不把本次耗时与之前 160 GB 部署混作性能对照。
- 精度测试路由使用 `consistent_hash`，PD 与直接推理分别经独立 router；直接推理使用同一组 D workers。
- 算子诊断固定 P20500/D20700，直接对该 D 发请求，排除随机选择 endpoint。CPU 快照仅用于诊断；正式回归前恢复原始实现并重启测试实例。

## 边界复现与状态核对

输入为重复 token IDs `[100,101,102,103]`，部分命中案例把中间一个 token 改为 104；temperature=0、ignore_eos=true，每次生成 32 tokens。这是合成边界压力输入，不等同于自然语言质量数据集。

长度 63/64/65/66/127/128/129/130/8192，各测试 first/repeat/partial，每个案例按 direct1、PD1、direct2、PD2 顺序执行，共 108 次成功请求。first 不代表主动清空缓存。

- 27 个案例中，13 个 PD1 与 direct1 不同。
- 7 个案例中 direct1 与 direct2 也不同，需结合实际缓存命中与分块变化解释，不能一律归于 PD。
- 全部 27 个案例的 PD1 与 PD2 相同。
- 8192-token 三个案例均逐 token 一致。

对 63-token、64-token 及 64-token 重复请求，各比较 148,520,704 bytes 的交接内容，全部逐字节相同。对 64-token 案例逐项取证：

| 检查点                                                           | 结果                                                     |
| ---------------------------------------------------------------- | -------------------------------------------------------- |
| 直接 64-token 前向的前 63 个 hidden states vs P 的 63-token 前向 | 完全相同                                                 |
| D 直接 63-token vs P 63-token 的所有目标状态                     | 完全相同                                                 |
| P 导出 vs D 续算前的 target/draft KV、所有交接状态               | 完全相同                                                 |
| 处理第 64 个 token 后，第一层卷积状态                            | 完全相同                                                 |
| 同一位置第一层 KDA recurrent state                               | max abs 0.00739104，RMS 7.57068e-6                       |
| 首输出 logits                                                    | max abs 0.65625，argmax 均为 token 271                   |
| 第二输出 logits（同一验证输入）                                  | max abs 3.265625，argmax 从直接路径 40 变成 PD 路径 4418 |

Graph/eager 六组请求的输出完全一致，排除了此案例由 graph replay 引入分叉的解释。

## 上游算子重放

从 eager 部署捕获第一层真实的 mixed QKV、gate、beta、卷积参数与初始状态。确认直接路径的前 63 tokens 和末尾 token 的原始投影分别与 P、D 对应输入完全相同。卷积使用部署的 indexed convolution，以保留实际 FP32 权重语义；whole/prefix/tail 对应位置的卷积输出也确认逐元素零误差；KDA 分别调用 LMDeploy indexed 实现和安装的上游 FLA `chunk_kda`。

1. 上游 FLA 的 63-token 最终状态与 D 接收的第一层状态逐元素零误差。
2. 对 whole64、prefix63、tail1，LMDeploy indexed KDA 输出与最终状态均与对应 FLA 路径逐元素零误差。
3. 上游 FLA whole64 与 prefix63+tail1 的最终状态差异，恰好复现部署中的 max abs 0.00739104、RMS 7.57068e-6。
4. whole/split 两条 FLA 结果分别与部署捕获的第一层最终状态逐元素零误差；整个重放不经过 Mooncake。

因此该复现的起点是 KDA 分块数学中的有限精度计算路径变化。生产 PD 传输与状态地址逻辑无需针对该案例修改；改变交接边界来追求逐 token 相同会改变 D 的工作量，也不能一般性保证多层模型的分块不变性。

## 回归范围与结果

H200 上运行：

```bash
python -m pytest -q tests/pytorch/kernel/test_indexed_prefill.py \
  -k "full_chunk_pipeline or prefill_split_and_checkpoint_continuation"
```

结果：**12 passed，30 deselected，207.27 s**；警告均为现有 pytest/PyTorch 弃用提示。

全量 GSM8K 使用 1,319 道 test、固定 4-shot、EvalScope 1.8.1、seed 42、并发 16、temperature 1.0、top_p 0.95、max_tokens 10240、stream=true。预热全部 16 个 P/D 连接后运行；PD 1284/1319，97.35%，所有响应正常 stop，最大输出 4254 tokens。

| 方式            | 正确 / 总数 | 准确率 | 达到输出上限 |
| --------------- | ----------- | ------ | ------------ |
| PD              | 1284 / 1319 | 97.35% | 0            |
| 同组 D 直接推理 | 1286 / 1319 | 97.50% | 2            |

两者均完成全部 1,319 个样本，无异常 API 响应。直接推理的样本 340、943 在 reasoning 阶段达到 10,240-token 上限，没有最终答案，按相同配置计为答错，没有删除或另行提高上限重算。两者共同答对 1278 题、共同答错 27 题，PD 独有正确 6 题，直接推理独有正确 8 题。

本次 PD 比直接推理少答对 2 题，差 0.15 个百分点。这里使用随机采样，是一次同版本对照，不把该小差值解释为确定的精度提升或下降。旧版记录 PD 1287/1319、直接推理 1282/1319，仅作为历史参考，不能替代本次同版本对照。

日志中有 2 次 P Store 异步 lookup 的 `Resource temporarily unavailable`，相应请求按 miss 回退计算并正常完成；D 无 error 日志，未出现 Store sender/receiver 隔离。首次评测客户端启动因继承 SOCKS 代理而失败，禁用本次内网评测的代理后从头运行，失败启动目录单独保留，未混入正式样本。

另外补测 32K 输入的 first/repeat/partial，各执行 direct1、PD1、direct2、PD2，共 12 次请求，均生成 32 tokens，全部一致。

只扩展现有 `test_indexed_full_chunk_pipeline` 的 packed 输入，补充 63/64/65 与 127/128/129，保留原有长度，调整独立输出/检查点槽位容量。没有放宽零容差断言。该测试验证 indexed pipeline 与 FLA 同一计算路径相等，不要求不同分块路径零误差。

## 复现与原始证据

[结果 JSON](../benchmark/results/mooncake_pd_accuracy_h200_20261120.json)记录全部边界案例的输出、缓存命中、首次分叉、传输字节检查、FLA 重放误差和 GSM8K 题目级差集。本机 `.git/pd_followup_20261120/` 保存原始预测、reviews、配置、客户端日志和诊断脚本；H200 归档还包含服务端日志及张量快照：

```text
/mnt/nvme1n1/ml_research/caikun/dev/code/v6/lmdeploy/runs/hybrid_attn/glm53_flash/perf_test/lmdeploy_pd_accuracy_20261120.tar.gz
```

归档内 `client_artifacts/launch.py` 固定模型与 P/D/Store 配置；`probe.py`、`debug_run.py` 和 `pd_probe.py`/`agent_debug.py` 是固定 endpoint 的诊断脚本；`replay_kda.py` 重放捕获的第一层输入；`verify_transfer_bytes.py` 校验交接原始字节。`gsm_pd.yaml`、`gsm_direct.yaml` 为全量评测输入，`full_accuracy.py` 调用 EvalScope 并记录退出状态。诊断插桩仅在独立源码快照中使用，未加入生产路径。

正式评测前已核对 agent、KDA backend 和启动脚本的 SHA256，与本地生产源码一致。所有原始样本保留，用户已暂存的周报未修改。
