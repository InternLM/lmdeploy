# 静态推理性能测试方法

我们把推理引擎在固定 batch、固定输入输出 token 数量的前提下的推理，称之为静态推理。

评测脚本是 `profile_generation.py`，在运行此脚本前，请安装 lmdeploy 预编译包，并下载评测脚本

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
```

测速时，需输入具体的模型。我们推荐把模型下载到本地，并通过 `lmdeploy convert` 把模型转换为 turbomind 格式，然后再进行测试。
这么做的原因是，方便调节推理引擎参数，以达到比较好的推理性能，比如批处理大小（max_batch_size），K/V cache缓存大小（max_cache_entry_count）等等。有关这些参数的详细说明，请参考[这里](../turbomind_config.md).

以下章节中，我们默认模型是 turbomind 格式的。

## 测量指标

LMDeploy 统计首token延时（first_token_latency）、吞吐量（tokens/s），每个token延时的百分位数据（P50，P75，P95，P99）、GPU mem 等测试结果。

`first_token_latency` 只有在流式推理的情况下才会输出。

吞吐量的计算公式为：$吞吐量 = 生成的token数量 / 总时间$。总时间包括 prefill 时间。

测试过程中，节点上所有的显卡不要运行其他任何程序，否则 GPU mem 的统计会不准确。

## 测试方法

```shell
python3 profile_generation.py <model_path> <optional arguments>
```

其中，`model_path` turbomind格式的模型在 localhost 上的路径。

可选测试参数如下：

- `--concurrency`

  代表请求线程的数量，并发请求会被推理引擎拼成 batch。默认值为`[1, 16, 32, 64]`，意味着默认测试 4 种不同并发度下的性能。并发量不能超过`config.ini`中的`max_batch_size`。否则，超出部分的请求会在推理队列中等待。

- `--prompt-tokens` 和 `--completion-tokens`

  输入token和输出token数量。它们是一个列表，列表中的元素是一一对应关系，即，`(--prompt-tokens[i]`, `--completion-tokens[i])` 是一组。比如在默认列表中，`[1, 128, 128, 2048, 2048]`和`[128, 128, 2048, 128, 2048]`，测试组合分别是，`(1, 128)`、`(128, 128)`、`(128, 2048)`、`(2048, 128)`和`(2048, 2048)`

- `--tp`

  模型在张量并行时，使用的显卡数量。必须是2的整数次幂。默认为 1。

- `--top_k`、`--top_p` 和 `temperature`

  这三个参数用来采样生成的 token_id。

- `--csv`

  一个 csv 文件路径，用来存放测试结果。默认是 `./profile_generation.csv`

- `--log-level`

  日志级别。默认是 'ERROR'

- `--test-round`

  测试的轮数，默认是 10。表示每组测试设置，会测试 10 轮，统计其平均结果。

我们把一组 `(并发数, prompt_token数量, completion-token数量)` 称为一组测试用例。所以，脚本执行的`测试用例总数 = 并发数列表长度 x prompt_token 列表长度`，`测试规模 = 测试用例总数 x 测试轮数`。用户可以根据自己的实际情况，灵活的调整测试参数。
