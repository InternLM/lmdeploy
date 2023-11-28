# Static Inference Performance Test Method

We view the performance of the inference engine under the fixed batch and fixed input/output token as static inference performance.

The evaluation script is `profile_generation.py`. Before running it, please install the lmdeploy precompiled package and download the evaluation script:

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
```

During performance test, a specific model needs to be inputted. We recommend converting the model into turbomind format via `lmdeploy convert`, then proceed with testing.
The reason is to conveniently adjust the parameters of the inference engine in order to achieve better performance, such as batch size (max_batch_size), K/V cache size (max_cache_entry_count), etc. For detailed explanations of these parameters, please refer to [here](../turbomind_config.md).

In the following sections, we assume the model is in turbomind format.

## Metrics

LMDeploy records test results like first token latency, throughput (tokens/s), percentile data of each token's latency (P50, P75, P95, P99), GPU mem, etc.

`first_token_latency` is only reported in the case of streaming inference.

The formula for calculating `throughput` is:
$$
Throughput=Number\\ of\\ generated\\ tokens/TotalTime
$$
Total time includes prefill time.

During the test process, all graphics cards on the node should not run any other programs, otherwise the statistics of GPU mem would be inaccurate.

## Method

```shell
python3 profile_generation.py <model_path> <optional arguments>
```

`model_path` refers to the path on localhost where the model in turbomind format is located.

Optional arguments are listed as below:

- `--concurrency`

  It represents the number of request threads. Requests of concurrent threads will be batched by the inference engine. It is a list with default value `[1, 16, 32, 64]`, which implies that the performance under 4 different levels of concurrency is tested. The level of concurrency should not exceed `max_batch_size` in [turbomind config](../turbomind_config.md#turbomind-20-config). Otherwise, there will be `max_batch_size - concurrency` number of threads randomly waiting almost at any time during test.

- `--prompt-tokens` and `--completion-tokens`

  Input token and output token numbers. They are lists of the same length. The elements in the list correspond one-to-one, that is,
  the pair `(prompt_tokens[i], completion_tokens[i])` is a test case. In the default list `[1, 128, 128, 2048, 2048]` and `[128, 128, 2048, 128, 2048]`, the test cases are `(1, 128)`, `(128, 128)`, `(128, 2048)`, `(2048, 128)` and `(2048, 2048)`

- `--tp`

  The number of GPUs used when the inference is in tensor parallel mode. It must be a power of 2. The default is 1.

- `--top_k`, `--top_p` and `temperature`

  They are used to sample the generated token_id.

- `--csv`

  A csv file path used to store test results. The default is `./profile_generation.csv`

- `--log-level`

  The log level. The default is 'ERROR'.

- `--test-round`

  The number of test rounds is set to 10 by default. This means that each case will undergo 10 rounds of testing, and the average result will be calculated.

We refer to a tuple of `(#concurrency, #prompt_token, #completion_token)` as a test case. Therefore, the total number of test cases (`#test_cases`) executed by the script is `len(concurrency) * len(prompt-tokens)`, and the total test rounds  are `#test_cases * #test_round`. Users can flexibly adjust test parameters according to their actual situation.
