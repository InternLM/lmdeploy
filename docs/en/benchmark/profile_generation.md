# Profile Token Latency and Throughput

We profile the latency and throughput of generated tokens with fixed batch size and fixed input/output token.

The profiling script is `profile_generation.py`. Before running it, please install the lmdeploy precompiled package and download the profiling script:

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
```

## Metrics

LMDeploy records test results like first token latency, token throughput (tokens/s), percentile data of each token's latency (P50, P75, P95, P99), GPU mem, etc.

`first_token_latency` is only reported in the case of streaming inference.

The formula for calculating `throughput` is:

$$
TokenThroughput = Number\\ of\\ generated\\ tokens/TotalTime
$$

Total time includes prefill time.

During the test process, all graphics cards on the node should not run any other programs, otherwise the statistics of GPU mem would be inaccurate.

## Profile

In this section, we take [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) as an example to show how to profile the inference engines of LMDeploy.

### Profile turbomind engine

```shell
cd lmdeploy/benchmark
python3 profile_generation.py internlm/internlm-7b
```

### Profile pytorch engine

```shell
cd lmdeploy/benchmark
python3 profile_generation.py internlm/internlm-7b --backend pytorch
```

For detailed argument specification of `profile_generation.py`, such as batch size, input and output token number an so on, please run the help command `python3 profile_generation.py -h`.
