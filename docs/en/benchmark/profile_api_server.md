# Profile API Server

The way to profiling `api_server` performance is similar to the method for [profiling throughput](./profile_throughput.md). The difference is `api_server` should be launched successfully before testing.

The profiling script is `profile_restful_api.py`. Before running it, please install the lmdeploy precompiled package, download the script and the test dataset:

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Metrics

LMDeploy records the performance metrics like first token latency, token throughput (tokens/s) and request throughput (RPM)

`first_token_latency` is only reported in the case of streaming inference.

The formula for calculating `token throughput` is:

$$
TokenThroughput = Number\\ of\\ generated\\ tokens/TotalTime
$$

And the formula for calculating `request throughput` is:

$$
RPM(request\\ per\\ minute)=Number\\ of\\ prompts/TotalTime * 60
$$

Total time includes prefill time.

## Profile

In this section, we take [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) as an example to show the benchmark procedure.

### Launch api_server

```shell
lmdeploy serve api_server internlm/internlm-7b
```

If you would like to change the server's port or other parameters, such as inference engine, max batch size and etc., please run `lmdeploy serve api_server -h` or read [this](../serving/api_server.md) guide to get the detailed explanation.

### Profile

```shell
python3 profile_restful_api.py http://0.0.0.0:23333 internlm/internlm-7b ./ShareGPT_V3_unfiltered_cleaned_split.json
```

For detailed argument specification of `profile_restful_api.py`, such as request concurrency, sampling parameters an so on, please run the help command `python3 profile_restful_api.py -h`.
