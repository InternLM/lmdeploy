# Profile Triton Inference Server

Triton Inference Server (TIS) is another serving method supported by LMDeploy besides `api_server`. Its performance testing methods and metrics are similar to those of [api_server](./profile_api_server.md).

The profiling script is `profile_serving.py`. Before running it, please install the lmdeploy precompiled package, download the profiling script and the test dataset:

```shell
pip install 'lmdeploy[serve]'
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Metrics

LMDeploy records the performance metrics like first token latency, token throughput (tokens/s) and request throughput (RPM)

`first_token_latency` is only reported in the case of streaming inference.

The formula for calculating `token throughput` is:

$$
TokenThroughput=Number\\ of\\ generated\\ tokens/TotalTime
$$

And the formula for calculating `request throughput` is:

$$
RPM(request\\ per\\ minute)=Number\\ of\\ prompts/TotalTime * 60
$$

Total time includes prefill time.

## Profile

In this section, we take [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) as an example to show the benchmark procedure.

### Launch triton inference server

Before launching the server, the LLM model must be converted to the turbomind format in advance.

```shell
lmdeploy convert internlm internlm/internlm-7b --dst-path ./internlm-7b --trust-remote-code
```

Then, the triton inference server can be launched by:

```shell
bash ./internlm-7b/service_docker_up.sh
```

### Profile

```shell
python3 profile_serving.py 0.0.0.0:33337 ./internlm-7b/triton_models/tokenizer ./ShareGPT_V3_unfiltered_cleaned_split.json
```

For detailed argument specification of `profile_serving.py`, such as request concurrency, sampling parameters an so on, please run the help command `python3 profile_serving.py -h`.
