# Request Throughput Test Method

In the applications, the length of the user's input prompt and the size of generated tokens are dynamic. The static inference performance is insufficient to reflect the inference engine's ability to handle the dynamic characteristics.

Therefore, it is necessary to use real dialogue data to evaluate the dynamic inference capabilities of the inference engine. This article will introduce how to test the dynamic inference performance of LMDeploy on localhost.

The evaluation script is `profile_throughput.py`. Before running it, please install the lmdeploy precompiled package, download the evaluation script and the test dataset:

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
RPM(request\\ per\\ minute) = Number\\ of\\ prompts/TotalTime * 60
$$

Total time includes prefill time.

## Example

We take `internlm-7b` as an example. The entire benchmark procedure is:

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 profile_throughput.py ./ShareGPT_V3_unfiltered_cleaned_split.json internlm/internlm-7b
```

## Command details

```shell
python3 profile_throughput.py --help
```
