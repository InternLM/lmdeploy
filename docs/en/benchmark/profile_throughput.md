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

python3 profile_throughput.py ./ShareGPT_V3_unfiltered_cleaned_split.json internlm/internlm-7b --cache-count 0.7
```

## Command details

```shell
python3 profile_throughput.py <dataset> <model_path> <optional arguments>
```

The required parameters are:

- `dataset`

  The path of the downloaded dataset

- `model_path`

  The path on localhost where the model in turbomind format is located.

Optional arguments are listed as below:

- `--concurrency`

  It represents the number of request threads. Defaults to 256.

- `--num-prompts`

  The number of sampled prompts from dataset to process. The default is 5000.

- `--tp`

  The number of GPUs used when the inference is in tensor parallel mode. It must be a power of 2. The default is 1.

- `--top_k`、`--top_p` and `--temperature`

  They are used to sample the generated token_id.

- `--stream_output`

  Indicator for streaming output. The default is `True`.

- `--csv`

  The path of a csv file to save the result with default value `./profile_throughput.csv`

- `--log-level`

  The log level. The default is `ERROR`.

- `--seed`

  It is the seed used in sampling prompts from dataset with default value 0.

- `--cache-count`

  The ratio of k/v cache memory. Default to 0.5.

- `--model-format`

  The layout of the model. Its value should be among \['hf', 'llama', 'awq', None\]. Defaults to `hf`. `llama` means `meta_llama` and `awq` indicates the quantized model by AWQ algorithm
