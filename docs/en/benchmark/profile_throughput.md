# Request Throughput Test Method

In the applications, the length of the user's input prompt and the size of generated tokens are dynamic. The static inference performance is insufficient to reflect the inference engine's ability to handle the dynamic characteristics.

Therefore, it is necessary to use real dialogue data to evaluate the dynamic inference capabilities of the inference engine. This article will introduce how to test the dynamic inference performance of LMDeploy on localhost.

The evaluation script is `profile_throughput.py`. Before running it, please install the lmdeploy precompiled package, download the evaluation script and the test dataset:

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

During performance test, a specific model needs to be inputted. We recommend converting the model into turbomind format via `lmdeploy convert`, then proceed with testing.
The reason is to conveniently adjust the parameters of the inference engine in order to achieve better performance, such as batch size (max_batch_size), K/V cache size (max_cache_entry_count), etc. For detailed explanations of these parameters, please refer to [here](../turbomind_config.md).

In the following sections, we assume the model is in turbomind format.

## Metrics

LMDeploy records the performance metrics like first token latency, token throughput (tokens/s) and request throughput (RPM)

`first_token_latency` is only reported in the case of streaming inference.

The formula for calculating `token throughput` is:

$$
TokenThroughput=Number\\ of\\ generated\\ tokens/TotalTime
$$

And the formula for calculating `request throughput` is:
$$
RPM(request per minute)=Number\\ of\\ generated\\ tokens/TotalTime * 60
$$

Total time includes prefill time.

## Methods

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

  It represents the number of request threads with default value 64. Requests of concurrent threads will be batched by the inference engine. Its value should not exceed `max_batch_size` in `config.ini`. Otherwise, the excess requests will wait in the inference queue.

- `--num-prompts`

  The number of sampled prompts from dataset to process. The default is 2000.

- `--tp`

  The number of GPUs used when the inference is in tensor parallel mode. It must be a power of 2. The default is 1.

- `--top_k`、`--top_p` and `temperature`

  They are used to sample the generated token_id.

- `--stream_output`

  Indicator for streaming output. The default is `True`.

- `--csv`

  The path of a csv file to save the result with default value `./profile_throughput.csv`

- `--log-level`

  The log level. The default is `ERROR`.

- `--seed`

  It is the seed used in sampling prompts from dataset with default value 0.
