# API Server Performance Test Method

The way to profiling api_server performance is similar to the method for [profiling throughput](./profile_throughput.md). The difference is api_server should be launched successfully before testing.

The evaluation script is `profile_restful_api.py`. Before running it, please install the lmdeploy precompiled package, download the evaluation script and the test dataset:

```shell
pip install 'lmdeploy[serve]>=0.1.0a1'
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
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
TokenThroughput = Number\\ of\\ generated\\ tokens/TotalTime
$$

And the formula for calculating `request throughput` is:

$$
RPM(request\\ per\\ minute)=Number\\ of\\ prompts/TotalTime * 60
$$

Total time includes prefill time.

## Example

We take `internlm-7b` as an example. The entire benchmark procedure is:

```shell
pip install 'lmdeploy[serve]>=0.1.0a1'
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# get internlm-7b from huggingface and convert it to turbomind format
lmdeploy convert internlm internlm/internlm-7b --dst-path ./internlm-7b

# launch server
lmdeploy serve api_server ./internlm-7b --server-port 23333

# open another terminal and run the following command in the directory `lmdeploy/benchmark`
python3 ./profile_restful_api.py http://0.0.0.0:23333 ./internlm-7b/triton_models/tokenizer ./ShareGPT_V3_unfiltered_cleaned_split.json
```

## Methods

Please refer to [this](../restful_api.md) guide to start `api_server`.
The argument `--instance-num` reflects the inference instance number. When more than `--instance-num` requests arrive at the `api_server` at the same time, the exceeding part of the requests will wait in the inference queue.

```shell
python3 profile_restful_api.py <server_addr> <tokenizer_path> <dataset> <optional arguments>
```

The required parameters are:

- `server_addr`

  The address of api_server with format `http://{server_ip}:{server_port}`

- `tokenizer_path`

  The path of the tokenizer model, which is used to encode the dataset to get the token size of prompts and responses

- `dataset`

  The path of the downloaded dataset

Optional arguments are listed as below:

- `--concurrency`

  It represents the number of request threads with default value 64. Requests of concurrent threads will be batched by the inference engine. Its value should not exceed the number of inference instances in the api_server.
  Otherwise, the excess requests will wait in the inference queue.

- `--num-prompts`

  The number of sampled prompts from dataset to process. The default is 2000.

- `--top_p` and `--temperature`

  They are used to sample the generated token_id.

- `--stream_output`

  Indicator for streaming output. The default is `False`.

- `--csv`

  The path of a csv file to save the result with default value `../profile_api_server.csv`

- `--seed`

  It is the seed used in sampling prompts from dataset with default value 0.
