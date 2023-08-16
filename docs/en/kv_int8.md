# KV Cache Quantization Benchmark Results

## Benchmark the Graphics Memory Usage

We take the [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) instruction model as the benchmark target. The benchmark process is as follows:

1. Use the `deploy.py` to convert the model, modify the maximum concurrent amount in `workspace`, and adjust the request amount in `llama_config.ini`
2. Compile the `bin/llama_triton_example` and get the graphics usage status of the fp16 version model under different batch_size settings
3. Execute the quantization script and get the quantization parameters. Then modify the config file to make [kCacheKVInt8](../../src/turbomind/models/llama/llama_utils.h) be effective
4. Re-execute the `bin/llama_triton_example` and get the graphics usage status of the int8 version model under different batch_size settings

Here is the benchmark result between the two versions of the model:

| batch_size | fp16 memory(MiB) | int8 memory(MiB) | diff(MiB) |
| :--------: | :--------------: | :--------------: | :-------: |
|     8      |      22337       |      18241       |   -4096   |
|     16     |      30593       |      22369       |   -8224   |
|     32     |      47073       |      30625       |  -16448   |
|     48     |      63553       |      38881       |  -24672   |

To compare with the weight quantization method such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/) , we benchmarked the memory usages of the 7B model between the two solutions, with part of the data from [llama.cpp](https://github.com/ggerganov/llama.cpp) . Here is the result:

![](../../resources/batch_memory.png)

It can be seen that each concurrency requires 1030MB of GPU memory to save kv_cache for 2048 tokens, so quantizing kv_cache can significantly reduce the speed of the memory growth at runtime.

Note that `kCacheKVInt8` and `WeightInt4` can be used simultaneously.

## Benchmark the Accuracy

The quantification method is PTQ, and the related formula is as follows:

```
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```

Here we take the [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) instruction model as the benchmark target again. The benchmark process is as follows:

1. Convert the model with `deploy.py` and run the docker service
2. Test the fp16 version accuracy with `client.py` using the dataset
3. Execute the quantization script to get the quantization parameters, and put them into the weights directory. Then modify the configuration file to make [kCacheKVInt8](../../src/turbomind/models/llama/llama_utils.h) option to be effective
4. Execute the `client.py` again to get the int8 version precision

The following table is the precision result obtained by the `kCacheKVInt8` method after quantizing 128 randomly selected data from the c4 dataset and testing it with [opencompass](https://github.com/InternLM/opencompass).

|     task      |     dataset     |    metric     | int8  | fp16  | diff  |
| :-----------: | :-------------: | :-----------: | :---: | :---: | :---: |
|   Language    |   winogrande    |   accuracy    | 60.77 | 61.48 | -0.71 |
|   Knowledge   |       nq        |     score     | 2.69  | 2.60  | +0.09 |
|   Reasoning   |      gsm8k      |   accuracy    | 33.28 | 34.72 | -1.44 |
|   Reasoning   |       bbh       | naive_average | 20.12 | 20.51 | -0.39 |
| Understanding | openbookqa_fact |   accuracy    | 82.40 | 82.20 | +0.20 |
| Understanding |   eprstmt-dev   |   accuracy    | 90.62 | 88.75 | +1.87 |
|    Safety     |   crows_pairs   |   accuracy    | 32.56 | 31.43 | +1.13 |
