# PTQ Quantization Benchmark Results

## Benchmark the Graphics Memory Usage

We take the [Chinese-LLaMa-Alpaca 7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) instruction model as the benchmark target. The benchmark process is as follows:

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

Since each concurrency requires 1030MB of the graphics memory to save the kv_cache for 2048 tokens, and the server side needs to consider the cost of high concurrency scenarios, it is more appropriate to run kv_cache quantization rather than directly quantize the weights.

Note that `kCacheKVInt8` and `WeightInt4` can be used simultaneously, and we will provide relevant implementations later.

## Benchmark the Accuracy

Here we take the [Chinese-LLaMa-Alpaca 7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) instruction model as the benchmark target again. The benchmark process is as follows:

1. Convert the model with `deploy.py` and run the docker service
2. Test the fp16 version accuracy with `client.py` using the dataset
3. Execute the quantization script to get the quantization parameters, and put them into the weights directory. Then modify the configuration file to make [kCacheKVInt8](../../src/turbomind/models/llama/llama_utils.h) option to be effective
4. Execute the `client.py` again to get the int8 version precision

The following table is the precision result obtained by the `kCacheKVInt8` method after quantizing 128 randomly selected data from the c4 dataset and testing it on the mmlu-social-science dataset, which has a total of 3065 multiple-choice questions:

| task |       dataset       | metric | fp16  | int8  | diff  |
| :--: | :-----------------: | :----: | :---: | :---: | :---: |
| Exam | mmlu-social-science | score  | 31.81 | 32.00 | +0.19 |

We noticed that there is a slight improvement in the precision, and the differences are as follows:

|                      Type                      | Number |
| :--------------------------------------------: | :----: |
| fp16 version failed but int8 version got right |   72   |
| fp16 version got right but int8 version failed |   66   |
|      failed in both fp16 and int8 version      |  118   |

We have validated the quantization implementation on more datasets and larger models and will keep updating the results.
