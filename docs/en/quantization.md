# PTQ test results

## Memory test

The test model is [Chinese-LLaMa-Alpaca 7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) instruction model.
Test step:

1. Use `deploy.py` to convert the model, modify the maximum number of concurrency in `workspace` configuration; adjust the number of requests in `llama_config.ini`
2. Compile and execute `bin/llama_triton_example` to obtain the memory footprint of the fp16 version at different batch_sizes
3. Execute the quantization script to obtain quantization parameters; modify the configuration file to make the [kCacheKVInt8](../../src/turbomind/models/llama/llama_utils.h) option take effect
4. Re-execute `bin/llama_triton_example` to get the int8 version in different batch_size video memory

The following is a video memory comparison of the two versions:


| batch_size | fp16 memory(MiB) | int8 memory(MiB) | diff(MiB) |
| :--------: | :--------------: | :--------------: | :-------: |
|     8      |      22337       |      18241       |   -4096   |
|     16     |      30593       |      22369       |   -8224   |
|     32     |      47073       |      30625       |  -16448   |
|     48     |      63553       |      38881       |  -24672   |

Compared with directly quantize Weight (such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/)), we made a comparative estimation of the memory growth of the two schemes in the 7B model , part of the data comes from [llama.cpp](https://github.com/ggerganov/llama.cpp).

![](../../resources/batch_memory.png)

Because each concurrency requires 1030MB to save kv_cache for 2048 tokens, and the server needs to consider the cost of high-concurrency scenarios, so quantizing kv_cache is more appropriate than directly quantizing weight.

It should be noted that `kCacheKVInt8` and `WeightInt4` can run at the same time, and we will provide relevant implementations later.

## Accuracy test

The test model is [Chinese-LLaMa-Alpaca 7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) instruction model.
Test step:

1. Use `deploy.py` to convert the model and run the docker service
2. Test the data set through `client.py` to obtain the accuracy of the fp16 version
3. Execute the quantization script to get the quantization parameters and put them in the weights directory; modify the configuration file to make the [kCacheKVInt8](../../src/turbomind/models/llama/llama_utils.h) option take effect
4. Execute `client.py` again to read the precision of the int8 version

The following is the `kCacheKVInt8` method only from the c4 dataset, randomly selecting 128 pieces of data to quantify, and the accuracy loss in the mmlu-social-science dataset.


| task |       dataset       | metric | fp16  | int8  | diff  |
| :--: | :-----------------: | :----: | :---: | :---: | :---: |
| Exam | mmlu-social-science | score  | 31.81 | 32.00 | +0.19 |

We noticed a slight improvement in accuracy. mmlu-social-science has a total of 3065 multiple-choice questions. The specific differences are as follows:

| Type | Quantity |
| :--------------------------: | :--: |
| The fp16 version is wrong, but the int8 version is correct | 72 |
| fp16 version is correct, int8 version is wrong | 66 |
| Both versions are wrong and the answers are different | 118 |

We have validated more datasets on larger models and will continue to update the results.
