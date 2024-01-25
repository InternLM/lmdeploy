# KV Cache Quantization and Test Results

For the LLaMa-7B fp16 model with a maximum length of 2048, the server requires approximately 1030MB of GPU memory to store kv_cache for each concurrent session created. This means that even an A100 80G can only serve a limited number of users.

To reduce runtime GPU memory usage, we have implemented PTQ quantization for kv cache, using the following formula:

```bash
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```

## How to Enable KV Cache INT8

### **Step One**

Get the quantization parameters and save them to the original HF model directory:

```bash
# get minmax
export HF_MODEL=internlm/internlm-chat-7b

lmdeploy lite calibrate \
  $HF_MODEL \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --work-dir $HF_MODEL
```

### **Step Two**

Test the chat performance. Note that setting `--quant-policy 4` would set to KV Cache int8 mode.

```bash
lmdeploy chat turbomind $HF_MODEL --model-format hf --quant-policy 4
```

## GPU Memory Test

The test object is the [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) model.
Testing method:

1. Use `deploy.py` to convert the model, modify the maximum concurrency in the `workspace` configuration; adjust the number of requests in `llama_config.ini`.
2. Compile and run `bin/llama_triton_example` to obtain the GPU memory situation of the fp16 version under different batch_size.
3. Enable quantization, re-run `bin/llama_triton_example` to obtain the GPU memory situation of the int8 version under different batch_size.

Below shows the comparison of GPU memory between the two versions:

| batch_size | fp16 memory(MiB) | int8 memory(MiB) | diff(MiB) |
| :--------: | :--------------: | :--------------: | :-------: |
|     8      |      22337       |      18241       |   -4096   |
|     16     |      30593       |      22369       |   -8224   |
|     32     |      47073       |      30625       |  -16448   |
|     48     |      63553       |      38881       |  -24672   |

Compared to directly quantizing Weight (such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/)), we have done a comparative estimation of memory growth in the 7B model for both methods, with some data from [llama.cpp](https://github.com/ggerganov/llama.cpp).

![](../../../resources/batch_memory.png)

As can be seen, the fp16 version requires 1030MB of GPU memory for each concurrency, so quantizing kv_cache can significantly reduce the rate of increase of runtime memory.

## Accuracy Test

The test object is the [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) command model.

Below is the result of PTQ quantization of `kCacheKVInt8` method with only 128 randomly selected data from the c4 dataset. The accuracy was tested using [opencompass](https://github.com/InternLM/opencompass) before and after quantization.

|     task      |     dataset     |    metric     | int8  | fp16  | diff  |
| :-----------: | :-------------: | :-----------: | :---: | :---: | :---: |
|   Language    |   winogrande    |   accuracy    | 60.77 | 61.48 | -0.71 |
|   Knowledge   |       nq        |     score     | 2.69  | 2.60  | +0.09 |
|   Reasoning   |      gsm8k      |   accuracy    | 33.28 | 34.72 | -1.44 |
|   Reasoning   |       bbh       | naive_average | 20.12 | 20.51 | -0.39 |
| Understanding | openbookqa_fact |   accuracy    | 82.40 | 82.20 | +0.20 |
| Understanding |   eprstmt-dev   |   accuracy    | 90.62 | 88.75 | +1.87 |
|    Safety     |   crows_pairs   |   accuracy    | 32.56 | 31.43 | +1.13 |

Note that both `kCacheKVInt8` and `WeightInt4` methods can be enabled at the same time.
Please refer to [w4a16](./w4a16.md) do `WeightInt4` and then
start chat like:

```shell
lmdeploy chat turbomind ./internlm-chat-7b-4bit --model-format awq --quant-policy 4
```
