# KV Cache 量化和测试结果

对于最大长度是 2048 的 LLaMa-7B fp16 模型，服务端每创建 1 个并发，都需要大约 1030MB 显存保存 kv_cache，即便是 A100 80G，能服务的用户也非常有限。

为了降低运行时显存，我们实现了 kv cache PTQ 量化，使用的公式如下：

```bash
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```

## 如何开启 KV Cache INT8

### **第一步**

通过以下命令，获取量化参数，并保存至原HF模型目录

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

### **第二步**

测试聊天效果。注意需要添加参数`--quant-policy 4`以开启KV Cache int8模式。

```bash
lmdeploy chat turbomind $HF_MODEL --model-format hf --quant-policy 4
```

## 显存测试

测试对象为 [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) 模型。
测试方法：

1. 使用 `deploy.py` 转换模型，修改 `workspace` 配置中的最大并发数；调整 `llama_config.ini` 中的请求数
2. 编译执行 `bin/llama_triton_example`，获取 fp16 版本在不同 batch_size 的显存情况
3. 开启量化，重新执行 `bin/llama_triton_example`，获取 int8 版本在不同 batch_size 显存情况

以下是两个版本的显存对比：

| batch_size | fp16 memory(MiB) | int8 memory(MiB) | diff(MiB) |
| :--------: | :--------------: | :--------------: | :-------: |
|     8      |      22337       |      18241       |   -4096   |
|     16     |      30593       |      22369       |   -8224   |
|     32     |      47073       |      30625       |  -16448   |
|     48     |      63553       |      38881       |  -24672   |

相对于直接量化 Weight（如 [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/)），我们做了两种方案在 7B 模型中的内存增长对比预估，部分数据来自 [llama.cpp](https://github.com/ggerganov/llama.cpp)。

![](../../../resources/batch_memory.png)

可以看到，fp16 版本每个并发需要 1030MB 显存，因此量化 kv_cache 能显著降低运行时的显存增长速度。

## 精度测试

测试对象为 [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b) 指令模型。

以下是 `kCacheKVInt8` 方法仅从 c4 数据集，随机选择 128 条数据 PTQ 量化。量化前后均使用 [opencompass](https://github.com/InternLM/opencompass) 测试精度。

|     task      |     dataset     |    metric     | int8  | fp16  | diff  |
| :-----------: | :-------------: | :-----------: | :---: | :---: | :---: |
|   Language    |   winogrande    |   accuracy    | 60.77 | 61.48 | -0.71 |
|   Knowledge   |       nq        |     score     | 2.69  | 2.60  | +0.09 |
|   Reasoning   |      gsm8k      |   accuracy    | 33.28 | 34.72 | -1.44 |
|   Reasoning   |       bbh       | naive_average | 20.12 | 20.51 | -0.39 |
| Understanding | openbookqa_fact |   accuracy    | 82.40 | 82.20 | +0.20 |
| Understanding |   eprstmt-dev   |   accuracy    | 90.62 | 88.75 | +1.87 |
|    Safety     |   crows_pairs   |   accuracy    | 32.56 | 31.43 | +1.13 |

需要注意的是，`kCacheKVInt8` 和 `WeightInt4` 两种方案可以同时开启。请参阅 [w4a16](./w4a16.md) 开启 `WeightInt4` ，然后测试聊天效果：

```shell
lmdeploy chat turbomind ./internlm-chat-7b-4bit --model-format awq --quant-policy 4
```
