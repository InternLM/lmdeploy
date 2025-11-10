# 支持的模型

以下列表分别为 LMDeploy TurboMind 引擎和 PyTorch 引擎在不同软硬件平台下支持的模型

## TurboMind CUDA 平台

|              Model               |      Size      | Type | FP16/BF16 | KV INT8 | KV INT4 | W4A16 |
| :------------------------------: | :------------: | :--: | :-------: | :-----: | :-----: | :---: |
|              Llama               |    7B - 65B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|              Llama2              |    7B - 70B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|              Llama3              |    8B, 70B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|             Llama3.1             |    8B, 70B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|     Llama3.2<sup>\[2\]</sup>     |     1B, 3B     | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|             InternLM             |    7B - 20B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            InternLM2             |    7B - 20B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|           InternLM2.5            |       7B       | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            InternLM3             |       8B       | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|       InternLM-XComposer2        |  7B, 4khd-7B   | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|      InternLM-XComposer2.5       |       7B       | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|            Intern-S1             |      241B      | MLLM |    Yes    |   Yes   |   Yes   |  No   |
|          Intern-S1-mini          |      8.3B      | MLLM |    Yes    |   Yes   |   Yes   |  No   |
|               Qwen               |   1.8B - 72B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|     Qwen1.5<sup>\[1\]</sup>      |  1.8B - 110B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|      Qwen2<sup>\[2\]</sup>       |   0.5B - 72B   | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|            Qwen2-MoE             |    57BA14B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|              Qwen3               |   0.6B-235B    | LLM  |    Yes    |   Yes   |  Yes\*  |  Yes  |
|     Qwen2.5<sup>\[2\]</sup>      |   0.5B - 72B   | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|     Mistral<sup>\[1\]</sup>      |       7B       | LLM  |    Yes    |   Yes   |   Yes   |  No   |
|             Mixtral              |  8x7B, 8x22B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|           DeepSeek-V2            |   16B, 236B    | LLM  |    Yes    |   Yes   |   Yes   |  No   |
|          DeepSeek-V2.5           |      236B      | LLM  |    Yes    |   Yes   |   Yes   |  No   |
|             Qwen-VL              |       7B       | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|           DeepSeek-VL            |       7B       | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|             Baichuan             |       7B       | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            Baichuan2             |       7B       | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            Code Llama            |    7B - 34B    | LLM  |    Yes    |   Yes   |   Yes   |  No   |
|                YI                |    6B - 34B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|          LLaVA(1.5,1.6)          |    7B - 34B    | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|             InternVL             |  v1.1 - v1.5   | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|            InternVL2             | 1-2B, 8B - 76B | MLLM |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
| InternVL2.5(MPO)<sup>\[2\]</sup> |    1 - 78B     | MLLM |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|    InternVL3<sup>\[2\]</sup>     |    1 - 78B     | MLLM |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|   InternVL3.5<sup>\[3\]</sup>    |  1 - 241BA28B  | MLLM |    Yes    |  Yes\*  |  Yes\*  |  No   |
|             ChemVLM              |    8B - 26B    | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|       MiniCPM-Llama3-V-2_5       |       -        | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|          MiniCPM-V-2_6           |       -        | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|               GLM4               |       9B       | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            CodeGeeX4             |       9B       | LLM  |    Yes    |   Yes   |   Yes   |   -   |
|              Molmo               |    7B-D,72B    | MLLM |    Yes    |   Yes   |   Yes   |  No   |
|             gpt-oss              |    20B,120B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |

“-” 表示还没有验证。

```{note}
* [1] turbomind 引擎不支持 window attention。所以，对于应用了 window attention，并开启了对应的开关"use_sliding_window"的模型，比如 Mistral、Qwen1.5 等，在推理时，请选择 pytorch engine
* [2] 当模型的 head_dim 非 128 时，turbomind 不支持它的 kv cache 4/8 bit 量化和推理。比如，llama3.2-1B，qwen2-0.5B，internvl2-1B 等等
```

## PyTorchEngine CUDA 平台

|             Model              |      Size       | Type | FP16/BF16 | KV INT8 | KV INT4 | W8A8 | W4A16 |
| :----------------------------: | :-------------: | :--: | :-------: | :-----: | :-----: | :--: | :---: |
|             Llama              |    7B - 65B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|             Llama2             |    7B - 70B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|             Llama3             |     8B, 70B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|            Llama3.1            |     8B, 70B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|            Llama3.2            |     1B, 3B      | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|          Llama3.2-VL           |    11B, 90B     | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|             Llama4             | Scout, Maverick | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|            InternLM            |    7B - 20B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|           InternLM2            |    7B - 20B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|          InternLM2.5           |       7B        | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|           InternLM3            |       8B        | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|           Intern-S1            |      241B       | MLLM |    Yes    |   Yes   |   Yes   | Yes  |   -   |
|         Intern-S1-mini         |      8.3B       | MLLM |    Yes    |   Yes   |   Yes   | Yes  |   -   |
|           Baichuan2            |       7B        | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  No   |
|           Baichuan2            |       13B       | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|            ChatGLM2            |       6B        | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|               YI               |    6B - 34B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|            Mistral             |       7B        | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|            Mixtral             |   8x7B, 8x22B   | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|              QWen              |   1.8B - 72B    | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|            QWen1.5             |   0.5B - 110B   | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|          QWen1.5-MoE           |      A2.7B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|             QWen2              |   0.5B - 72B    | LLM  |    Yes    |   Yes   |   No    | Yes  |  Yes  |
|            Qwen2.5             |   0.5B - 72B    | LLM  |    Yes    |   Yes   |   No    | Yes  |  Yes  |
|             Qwen3              |   0.6B - 235B   | LLM  |    Yes    |   Yes   |  Yes\*  |  -   |  Yes  |
|           QWen3-Next           |       80B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|            QWen2-VL            |     2B, 7B      | MLLM |    Yes    |   Yes   |   No    |  No  |  Yes  |
|           QWen2.5-VL           |    3B - 72B     | MLLM |    Yes    |   No    |   No    |  No  |  No   |
|            QWen3-VL            |    2B - 235B    | MLLM |    Yes    |   No    |   No    |  No  |  No   |
|          DeepSeek-MoE          |       16B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|          DeepSeek-V2           |    16B, 236B    | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|         DeepSeek-V2.5          |      236B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|          DeepSeek-VL2          |    3B - 27B     | MLLM |    Yes    |   No    |   No    |  No  |  No   |
|            MiniCPM3            |       4B        | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|         MiniCPM-V-2_6          |       8B        | LLM  |    Yes    |   No    |   No    |  No  |  Yes  |
|             Gemma              |      2B-7B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|           StarCoder2           |     3B-15B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|           Phi-3-mini           |      3.8B       | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|          Phi-3-vision          |      4.2B       | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|           Phi-4-mini           |      3.8B       | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|          CogVLM-Chat           |       17B       | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|          CogVLM2-Chat          |       19B       | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
| LLaVA(1.5,1.6)<sup>\[2\]</sup> |     7B-34B      | MLLM |    No     |   No    |   No    |  No  |  No   |
|         InternVL(v1.5)         |     2B-26B      | MLLM |    Yes    |   Yes   |   Yes   |  No  |  Yes  |
|           InternVL2            |     1B-76B      | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|        InternVL2.5(MPO)        |     1B-78B      | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|           InternVL3            |     1B-78B      | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|          InternVL3.5           |   1B-241BA28B   | MLLM |    Yes    |   Yes   |   Yes   |  No  |  No   |
| Mono-InternVL<sup>\[1\]</sup>  |       2B        | MLLM |   Yes\*   |   Yes   |   Yes   |  -   |   -   |
|            ChemVLM             |     8B-26B      | MLLM |    Yes    |   Yes   |   No    |  -   |   -   |
|             Gemma2             |     9B-27B      | LLM  |    Yes    |   Yes   |   Yes   |  -   |   -   |
|             Gemma3             |     1B-27B      | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|             GLM-4              |       9B        | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|           GLM-4-0414           |       9B        | LLM  |    Yes    |   Yes   |   Yes   |  -   |   -   |
|             GLM-4V             |       9B        | MLLM |    Yes    |   Yes   |   Yes   |  No  |  Yes  |
|       GLM-4.1V-Thinking        |       9B        | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|            GLM-4.5             |      355B       | LLM  |    Yes    |   Yes   |   Yes   |  -   |   -   |
|          GLM-4.5-Air           |      106B       | LLM  |    Yes    |   Yes   |   Yes   |  -   |   -   |
|           CodeGeeX4            |       9B        | LLM  |    Yes    |   Yes   |   Yes   |  -   |   -   |
|          Phi-3.5-mini          |      3.8B       | LLM  |    Yes    |   Yes   |   No    |  -   |   -   |
|          Phi-3.5-MoE           |     16x3.8B     | LLM  |    Yes    |   Yes   |   No    |  -   |   -   |
|         Phi-3.5-vision         |      4.2B       | MLLM |    Yes    |   Yes   |   No    |  -   |   -   |
|              SDAR              |    1.7B-30B     | LLM  |    Yes    |   Yes   |   No    |  -   |   -   |

```{note}
* [1] 目前，Mono-InternVL不支持FP16，因为数值不稳定。请改用BF16
* [2] 自 0.6.4 之后，PyTorch 引擎移除了对 llava 模型原始格式的支持。我们建议使用它们对应的 transformers 格式的模型。这些模型可以在 https://huggingface.co/llava-hf 中找到
```

## PyTorchEngine 其他平台

|                |           |      |  Atlas 800T A2   |  Atlas 800T A2   | Atlas 800T A2 | Atlas 800T A2 | Atlas 300I Duo |  Atlas 800T A3   | Maca C500 | Cambricon |
| :------------: | :-------: | :--: | :--------------: | :--------------: | :-----------: | :-----------: | :------------: | :--------------: | :-------: | :-------: |
|     Model      |   Size    | Type | FP16/BF16(eager) | FP16/BF16(graph) |  W8A8(graph)  | W4A16(eager)  |  FP16(graph)   | FP16/BF16(eager) |  BF/FP16  |  BF/FP16  |
|     Llama2     | 7B - 70B  | LLM  |       Yes        |       Yes        |      Yes      |      Yes      |       -        |       Yes        |    Yes    |    Yes    |
|     Llama3     |    8B     | LLM  |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |       Yes        |    Yes    |    Yes    |
|    Llama3.1    |    8B     | LLM  |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |       Yes        |    Yes    |    Yes    |
|   InternLM2    | 7B - 20B  | LLM  |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |       Yes        |    Yes    |    Yes    |
|  InternLM2.5   | 7B - 20B  | LLM  |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |       Yes        |    Yes    |    Yes    |
|   InternLM3    |    8B     | LLM  |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |       Yes        |    Yes    |    Yes    |
|    Mixtral     |   8x7B    | LLM  |       Yes        |       Yes        |      No       |      No       |      Yes       |        -         |    Yes    |    Yes    |
|  QWen1.5-MoE   |   A2.7B   | LLM  |       Yes        |        -         |      No       |      No       |       -        |        -         |    Yes    |     -     |
|   QWen2(.5)    |    7B     | LLM  |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |        -         |    Yes    |    Yes    |
|    QWen2-VL    |  2B, 7B   | MLLM |       Yes        |       Yes        |       -       |       -       |       -        |        -         |    Yes    |    No     |
|   QWen2.5-VL   | 3B - 72B  | MLLM |       Yes        |       Yes        |       -       |       -       |      Yes       |        -         |    Yes    |    No     |
|   QWen2-MoE    |  A14.57B  | LLM  |       Yes        |        -         |      No       |      No       |       -        |        -         |    Yes    |     -     |
|     QWen3      | 0.6B-235B | LLM  |       Yes        |       Yes        |      No       |      No       |      Yes       |       Yes        |    Yes    |    Yes    |
|  DeepSeek-V2   |    16B    | LLM  |        No        |       Yes        |      No       |      No       |       -        |        -         |     -     |     -     |
| InternVL(v1.5) |  2B-26B   | MLLM |       Yes        |        -         |      Yes      |      Yes      |       -        |        -         |    Yes    |     -     |
|   InternVL2    |  1B-40B   | MLLM |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |        -         |    Yes    |    Yes    |
|  InternVL2.5   |  1B-78B   | MLLM |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |        -         |    Yes    |    Yes    |
|   InternVL3    |  1B-78B   | MLLM |       Yes        |       Yes        |      Yes      |      Yes      |      Yes       |        -         |    Yes    |    Yes    |
|  CogVLM2-chat  |    19B    | MLLM |       Yes        |        No        |       -       |       -       |       -        |        -         |    Yes    |     -     |
|     GLM4V      |    9B     | MLLM |       Yes        |        No        |       -       |       -       |       -        |        -         |     -     |     -     |
