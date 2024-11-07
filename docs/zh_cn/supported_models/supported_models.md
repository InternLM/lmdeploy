# 支持的模型

以下列表分别为 LMDeploy TurboMind 引擎和 PyTorch 引擎在不同软硬件平台下支持的模型

## TurboMind CUDA 平台

|         Model         |     Size     | Type | FP16/BF16 | KV INT8 | KV INT4 | W4A16 |
| :-------------------: | :----------: | :--: | :-------: | :-----: | :-----: | :---: |
|         Llama         |   7B - 65B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|        Llama2         |   7B - 70B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|        Llama3         |   8B, 70B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|       Llama3.1        |   8B, 70B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|       Llama3.2        |      3B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|       InternLM        |   7B - 20B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|       InternLM2       |   7B - 20B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|      InternLM2.5      |      7B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|  InternLM-XComposer2  | 7B, 4khd-7B  | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
| InternLM-XComposer2.5 |      7B      | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|         Qwen          |  1.8B - 72B  | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|        Qwen1.5        | 1.8B - 110B  | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|         Qwen2         |  1.5B - 72B  | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|        Mistral        |      7B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|        Mixtral        | 8x7B, 8x22B  | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|        Qwen-VL        |      7B      | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|      DeepSeek-VL      |      7B      | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|       Baichuan        |      7B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|       Baichuan2       |      7B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|      Code Llama       |   7B - 34B   | LLM  |    Yes    |   Yes   |   Yes   |  No   |
|          YI           |   6B - 34B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|    LLaVA(1.5,1.6)     |   7B - 34B   | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|       InternVL        | v1.1 - v1.5  | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|       InternVL2       | 2B, 8B - 76B | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
| MiniCPM-Llama3-V-2_5  |      -       | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|     MiniCPM-V-2_6     |      -       | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|    MiniGeminiLlama    |      7B      | MLLM |    Yes    |    -    |    -    |  Yes  |
|         GLM4          |      9B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|       CodeGeeX4       |      9B      | LLM  |    Yes    |   Yes   |   Yes   |   -   |

“-” 表示还没有验证。

```{note}
turbomind 引擎不支持 window attention。所以，对于应用了 window attention，并开启了对应的开关"use_sliding_window"的模型，比如 Mistral、Qwen1.5 等，在推理时，请选择 pytorch engine
```

## PyTorchEngine CUDA 平台

|     Model      |    Size     | Type | FP16/BF16 | KV INT8 | KV INT4 | W8A8 | W4A16 |
| :------------: | :---------: | :--: | :-------: | :-----: | :-----: | :--: | :---: |
|     Llama      |  7B - 65B   | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|     Llama2     |  7B - 70B   | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|     Llama3     |   8B, 70B   | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|    Llama3.1    |   8B, 70B   | LLM  |    Yes    |   Yes   |   Yes   |  No  |   -   |
|    Llama3.2    |   1B, 3B    | LLM  |    Yes    |   Yes   |   Yes   |  No  |   -   |
|  Llama3.2-VL   |  11B, 90B   | MLLM |    Yes    |   Yes   |   Yes   |  No  |   -   |
|    InternLM    |  7B - 20B   | LLM  |    Yes    |   Yes   |   Yes   | Yes  |   -   |
|   InternLM2    |  7B - 20B   | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|  InternLM2.5   |     7B      | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|   Baichuan2    |     7B      | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  No   |
|   Baichuan2    |     13B     | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|    ChatGLM2    |     6B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|     Falcon     |  7B - 180B  | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|       YI       |  6B - 34B   | LLM  |    Yes    |   Yes   |   Yes   |  No  |  Yes  |
|    Mistral     |     7B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|    Mixtral     | 8x7B, 8x22B | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|      QWen      | 1.8B - 72B  | LLM  |    Yes    |   Yes   |   Yes   |  No  |  Yes  |
|    QWen1.5     | 0.5B - 110B | LLM  |    Yes    |   Yes   |   Yes   |  No  |  Yes  |
|  QWen1.5-MoE   |    A2.7B    | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|     QWen2      | 0.5B - 72B  | LLM  |    Yes    |   Yes   |   No    |  No  |  Yes  |
|    QWen2-VL    |   2B, 7B    | MLLM |    Yes    |   Yes   |   No    |  No  |  No   |
|  DeepSeek-MoE  |     16B     | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|  DeepSeek-V2   |  16B, 236B  | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|    MiniCPM3    |     4B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|     Gemma      |    2B-7B    | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|      Dbrx      |    132B     | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|   StarCoder2   |   3B-15B    | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|   Phi-3-mini   |    3.8B     | LLM  |    Yes    |   Yes   |   Yes   |  No  |  Yes  |
|  Phi-3-vision  |    4.2B     | MLLM |    Yes    |   Yes   |   Yes   |  No  |   -   |
|  CogVLM-Chat   |     17B     | MLLM |    Yes    |   Yes   |   Yes   |  No  |   -   |
|  CogVLM2-Chat  |     19B     | MLLM |    Yes    |   Yes   |   Yes   |  No  |   -   |
| LLaVA(1.5,1.6) |   7B-34B    | MLLM |    Yes    |   Yes   |   Yes   |  No  |   -   |
| InternVL(v1.5) |   2B-26B    | MLLM |    Yes    |   Yes   |   Yes   |  No  |  Yes  |
|   InternVL2    |   1B-40B    | MLLM |    Yes    |   Yes   |   Yes   |  No  |   -   |
| Mono-InternVL  |     2B      | MLLM |    Yes    |   Yes   |   Yes   |  No  |   -   |
|     Gemma2     |   9B-27B    | LLM  |    Yes    |   Yes   |   Yes   |  No  |   -   |
|      GLM4      |     9B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |  No   |
|     GLM-4V     |     9B      | MLLM |    Yes    |   Yes   |   Yes   |  No  |  No   |
|   CodeGeeX4    |     9B      | LLM  |    Yes    |   Yes   |   Yes   |  No  |   -   |
|  Phi-3.5-mini  |    3.8B     | LLM  |    Yes    |   Yes   |   No    |  No  |   -   |
|  Phi-3.5-MoE   |   16x3.8B   | LLM  |    Yes    |   Yes   |   No    |  No  |   -   |
| Phi-3.5-vision |    4.2B     | MLLM |    Yes    |   Yes   |   No    |  No  |   -   |

## PyTorchEngine 华为昇腾平台

|     Model      |   Size   | Type | FP16/BF16 | W4A16 |
| :------------: | :------: | :--: | :-------: | :---: |
|     Llama2     | 7B - 70B | LLM  |    Yes    |  Yes  |
|     Llama3     |    8B    | LLM  |    Yes    |  Yes  |
|    Llama3.1    |    8B    | LLM  |    Yes    |  Yes  |
|   InternLM2    | 7B - 20B | LLM  |    Yes    |  Yes  |
|  InternLM2.5   | 7B - 20B | LLM  |    Yes    |  Yes  |
|    Mixtral     |   8x7B   | LLM  |    Yes    |  No   |
|  QWen1.5-MoE   |  A2.7B   | LLM  |    Yes    |  No   |
|     QWen2      |    7B    | LLM  |    Yes    |  No   |
|   QWen2-MoE    | A14.57B  | LLM  |    Yes    |  No   |
| InternVL(v1.5) |  2B-26B  | MLLM |    Yes    |  Yes  |
|   InternVL2    |  1B-40B  | MLLM |    Yes    |  Yes  |
