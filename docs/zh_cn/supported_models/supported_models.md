# 支持的模型

## TurboMind 支持的模型

|        模型        |  模型规模  | FP16/BF16 | KV INT8 | KV INT4 | W4A16 |
| :----------------: | :--------: | :-------: | :-----: | :-----: | :---: |
|       Llama        |  7B - 65B  |    Yes    |   Yes   |   Yes   |  Yes  |
|       Llama2       |  7B - 70B  |    Yes    |   Yes   |   Yes   |  Yes  |
|       Llama3       |  8B, 70B   |    Yes    |   Yes   |   Yes   |  Yes  |
|      InternLM      |  7B - 20B  |    Yes    |   Yes   |   Yes   |  Yes  |
|     InternLM2      |  7B - 20B  |    Yes    |   Yes   |   Yes   |  Yes  |
| InternLM-XComposer |     7B     |    Yes    |    -    |    -    |   -   |
|        QWen        | 1.8B - 72B |    Yes    |   Yes   |   Yes   |  Yes  |
|      QWen1.5       | 1.8B - 72B |    Yes    |   Yes   |   Yes   |  Yes  |
|      QWen-VL       |     7B     |    Yes    |    -    |    -    |   -   |
|    DeepSeek-VL     |     7B     |    Yes    |    -    |    -    |   -   |
|      Baichuan      |     7B     |    Yes    |   Yes   |   Yes   |  Yes  |
|     Baichuan2      |     7B     |    Yes    |   Yes   |   Yes   |  Yes  |
|     Code Llama     |  7B - 34B  |    Yes    |   Yes   |   Yes   |  No   |
|         YI         |  6B - 34B  |    Yes    |   Yes   |   Yes   |  No   |
|   LLaVA(1.5,1.6)   |  7B - 34B  |    Yes    |    -    |    -    |   -   |
|   InternVL-Chat    |     -      |    Yes    |    -    |    -    |   -   |
|  MiniGeminiLlama   |     7B     |    Yes    |   No    |   No    |  No   |

“-” 表示还没有验证。

```{note}
turbomind 引擎不支持 window attention。所以，对于应用了 window attention，并开启了对应的开关"use_sliding_window"的模型，在推理时，请选择 pytorch engine
```

### PyTorch 支持的模型

|     模型     |  模型规模  | FP16/BF16 | KV INT8 | W8A8 |
| :----------: | :--------: | :-------: | :-----: | :--: |
|    Llama     |  7B - 65B  |    Yes    |   No    | Yes  |
|    Llama2    |  7B - 70B  |    Yes    |   No    | Yes  |
|    Llama3    |  8B, 70B   |    Yes    |   No    | Yes  |
|   InternLM   |  7B - 20B  |    Yes    |   No    | Yes  |
|  InternLM2   |  7B - 20B  |    Yes    |   No    |  -   |
|  Baichuan2   |  7B - 13B  |    Yes    |   No    | Yes  |
|   ChatGLM2   |     6B     |    Yes    |   No    |  No  |
|    Falcon    | 7B - 180B  |    Yes    |   No    |  No  |
|      YI      |  6B - 34B  |    Yes    |   No    |  No  |
|   Mistral    |     7B     |    Yes    |   No    |  No  |
|   Mixtral    |    8x7B    |    Yes    |   No    |  No  |
|     QWen     | 1.8B - 72B |    Yes    |   No    |  No  |
|   QWen1.5    | 0.5B - 72B |    Yes    |   No    |  No  |
| QWen1.5-MoE  |   A2.7B    |    Yes    |   No    |  No  |
| DeepSeek-MoE |    16B     |    Yes    |   No    |  No  |
|    Gemma     |   2B-7B    |    Yes    |   No    |  No  |
|     Dbrx     |    132B    |    Yes    |   No    |  No  |
