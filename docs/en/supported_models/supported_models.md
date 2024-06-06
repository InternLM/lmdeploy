# Supported Models

## Models supported by TurboMind

|        Model        |    Size     | FP16/BF16 | KV INT8 | KV INT4 | W4A16 |
| :-----------------: | :---------: | :-------: | :-----: | :-----: | :---: |
|        Llama        |  7B - 65B   |    Yes    |   Yes   |   Yes   |  Yes  |
|       Llama2        |  7B - 70B   |    Yes    |   Yes   |   Yes   |  Yes  |
|       Llama3        |   8B, 70B   |    Yes    |   Yes   |   Yes   |  Yes  |
|      InternLM       |  7B - 20B   |    Yes    |   Yes   |   Yes   |  Yes  |
|      InternLM2      |  7B - 20B   |    Yes    |   Yes   |   Yes   |  Yes  |
| InternLM-XComposer  |     7B      |    Yes    |    -    |    -    |  Yes  |
| InternLM-XComposer2 | 7B, 4khd-7B |    Yes    |    -    |    -    |  Yes  |
|        QWen         | 1.8B - 72B  |    Yes    |   Yes   |   Yes   |  Yes  |
|       QWen1.5       | 1.8B - 72B  |    Yes    |   Yes   |   Yes   |  Yes  |
|       Mistral       |     7B      |    Yes    |   Yes   |   Yes   |  No   |
|       QWen-VL       |     7B      |    Yes    |    -    |    -    |  Yes  |
|     DeepSeek-VL     |     7B      |    Yes    |    -    |    -    |  Yes  |
|      Baichuan       |     7B      |    Yes    |   Yes   |   Yes   |  Yes  |
|      Baichuan2      |     7B      |    Yes    |   Yes   |   Yes   |  Yes  |
|     Code Llama      |  7B - 34B   |    Yes    |   Yes   |   Yes   |  No   |
|         YI          |  6B - 34B   |    Yes    |   Yes   |   Yes   |  No   |
|   LLaVA(1.5,1.6)    |  7B - 34B   |    Yes    |    -    |    -    |  Yes  |
|    InternVL-Chat    | v1.1- v1.5  |    Yes    |    -    |    -    |  Yes  |
|   MiniGeminiLlama   |     7B      |    Yes    |   No    |   No    |  Yes  |
|        GLM-4        |     9B      |    Yes    |   Yes   |   Yes   |  No   |

"-" means not verified yet.

```{note}
The TurboMind engine doesn't support window attention. Therefore, for models that have applied window attention and have the corresponding switch "use_sliding_window" enabled, such as Mistral, QWen1.5 and etc., please choose the PyTorch engine for inference.
```

## Models supported by PyTorch

|    Model     |    Size    | FP16/BF16 | KV INT8 | W8A8 |
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
|  StarCoder2  |   3B-15B   |    Yes    |   No    |  No  |
|  Phi-3-mini  |    3.8B    |    Yes    |   No    |  No  |
| CogVLM-Chat  |    17B     |    Yes    |   No    |  No  |
| CogVLM2-Chat |    19B     |    Yes    |   No    |  No  |
