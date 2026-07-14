# 支持的模型

以下列表分别为 LMDeploy TurboMind 引擎和 PyTorch 引擎在不同软硬件平台下支持的模型。

## 引擎支持矩阵

此表提供了每个模型系列支持的推理引擎的快速参考。使用此表选择适合您部署的引擎。

**图例：**
- ✅ = 完全支持
- ⚠️ = 部分支持（见注释）
- ❌ = 不支持
- - = 未验证/测试

### 模型系列支持概览

| 模型系列 | TurboMind (CUDA) | PyTorch (CUDA) | PyTorch (昇腾) | PyTorch (Maca) | PyTorch (寒武纪) | 推荐引擎 |
|:------------:|:----------------:|:--------------:|:----------------:|:--------------:|:-------------------:|:------------------:|
| Llama 1/2/3/3.1/3.2 | ✅ | ✅ | ✅ | ✅ | ✅ | TurboMind（更快） |
| Llama 4 | ❌ | ✅ | - | - | - | 仅 PyTorch |
| InternLM 1/2/2.5/3 | ✅ | ✅ | ✅ | ✅ | ✅ | TurboMind（更快） |
| Intern-S1/S2 | ✅ | ✅ | - | - | - | 两者都支持 |
| Qwen 1/1.5/2/2.5 | ✅ | ✅ | ✅ | ✅ | ✅ | TurboMind（更快） |
| Qwen 3/3.5 | ✅* | ✅ | ✅ | ✅ | ✅ | PyTorch（完整功能） |
| Mistral/Mixtral | ⚠️¹ | ✅ | ✅ | ✅ | ✅ | PyTorch（窗口注意力） |
| DeepSeek V2/V2.5 | ✅ | ✅ | ⚠️ | - | - | TurboMind (CUDA) |
| DeepSeek V3/V3.2 | ❌ | ✅ | ❌ | - | - | 仅 PyTorch |
| DeepSeek-VL/VL2 | ✅ / ❌ | ✅ | - | - | - | VL2 用 PyTorch |
| Baichuan 1/2 | ✅ | ✅ | - | - | - | TurboMind（更快） |
| Yi | ✅ | ✅ | - | - | - | TurboMind（更快） |
| Code Llama | ✅ | ✅ | - | - | - | TurboMind（更快） |
| GLM-4/4V/4.5/5 | ✅ | ✅ | - | ✅ | - | 两者都支持 |
| Gemma 2/3 | ❌ | ✅ | - | ✅ | - | 仅 PyTorch |
| Phi-3/4 | ❌ | ✅ | - | - | - | 仅 PyTorch |
| LLaVA 1.5/1.6 | ✅ | ⚠️² | - | - | - | TurboMind 或 PyTorch |
| Qwen-VL 系列 | ✅ | ✅ | ✅ | ✅ | ❌ | TurboMind (VL), PyTorch (VL2+) |
| InternVL 1/2/2.5/3/3.5 | ✅ | ✅ | ✅ | ✅ | - | TurboMind（更快） |
| MiniCPM-V | ✅ | ✅ | - | - | - | 两者都支持 |
| Molmo | ✅ | ✅ | - | - | - | 两者都支持 |
| CogVLM 1/2 | ❌ | ✅ | ✅ | ✅ | - | 仅 PyTorch |
| StarCoder2 | ❌ | ✅ | - | - | - | 仅 PyTorch |
| gpt-oss | ✅ | ✅ | - | - | - | TurboMind（更快） |

**注释：**
1. **Mistral/Qwen1.5**: TurboMind 不支持窗口注意力。如果启用了 `use_sliding_window`，请使用 PyTorch 引擎。
2. **LLaVA**: PyTorch 引擎在 v0.6.4 之后移除了对原始 LLaVA 模型的支持。使用来自 https://huggingface.co/llava-hf 的 transformers 模型
3. **Qwen3.5**: TurboMind 目前不支持视觉编码器。使用 PyTorch 获得完整的 VLM 功能。
4. **DeepSeek V3**: 由于模型架构复杂性，需要 PyTorch 引擎。
5. 从版本 0.11.1 开始，PyTorchEngine 不再提供对 mllama 的支持。

### 量化支持对比

| 量化类型 | TurboMind | PyTorch 引擎 | 说明 |
|:-----------------:|:---------:|:--------------:|:------|
| FP16/BF16 | ✅ 所有模型 | ✅ 所有模型 | 基础精度 |
| W4A16 (AWQ/GPTQ) | ✅ 大多数 LLM | ✅ 选定模型 | 4 位权重 |
| W8A8 | ❌ 有限 | ✅ 选定模型 | 8 位权重和激活值 |
| KV Cache INT8 | ✅ 大多数模型 | ✅ 大多数模型 | 需要 head_dim=128* |
| KV Cache INT4 | ✅ 大多数模型 | ✅ 选定模型 | 需要 head_dim=128* |

*\* 当模型的 head_dim 不为 128 时（例如 llama3.2-1B、qwen2-0.5B），TurboMind 不支持 KV cache 4/8 位量化。*

### 功能支持矩阵

| 功能 | TurboMind | PyTorch 引擎 |
|:-------:|:---------:|:--------------:|
| 连续批处理 | ✅ | ✅ |
| 分页注意力 | ✅ | ✅ |
| 张量并行 | ✅ | ✅ |
| 前缀缓存 | ✅ | ✅ |
| LoRA | ✅ | ✅ |
| 推测解码 | ✅ | ✅ |
| 结构化输出 | ✅ | ✅ |
| 多 GPU (TP) | ✅ | ✅ |
| 多节点 | ✅ | ✅ (Ray) |
| 视觉语言 | ✅ 有限 | ✅ 完整 |
| MoE 模型 | ✅ 选定 | ✅ 完整 |
| 动态形状 | ⚠️ 有限 | ✅ 完整 |
| 自定义内核 | ✅ CUDA | ✅ Triton/CUDA |
| 易于添加模型 | ❌ 复杂 | ✅ 简单 |

**何时选择哪个引擎：**

**选择 TurboMind 当：**
- 您需要在 CUDA GPU 上获得最大性能
- 使用标准 LLM 架构（Llama、Qwen、InternLM）
- 在高吞吐量要求的生产环境中部署
- 使用 4 位量化（W4A16）

**选择 PyTorch 引擎当：**
- 使用 VLM（视觉语言模型）
- 使用 TurboMind 尚未支持的新模型
- 需要动态形状支持
- 在非 CUDA 平台上开发（昇腾、Maca、寒武纪）
- 添加自定义模型支持（更容易集成）
- 使用窗口注意力模型（Mistral、某些 Qwen 变体）

---

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
|          Intern-S1-Pro           |      1TB       | MLLM |    Yes    |    -    |    -    |  No   |
|               Qwen               |   1.8B - 72B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|     Qwen1.5<sup>\[1\]</sup>      |  1.8B - 110B   | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|      Qwen2<sup>\[2\]</sup>       |   0.5B - 72B   | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|            Qwen2-MoE             |    57BA14B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|     Qwen2.5<sup>\[2\]</sup>      |   0.5B - 72B   | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|              Qwen3               |   0.6B-235B    | LLM  |    Yes    |   Yes   |  Yes\*  |  Yes  |
|     Qwen3.5<sup>\[3\]</sup>      |   0.8B-397B    | LLM  |    Yes    |   Yes   |   No    |  Yes  |
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
|          GLM-4.7-Flash           |      30B       | LLM  |    Yes    |   No    |   No    |  No   |
|            CodeGeeX4             |       9B       | LLM  |    Yes    |   Yes   |   Yes   |   -   |
|              Molmo               |    7B-D,72B    | MLLM |    Yes    |   Yes   |   Yes   |  No   |
|             gpt-oss              |    20B,120B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |

“-” 表示还没有验证。

```{note}
* [1] turbomind 引擎不支持 window attention。所以，对于应用了 window attention，并开启了对应的开关"use_sliding_window"的模型，比如 Mistral、Qwen1.5 等，在推理时，请选择 pytorch engine
* [2] 当模型的 head_dim 非 128 时，turbomind 不支持它的 kv cache 4/8 bit 量化和推理。比如，llama3.2-1B，qwen2-0.5B，internvl2-1B 等等
* [3] turbomind 目前暂不支持 Qwen3.5 系列的视觉编码器。
```

## PyTorchEngine CUDA 平台

|             Model              |      Size       | Type | FP16/BF16 | KV INT8 | KV INT4 | W8A8 | W4A16 |
| :----------------------------: | :-------------: | :--: | :-------: | :-----: | :-----: | :--: | :---: |
|             Llama              |    7B - 65B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|             Llama2             |    7B - 70B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|             Llama3             |     8B, 70B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|            Llama3.1            |     8B, 70B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|            Llama3.2            |     1B, 3B      | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|             Llama4             | Scout, Maverick | MLLM |    Yes    |   Yes   |   Yes   |  -   |   -   |
|            InternLM            |    7B - 20B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|           InternLM2            |    7B - 20B     | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|          InternLM2.5           |       7B        | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|           InternLM3            |       8B        | LLM  |    Yes    |   Yes   |   Yes   | Yes  |  Yes  |
|           Intern-S1            |      241B       | MLLM |    Yes    |   Yes   |   Yes   | Yes  |   -   |
|         Intern-S1-mini         |      8.3B       | MLLM |    Yes    |   Yes   |   Yes   | Yes  |   -   |
|         Intern-S1-Pro          |       1TB       | MLLM |    Yes    |    -    |    -    |  -   |  No   |
|       Intern-S2-Preview        |     35B-A3B     | MLLM |    Yes    |   No    |   No    |  No  |  No   |
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
|            QWen3.5             |    0.8B-397B    | MLLM |    Yes    |   No    |   No    |  No  |  No   |
|           Qwen3-Omni           |     30B-A3B     | MLLM |    Yes    |   No    |   No    |  No  |  No   |
|           QWen3-Next           |       80B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|            QWen2-VL            |     2B, 7B      | MLLM |    Yes    |   Yes   |   No    |  No  |  Yes  |
|           QWen2.5-VL           |    3B - 72B     | MLLM |    Yes    |   No    |   No    |  No  |  No   |
|            QWen3-VL            |    2B - 235B    | MLLM |    Yes    |   No    |   No    |  No  |  No   |
|          DeepSeek-MoE          |       16B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|          DeepSeek-V2           |    16B, 236B    | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|         DeepSeek-V2.5          |      236B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|          DeepSeek-V3           |      685B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|         DeepSeek-V3.2          |      685B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
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
|         GLM-4.7-Flash          |       30B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|             GLM-5              |      754B       | LLM  |    Yes    |   No    |   No    |  No  |  No   |
|           CodeGeeX4            |       9B        | LLM  |    Yes    |   Yes   |   Yes   |  -   |   -   |
|          Phi-3.5-mini          |      3.8B       | LLM  |    Yes    |   Yes   |   No    |  -   |   -   |
|          Phi-3.5-MoE           |     16x3.8B     | LLM  |    Yes    |   Yes   |   No    |  -   |   -   |
|         Phi-3.5-vision         |      4.2B       | MLLM |    Yes    |   Yes   |   No    |  -   |   -   |
|              SDAR              |    1.7B-30B     | LLM  |    Yes    |   Yes   |   No    |  -   |   -   |

```{note}
* [1] 目前，Mono-InternVL不支持FP16，因为数值不稳定。请改用BF16
* [2] 自 0.6.4 之后，PyTorch 引擎移除了对 llava 模型原始格式的支持。我们建议使用它们对应的 transformers 格式的模型。这些模型可以在 https://huggingface.co/llava-hf 中找到
自 0.11.1 起，PytorchEngine 移除了 mllama 的支持
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
