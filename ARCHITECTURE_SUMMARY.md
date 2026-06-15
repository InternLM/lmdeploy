# LMDeploy PyTorch Backend 架构总结

## 核心设计理念

LMDeploy 采用**双后端架构**，通过统一的 `pipeline()` API 提供一致的用户体验。

```
用户代码
    |
    v
lmdeploy.pipeline()  (api.py)
    |
    +---> PyTorch Engine (lmdeploy/pytorch/)
    |       - 纯 Python 实现
    |       - 动态模型层替换
    |       - 62+ 模型支持
    |
    +---> TurboMind Engine (lmdeploy/turbomind/)
            - C++/CUDA 实现
            - 高性能推理引擎
            - 需要编译
```

---

## PyTorch Backend 核心组件

### 1. 模型补丁机制 (Model Patching)

**核心思想**: 加载 HuggingFace 模型后，动态替换其层为 LMDeploy 优化实现

关键文件:
- `pytorch/models/module_map.py` - 注册表：HF 类名 -> LMDeploy 替换类
- `pytorch/models/patch.py` - 运行时应用替换
- `pytorch/models/*.py` - 40+ 模型实现 (llama.py, qwen.py, deepseek_v2.py 等)

工作流程:
```
HF Model (LlamaForCausalLM)
    |
    v
[patch.py] 动态替换
    |
    v
LMDeploy Model (optimized attention, MLP, embeddings)
```

### 2. 优化模块库 (nn/)

可重用的优化模块:
- `linear/` - 量化线性层 (AWQ, W8A8, blocked-FP8, LoRA)
- `attention.py` - 优化的注意力机制
- `norm.py` - 归一化层
- `rotary_embedding.py` - RoPE 旋转位置编码
- `moe/` - MoE 专家混合模块

### 3. 内核层 (kernels/)

Triton/CUDA 自定义内核:
- `w8a8_triton_kernels.py` - W8A8 量化内核
- `cuda/` - CUDA 特定内核 (flashattention, pagedattention 等)

### 4. 后端调度器 (backends/)

按量化类型分发的内核/操作符:
- `cuda/` - CUDA 后端
- `awq/` - AWQ 量化后端
- `fp8/` - FP8 量化后端

---

## 引擎执行流程

### 主要组件

```
engine.py (主引擎)
    |
    +---> paging/scheduler.py (调度器)
    |       - 序列 -> 批次管理
    |       - prefill/decode 阶段
    |       - block eviction
    |       - prefix caching (BlockTrie)
    |
    +---> engine/engine_loop.py (异步推理循环)
    |
    +---> engine/model_agent/ (模型代理)
            - 模型加载和管理
            - 推理执行
```

### 配置数据类 (config.py)

| 配置类 | 用途 |
|--------|------|
| `ModelConfig` | 模型参数 (hidden_size, num_layers, dtype 等) |
| `CacheConfig` | KV Cache 配置 (block_size, max_batches 等) |
| `SchedulerConfig` | 调度器配置 (max_batches, max_session_len 等) |
| `BackendConfig` | 后端配置 (eager_mode, device_type) |
| `DistConfig` | 分布式配置 |

---

## 量化系统 (Lite/)

### 入口点 (`lite/apis/`)

- `calibrate.py` - 主校准入口
- `auto_awq.py` - AWQ 自动量化
- `gptq.py` - GPTQ 量化
- `smooth_quant.py` - SmoothQuant 量化

### 量化流程

```
加载 HF 模型
    |
    v
CalibrationContext 收集激活统计
    |
    v
计算缩放因子 (lite/quantization/)
    |
    v
写入量化权重
```

### 关键文件

- `lite/quantization/awq.py` - AWQ 实现
- `lite/quantization/weight/quantizer.py` - 权重量化器
- `lite/quantization/activation/observer.py` - 激活统计观察者
- `lite/modeling/` - 模型特定的 GPTQ 实现

---

## 视觉语言模型 (VLM/)

- `vl/model/` - VLM 预处理 (InternVL, Qwen-VL, LLaVA, CogVLM)
- `vl/media/` - 图像/视频加载器
- `pytorch/multimodal/` - PyTorch 引擎的多模态输入处理

---

## 其他关键文件

| 文件 | 用途 |
|------|------|
| `messages.py` | 核心类型 (GenerationConfig, EngineConfig 等) |
| `model.py` | 聊天模板 (对话格式化) |
| `archs.py` | 架构注册表 (模型架构名 -> 运行时补丁) |
| `tokenizer.py` | Tokenizer 包装器 |
| `serve/openai/` | OpenAI 兼容 API 服务器 |

---

## 添加新模型的步骤

参考 `/support-new-model` skill，一般步骤:

1. 在 `pytorch/models/` 创建新模型文件
2. 在 `module_map.py` 注册模型映射
3. 实现优化的 attention、MLP、embeddings
4. 在 `archs.py` 注册架构
5. 添加测试

---

## 性能优化关键技术

1. **Persistent Batch (Continuous Batching)** - 持续批处理
2. **Blocked KV Cache** - 分块 KV 缓存
3. **Dynamic Split & Fuse** - 动态分割和融合
4. **Tensor Parallelism** - 张量并行
5. **Paged Attention** - 分页注意力
6. **Flash Attention** - 快速注意力内核
7. **4-bit Quantization (AWQ)** - 4 比特量化

---

## 当前状态 (2026-06)

- 版本: v0.13.0
- PyTorch 后端: 完全可用 (CPU/GPU)
- TurboMind 后端: 需要 C++ 编译
- 支持模型: 62+ (LLaMA, Qwen, DeepSeek, InternLM 等)
- 最新特性: Qwen3.5, llm-compressor 4bit 量化
