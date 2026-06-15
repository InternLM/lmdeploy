# GEMM 调优指南

GEMM（通用矩阵乘法）是基于 Transformer 的大语言模型推理中最关键的操作之一。优化 GEMM 性能可以显著提升整体推理吞吐量和降低延迟。本指南介绍如何在 LMDeploy 中调优 GEMM 操作以获得最佳性能。

## 概述

LMDeploy 提供了多种针对不同场景优化的 GEMM 实现：

- **FP16/FP32 GEMM**: 标准精度矩阵乘法
- **INT8 W8A8 GEMM**: 8 位权重和激活值量化
- **FP8 GEMM**: 混合精度工作负载的 FP8 精度
- **Blocked GEMM**: 针对分页注意力和 KV 缓存操作优化

GEMM 内核的选择及其配置取决于您的硬件、模型架构和部署场景。

## 硬件特定优化

### NVIDIA GPU

LMDeploy 在 NVIDIA GPU 上使用 CUDA 内核和 cuBLAS/cuBLASLt 进行 GEMM 操作。不同 GPU 架构的性能表现：

| GPU 架构 | 推荐方案 |
|---------|---------|
| Volta (V100) | FP16 + cuBLAS，支持 MXFP4 |
| Ampere (A100/A800) | FP16/BF16，INT8 + Tensor Cores |
| Hopper (H100/H800) | FP8，增强的 FP16 Tensor Cores |
| Blackwell (B200) | FP8/FP4 + 最新 Tensor Cores |

#### 启用 Tensor Cores

Tensor Cores 为矩阵运算提供显著加速。确保已启用：

```python
import torch

# 为 Ampere+ GPU 启用 TF32（提升 FP32 性能）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### AMD ROCm GPU

对于 AMD GPU，LMDeploy 使用 rocBLAS 和自定义 Triton 内核。设置适当的后端：

```python
from lmdeploy import pipeline
from lmdeploy.pytorch.config import PytorchEngineConfig

engine_config = PytorchEngineConfig(
    device_type='cuda',  # 也适用于 ROCm
    dtype='float16'
)

pipe = pipeline('your-model-path', engine_config=engine_config)
```

## 量化感知 GEMM 调优

### W8A8 量化

仅权重 8 位量化减少内存带宽需求，同时保持精度：

```python
from lmdeploy import pipeline
from lmdeploy.messages import QuantPolicy

# 启用 W8A8 量化
quant_policy = QuantPolicy(
    w_bits=8,      # 8 位权重
    a_bits=8,      # 8 位激活值
    calib_data='calibration_dataset.json'
)

pipe = pipeline(
    'meta-llama/Llama-3-8b-Instruct',
    quant_policy=quant_policy
)
```

**性能提示：**
- 使用代表您工作负载的校准数据
- W8A8 最适合批处理大小 > 4 的场景
- 预期内存减少约 1.5-2 倍，精度损失最小

### FP8 量化

FP8 为某些模型提供比 INT8 更好的动态范围：

```python
from lmdeploy.messages import QuantPolicy

# 启用 FP8 量化
quant_policy = QuantPolicy(
    w_bits=8,
    a_bits=8,
    quant_type='fp8'  # 使用 FP8 E4M3 格式
)
```

**何时使用 FP8：**
- 对 INT8 量化误差敏感的模型
- 需要更高数值精度的工作负载
- 具有原生 FP8 支持的 Hopper (H100/H800) GPU

### AWQ（激活感知权重量化）

AWQ 实现 4 位量化，精度下降最小：

```bash
# 使用 AWQ 校准和量化模型
lmdeploy lite auto_awq \
    --model meta-llama/Llama-3-8b-Instruct \
    --calib-dataset c4 \
    --calib-samples 128 \
    --calib-seqlen 2048 \
    --w-bits 4 \
    --w-group-size 128 \
    --export-dir ./awq_quantized_model
```

然后部署量化后的模型：

```python
from lmdeploy import pipeline

pipe = pipeline('./awq_quantized_model')
```

## 缓存配置对 GEMM 的影响

KV 缓存配置通过内存访问模式直接影响 GEMM 性能：

```python
from lmdeploy import pipeline
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

# 为您的工作负载优化缓存
cache_config = CacheConfig(
    max_batches=64,           # 根据可用 GPU 内存调整
    block_size=16,            # 短序列使用较小的块
    num_gpu_blocks=2000,      # 更长上下文则增加
    cache_max_entry_count=0.8 # 预留 20% 给激活值
)

scheduler_config = SchedulerConfig(
    max_batches=64,
    max_session_len=8192,     # 匹配您的典型上下文长度
    max_request_output_len=512
)

pipe = pipeline(
    'meta-llama/Llama-3-8b-Instruct',
    cache_config=cache_config,
    scheduler_config=scheduler_config
)
```

**调优指南：**
- **block_size**: 混合长度工作负载使用 16，统一长序列使用 64-128
- **max_batches**: 较高值提高吞吐量但可能降低单请求延迟
- **cache_max_entry_count**: 较低值（0.6-0.7）为 GEMM 工作区留出更多内存

## 批处理大小优化

由于更好的 GPU 利用率，GEMM 效率随批处理大小增大而提高：

```python
# 示例：寻找最优批处理大小
import time
from lmdeploy import pipeline

pipe = pipeline('meta-llama/Llama-3-8b-Instruct')

prompts = ["What is AI?"] * 32  # 测试批大小为 32

start = time.time()
responses = pipe(prompts)
end = time.time()

print(f"批大小: {len(prompts)}")
print(f"总时间: {end - start:.2f}s")
print(f"吞吐量: {len(prompts) / (end - start):.2f} req/s")
```

**建议：**
- 小批量（1-4）：优化延迟，使用 FP16
- 中等批量（8-32）：平衡延迟/吞吐量，考虑 INT8
- 大批量（64+）：最大化吞吐量，启用连续批处理

## GEMM 性能分析

使用内置分析器识别 GEMM 瓶颈：

### PyTorch Profiler

```bash
# 启用 CPU 分析
export LMDEPLOY_PROFILE_CPU=1

# 启用 CUDA 内核分析
export LMDEPLOY_PROFILE_CUDA=1

# 3 秒后开始分析（预热）
export LMDEPLOY_PROFILE_DELAY=3

# 分析 10 秒
export LMDEPLOY_PROFILE_DURATION=10

# 保存分析数据
export LMDEPLOY_PROFILE_OUT_PREFIX="/path/to/profile_"

python your_inference_script.py
```

使用 PyTorch Profiler 或 TensorBoard 分析生成的配置文件：

```bash
tensorboard --logdir /path/to/profile_
```

### Nsight Systems（NVIDIA）

详细的 CUDA 内核分析：

```bash
# 单 GPU
nsys profile python your_script.py

# 多 GPU + Ray
export LMDEPLOY_RAY_NSYS_ENABLE=1
export LMDEPLOY_RAY_NSYS_OUT_PREFIX="/path/to/nsight_"
python your_script.py
```

## 模型特定调优

### LLaMA 系列

LLaMA 模型受益于：
- 分组查询注意力优化
- SwiGLU 激活函数融合
- RMSNorm 内核融合

```python
# LLaMA 特定优化
from lmdeploy import pipeline

pipe = pipeline(
    'meta-llama/Llama-3-8b-Instruct',
    tp=1,  # 张量并行（多 GPU 使用 >1）
    session_len=8192
)
```

### Qwen 系列

Qwen 模型支持高级特性：
- 混合专家（MoE）路由优化
- 多模态融合（Qwen-VL）

```python
# Qwen MoE 优化
pipe = pipeline(
    'Qwen/Qwen1.5-MoE-A2.7B-Chat',
    moe_tp=2  # MoE 层独立的张量并行
)
```

### DeepSeek 模型

DeepSeek V2/V3 需要：
- MLA（多头潜在注意力）优化
- DeepGemm 集成用于 FP8

```python
# DeepSeek V3 + FP8
from lmdeploy.messages import QuantPolicy

quant_policy = QuantPolicy(w_bits=8, a_bits=8, quant_type='fp8')

pipe = pipeline(
    'deepseek-ai/DeepSeek-V3',
    quant_policy=quant_policy,
    tp=8  # 需要多 GPU
)
```

## 常见性能问题及解决方案

### 问题 1：GPU 利用率低

**症状：** GPU 使用率