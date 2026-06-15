# GEMM Tuning Guide

GEMM (General Matrix Multiply) is one of the most critical operations in transformer-based LLM inference. Optimizing GEMM performance can significantly improve overall inference throughput and latency. This guide explains how to tune GEMM operations in LMDeploy for optimal performance.

## Overview

LMDeploy provides multiple GEMM implementations optimized for different scenarios:

- **FP16/FP32 GEMM**: Standard precision matrix multiplication
- **INT8 W8A8 GEMM**: 8-bit weight and activation quantization
- **FP8 GEMM**: FP8 precision for mixed-precision workloads
- **Blocked GEMM**: Optimized for paged attention and KV cache operations

The choice of GEMM kernel and its configuration depends on your hardware, model architecture, and deployment scenario.

## Hardware-Specific Optimization

### NVIDIA GPUs

LMDeploy leverages CUDA kernels and cuBLAS/cuBLASLt for GEMM operations on NVIDIA GPUs. Performance varies across GPU architectures:

| GPU Architecture | Recommended Approach |
|-----------------|---------------------|
| Volta (V100)    | FP16 with cuBLAS, MXFP4 support |
| Ampere (A100/A800) | FP16/BF16, INT8 with Tensor Cores |
| Hopper (H100/H800) | FP8, FP16 with enhanced Tensor Cores |
| Blackwell (B200) | FP8/FP4 with latest Tensor Cores |

#### Enabling Tensor Cores

Tensor Cores provide significant speedup for matrix operations. Ensure they are enabled:

```python
import torch

# Enable TF32 for Ampere+ GPUs (improves FP32 performance)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### AMD ROCm GPUs

For AMD GPUs, LMDeploy uses rocBLAS and custom Triton kernels. Set the appropriate backend:

```python
from lmdeploy import pipeline
from lmdeploy.pytorch.config import PytorchEngineConfig

engine_config = PytorchEngineConfig(
    device_type='cuda',  # Works with ROCm as well
    dtype='float16'
)

pipe = pipeline('your-model-path', engine_config=engine_config)
```

## Quantization-Aware GEMM Tuning

### W8A8 Quantization

Weight-only 8-bit quantization reduces memory bandwidth requirements while maintaining accuracy:

```python
from lmdeploy import pipeline
from lmdeploy.messages import QuantPolicy

# Enable W8A8 quantization
quant_policy = QuantPolicy(
    w_bits=8,      # 8-bit weights
    a_bits=8,      # 8-bit activations
    calib_data='calibration_dataset.json'
)

pipe = pipeline(
    'meta-llama/Llama-3-8b-Instruct',
    quant_policy=quant_policy
)
```

**Performance Tips:**
- Use calibration data representative of your workload
- W8A8 works best for batch sizes > 4
- Expect ~1.5-2x memory reduction with minimal accuracy loss

### FP8 Quantization

FP8 provides better dynamic range than INT8 for certain models:

```python
from lmdeploy.messages import QuantPolicy

# Enable FP8 quantization
quant_policy = QuantPolicy(
    w_bits=8,
    a_bits=8,
    quant_type='fp8'  # Use FP8 E4M3 format
)
```

**When to use FP8:**
- Models sensitive to INT8 quantization errors
- Workloads requiring higher numerical precision
- Hopper (H100/H800) GPUs with native FP8 support

### AWQ (Activation-aware Weight Quantization)

AWQ achieves 4-bit quantization with minimal accuracy degradation:

```bash
# Calibrate and quantize model with AWQ
lmdeploy lite auto_awq \
    --model meta-llama/Llama-3-8b-Instruct \
    --calib-dataset c4 \
    --calib-samples 128 \
    --calib-seqlen 2048 \
    --w-bits 4 \
    --w-group-size 128 \
    --export-dir ./awq_quantized_model
```

Then deploy the quantized model:

```python
from lmdeploy import pipeline

pipe = pipeline('./awq_quantized_model')
```

## Cache Configuration Impact on GEMM

KV cache configuration directly affects GEMM performance through memory access patterns:

```python
from lmdeploy import pipeline
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

# Optimize cache for your workload
cache_config = CacheConfig(
    max_batches=64,           # Adjust based on available GPU memory
    block_size=16,            # Smaller blocks for short sequences
    num_gpu_blocks=2000,      # Increase for longer contexts
    cache_max_entry_count=0.8 # Reserve 20% for activations
)

scheduler_config = SchedulerConfig(
    max_batches=64,
    max_session_len=8192,     # Match your typical context length
    max_request_output_len=512
)

pipe = pipeline(
    'meta-llama/Llama-3-8b-Instruct',
    cache_config=cache_config,
    scheduler_config=scheduler_config
)
```

**Tuning Guidelines:**
- **block_size**: Use 16 for mixed-length workloads, 64-128 for uniform long sequences
- **max_batches**: Higher values increase throughput but may reduce per-request latency
- **cache_max_entry_count**: Lower values (0.6-0.7) leave more memory for GEMM workspace

## Batch Size Optimization

GEMM efficiency improves with larger batch sizes due to better GPU utilization:

```python
# Example: Finding optimal batch size
import time
from lmdeploy import pipeline

pipe = pipeline('meta-llama/Llama-3-8b-Instruct')

prompts = ["What is AI?"] * 32  # Test with batch of 32

start = time.time()
responses = pipe(prompts)
end = time.time()

print(f"Batch size: {len(prompts)}")
print(f"Total time: {end - start:.2f}s")
print(f"Throughput: {len(prompts) / (end - start):.2f} req/s")
```

**Recommendations:**
- Small batches (1-4): Optimize for latency, use FP16
- Medium batches (8-32): Balance latency/throughput, consider INT8
- Large batches (64+): Maximize throughput, enable continuous batching

## Profiling GEMM Performance

Use built-in profilers to identify GEMM bottlenecks:

### PyTorch Profiler

```bash
# Enable CPU profiling
export LMDEPLOY_PROFILE_CPU=1

# Enable CUDA kernel profiling
export LMDEPLOY_PROFILE_CUDA=1

# Start profiling after 3 seconds (warmup)
export LMDEPLOY_PROFILE_DELAY=3

# Profile for 10 seconds
export LMDEPLOY_PROFILE_DURATION=10

# Save profile data
export LMDEPLOY_PROFILE_OUT_PREFIX="/path/to/profile_"

python your_inference_script.py
```

Analyze the generated profile files with PyTorch Profiler or TensorBoard:

```bash
tensorboard --logdir /path/to/profile_
```

### Nsight Systems (NVIDIA)

For detailed CUDA kernel analysis:

```bash
# Single GPU
nsys profile python your_script.py

# Multi-GPU with Ray
export LMDEPLOY_RAY_NSYS_ENABLE=1
export LMDEPLOY_RAY_NSYS_OUT_PREFIX="/path/to/nsight_"
python your_script.py
```

## Model-Specific Tuning

### LLaMA Family

LLaMA models benefit from:
- Grouped-query attention optimization
- SwiGLU activation function fusion
- RMSNorm kernel fusion

```python
# LLaMA-specific optimization
from lmdeploy import pipeline

pipe = pipeline(
    'meta-llama/Llama-3-8b-Instruct',
    tp=1,  # Tensor parallelism (use >1 for multi-GPU)
    session_len=8192
)
```

### Qwen Series

Qwen models support advanced features:
- Mixture-of-Experts (MoE) routing optimization
- Multi-modal fusion (for Qwen-VL)

```python
# Qwen MoE optimization
pipe = pipeline(
    'Qwen/Qwen1.5-MoE-A2.7B-Chat',
    moe_tp=2  # Separate TP for MoE layers
)
```

### DeepSeek Models

DeepSeek V2/V3 require:
- MLA (Multi-head Latent Attention) optimization
- DeepGemm integration for FP8

```python
# DeepSeek V3 with FP8
from lmdeploy.messages import QuantPolicy

quant_policy = QuantPolicy(w_bits=8, a_bits=8, quant_type='fp8')

pipe = pipeline(
    'deepseek-ai/DeepSeek-V3',
    quant_policy=quant_policy,
    tp=8  # Requires multi-GPU
)
```

## Common Performance Issues and Solutions

### Issue 1: Low GPU Utilization

**Symptoms:** GPU usage < 50%, high CPU overhead

**Solutions:**
- Increase batch size
- Enable continuous batching
- Reduce I/O bottleneck (preload data)
- Check for CPU-bound tokenization

### Issue 2: Memory Fragmentation

**Symptoms:** OOM errors despite available memory

**Solutions:**
```python
# Compact KV cache allocation
cache_config = CacheConfig(
    block_size=64,           # Larger blocks reduce fragmentation
    cache_max_entry_count=0.7
)
```

### Issue 3: Slow First Token (TTFT)

**Symptoms:** High time-to-first-token latency

**Solutions:**
- Enable prefix caching for repeated prompts
- Use smaller block_size for prefill phase
- Optimize prompt length (truncate if possible)

```python
cache_config = CacheConfig(
    enable_prefix_caching=True,
    block_size=16  # Smaller for better prefill
)
```

### Issue 4: Throughput Degradation with Long Contexts

**Symptoms:** Performance drops significantly for sequences > 4K tokens

**Solutions:**
- Increase `max_session_len` appropriately
- Use sliding window attention if supported
- Consider quantization to reduce memory pressure

```python
scheduler_config = SchedulerConfig(
    max_session_len=32768,  # For 32K context
    eviction_type='recompute'  # Recompute vs evict trade-off
)
```

## Advanced: Custom Kernel Selection

For advanced users, LMDeploy allows selecting specific GEMM kernels:

```python
import os

# Force specific GEMM implementation
os.environ['LMDEPLOY_GEMM_BACKEND'] = 'cublas'  # Options: cublas, triton, custom

# Enable Flash Attention for better memory efficiency
os.environ['LMDEPLOY_USE_FLASH_ATTN'] = '1'

# Tune block size for your specific GPU
os.environ['LMDEPLOY_CACHE_BLOCK_SIZE'] = '64'
```

## Benchmarking Your Configuration

Use LMDeploy's benchmark tools to validate tuning:

```bash
# Throughput benchmark
python benchmark/benchmark_throughput.py \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --model meta-llama/Llama-3-8b-Instruct \
    --num-prompts 1000 \
    --request-rate 10

# Latency benchmark
python benchmark/benchmark_pipeline.py \
    --model meta-llama/Llama-3-8b-Instruct \
    --batch-size 32 \
    --seq-len 512
```

Compare results before and after tuning to quantify improvements.

## Best Practices Summary

1. **Start with defaults**: LMDeploy's default configurations are well-tuned for common scenarios
2. **Profile first**: Use profilers to identify actual bottlenecks before tuning
3. **Quantize when possible**: INT8/FP8 can provide 1.5-2x speedup with minimal accuracy loss
4. **Optimize for your workload**: Batch size, sequence length, and concurrency matter
5. **Monitor continuously**: Performance can vary with model updates and library versions
6. **Consider hardware limits**: Don't over-provision beyond GPU capabilities
7. **Test with realistic data**: Use production-like prompts for benchmarking

## Additional Resources

- [PyTorch Profiling Guide](pytorch_profiling.md)
- [Quantization Guide](../quantization/w4a16.md)
- [TurboMind Configuration](../inference/turbomind_config.md)
- [Performance Metrics](metrics.md)

For hardware-specific optimizations and latest kernel improvements, refer to the [LMDeploy GitHub repository](https://github.com/InternLM/lmdeploy).
