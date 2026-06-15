# LMDeploy Examples

This directory contains practical examples demonstrating various features and use cases of LMDeploy.

## Directory Structure

```
examples/
├── README.md                    # This file
├── lite/                        # Quantization examples (existing)
│   ├── qwen3_30b_a3b_awq.py
│   └── qwen3_30b_a3b_gptq.py
├── serving/                     # API server and deployment examples
│   ├── simple_api_server.py     # Basic API server with client example
│   └── multi_gpu_deployment.py  # Multi-GPU and load balancing
├── inference/                   # Inference examples
│   ├── batch_inference.py       # Batch processing and streaming
│   └── vlm_inference.py         # Vision-Language Model examples
└── quantization/                # Quantization workflows
    └── quantization_workflow.py # Complete quantization guide
```

## Quick Start

### 1. API Server Example

Launch an OpenAI-compatible API server:

```bash
# Start server
python examples/serving/simple_api_server.py --model internlm/internlm2_5-7b-chat

# In another terminal, run client example
python examples/serving/simple_api_server.py --client
```

### 2. Batch Inference Example

Run various inference patterns:

```bash
# Run all examples
python examples/inference/batch_inference.py --model meta-llama/Llama-3-8b-Instruct

# Run specific example (1-5)
python examples/inference/batch_inference.py --model meta-llama/Llama-3-8b-Instruct --example 1
```

Examples include:
- Basic batch inference
- Configured inference with custom parameters
- Streaming inference
- Multi-turn conversation
- Performance benchmarking

### 3. Multi-GPU Deployment

Deploy models across multiple GPUs:

```bash
# Single GPU
python examples/serving/multi_gpu_deployment.py --model meta-llama/Llama-3-8b --gpus 1

# Multi-GPU (tensor parallelism)
python examples/serving/multi_gpu_deployment.py --model meta-llama/Llama-3-70b --gpus 2
```

### 4. Vision-Language Models

Use VLMs for image understanding:

```bash
# Basic VLM inference
python examples/inference/vlm_inference.py --model Qwen/Qwen2-VL-7B-Instruct

# With local image
python examples/inference/vlm_inference.py --model Qwen/Qwen2-VL-7B-Instruct --image-path ./image.jpg
```

Supported tasks:
- Image description
- Visual question answering (VQA)
- Document understanding
- Multi-image comparison

### 5. Quantization Workflow

Complete quantization examples:

```bash
# AWQ quantization
python examples/quantization/quantization_workflow.py --method awq --model meta-llama/Llama-3-8b

# GPTQ quantization
python examples/quantization/quantization_workflow.py --method gptq --model meta-llama/Llama-3-8b

# W8A8 quantization (runtime)
python examples/quantization/quantization_workflow.py --method w8a8 --model meta-llama/Llama-3-8b

# Compare methods
python examples/quantization/quantization_workflow.py --method compare
```

## Requirements

Install required packages:

```bash
pip install lmdeploy openai torch transformers

# For quantization examples
pip install lmdeploy[lite] llmcompressor datasets

# For VLM examples
pip install timm torchvision
```

## Configuration Tips

### GPU Memory Management

If you encounter OOM errors, reduce cache allocation:

```python
from lmdeploy import PytorchEngineConfig, CacheConfig

cache_config = CacheConfig(
    cache_max_entry_count=0.5,  # Reduce from default 0.8
)

engine_config = PytorchEngineConfig(
    cache_config=cache_config,
)
```

### Performance Optimization

For better throughput:

```python
engine_config = PytorchEngineConfig(
    max_batch_size=64,        # Increase batch size
    session_len=8192,         # Adjust to your needs
    enable_prefix_caching=True,  # Enable if repeated prompts
)
```

### Tensor Parallelism

For large models requiring multiple GPUs:

```python
engine_config = PytorchEngineConfig(
    tp=2,  # Split across 2 GPUs
)
```

## Common Issues

### Issue: Out of Memory

**Solution:** Reduce `cache_max_entry_count` or `max_batch_size`

### Issue: Slow First Token

**Solution:** Enable prefix caching for repeated system prompts

### Issue: Low GPU Utilization

**Solution:** Increase batch size or enable continuous batching

## Additional Resources

- [Documentation](https://lmdeploy.readthedocs.io/)
- [API Reference](https://lmdeploy.readthedocs.io/en/latest/api.html)
- [GitHub Repository](https://github.com/InternLM/lmdeploy)
- [GEMM Tuning Guide](../docs/en/advance/gemm_tuning.md)

## Contributing

Feel free to add more examples! Please follow these guidelines:

1. Include comprehensive docstrings
2. Add argument parser for flexibility
3. Handle errors gracefully
4. Provide usage examples in comments
5. Test with at least one model

## License

These examples are licensed under the Apache 2.0 License - see the main repository LICENSE file for details.
