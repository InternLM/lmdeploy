---
name: code-navigation
description: LMDeploy codebase directory map for fast orientation.
---

# LMDeploy Project Structure

```text
lmdeploy/
├── cli/                        # Command line interface implementations
├── lib/                        # Shared libraries/binary assets
├── lite/                       # Quantization Toolkit
│   ├── apis/                   # Calibration, AWQ, and SmoothQuant entry points
│   ├── modeling/               # GPTQ/quantized model specific logic
│   ├── quantization/           # Scaling calculation (activations/weights)
│   └── utils/                  # Quantization helper functions (cal_qparams.py)
├── metrics/                    # Statistics and performance monitoring
├── monitoring/                 # Monitoring configs (Docker/Grafana)
├── pytorch/                    # PyTorch inference backend
│   ├── adapter/                # LoRA and adapter logic
│   ├── backends/               # Kernel/Operator Dispatchers (FP8, AWQ, CUDA)
│   ├── check_env/              # Environment/GPU capability sanity checks
│   ├── configurations/         # Per-model engine configurations (Llama, etc.)
│   ├── devices/                # Device management (CUDA)
│   ├── disagg/                 # Disaggregated prefill/decode logic
│   ├── engine/                 # Main Scheduler and Execution Loop
│   ├── kernels/                # Triton/CUDA Kernels (w8a8_triton_kernels.py)
│   ├── models/                 # Model Patches: Replacing HF layers with kernels
│   ├── multimodal/             # Multi-modal input types for Pytorch engine
│   ├── nn/                     # Reusable PyTorch modules
│   ├── paging/                 # PagedAttention: KV cache block management
│   ├── spec_decode/            # Speculative decoding logic
│   ├── strategies/             # Execution and dispatch strategies
│   ├── third_party/            # External dependencies/repos
│   ├── tools/                  # Internal engine debugging tools
│   ├── transformers/           # HF Transformers integration depth
│   └── weight_loader/          # Sharded/quantized weight loading engine
├── serve/                      # Serving: OpenAI-compatible API and gRPC
├── turbomind/                  # C++ TurboMind inference backend
├── vl/                         # Vision-Language (VL) Support and Image Processing
│   ├── media/                  # Image/Video/... loaders and base classes
│   └── model/                  # VL Archs (InternVL, Qwen-VL, LLaVA, etc.) and preprocess
├── api.py                      # High-level entry for model interaction
├── archs.py                    # Registry: Maps architectures to runtime patches
├── messages.py                 # Core Types: GenerationConfig, EngineConfig
├── model.py                    # Chat Templates: CRITICAL for conversation logic
├── pipeline.py                 # Main Orchestrator: Engine + Tokenizer
└── tokenizer.py                # Wrapper for HF/SentencePiece tokenizers
```
