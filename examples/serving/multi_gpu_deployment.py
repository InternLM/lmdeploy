"""
Multi-GPU Deployment Example

This example demonstrates how to deploy models across multiple GPUs
using tensor parallelism for improved performance and larger model support.

Usage:
    # Single GPU
    python examples/serving/multi_gpu_deployment.py --model meta-llama/Llama-3-8b-Instruct --gpus 1

    # Multi-GPU (e.g., 2 GPUs)
    python examples/serving/multi_gpu_deployment.py --model meta-llama/Llama-3-70b-Instruct --gpus 2
"""

import argparse
import os


def single_gpu_deployment(model_path: str):
    """Deploy model on a single GPU."""
    print("=" * 60)
    print("Single GPU Deployment")
    print("=" * 60)

    from lmdeploy import pipeline, PytorchEngineConfig

    # Basic single GPU configuration
    engine_config = PytorchEngineConfig(
        tp=1,  # Tensor parallelism = 1 (single GPU)
        session_len=8192,
        max_batch_size=32,
    )

    print(f"\nDeploying model on 1 GPU: {model_path}")
    print(f"Configuration:")
    print(f"  Tensor Parallelism: {engine_config.tp}")
    print(f"  Session Length: {engine_config.session_len}")
    print(f"  Max Batch Size: {engine_config.max_batch_size}")

    pipe = pipeline(model_path, backend_config=engine_config)

    # Test inference
    response = pipe(["Hello! How are you today?"])
    print(f"\nTest response: {response.text[:100]}...")

    return pipe


def multi_gpu_tensor_parallel(model_path: str, num_gpus: int = 2):
    """Deploy model using tensor parallelism across multiple GPUs."""
    print("\n" + "=" * 60)
    print(f"Multi-GPU Tensor Parallel Deployment ({num_gpus} GPUs)")
    print("=" * 60)

    from lmdeploy import pipeline, PytorchEngineConfig

    # Check available GPUs
    import torch
    available_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {available_gpus}")

    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available.")
        num_gpus = available_gpus

    # Configure tensor parallelism
    engine_config = PytorchEngineConfig(
        tp=num_gpus,  # Split model across GPUs
        session_len=16384,  # Can use longer context with more GPUs
        max_batch_size=64,  # Higher batch size with more compute
        cache_max_entry_count=0.8,
    )

    print(f"\nDeploying model across {num_gpus} GPUs: {model_path}")
    print(f"Configuration:")
    print(f"  Tensor Parallelism: {engine_config.tp}")
    print(f"  Session Length: {engine_config.session_len}")
    print(f"  Max Batch Size: {engine_config.max_batch_size}")
    print(f"  Cache Entry Count: {engine_config.cache_max_entry_count}")

    pipe = pipeline(model_path, backend_config=engine_config)

    # Test with batch inference
    prompts = [
        "Explain quantum entanglement",
        "What is the meaning of life?",
        "How does machine learning work?",
    ]

    print(f"\nTesting with {len(prompts)} prompts...")
    responses = pipe(prompts)

    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"\nPrompt {i}: {prompt}")
        print(f"Response: {response.text[:150]}...")

    return pipe


def multi_node_deployment_example(model_path: str):
    """Example configuration for multi-node deployment (documentation)."""
    print("\n" + "=" * 60)
    print("Multi-Node Deployment Example (Documentation)")
    print("=" * 60)

    print("""
For multi-node deployment, use Ray to distribute across machines:

1. Install Ray:
   pip install ray

2. Start Ray cluster:
   # On head node:
   ray start --head --port=6379

   # On worker nodes:
   ray start --address='<head-node-ip>:6379'

3. Deploy with Ray:
   ```python
   from lmdeploy import pipeline, PytorchEngineConfig

   engine_config = PytorchEngineConfig(
       tp=4,  # Total GPUs across all nodes
       device_type='cuda',
   )

   pipe = pipeline(
       'meta-llama/Llama-3-70b-Instruct',
       backend_config=engine_config,
   )
   ```

4. Environment variables for Ray:
   export RAY_ADDRESS='<head-node-ip>:6379'
   export LMDEPLOY_RAY_LOG_LEVEL='INFO'

Note: Ensure all nodes have the same model files accessible
(via shared storage or pre-downloaded to each node).
""")


def gpu_memory_optimization(model_path: str):
    """Optimize GPU memory usage for better performance."""
    print("\n" + "=" * 60)
    print("GPU Memory Optimization")
    print("=" * 60)

    from lmdeploy import pipeline, PytorchEngineConfig, CacheConfig

    import torch
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nTotal GPU Memory: {total_memory:.2f} GB")

    # Calculate optimal cache configuration
    # Reserve ~40% for weights and activations, ~60% for KV cache
    cache_ratio = 0.6

    cache_config = CacheConfig(
        max_batches=64,
        block_size=16,
        cache_max_entry_count=cache_ratio,
    )

    engine_config = PytorchEngineConfig(
        tp=1,
        session_len=8192,
        max_batch_size=32,
        cache_config=cache_config,
    )

    print(f"\nOptimized Configuration:")
    print(f"  Cache Ratio: {cache_ratio * 100:.0f}%")
    print(f"  Max Batches: {cache_config.max_batches}")
    print(f"  Block Size: {cache_config.block_size}")
    print(f"  Estimated KV Cache: {total_memory * cache_ratio:.2f} GB")

    pipe = pipeline(model_path, backend_config=engine_config)

    # Monitor memory usage
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    print(f"\nMemory Usage After Loading:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Utilization: {allocated / total_memory * 100:.1f}%")

    return pipe


def load_balanced_serving(model_path: str, num_instances: int = 2):
    """Example of setting up load-balanced serving."""
    print("\n" + "=" * 60)
    print("Load-Balanced Serving Setup")
    print("=" * 60)

    print(f"""
To set up load-balanced serving with {num_instances} instances:

Option 1: Multiple API Servers with Nginx

1. Start multiple API servers on different ports:
   # Instance 1
   lmdeploy serve api_server {model_path} --server-port 23333 --tp 1

   # Instance 2
   lmdeploy serve api_server {model_path} --server-port 23334 --tp 1

2. Configure Nginx as load balancer:
   ```nginx
   upstream lmdeploy_servers {{
       server localhost:23333;
       server localhost:23334;
   }}

   server {{
       listen 8000;
       location / {{
           proxy_pass http://lmdeploy_servers;
       }}
   }}
   ```

3. Access via: http://localhost:8000/v1/chat/completions

Option 2: Use LMDeploy Proxy Server

   lmdeploy serve proxy_server \\
       --server-name localhost \\
       --server-port 8000 \\
       --upstream-servers http://localhost:23333,http://localhost:23334

Benefits:
- Higher throughput through parallel processing
- Better fault tolerance
- Ability to scale horizontally
- Reduced latency under high load
""")


def main():
    parser = argparse.ArgumentParser(description="LMDeploy Multi-GPU Deployment Examples")
    parser.add_argument("--model", type=str, default="internlm/internlm2_5-7b-chat",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use (default: 1)")
    parser.add_argument("--example", type=str,
                        choices=["single", "multi", "memory", "loadbalance", "all"],
                        default="all",
                        help="Which example to run (default: all)")

    args = parser.parse_args()

    examples = {
        "single": ("Single GPU Deployment", lambda: single_gpu_deployment(args.model)),
        "multi": ("Multi-GPU Tensor Parallel", lambda: multi_gpu_tensor_parallel(args.model, args.gpus)),
        "memory": ("GPU Memory Optimization", lambda: gpu_memory_optimization(args.model)),
        "loadbalance": ("Load-Balanced Serving", lambda: load_balanced_serving(args.model)),
    }

    if args.example == "all":
        # Run all applicable examples
        for name, func in examples.items():
            try:
                print(f"\n{'#' * 60}")
                print(f"# {func[0]}")
                print(f"{'#' * 60}\n")
                func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
                import traceback
                traceback.print_exc()

        # Always show multi-node documentation
        multi_node_deployment_example(args.model)
    else:
        # Run specific example
        name, func = examples[args.example]
        try:
            func()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Deployment examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
