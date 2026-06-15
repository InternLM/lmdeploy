"""
Batch Inference Example

This example demonstrates efficient batch inference with LMDeploy,
including various configuration options for optimal performance.

Usage:
    python examples/inference/batch_inference.py --model internlm/internlm2_5-7b-chat
"""

import argparse
import time
from typing import List


def basic_batch_inference(model_path: str):
    """Basic batch inference with multiple prompts."""
    print("=" * 60)
    print("Example 1: Basic Batch Inference")
    print("=" * 60)

    from lmdeploy import pipeline

    # Create pipeline
    print(f"\nLoading model: {model_path}")
    pipe = pipeline(model_path)

    # Define multiple prompts for batch processing
    prompts = [
        "What is machine learning?",
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to calculate factorial.",
        "What are the benefits of renewable energy?",
        "How does blockchain technology work?",
    ]

    print(f"\nProcessing {len(prompts)} prompts in batch...")
    start_time = time.time()

    # Run batch inference
    responses = pipe(prompts)

    elapsed_time = time.time() - start_time

    # Print results
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Average time per request: {elapsed_time / len(prompts):.2f} seconds")
    print("-" * 60)

    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"\nPrompt {i}: {prompt}")
        print(f"Response: {response.text[:200]}...")

    return elapsed_time


def configured_batch_inference(model_path: str):
    """Batch inference with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Configured Batch Inference")
    print("=" * 60)

    from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig

    # Configure engine for better performance
    engine_config = PytorchEngineConfig(
        max_batch_size=32,
        session_len=4096,
        cache_max_entry_count=0.7,  # Reduce if OOM
    )

    print(f"\nLoading model with custom config: {model_path}")
    pipe = pipeline(model_path, backend_config=engine_config)

    # Configure generation parameters
    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        top_k=40,
        stop_words=["\n\n"],
    )

    prompts = [
        "Write a haiku about programming",
        "What is the capital of France?",
        "Explain Docker in one sentence",
    ]

    print(f"\nGenerating with custom parameters...")
    print(f"  Max tokens: {gen_config.max_new_tokens}")
    print(f"  Temperature: {gen_config.temperature}")
    print(f"  Top-p: {gen_config.top_p}")
    print("-" * 60)

    responses = pipe(prompts, gen_config=gen_config)

    for prompt, response in zip(prompts, responses):
        print(f"\nQ: {prompt}")
        print(f"A: {response.text}")

    return responses


def streaming_inference(model_path: str):
    """Streaming inference for real-time output."""
    print("\n" + "=" * 60)
    print("Example 3: Streaming Inference")
    print("=" * 60)

    from lmdeploy import pipeline

    pipe = pipeline(model_path)

    prompt = "Write a short story about a robot learning to paint"

    print(f"\nPrompt: {prompt}")
    print("\nStreaming response:")
    print("-" * 60)

    # Stream responses token by token
    for response in pipe.stream_infer([prompt]):
        print(response.text, end="", flush=True)

    print("\n" + "-" * 60)
    print("Streaming completed!")


def multi_turn_conversation(model_path: str):
    """Multi-turn conversation example."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-turn Conversation")
    print("=" * 60)

    from lmdeploy import pipeline

    pipe = pipeline(model_path)

    # Simulate a conversation
    messages = [
        {"role": "user", "content": "What is Python?"},
    ]

    print("\nConversation:")
    print("-" * 60)

    # First turn
    print(f"User: {messages[-1]['content']}")
    response = pipe(messages)
    assistant_reply = response.text
    print(f"Assistant: {assistant_reply[:200]}...")

    # Second turn
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "Can you show me a simple example?"})

    print(f"\nUser: {messages[-1]['content']}")
    response = pipe(messages)
    print(f"Assistant: {response.text[:200]}...")

    # Third turn
    messages.append({"role": "assistant", "content": response.text})
    messages.append({"role": "user", "content": "How do I run this code?"})

    print(f"\nUser: {messages[-1]['content']}")
    response = pipe(messages)
    print(f"Assistant: {response.text[:200]}...")

    print("\n" + "-" * 60)
    print("Conversation completed!")


def performance_benchmark(model_path: str, num_requests: int = 10):
    """Simple performance benchmark."""
    print("\n" + "=" * 60)
    print("Example 5: Performance Benchmark")
    print("=" * 60)

    from lmdeploy import pipeline

    pipe = pipeline(model_path)

    # Create test prompts
    prompts = [f"What is question number {i+1}?" for i in range(num_requests)]

    print(f"\nRunning benchmark with {num_requests} requests...")
    print("-" * 60)

    # Warm-up
    print("Warming up...")
    _ = pipe(prompts[:2])

    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    responses = pipe(prompts)
    elapsed_time = time.time() - start_time

    # Calculate metrics
    total_tokens = sum(len(r.text.split()) for r in responses)
    throughput = num_requests / elapsed_time
    token_throughput = total_tokens / elapsed_time

    print(f"\nBenchmark Results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Request throughput: {throughput:.2f} req/s")
    print(f"  Estimated token throughput: {token_throughput:.2f} tokens/s")
    print(f"  Average latency: {elapsed_time / num_requests * 1000:.2f} ms/req")

    return {
        "total_time": elapsed_time,
        "throughput": throughput,
        "avg_latency": elapsed_time / num_requests,
    }


def main():
    parser = argparse.ArgumentParser(description="LMDeploy Batch Inference Examples")
    parser.add_argument("--model", type=str, default="internlm/internlm2_5-7b-chat",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4, 5, 0], default=0,
                        help="Run specific example (1-5) or all (0)")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of requests for benchmark (default: 10)")

    args = parser.parse_args()

    examples = {
        1: ("Basic Batch Inference", lambda: basic_batch_inference(args.model)),
        2: ("Configured Batch Inference", lambda: configured_batch_inference(args.model)),
        3: ("Streaming Inference", lambda: streaming_inference(args.model)),
        4: ("Multi-turn Conversation", lambda: multi_turn_conversation(args.model)),
        5: ("Performance Benchmark", lambda: performance_benchmark(args.model, args.num_requests)),
    }

    if args.example == 0:
        # Run all examples
        for ex_id, (name, func) in examples.items():
            try:
                print(f"\n{'#' * 60}")
                print(f"# Running Example {ex_id}: {name}")
                print(f"{'#' * 60}\n")
                func()
            except Exception as e:
                print(f"\nError in example {ex_id}: {e}")
                import traceback
                traceback.print_exc()
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
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
