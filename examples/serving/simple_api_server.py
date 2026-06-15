"""
Simple API Server Example

This example demonstrates how to launch an OpenAI-compatible API server
using LMDeploy and interact with it using the OpenAI Python client.

Usage:
    # Terminal 1: Start the server
    python examples/serving/simple_api_server.py --model internlm/internlm2_5-7b-chat

    # Terminal 2: Run the client (or use curl/requests)
    python examples/serving/simple_api_server.py --client --model internlm/internlm2_5-7b-chat
"""

import argparse
import subprocess
import sys
import time


def launch_server(model_path, server_port=23333, tp=1, session_len=8192):
    """Launch the LMDeploy API server."""
    print(f"Starting API server for model: {model_path}")
    print(f"Server will be available at: http://localhost:{server_port}")
    print(f"Tensor parallelism: {tp}")
    print(f"Session length: {session_len}")
    print("-" * 60)

    cmd = [
        "lmdeploy", "serve", "api_server",
        model_path,
        "--server-port", str(server_port),
        "--tp", str(tp),
        "--session-len", str(session_len),
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)


def run_client_example(model_name="default", base_url="http://localhost:23333/v1"):
    """Run a simple client example using OpenAI SDK."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Install with: pip install openai")
        sys.exit(1)

    print(f"Connecting to API server at: {base_url}")
    print("-" * 60)

    # Initialize client
    client = OpenAI(
        api_key="YOUR_API_KEY",  # Not required for local deployment
        base_url=base_url
    )

    # List available models
    print("\n1. Listing available models:")
    try:
        models = client.models.list()
        model_id = models.data[0].id if models.data else model_name
        print(f"   Available model: {model_id}")
    except Exception as e:
        print(f"   Error listing models: {e}")
        model_id = model_name

    # Simple chat completion
    print("\n2. Simple chat completion:")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is artificial intelligence?"},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        print(f"   Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    # Streaming chat completion
    print("\n3. Streaming chat completion:")
    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "Explain quantum computing in 3 sentences"},
            ],
            temperature=0.8,
            stream=True,
        )
        print("   Response: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"\n   Error: {e}")

    # Multi-turn conversation
    print("\n4. Multi-turn conversation:")
    try:
        messages = [
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": "How do I reverse a list in Python?"},
        ]

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.5,
        )

        answer = response.choices[0].message.content
        print(f"   Assistant: {answer}")

        # Continue conversation
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": "Can you show me an example?"})

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.5,
        )
        print(f"   Assistant: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Client example completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="LMDeploy API Server Example")
    parser.add_argument("--model", type=str, default="internlm/internlm2_5-7b-chat",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--port", type=int, default=23333,
                        help="Server port (default: 23333)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism size (default: 1)")
    parser.add_argument("--session-len", type=int, default=8192,
                        help="Maximum session length (default: 8192)")
    parser.add_argument("--client", action="store_true",
                        help="Run in client mode instead of server mode")
    parser.add_argument("--base-url", type=str, default="http://localhost:23333/v1",
                        help="Base URL for client mode (default: http://localhost:23333/v1)")

    args = parser.parse_args()

    if args.client:
        # Wait a moment to ensure server is ready
        print("Waiting for server to be ready...")
        time.sleep(2)
        run_client_example(model_name=args.model, base_url=args.base_url)
    else:
        launch_server(args.model, args.port, args.tp, args.session_len)


if __name__ == "__main__":
    main()
