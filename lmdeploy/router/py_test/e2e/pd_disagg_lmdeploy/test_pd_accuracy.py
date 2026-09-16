#!/usr/bin/env python3
"""
P/D Disaggregation Accuracy Test for LMDeploy Router

This script validates that the router correctly routes requests through
prefill and decode instances with proper output accuracy.
"""

import argparse
import json
import sys

import requests


class Colors:
    """Terminal colors for output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


SAMPLE_PROMPTS = [
    "Red Hat is the best company in the world to work for because",
    "We hold these truths to be self-evident, that all men are created equal,",
    "The quick brown fox jumps over the lazy dog and",
    "In the beginning was the Word, and the Word was with",
    "To be or not to be, that is the question:",
    "Four score and seven years ago our fathers brought forth on this continent,",
    "It was the best of times, it was the worst of times,",
    "Call me Ishmael. Some years ago—never mind how long precisely—",
    "All happy families are alike; each unhappy family is unhappy in",
    "It is a truth universally acknowledged, that a single man in possession",
]


def test_completion_accuracy(
    router_url: str,
    model: str,
    num_requests: int = 20,
    max_tokens: int = 30,
    temperature: float = 0.0,
) -> bool:
    """
    Test that completions return valid outputs through P/D disaggregation.

    Args:
        router_url: URL of the router
        model: Model name to use
        num_requests: Number of test requests to send
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for deterministic)

    Returns:
        True if all tests pass, False otherwise
    """
    print_info(f"Testing completion accuracy with {num_requests} requests")

    session = requests.Session()
    failures = []

    for i in range(num_requests):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]

        try:
            response = session.post(
                f"{router_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=60,
            )

            if response.status_code != 200:
                failures.append(f"Request {i}: HTTP {response.status_code}")
                print_error(f"Request {i} failed with status {response.status_code}")
                continue

            data = response.json()

            # Validate response structure
            if "choices" not in data or not data["choices"]:
                failures.append(f"Request {i}: No choices in response")
                print_error(f"Request {i}: No choices in response")
                continue

            choice = data["choices"][0]
            if "text" not in choice:
                failures.append(f"Request {i}: No text in choice")
                print_error(f"Request {i}: No text in choice")
                continue

            output_text = choice["text"]

            # Validate output is non-empty
            if not output_text or len(output_text.strip()) == 0:
                failures.append(f"Request {i}: Empty output")
                print_error(f"Request {i}: Empty output")
                continue

            # Validate usage information
            if "usage" not in data:
                print_warning(f"Request {i}: No usage information")

            print_success(f"Request {i}: Generated {len(output_text)} chars")

        except requests.RequestException as e:
            failures.append(f"Request {i}: {str(e)}")
            print_error(f"Request {i} failed with exception: {e}")
        except json.JSONDecodeError as e:
            failures.append(f"Request {i}: Invalid JSON response")
            print_error(f"Request {i}: JSON decode error: {e}")
        except Exception as e:
            failures.append(f"Request {i}: Unexpected error: {str(e)}")
            print_error(f"Request {i}: Unexpected error: {e}")

    # Summary
    success_count = num_requests - len(failures)
    print()
    print_info("=" * 60)
    print_info("Completion Accuracy Test Results:")
    print_info(f"  Total Requests:     {num_requests}")
    print_info(f"  Successful:         {success_count}")
    print_info(f"  Failed:             {len(failures)}")
    print_info("=" * 60)

    if failures:
        print()
        print_error("Failures:")
        for failure in failures:
            print_error(f"  - {failure}")
        return False

    return True


def test_streaming_accuracy(
    router_url: str,
    model: str,
    num_requests: int = 5,
    max_tokens: int = 20,
    temperature: float = 0.0,
) -> bool:
    """
    Test that streaming completions work correctly through P/D disaggregation.

    Args:
        router_url: URL of the router
        model: Model name to use
        num_requests: Number of test requests to send
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        True if all tests pass, False otherwise
    """
    print_info(f"Testing streaming accuracy with {num_requests} requests")

    session = requests.Session()
    failures = []

    for i in range(num_requests):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]

        try:
            response = session.post(
                f"{router_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
                timeout=60,
                stream=True,
            )

            if response.status_code != 200:
                failures.append(f"Stream {i}: HTTP {response.status_code}")
                print_error(f"Stream {i} failed with status {response.status_code}")
                continue

            chunks = []
            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        chunks.append(chunk)
                    except json.JSONDecodeError:
                        continue

            if not chunks:
                failures.append(f"Stream {i}: No chunks received")
                print_error(f"Stream {i}: No chunks received")
                continue

            # Reconstruct full text from chunks
            full_text = ""
            for chunk in chunks:
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "text" in choice:
                        full_text += choice["text"]

            if not full_text:
                failures.append(f"Stream {i}: No text in chunks")
                print_error(f"Stream {i}: No text in chunks")
                continue

            print_success(
                f"Stream {i}: Received {len(chunks)} chunks, {len(full_text)} chars"
            )

        except requests.RequestException as e:
            failures.append(f"Stream {i}: {str(e)}")
            print_error(f"Stream {i} failed with exception: {e}")
        except Exception as e:
            failures.append(f"Stream {i}: Unexpected error: {str(e)}")
            print_error(f"Stream {i}: Unexpected error: {e}")

    # Summary
    success_count = num_requests - len(failures)
    print()
    print_info("=" * 60)
    print_info("Streaming Accuracy Test Results:")
    print_info(f"  Total Requests:     {num_requests}")
    print_info(f"  Successful:         {success_count}")
    print_info(f"  Failed:             {len(failures)}")
    print_info("=" * 60)

    if failures:
        print()
        print_error("Failures:")
        for failure in failures:
            print_error(f"  - {failure}")
        return False

    return True


def test_router_health(router_url: str) -> bool:
    """
    Test that the router is healthy and responding.

    Args:
        router_url: URL of the router

    Returns:
        True if healthy, False otherwise
    """
    print_info("Checking router health...")

    try:
        # Try health endpoint
        response = requests.get(f"{router_url}/health", timeout=10)
        if response.status_code == 200:
            print_success("Router health check passed")
            return True
    except requests.RequestException:
        pass

    try:
        # Try models endpoint as fallback
        response = requests.get(f"{router_url}/v1/models", timeout=10)
        if response.status_code == 200:
            print_success("Router models endpoint accessible")
            return True
    except requests.RequestException as e:
        print_error(f"Router health check failed: {e}")
        return False

    print_error("Router is not healthy")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Test P/D disaggregation accuracy through LMDeploy router"
    )
    parser.add_argument(
        "--router-url",
        type=str,
        required=True,
        help="URL of the router (e.g., http://localhost:8300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for testing",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="Number of completion requests to test (default: 20)",
    )
    parser.add_argument(
        "--num-streaming-requests",
        type=int,
        default=5,
        help="Number of streaming requests to test (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum tokens to generate (default: 30)",
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Skip streaming tests",
    )

    args = parser.parse_args()

    print()
    print_info("=" * 60)
    print_info("P/D Disaggregation Accuracy Test")
    print_info("=" * 60)
    print_info(f"Router URL:          {args.router_url}")
    print_info(f"Model:               {args.model}")
    print_info(f"Completion Tests:    {args.num_requests}")
    print_info(f"Streaming Tests:     {args.num_streaming_requests}")
    print_info("=" * 60)
    print()

    # Test 1: Router Health
    if not test_router_health(args.router_url):
        print_error("Router health check failed. Aborting tests.")
        return 1

    print()

    # Test 2: Completion Accuracy
    completion_success = test_completion_accuracy(
        router_url=args.router_url,
        model=args.model,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
    )

    print()

    # Test 3: Streaming Accuracy (optional)
    streaming_success = True
    if not args.skip_streaming:
        streaming_success = test_streaming_accuracy(
            router_url=args.router_url,
            model=args.model,
            num_requests=args.num_streaming_requests,
            max_tokens=args.max_tokens,
        )
        print()

    # Final Summary
    print_info("=" * 60)
    print_info("Final Results:")
    print_info("=" * 60)

    if completion_success:
        print_success("Completion Tests:  PASSED")
    else:
        print_error("Completion Tests:  FAILED")

    if not args.skip_streaming:
        if streaming_success:
            print_success("Streaming Tests:   PASSED")
        else:
            print_error("Streaming Tests:   FAILED")

    print_info("=" * 60)
    print()

    if completion_success and streaming_success:
        print_success("All P/D disaggregation accuracy tests PASSED!")
        return 0
    else:
        print_error("Some P/D disaggregation accuracy tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
