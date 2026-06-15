"""
Vision-Language Model (VLM) Inference Example

This example demonstrates how to use LMDeploy with vision-language models
for image understanding, visual question answering, and multimodal tasks.

Supported VLMs:
- Qwen-VL series
- InternVL series
- LLaVA series
- MiniCPM-V series
- DeepSeek-VL2

Usage:
    python examples/inference/vlm_inference.py --model Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import os


def basic_vlm_inference(model_path: str):
    """Basic VLM inference with image URL."""
    print("=" * 60)
    print("Example 1: Basic VLM Inference")
    print("=" * 60)

    from lmdeploy import pipeline

    print(f"\nLoading VLM model: {model_path}")
    pipe = pipeline(model_path)

    # Example with image URL
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://pytorch.org/assets/images/pytorch-logo.png"}},
        ]
    }]

    print("\nSending request with image...")
    response = pipe(messages)

    print(f"\nResponse: {response.text}")

    return response


def local_image_inference(model_path: str, image_path: str = None):
    """Inference with local image file."""
    print("\n" + "=" * 60)
    print("Example 2: Local Image Inference")
    print("=" * 60)

    from lmdeploy import pipeline

    # Create a sample image path if not provided
    if image_path is None:
        # Use a test image or create placeholder
        image_path = "sample_image.jpg"
        print(f"Note: Using placeholder image path: {image_path}")
        print("Replace with actual image path for real usage")

    if not os.path.exists(image_path):
        print(f"\nWarning: Image file not found: {image_path}")
        print("Skipping this example. Provide a valid image path with --image-path")
        return None

    pipe = pipeline(model_path)

    # Read local image
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail"},
            {"type": "image", "image": image_path},
        ]
    }]

    print(f"\nProcessing image: {image_path}")
    response = pipe(messages)

    print(f"\nDescription: {response.text}")

    return response


def multi_image_inference(model_path: str):
    """Inference with multiple images."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Image Inference")
    print("=" * 60)

    from lmdeploy import pipeline

    pipe = pipeline(model_path)

    # Example with multiple images
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two images"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
        ]
    }]

    print("\nSending request with multiple images...")
    print("(Note: Replace URLs with actual images)")

    try:
        response = pipe(messages)
        print(f"\nComparison: {response.text}")
        return response
    except Exception as e:
        print(f"Error (expected with placeholder URLs): {e}")
        print("Provide valid image URLs to test this feature")
        return None


def visual_question_answering(model_path: str):
    """Visual Question Answering (VQA) example."""
    print("\n" + "=" * 60)
    print("Example 4: Visual Question Answering")
    print("=" * 60)

    from lmdeploy import pipeline

    pipe = pipeline(model_path)

    # VQA with specific questions
    questions = [
        "What color is the object?",
        "How many people are in the image?",
        "What is the text in this image?",
    ]

    image_url = "https://pytorch.org/assets/images/pytorch-logo.png"

    for question in questions:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        }]

        print(f"\nQ: {question}")
        response = pipe(messages)
        print(f"A: {response.text}")

    return True


def document_understanding(model_path: str):
    """Document understanding and OCR example."""
    print("\n" + "=" * 60)
    print("Example 5: Document Understanding")
    print("=" * 60)

    from lmdeploy import pipeline

    pipe = pipeline(model_path)

    # Document analysis
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this document and summarize its content"},
            {"type": "image_url", "image_url": {"url": "https://example.com/document.png"}},
        ]
    }]

    print("\nAnalyzing document...")
    print("(Note: Replace with actual document image)")

    try:
        response = pipe(messages)
        print(f"\nDocument Analysis: {response.text[:500]}...")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None


def batch_vlm_inference(model_path: str):
    """Batch processing of multiple images."""
    print("\n" + "=" * 60)
    print("Example 6: Batch VLM Inference")
    print("=" * 60)

    from lmdeploy import pipeline

    pipe = pipeline(model_path)

    # Prepare multiple image-text pairs
    prompts = [
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img1.jpg"}},
            ]
        }],
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this scene"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img2.jpg"}},
            ]
        }],
    ]

    print(f"\nProcessing {len(prompts)} image queries in batch...")

    try:
        responses = pipe(prompts)
        for i, response in enumerate(responses, 1):
            print(f"\nResponse {i}: {response.text[:200]}...")
        return responses
    except Exception as e:
        print(f"Error: {e}")
        return None


def vlm_with_configuration(model_path: str):
    """VLM inference with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 7: Configured VLM Inference")
    print("=" * 60)

    from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig

    # Configure engine
    engine_config = PytorchEngineConfig(
        session_len=4096,
        max_batch_size=8,
        cache_max_entry_count=0.6,
    )

    # Configure generation
    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
    )

    print(f"\nLoading VLM with custom config: {model_path}")
    pipe = pipeline(model_path, backend_config=engine_config)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image and provide insights"},
            {"type": "image_url", "image_url": {"url": "https://example.com/analysis.jpg"}},
        ]
    }]

    print("\nGenerating response with custom parameters...")
    print(f"  Max tokens: {gen_config.max_new_tokens}")
    print(f"  Temperature: {gen_config.temperature}")

    try:
        response = pipe(messages, gen_config=gen_config)
        print(f"\nAnalysis: {response.text}")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="LMDeploy VLM Inference Examples")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="VLM model path or HuggingFace model ID")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to local image file")
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 0], default=0,
                        help="Run specific example (1-7) or all (0)")

    args = parser.parse_args()

    examples = {
        1: ("Basic VLM Inference", lambda: basic_vlm_inference(args.model)),
        2: ("Local Image Inference", lambda: local_image_inference(args.model, args.image_path)),
        3: ("Multi-Image Inference", lambda: multi_image_inference(args.model)),
        4: ("Visual Question Answering", lambda: visual_question_answering(args.model)),
        5: ("Document Understanding", lambda: document_understanding(args.model)),
        6: ("Batch VLM Inference", lambda: batch_vlm_inference(args.model)),
        7: ("Configured VLM Inference", lambda: vlm_with_configuration(args.model)),
    }

    print(f"\nUsing VLM model: {args.model}")
    print("Note: Some examples use placeholder URLs. Replace with actual images for testing.\n")

    if args.example == 0:
        # Run all examples
        for ex_id, (name, func) in examples.items():
            try:
                print(f"\n{'#' * 60}")
                print(f"# Example {ex_id}: {name}")
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
    print("VLM examples completed!")
    print("=" * 60)
    print("\nTip: For better results, replace placeholder URLs with actual image URLs or local paths.")


if __name__ == "__main__":
    main()
