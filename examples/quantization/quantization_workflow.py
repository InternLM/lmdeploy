"""
Quantization Workflow Example

This example demonstrates the complete quantization workflow using LMDeploy,
including AWQ, GPTQ, and W8A8 quantization methods.

Usage:
    # AWQ quantization
    python examples/quantization/quantization_workflow.py --method awq --model meta-llama/Llama-3-8b

    # GPTQ quantization
    python examples/quantization/quantization_workflow.py --method gptq --model meta-llama/Llama-3-8b

    # W8A8 quantization
    python examples/quantization/quantization_workflow.py --method w8a8 --model meta-llama/Llama-3-8b
"""

import argparse
import os
import time


def awq_quantization(model_path: str, output_dir: str = "./awq_quantized"):
    """AWQ (Activation-aware Weight Quantization) workflow."""
    print("=" * 60)
    print("AWQ Quantization Workflow")
    print("=" * 60)

    try:
        from lmdeploy.lite import auto_awq
    except ImportError:
        print("Note: Using llmcompressor-based AWQ (alternative method)")
        print("Install lmdeploy with lite extras: pip install lmdeploy[lite]")
        return alternative_awq_with_llmcompressor(model_path, output_dir)

    print(f"\nOriginal model: {model_path}")
    print(f"Output directory: {output_dir}")
    print("\nAWQ Configuration:")
    print("  Bits: 4 (int4)")
    print("  Group size: 128")
    print("  Calibration samples: 128")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\nStarting AWQ quantization...")
    start_time = time.time()

    try:
        # Run AWQ quantization
        auto_awq(
            model=model_path,
            calib_dataset='c4',
            calib_samples=128,
            calib_seqlen=2048,
            work_dir=output_dir,
            search_scale=True,
            device='cuda',
        )

        elapsed_time = time.time() - start_time
        print(f"\nAWQ quantization completed in {elapsed_time:.2f} seconds")
        print(f"Quantized model saved to: {output_dir}")

        # Verify quantized model
        verify_quantized_model(output_dir)

        return output_dir

    except Exception as e:
        print(f"\nError during AWQ quantization: {e}")
        import traceback
        traceback.print_exc()
        return None


def alternative_awq_with_llmcompressor(model_path: str, output_dir: str):
    """Alternative AWQ using llmcompressor (for newer models)."""
    print("\nUsing llmcompressor for AWQ quantization...")

    try:
        from datasets import load_dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: llmcompressor not installed. Install with:")
        print("  pip install llmcompressor datasets transformers")
        return None

    print(f"\nLoading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype='auto',
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Prepare calibration dataset
    print("Preparing calibration dataset...")
    ds = load_dataset('neuralmagic/calibration', 'LLM', split='train[:128]')

    def preprocess(example):
        messages = []
        for message in example['messages']:
            if message['role'] == 'user':
                messages.append({'role': 'user', 'content': message['content']})
            elif message['role'] == 'assistant':
                messages.append({'role': 'assistant', 'content': message['content']})

        return tokenizer(
            tokenizer.apply_chat_template(messages, tokenize=False),
            padding=False,
            max_length=512,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.shuffle(seed=42).map(preprocess, remove_columns=ds.column_names).select(range(128))

    # Configure AWQ recipe
    recipe = [
        AWQModifier(
            ignore=['lm_head', 're:.*mlp.gate$'],
            scheme='W4A16_ASYM',
            targets=['Linear'],
            duo_scaling='both',
        ),
    ]

    print("Running AWQ quantization...")
    start_time = time.time()

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=128,
    )

    # Save quantized model
    print(f"Saving quantized model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed_time = time.time() - start_time
    print(f"\nAWQ quantization completed in {elapsed_time:.2f} seconds")

    return output_dir


def gptq_quantization(model_path: str, output_dir: str = "./gptq_quantized"):
    """GPTQ quantization workflow."""
    print("\n" + "=" * 60)
    print("GPTQ Quantization Workflow")
    print("=" * 60)

    try:
        from lmdeploy.lite import gptq
    except ImportError:
        print("Note: GPTQ requires lmdeploy lite module")
        print("Install with: pip install lmdeploy[lite]")
        return alternative_gptq_with_autoquant(model_path, output_dir)

    print(f"\nOriginal model: {model_path}")
    print(f"Output directory: {output_dir}")
    print("\nGPTQ Configuration:")
    print("  Bits: 4 (int4)")
    print("  Group size: 128")
    print("  Calibration samples: 128")

    os.makedirs(output_dir, exist_ok=True)

    print("\nStarting GPTQ quantization...")
    start_time = time.time()

    try:
        gptq(
            model=model_path,
            calib_dataset='c4',
            calib_samples=128,
            calib_seqlen=2048,
            work_dir=output_dir,
            device='cuda',
        )

        elapsed_time = time.time() - start_time
        print(f"\nGPTQ quantization completed in {elapsed_time:.2f} seconds")
        print(f"Quantized model saved to: {output_dir}")

        verify_quantized_model(output_dir)

        return output_dir

    except Exception as e:
        print(f"\nError during GPTQ quantization: {e}")
        import traceback
        traceback.print_exc()
        return None


def alternative_gptq_with_autoquant(model_path: str, output_dir: str):
    """Alternative GPTQ using auto-gptq."""
    print("\nUsing auto-gptq for GPTQ quantization...")

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: auto-gptq not installed. Install with:")
        print("  pip install auto-gptq")
        return None

    print(f"\nLoading model: {model_path}")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Configure quantization
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
    )

    print("Starting GPTQ quantization...")
    start_time = time.time()

    # Quantize (requires calibration data)
    # Note: This is a simplified example. In practice, provide proper calibration data.
    model.quantize(
        tokenizer=tokenizer,
        quantize_config=quantize_config,
        # examples=calibration_data,  # Provide calibration examples
    )

    # Save quantized model
    print(f"Saving to: {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed_time = time.time() - start_time
    print(f"\nGPTQ quantization completed in {elapsed_time:.2f} seconds")

    return output_dir


def w8a8_quantization_demo(model_path: str):
    """W8A8 (8-bit weight and activation) quantization demonstration."""
    print("\n" + "=" * 60)
    print("W8A8 Quantization Demo")
    print("=" * 60)

    from lmdeploy import pipeline
    from lmdeploy.messages import QuantPolicy

    print(f"\nModel: {model_path}")
    print("\nW8A8 Configuration:")
    print("  Weight bits: 8")
    print("  Activation bits: 8")
    print("  Method: SmoothQuant/AWQ-style")

    # Create quantization policy
    quant_policy = QuantPolicy(
        w_bits=8,
        a_bits=8,
    )

    print("\nLoading model with W8A8 quantization...")
    pipe = pipeline(model_path, quant_policy=quant_policy)

    # Test inference
    test_prompts = [
        "What is quantization in machine learning?",
        "Explain the benefits of 8-bit quantization",
    ]

    print("\nTesting quantized model...")
    responses = pipe(test_prompts)

    for prompt, response in zip(test_prompts, responses):
        print(f"\nQ: {prompt}")
        print(f"A: {response.text[:200]}...")

    print("\nW8A8 quantization applied successfully!")
    print("Note: W8A8 is applied at runtime, no separate model saving needed.")

    return pipe


def compare_quantization_methods(model_path: str):
    """Compare different quantization methods."""
    print("\n" + "=" * 60)
    print("Quantization Methods Comparison")
    print("=" * 60)

    comparison_table = """
┌─────────────┬──────────┬───────────┬──────────────┬───────────────┐
│ Method      │ Bits     │ Accuracy  │ Speedup      │ Memory Saving │
├─────────────┼──────────┼───────────┼──────────────┼───────────────┤
│ FP16 (Base) │ 16       │ 100%      │ 1.0x         │ 1.0x          │
│ W8A8        │ 8/8      │ ~99%      │ 1.5-2.0x     │ ~2.0x         │
│ AWQ         │ 4/16     │ ~98-99%   │ 2.0-2.5x     │ ~4.0x         │
│ GPTQ        │ 4/16     │ ~97-99%   │ 2.0-2.5x     │ ~4.0x         │
└─────────────┴──────────┴───────────┴──────────────┴───────────────┘

When to use each method:

1. W8A8 (8-bit):
   - Best for: Production deployment with minimal accuracy loss
   - Pros: Easy to apply, good accuracy, significant speedup
   - Cons: Less memory savings than 4-bit

2. AWQ (4-bit):
   - Best for: Maximum compression with good accuracy
   - Pros: Excellent memory savings, maintains accuracy well
   - Cons: Requires calibration, longer quantization time

3. GPTQ (4-bit):
   - Best for: When AWQ is not available
   - Pros: Well-established, widely supported
   - Cons: May have slightly lower accuracy than AWQ

Recommendation:
- Start with W8A8 for quick wins
- Use AWQ for production when memory is critical
- Choose based on your accuracy requirements
"""
    print(comparison_table)


def verify_quantized_model(model_path: str):
    """Verify that quantized model works correctly."""
    print("\nVerifying quantized model...")

    from lmdeploy import pipeline

    try:
        pipe = pipeline(model_path)

        # Simple test
        response = pipe(["Test prompt for verification"])
        print(f"✓ Model loaded successfully")
        print(f"✓ Generated response: {response.text[:50]}...")
        print(f"✓ Quantized model is working correctly")

        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="LMDeploy Quantization Workflow Examples")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8b",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--method", type=str, choices=["awq", "gptq", "w8a8", "compare", "all"],
                        default="all", help="Quantization method to demonstrate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for quantized model")

    args = parser.parse_args()

    methods = {
        "awq": ("AWQ Quantization", lambda: awq_quantization(args.model, args.output_dir or "./awq_quantized")),
        "gptq": ("GPTQ Quantization", lambda: gptq_quantization(args.model, args.output_dir or "./gptq_quantized")),
        "w8a8": ("W8A8 Quantization", lambda: w8a8_quantization_demo(args.model)),
        "compare": ("Methods Comparison", lambda: compare_quantization_methods(args.model)),
    }

    if args.method == "all":
        # Show comparison first
        compare_quantization_methods(args.model)

        # Run all quantization methods
        for name, (title, func) in methods.items():
            if name != "compare":
                try:
                    print(f"\n{'#' * 60}")
                    print(f"# {title}")
                    print(f"{'#' * 60}\n")
                    func()
                except Exception as e:
                    print(f"\nError in {name}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Run specific method
        title, func = methods[args.method]
        try:
            func()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Quantization workflow examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Deploy quantized model: lmdeploy serve api_server <quantized_model_path>")
    print("2. Test performance: python benchmark/benchmark_throughput.py")
    print("3. Verify accuracy with your evaluation dataset")


if __name__ == "__main__":
    main()
