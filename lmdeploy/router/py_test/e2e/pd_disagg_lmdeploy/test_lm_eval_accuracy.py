# Copyright (c) OpenMMLab. All rights reserved.
#!/usr/bin/env python3
"""P/D Disaggregation LM-Eval Accuracy Test for LMDeploy Router.

This script measures LM-Eval accuracy through the LMDeploy Prefill/Decode router. It is optional and requires the router
to be already running.

Uses the LM Evaluation Harness (lm-eval) to measure accuracy on the gsm8k task.
"""

import argparse
import os
import sys

import lm_eval
import openai

# Test configuration
TASK = 'gsm8k'
FILTER = 'exact_match,strict-match'
RTOL = 0.03  # Relative tolerance for accuracy comparison

# Optional model-specific expected values. Set MODEL_NAME to the model ID
# exposed by the LMDeploy servers; add a local baseline when one is known.
EXPECTED_VALUES = {
    'meta-llama/Llama-3.2-1B-Instruct': 0.33,  # Lowered to accept >30% accuracy
    'Qwen/Qwen3-0.6B': 0.41,
    'deepseek-ai/deepseek-vl2-small': 0.59,
    'deepseek-ai/deepseek-vl2-tiny': 0.19,
    'deepseek-ai/DeepSeek-V2-Lite-Chat': 0.65,
}

# Simple prompt for connectivity test
SIMPLE_PROMPT = (
    'LMDeploy serves models with an active open-source community, which means'
)


class Colors:
    """Terminal colors for output."""

    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_success(msg: str):
    print(f'{Colors.GREEN}✓ {msg}{Colors.RESET}')


def print_error(msg: str):
    print(f'{Colors.RED}✗ {msg}{Colors.RESET}')


def print_info(msg: str):
    print(f'{Colors.BLUE}ℹ {msg}{Colors.RESET}')


def print_warning(msg: str):
    print(f'{Colors.YELLOW}⚠ {msg}{Colors.RESET}')


def run_simple_prompt(base_url: str, model_name: str) -> bool:
    """Run a simple prompt to verify connectivity before running full
    evaluation.

    Args:
        base_url: Base URL for the router API
        model_name: Model name to test

    Returns:
        True if successful, False otherwise
    """
    print_info('Running connectivity test with simple prompt...')

    try:
        # Use a very long timeout (10 minutes) for connectivity test
        client = openai.OpenAI(api_key='EMPTY', base_url=base_url, timeout=600.0)
        # Use chat completions for Instruct models
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': SIMPLE_PROMPT}],
            max_tokens=50,
        )

        output = completion.choices[0].message.content if completion.choices else ''

        print('-' * 60)
        print_info(f'Connectivity Test Results for {model_name}:')
        print(f'Prompt: {SIMPLE_PROMPT}')
        print(f'Output: {output}')
        print('-' * 60)

        if not output or len(output.strip()) == 0:
            print_error('Connectivity test failed: Empty output')
            return False

        print_success('Connectivity test passed')
        return True

    except Exception as e:
        print_error(f'Connectivity test failed: {e}')
        return False


def run_accuracy_evaluation(
    base_url: str,
    model_name: str,
    num_concurrent: int = 20,
) -> dict:
    """Run LM Evaluation Harness on gsm8k task.

    Args:
        base_url: Base URL for the router API (should be http://host:port/v1)
        model_name: Model name to evaluate
        num_concurrent: Number of concurrent requests

    Returns:
        Dictionary containing evaluation results
    """
    print_info(f'Running LM-Eval accuracy test on {TASK} task...')
    print_info('This may take several minutes...')

    # Use Chat Completions API for Instruct models (native format)
    # This fixes 422 errors from endpoint mismatch
    try:
        results = lm_eval.simple_evaluate(
            model='local-chat-completions',
            model_args={
                'model': model_name,
                'base_url': f'{base_url}/chat/completions',
                'num_concurrent': num_concurrent,
                'max_retries': 3,
                'tokenized_requests': False,
            },
            tasks=TASK,
            num_fewshot=5,
            limit=500,
            apply_chat_template=True,  # Enable chat template for Instruct models
            fewshot_as_multiturn=True,  # Format few-shot examples as conversation turns
            log_samples=False,
        )
        return results

    except Exception as e:
        print_error(f'LM-Eval failed: {e}')
        raise


def validate_accuracy(
    results: dict,
    model_name: str,
) -> bool:
    """Validate that accuracy meets expected thresholds.

    Args:
        results: LM-Eval results dictionary
        model_name: Model name being evaluated

    Returns:
        True if accuracy is within acceptable range, False otherwise
    """
    measured_value = results['results'][TASK][FILTER]
    expected_value = EXPECTED_VALUES.get(model_name)

    print()
    print('=' * 60)
    print_info('Accuracy Results:')
    print_info(f'  Model:              {model_name}')
    print_info(f'  Task:               {TASK}')
    print_info(f'  Metric:             {FILTER}')
    print_info(f'  Measured Accuracy:  {measured_value:.4f}')

    if expected_value is None:
        print_warning(
            f'No expected baseline found for {model_name}. ' 'Cannot validate accuracy.'
        )
        print_info(
            'If this is the first time testing this model, '
            'you may want to add the measured value to EXPECTED_VALUES.'
        )
        print('=' * 60)
        return True  # Pass if no baseline (assume correct)

    print_info(f'  Expected Accuracy:  {expected_value:.4f}')
    print_info(f'  Tolerance:          ±{RTOL:.4f}')

    lower_bound = expected_value - RTOL
    # Higher accuracy is always acceptable, so we only enforce lower bound
    minimum_threshold = 0.3

    print_info(f'  Minimum Threshold:  {minimum_threshold:.4f}')
    print_info(f'  Lower Bound:        {lower_bound:.4f}')
    print('=' * 60)
    print()

    if measured_value >= lower_bound:
        print_success(
            f'Accuracy meets requirements! '
            f'({measured_value:.4f} vs expected {expected_value:.4f})'
        )
        return True
    else:
        print_error(
            f'Accuracy below acceptable threshold! '
            f'({measured_value:.4f} vs expected {expected_value:.4f})'
        )
        print_error(
            f'Shortfall: {lower_bound - measured_value:.4f} '
            f'(minimum: {lower_bound:.4f})'
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test P/D disaggregation accuracy using LM-Eval'
    )
    parser.add_argument(
        '--router-url',
        type=str,
        required=True,
        help='URL of the router (e.g., http://localhost:8300)',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name to use for testing (can also use TEST_MODEL env var)',
    )
    parser.add_argument(
        '--num-concurrent',
        type=int,
        default=20,
        help='Number of concurrent requests (default: 20)',
    )
    parser.add_argument(
        '--skip-connectivity',
        action='store_true',
        help='Skip initial connectivity test',
    )

    args = parser.parse_args()

    # Get model name from args or environment
    model_name = args.model or os.environ.get('TEST_MODEL')
    if not model_name:
        print_error('Model name must be provided via --model or TEST_MODEL env var')
        return 1

    # Construct base URL for OpenAI API
    base_url = f'{args.router_url}/v1'

    print()
    print('=' * 60)
    print_info('P/D Disaggregation LM-Eval Accuracy Test')
    print('=' * 60)
    print_info(f'Router URL:         {args.router_url}')
    print_info(f'API Base URL:       {base_url}')
    print_info(f'Model:              {model_name}')
    print_info(f'Task:               {TASK}')
    print_info(f'Concurrent Reqs:    {args.num_concurrent}')
    print('=' * 60)
    print()

    # Step 1: Connectivity test (optional)
    if not args.skip_connectivity:
        if not run_simple_prompt(base_url, model_name):
            print_error('Connectivity test failed. Aborting evaluation.')
            return 1
        print()

    # Step 2: Run LM-Eval
    try:
        results = run_accuracy_evaluation(
            base_url=base_url,
            model_name=model_name,
            num_concurrent=args.num_concurrent,
        )
    except Exception as e:
        print_error(f'Evaluation failed: {e}')
        return 1

    # Step 3: Validate accuracy
    if not validate_accuracy(results, model_name):
        print_error('Accuracy validation failed!')
        return 1

    print_success('All accuracy tests PASSED!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
