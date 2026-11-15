"""Generate reproducible token traces for A3/CUDA eager, graph and MTP
comparison.

Run each mode in a separate process. This checks deterministic token agreement, not throughput. Full benchmark and task-
quality evaluation remain separate.
"""

import argparse
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', choices=('ascend', 'cuda'), default='ascend')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--ep', type=int, default=1)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--prefix-cache', action='store_true')
    parser.add_argument('--spec-method', choices=('qwen3_5_mtp', 'deepseek_mtp'))
    parser.add_argument('--spec-model', default='')
    parser.add_argument('--spec-tokens', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--prefill-tokens', type=int, default=256)
    parser.add_argument('--new-tokens', type=int, default=32)
    parser.add_argument('--sampling', choices=('greedy', 'mixed', 'random'), default='greedy')
    parser.add_argument('--bad-word', action='append', default=[])
    parser.add_argument('--quant-policy', type=int, choices=(0, 8), default=0)
    parser.add_argument('--prompts', type=Path, help='JSON list of strings or LMDeploy multimodal prompt objects')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.compare and args.sampling != 'greedy':
        parser.error('--compare checks exact greedy tokens; use --sampling greedy')
    if args.device == 'ascend':
        if importlib.util.find_spec('dlinfer') is not None:
            raise RuntimeError('Use a clean environment with no dlinfer installation.')

        class BlockDlinfer(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname.split('.')[0] == 'dlinfer':
                    raise RuntimeError('Unexpected import: ' + fullname)

        sys.meta_path.insert(0, BlockDlinfer())
    from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
    from lmdeploy.messages import SpeculativeConfig

    config = PytorchEngineConfig(
        device_type=args.device,
        tp=args.tp,
        dp=args.dp,
        ep=args.ep,
        eager_mode=args.eager,
        enable_prefix_caching=args.prefix_cache,
        max_batch_size=args.batch_size,
        block_size=args.block_size,
        max_prefill_token_num=args.prefill_tokens,
        quant_policy=args.quant_policy,
    )
    spec = (
        None
        if args.spec_method is None
        else SpeculativeConfig(method=args.spec_method, model=args.spec_model, num_speculative_tokens=args.spec_tokens)
    )
    prompts = (
        json.loads(args.prompts.read_text())
        if args.prompts
        else [
            'Write a short Python function that sums a list of integers.',
            'Explain why the sky is blue in one sentence.',
            'One two three four five. ' * 180 + 'Continue with a short summary.',
            'Translate into Chinese: Efficient inference needs careful memory management.',
        ]
    )
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('--prompts must contain a nonempty JSON list')

    def generation(index):
        sample = args.sampling == 'random' or (args.sampling == 'mixed' and index % 2 == 1)
        return GenerationConfig(
            do_sample=sample,
            top_k=10 if sample else 1,
            temperature=0.8 if sample else 1.0,
            max_new_tokens=args.new_tokens,
            ignore_eos=True,
            random_seed=42 + index,
            logprobs=1,
            bad_words=args.bad_word or None,
        )

    traces = []
    with pipeline(args.model, backend_config=config, speculative_config=spec) as pipe:
        # Repeat and change the batch shape to exercise replay and prefix/state reuse.
        batches = [prompts[:1], prompts, prompts[:1], prompts[: min(3, len(prompts))]]
        for batch in batches:
            responses = pipe(batch, gen_config=[generation(i) for i in range(len(batch))])
            for response in responses:
                if response.finish_reason in ('error', 'abort') or not response.token_ids:
                    raise RuntimeError(f"Generation failed: {response}")
            traces.append(
                [
                    dict(token_ids=r.token_ids, text=r.text, cached_tokens=r.cached_tokens, logprobs=r.logprobs)
                    for r in responses
                ]
            )
    payload = dict(
        options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        traces=traces,
    )
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    # Identical prompt, fresh request: catch state leakage on replay/prefix hits.
    if args.sampling != 'random' and traces[0][0]['token_ids'] != traces[2][0]['token_ids']:
        raise AssertionError('Repeated prompt changed greedy tokens; inspect the saved trace.')
    if args.compare:
        previous = json.loads(args.compare.read_text())['traces']
        actual = [[r['token_ids'] for r in batch] for batch in traces]
        expected = [[r['token_ids'] for r in batch] for batch in previous]
        if actual != expected:
            raise AssertionError(
                f"Greedy tokens differ from {args.compare}; inspect logits/tolerances before acceptance."
            )
    print(f"Saved {args.output}; generation completed. Random sampling requires separate distribution checks.")


if __name__ == '__main__':
    main()
