# Copyright (c) OpenMMLab. All rights reserved.
"""Run the vLLM PR #59246 load matrix against an LMDeploy server.

Start two otherwise identical TP8 servers in turn, with MTP disabled and
``--quant-policy none`` or ``--quant-policy fp8``. Use a separate output
directory for each policy. Requires a vLLM client environment; this script
does not start a vLLM model server.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:23333')
    parser.add_argument('--model', required=True, help='Served model name')
    parser.add_argument('--tokenizer', required=True, help='Checkpoint path or tokenizer ID')
    parser.add_argument('--vllm-python', default=sys.executable)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    for input_len, output_len in [(1024, 128), (16384, 256)]:
        for concurrency, prompts in [(1, 16), (8, 32), (32, 128)]:
            name = f'p{input_len}_d{output_len}_c{concurrency}'
            command = [
                args.vllm_python, '-m', 'vllm.entrypoints.cli.main', 'bench', 'serve',
                '--backend', 'openai', '--base-url', args.base_url, '--endpoint', '/v1/completions',
                '--model', args.model, '--tokenizer', args.tokenizer, '--dataset-name', 'random',
                '--random-input-len', str(input_len), '--random-output-len', str(output_len),
                '--random-range-ratio', '0', '--num-prompts', str(prompts),
                '--max-concurrency', str(concurrency), '--ignore-eos', '--seed', '42',
                '--num-warmups', str(concurrency), '--disable-tqdm', '--save-result',
                '--result-dir', str(args.output_dir), '--result-filename', f'{name}.json',
            ]
            (args.output_dir / f'{name}.command.json').write_text(json.dumps(command, indent=2))
            with (args.output_dir / f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((args.output_dir / f'{name}.json').read_text())
            if (result['completed'] != prompts
                    or result['total_input_tokens'] != prompts * input_len
                    or result['total_output_tokens'] != prompts * output_len):
                raise RuntimeError(f'Unexpected requests/input/output lengths: {name}')
            row = dict(case=name, **{key: result[key] for key in (
                'completed', 'total_input_tokens', 'total_output_tokens',
                'output_throughput', 'mean_tpot_ms', 'mean_ttft_ms')})
            summary.append(row)
            print(json.dumps(row), flush=True)
            (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
