#!/usr/bin/env python3
"""End-to-end smoke test for MemDecode dual-checkpoint setup in lmdeploy
PyTorch.

Starts an api_server with a base model plus MemDecode ``memory_model_path`` and
optional router settings via ``--hf-overrides``, then runs a simple Chat
Completion call.

Default paths target Intern-S2-Preview + fine-tuned 4B memory with **fixed** fusion
(``lambda_value``). The bundled ``DEFAULT_ROUTER_PATH`` is trained for Intern-S2-397B
base hidden size and does **not** match Preview; use ``--mode fixed`` for Preview.

Example (Intern-S2-Preview + memory, single GPU):

    CUDA_VISIBLE_DEVICES=0 python examples/memdec/main.py \\
        --base-model-path /path/to/Intern-S2-Preview \\
        --memory-model-path /path/to/memory/checkpoint \\
        --mode fixed --tp 1 --max-new-tokens 64

Adaptive mode requires a router checkpoint whose input dim matches
``base_hidden_size + memory_hidden_size + 4 * scalar_proj_dim`` for your pair.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

DEFAULT_BASE_MODEL_PATH = (
    '/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--internlm--Intern-S2-Preview/'
    'snapshots/4f57cab513689b089019fce4ad24e26520df183c'
)
DEFAULT_MEM_MODEL_PATH = (
    '/mnt/shared-storage-user/llmrazor-share/memdecode/'
    'interns2_4bmemory_m4match_clean8n_tp2_mbs1_gbs1024_lr4e5_wu5p_3epoch_hf'
)
DEFAULT_ROUTER_PATH = (
    # Trained for Intern-S2-397B + 4B memory (mlp input dim 6912). Not for Preview (4864).
    '/mnt/shared-storage-user/llmrazor-share/memdecode/'
    'interns2_397b_base02_2806_plus_interns2_4b_mem_m4match_clean8n_sft32k_bio128k_realmm_mlp4_lr4e4_epoch1_20260622'
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--base-model-path', default=DEFAULT_BASE_MODEL_PATH)
    p.add_argument('--memory-model-path', default=DEFAULT_MEM_MODEL_PATH)
    p.add_argument(
        '--router-path',
        default=DEFAULT_ROUTER_PATH,
        help='Adaptive mode only. Router must match base/memory hidden sizes (not Preview + 397B router).',
    )
    p.add_argument(
        '--mode',
        choices=['fixed', 'adaptive'],
        default='fixed',
        help='Fusion mode. Use fixed for Intern-S2-Preview unless you have a matching router.',
    )
    p.add_argument('--server-name', default='127.0.0.1')
    p.add_argument('--server-port', type=int, default=23333)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--dp', type=int, default=1)
    p.add_argument('--ep', type=int, default=1)
    p.add_argument('--model-format', default=None)
    p.add_argument('--cache-max-entry-count', type=float, default=0.8)
    p.add_argument('--max-batch-size', type=int, default=4)
    p.add_argument(
        '--lambda-value',
        type=float,
        default=0.5,
        help='Fixed lambda for fused logits.',
        dest='lambda_value',
    )
    p.add_argument('--lambda-base-only-threshold', type=float, default=-1.0,
                   help='Adaptive mode only; if >=0.0, force base-only when lambda < threshold.')
    p.add_argument('--model-name', default='memdecode-intern-s2-preview')
    p.add_argument('--max-new-tokens', type=int, default=None)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--prompt', default='Explain MemDecode in one concise sentence.')
    p.add_argument('--timeout', type=int, default=3600)
    return p.parse_args()


def build_hf_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        'memory_model_path': args.memory_model_path,
        'lambda_value': args.lambda_value,
        'adaptive_router': False,
    }
    if args.mode == 'adaptive':
        overrides['router_path'] = args.router_path
        overrides['adaptive_router'] = True
        overrides['lambda_base_only_threshold'] = args.lambda_base_only_threshold
    return overrides


def wait_for_server_ready(base_url: str, timeout_s: int = 120) -> None:
    models_url = f'{base_url}/v1/models'
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            resp = requests.get(models_url, timeout=3)
            if resp.status_code == 200:
                return
            last_error = f'status={resp.status_code}, body={resp.text[:200]}'
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
        time.sleep(2)
    raise RuntimeError(f'server did not become ready in {timeout_s}s, last_error={last_error}')


def query_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
):
    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'top_p': top_p,
        'stream': False,
    }
    if max_tokens is not None:
        payload['max_tokens'] = max_tokens
    resp = requests.post(f'{base_url}/v1/chat/completions', json=payload, timeout=120)
    resp.raise_for_status()
    out = resp.json()
    text = out['choices'][0]['message']['content']
    return out, text


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.base_model_path):
        raise FileNotFoundError(f'base model path does not exist: {args.base_model_path}')
    if not os.path.exists(args.memory_model_path):
        raise FileNotFoundError(f'memory model path does not exist: {args.memory_model_path}')
    if args.mode == 'adaptive' and not os.path.exists(args.router_path):
        raise FileNotFoundError(f'router path does not exist: {args.router_path}')

    env = os.environ.copy()

    hf_overrides = build_hf_overrides(args)
    server_url = f'http://{args.server_name}:{args.server_port}'

    cmd = [
        sys.executable,
        '-m', 'lmdeploy',
        'serve', 'api_server', args.base_model_path,
        '--backend', 'pytorch',
        '--server-name', args.server_name,
        '--server-port', str(args.server_port),
        '--tp', str(args.tp),
        '--dp', str(args.dp),
        '--ep', str(args.ep),
        '--cache-max-entry-count', str(args.cache_max_entry_count),
        '--max-batch-size', str(args.max_batch_size),
        '--trust-remote-code',
        '--model-name', args.model_name,
        '--log-level', 'INFO',
        '--hf-overrides', json.dumps(hf_overrides),
    ]
    if args.model_format is not None:
        cmd.extend(['--model-format', args.model_format])

    print('Starting server:')
    print('  ' + ' '.join(cmd))
    print('with hf_overrides:')
    print('  ' + json.dumps(hf_overrides, indent=2))

    proc = subprocess.Popen(cmd, env=env)
    try:
        wait_for_server_ready(server_url, timeout_s=args.timeout)
        print('Server ready. Running smoke test...')
        model_resp = requests.get(f'{server_url}/v1/models').json()
        print('Models:')
        print(model_resp)

        out, text = query_chat(
            server_url,
            model_name=args.model_name,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print('Response:')
        print(text)
        print('Full response:')
        print(json.dumps(out, indent=2))
        print('PASS: chat completion returned text.')
        return 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == '__main__':
    raise SystemExit(main())
