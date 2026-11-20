# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Golden sample gate for the W4A16 GEMM routes.

A fixed prompt set (8 zh, 8 en, 8 code, 8 math) drives two engines on the
same model: the AWQ Triton path (`LMDEPLOY_W4A16_GEMM_BACKEND` off the
TurboMind values) dumps the reference, and the TurboMind path must match it.

The gate carries two numbers per run:

- top1 agreement of the first generated token over the full vocabulary;
- truncated KL between the first-token distributions, restricted to the
  reference top-20 support.

The reference lives in ``awq_ref.json`` so later engine changes still face
the frozen external judge (self-consistent parity never proves semantics).
"""

import json
import os

import torch

# 8 zh, 8 en, 8 code, 8 math. The set is literal so any future run sees the
# exact same inputs. The fixed seed in callers covers sampling paths.
PROMPTS = [
    # zh
    '请用一句话介绍长城的历史意义。',
    '简单解释一下什么是光合作用。',
    '写一句鼓励正在进行毕业答辩学生的中文句子。',
    '把"天气真好"翻译成英文并造一个对话场景。',
    '中文里"塞翁失马"是什么意思？',
    '用中文列出三个健康的早餐建议。',
    '写一首关于秋天的中文俳句。',
    '说明普通话和粤语的一个主要区别。',
    # en
    'Summarize the plot of Romeo and Juliet in one sentence.',
    'Explain the difference between weather and climate.',
    'What are three tips for writing a clear technical document?',
    'Name the largest ocean on Earth and one fact about it.',
    'Rewrite this sentence politely: "Send me the report now."',
    'What is the capital of Australia, and when was it founded?',
    'Give a short definition of machine learning.',
    'Suggest a title for a blog post about remote work.',
    # code
    'Write a Python function that reverses a list without using slicing.',
    'What does the SQL statement SELECT COUNT(*) FROM users return?',
    'Fix this Python bug: for i in range(len(arr)): print(arr[i+1])',
    'Write a bash one-liner that counts lines in all .py files.',
    'Explain what git rebase does in two sentences.',
    'Write a JavaScript function that debounces another function.',
    'What is the time complexity of binary search?',
    'Write a C snippet that swaps two integers with XOR.',
    # math
    'What is 17 multiplied by 23?',
    'Solve for x: 3x + 7 = 22.',
    'What is the derivative of x^3 + 2x?',
    'Compute the sum of the first 10 positive integers.',
    'What is 20% of 150?',
    'If a train travels 60 km in 45 minutes, what is its speed in km/h?',
    'What is the area of a circle with radius 3? Use pi = 3.14.',
    'Simplify the fraction 42/56.',
]

MAX_NEW_TOKENS = 16
TOP_K_FIRST = 20
# Measured: worst impl-level tm-vs-awq KL 1.9e-4 (large-K layers, m>=16).
# The engine-level threshold starts one order above that measurement.
KL_THRESHOLD = 2e-3


def top20_of(logits: torch.Tensor):
    """Return the ids and raw logits of the top-20 next tokens."""
    values, ids = torch.topk(logits, TOP_K_FIRST)
    return ids.tolist(), values.tolist()


def truncated_kl(ref_top20_logits, tm_logits: torch.Tensor, ids: list[int]) -> float:
    """KL(ref || tm) restricted to the reference top-20 support.

    ``ref_top20_logits`` holds the reference logits aligned with ``ids``;
    ``tm_logits`` is the full candidate next-token logits.
    """
    index = torch.tensor(ids, dtype=torch.long)
    ref = torch.log_softmax(ref_top20_logits, dim=-1)
    tm = torch.log_softmax(tm_logits[index], dim=-1)
    return (ref.exp() * (ref - tm)).sum().item()


def run_engine(model_path: str,
               prompts: list[str],
               backend: str | None,
               cache_ratio: float = 0.15,
               log_level: str = 'WARNING'):
    """Run the prompt set on one pytorch engine and collect gate artifacts.

    ``backend`` selects ``LMDEPLOY_W4A16_GEMM_BACKEND`` for the engine. None
    keeps the ambient default (``auto``). Prompts run without the chat
    template on both paths, so the reference and the candidate see the same
    raw token ids. Returns one dict per prompt with the greedy token ids and
    the first-token logits.
    """
    import lmdeploy.pytorch.envs as envs
    from lmdeploy import pipeline
    from lmdeploy.messages import GenerationConfig, PytorchEngineConfig

    # The attribute patch covers an in-process (uni) executor; the environ
    # copy covers an mp executor whose workers re-read the env on spawn.
    original = envs.w4a16_gemm_backend
    original_env = os.environ.get('LMDEPLOY_W4A16_GEMM_BACKEND')
    original_kernels = os.environ.get('LMDEPLOY_TM_KERNELS')
    if backend is not None:
        envs.w4a16_gemm_backend = backend
        os.environ['LMDEPLOY_W4A16_GEMM_BACKEND'] = backend
        # The unified-env class entry wins over the legacy env, so pin it too:
        # an ambient LMDEPLOY_TM_KERNELS must not silently re-route the run.
        state = 'on' if backend in ('auto', 'turbomind') else 'off'
        os.environ['LMDEPLOY_TM_KERNELS'] = f'w4a16={state}'
    try:
        engine_config = PytorchEngineConfig(session_len=1024,
                                            max_batch_size=8,
                                            cache_max_entry_count=cache_ratio,
                                            enable_prefix_caching=False,
                                            distributed_executor_backend='uni')
        pipe = pipeline(model_path, backend_config=engine_config, log_level=log_level)
        try:
            gen_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS,
                                          do_sample=False,
                                          ignore_eos=True,
                                          random_seed=0)
            greedy = pipe(prompts, gen_config=gen_config, do_preprocess=False)
            tokenizer = pipe.async_engine.tokenizer
            input_ids = [tokenizer.encode(prompt) for prompt in prompts]
            logits = pipe._run(coro=pipe.async_engine.async_get_logits(input_ids=input_ids)).result()
        finally:
            pipe.close()
    finally:
        envs.w4a16_gemm_backend = original
        if original_env is None:
            os.environ.pop('LMDEPLOY_W4A16_GEMM_BACKEND', None)
        else:
            os.environ['LMDEPLOY_W4A16_GEMM_BACKEND'] = original_env
        if original_kernels is None:
            os.environ.pop('LMDEPLOY_TM_KERNELS', None)
        else:
            os.environ['LMDEPLOY_TM_KERNELS'] = original_kernels

    results = []
    for greedy_resp, logits_tensor in zip(greedy, logits):
        results.append(
            dict(
                greedy_ids=list(greedy_resp.token_ids),
                first_logits=logits_tensor.detach().to(torch.float32).reshape(-1,
                                                                              logits_tensor.shape[-1])[-1].cpu(),
            ))
    return results


def dump_reference(model_path: str, out_path: str, backend: str = 'triton'):
    """Dump the AWQ Triton reference for the fixed prompt set."""
    results = run_engine(model_path, PROMPTS, backend=backend)
    payload = {
        'model': os.path.basename(os.path.normpath(model_path)),
        'prompts': PROMPTS,
        'max_new_tokens': MAX_NEW_TOKENS,
        'first_top20_ids': [],
        'first_top20_logits': [],
        'greedy_ids': [],
    }
    for item in results:
        ids, values = top20_of(item['first_logits'])
        payload['first_top20_ids'].append([int(i) for i in ids])
        payload['first_top20_logits'].append([float(v) for v in values])
        payload['greedy_ids'].append([int(t) for t in item['greedy_ids']])
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def gate_against(reference: dict, results: list) -> dict:
    """Compare one engine run against the reference payload.

    Returns the gate numbers; the caller decides pass or fail against the thresholds, so reports can carry the measured
    distribution.
    """
    assert len(results) == len(reference['prompts'])
    top1_hits = 0
    greedy_hits = 0
    worst_kl = 0.0
    per_prompt = []
    for idx, (ref_item, item) in enumerate(zip(zip(reference['first_top20_ids'], reference['first_top20_logits'],
                                                   reference['greedy_ids']), results)):
        ref_ids, ref_values, ref_greedy = ref_item
        tm_logits = item['first_logits']
        tm_top1 = int(torch.argmax(tm_logits).item())
        ref_top1 = ref_ids[0]
        hit = tm_top1 == ref_top1
        top1_hits += hit
        kl = truncated_kl(torch.tensor(ref_values, dtype=torch.float32), tm_logits, ref_ids)
        worst_kl = max(worst_kl, kl)
        same_greedy = list(item['greedy_ids']) == ref_greedy
        greedy_hits += same_greedy
        per_prompt.append(dict(index=idx, top1_match=hit, kl=kl, greedy_match=same_greedy))
    return dict(
        num_prompts=len(results),
        top1_agreement=top1_hits / len(results),
        greedy_agreement=greedy_hits / len(results),
        worst_truncated_kl=worst_kl,
        kl_threshold=KL_THRESHOLD,
        per_prompt=per_prompt,
    )
