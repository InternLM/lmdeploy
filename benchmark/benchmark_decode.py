import json
import pickle
import time

import fire
from transformers import AutoTokenizer

from lmdeploy.pytorch.decode import Engine


def benchmark(model_path,
              share_gpt_path,
              downsample=100,
              accel=None,
              save_to='decode_result.pkl'):
    """Benchmark using ShareGPT data.

    Please download `ShareGPT_V3_unfiltered_cleaned_split.json` as data for
    this benchmark.
    """

    start = time.monotonic()
    content = json.load(open(share_gpt_path, 'r'))

    texts = []
    for c in content:
        for cc in c['conversations']:
            texts.append(cc['value'])

    print(f'Parse json in {time.monotonic() - start} seconds.')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    start = time.monotonic()
    # Init an engine
    engine = Engine(model_path, tokenizer=tokenizer, accel=accel)
    # decode prompts
    probs = engine.decode(texts[::downsample], sort=True)
    total_tokens = sum(p.numel() for p in probs)

    elapsed = time.monotonic() - start
    print(f'Decoded {total_tokens} tokens in {elapsed:.1f} seconds, '
          f'{total_tokens / elapsed} tokens/s.')
    print(f'Decoded {len(probs)} prompts in {elapsed:.1f} seconds, '
          f'{len(probs) / elapsed} requests/s.')

    with open(save_to, 'wb') as f:
        pickle.dump(probs, f)


if __name__ == '__main__':
    fire.Fire(benchmark)
    # llama-2 on 3 A100:
    # Decoded 2665438 tokens in 111.4 seconds, 23922.215803221567 tokens/s.
    # Decoded 7022 prompts in 111.4 seconds, 63.022212248126515 requests/s.
