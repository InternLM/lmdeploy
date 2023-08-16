import json
import pickle
import time
from pathlib import Path

import fire
import numpy as np
from transformers import AutoTokenizer

from lmdeploy.pytorch.decode import Engine


def benchmark(model_path,
              share_gpt_path,
              downsample=100,
              accel=None,
              save_to='decode_result'):
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

    texts = texts[::downsample]
    input_ids = tokenizer(texts, padding=False).input_ids
    print(F'Number of prompts: {len(input_ids)}')
    print(F'Maximum length: {max(map(len, input_ids))}')
    print(F'Total length: {sum(map(len, input_ids))}')

    start = time.monotonic()
    # Init an engine
    engine = Engine(model_path, tokenizer=tokenizer, accel=accel)
    # decode prompts
    probs = engine.decode(input_ids)
    total_tokens = sum(p.size for p in probs)

    elapsed = time.monotonic() - start
    print(f'Decoded {total_tokens} tokens in {elapsed:.1f} seconds, '
          f'{total_tokens / elapsed} tokens/s.')
    print(f'Decoded {len(probs)} prompts in {elapsed:.1f} seconds, '
          f'{len(probs) / elapsed} requests/s.')

    pkl_path = Path(save_to).with_suffix('.pkl')

    with pkl_path.open('wb') as f:
        pickle.dump(probs, f)

    txt_path = Path(save_to).with_suffix('.txt')
    np.savetxt(txt_path.as_posix(), probs, fmt='%.4e')


if __name__ == '__main__':
    fire.Fire(benchmark)
    # llama-2 on 3 A100:
    # Decoded 2665438 tokens in 111.4 seconds, 23922.215803221567 tokens/s.
    # Decoded 7022 prompts in 111.4 seconds, 63.022212248126515 requests/s.
