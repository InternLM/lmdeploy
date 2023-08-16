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
    total_tokens = sum(map(len, input_ids))

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

    # llama-2 on 1 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 175.3 seconds, 9012.821089984884 tokens/s.
    # Decoded 7022 prompts in 175.3 seconds, 40.067481648961376 requests/s.

    # llama-2 on 3 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 77.9 seconds, 20268.736076299527 tokens/s.
    # Decoded 7022 prompts in 77.9 seconds, 90.10688248180179 requests/s.

    # llama-2 on 8 A100:
    # data = ShareGPT, downsample = 100
    # Decoded 1579536 tokens in 55.2 seconds, 28630.35872677815 tokens/s.
    # Decoded 7022 prompts in 55.2 seconds, 127.27939026361929 requests/s.

    # llama-2 on 8 A100:
    # data = ShareGPT, downsample = 10
    # Decoded 15991314 tokens in 242.7 seconds, 65893.38488718234 tokens/s.
    # Decoded 70216 prompts in 242.7 seconds, 289.33018970413536 requests/s.

    # Above time all includes time for workers to load model.
