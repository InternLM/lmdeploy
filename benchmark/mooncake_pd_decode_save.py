# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise decode checkpoints and reuse generated tokens through the PD
router.

Confirm persistence using matching state_hash values in store_save_scheduled / store_load traces and a successful
store_save_finished for the same save_id. HTTP success alone does not prove a Store hit. Run outside performance
measurements.
"""
import argparse
import json
import time

from mooncake_pd_validate import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pd-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--max-tokens', type=int, default=1088)
    args = parser.parse_args()
    tokens = ([149, 150, 151, 152, 153, 154, 155] * ((args.prompt_tokens + 6) // 7))[:args.prompt_tokens]
    begin = time.time()
    source = generate(args.pd_url, args.model, tokens, args.max_tokens)
    continuation = tokens + source['output_ids'] + [137, 138, 139, 140, 141, 142, 143, 144]
    resumed = generate(args.pd_url, args.model, continuation, 16)
    print(json.dumps(dict(begin=begin, end=time.time(), input_length=len(tokens),
                          continuation_length=len(continuation), source=source, resumed=resumed)))


if __name__ == '__main__':
    main()
