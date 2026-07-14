#!/usr/bin/env python3
"""Smoke-test InternVL3.5 VLM with an image."""
import os
import sys
import time

import huggingface_hub.constants as hf_constants


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'OpenGVLab/InternVL3_5-8B'
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else '/nvme2/huggingface_hub/hub'
    image_path = sys.argv[3] if len(sys.argv) > 3 else '/data/lmdeploy-modeling/resources/batch_memory.png'
    gpus = sys.argv[4] if len(sys.argv) > 4 else '0'

    hf_constants.HF_HUB_CACHE = cache_dir
    hf_constants.HF_HUB_OFFLINE = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
    from lmdeploy.vl import load_image

    engine_config = TurbomindEngineConfig(
        async_=1,
        max_batch_size=4,
        session_len=8192,
        cache_max_entry_count=0.5,
        max_prefill_token_num=1024,
        tp=1,
        dp=1,
        enable_metrics=False,
        communicator='nccl',
    )
    gen_config = GenerationConfig(max_new_tokens=256, do_sample=False)

    image = load_image(image_path)
    prompt = 'Describe this image in detail. What do you see?'

    print('--- setup ---')
    print(f'model: {model_path}')
    print(f'image: {image_path}')
    print(f'gpus: {gpus}')
    print()

    t0 = time.perf_counter()
    with pipeline(model_path, backend_config=engine_config, log_level='WARNING') as pipe:
        load_s = time.perf_counter() - t0
        print('--- timing ---')
        print(f'pipeline load: {load_s:.2f} s')

        t1 = time.perf_counter()
        out = pipe([(prompt, image)], gen_config=gen_config, do_preprocess=True)
        infer_s = time.perf_counter() - t1
        print(f'inference: {infer_s:.2f} s')
        print()

        res = out[0]
        text = res.text if hasattr(res, 'text') else str(res)
        input_tokens = getattr(res, 'input_token_len', -1)
        gen_tokens = getattr(res, 'generate_token_len', -1)

        print('--- tokens ---')
        print(f'input: {input_tokens}')
        print(f'generated: {gen_tokens}')
        print()
        print('--- response begin ---')
        print(text)
        print('--- response end ---')


if __name__ == '__main__':
    main()
