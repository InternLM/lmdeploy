import time

import huggingface_hub.constants as hf_constants

hf_constants.HF_HUB_OFFLINE = 1
hf_constants.HF_HUB_CACHE = '/mnt_cfs/huggingface_hub/hub'

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image

model_id = 'Qwen/Qwen3.5-35B-A3B-FP8'
image_url = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
prompt = (
    'Describe this image accurately in at least 150 words. Identify the main '
    'subject, its appearance and pose, and the surrounding environment.'
)

image = load_image(image_url)
engine_config = TurbomindEngineConfig(
    tp=1,
    session_len=16384,
    max_batch_size=1,
    max_prefill_token_num=4096,
    cache_max_entry_count=0.5,
    enable_metrics=False,
)
gen_config = GenerationConfig(max_new_tokens=256, do_sample=False)

start = time.perf_counter()
with pipeline(
        model_id,
        backend_config=engine_config,
        log_level='WARNING',
        trust_remote_code=True) as pipe:
    loaded = time.perf_counter()
    response = pipe((prompt, image), gen_config=gen_config)
    finished = time.perf_counter()

print(f'pipeline load: {loaded - start:.2f} s')
print(f'inference: {finished - loaded:.2f} s')
print(f'input tokens: {response.input_token_len}')
print(f'generated tokens: {response.generate_token_len}')
print('--- response begin ---')
print(response.text)
print('--- response end ---')
