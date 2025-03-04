import json
import os

import fire
import yaml

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.utils import is_bf16_supported

gen_config = GenerationConfig(max_new_tokens=500)


def run_pipeline_chat_test(model_path, cases_path, tp, backend_type, is_pr_test, extra: object = None):

    if 'pytorch' in backend_type:
        backend_config = PytorchEngineConfig(tp=tp)
    else:
        backend_config = TurbomindEngineConfig(tp=tp)

    if 'lora' in backend_type:
        backend_config.adapters = extra.get('adapters')
    if 'kvint' in backend_type:
        backend_config.quant_policy = extra.get('quant_policy')

    if 'w4' in model_path or ('4bits' in model_path or 'awq' in model_path.lower()):
        backend_config.model_format = 'awq'
    if 'gptq' in model_path.lower():
        backend_config.model_format = 'gptq'
    if not is_bf16_supported():
        backend_config.dtype = 'float16'

    pipe = pipeline(model_path, backend_config=backend_config)

    cases_path = os.path.join(cases_path)
    with open(cases_path) as f:
        cases_info = yaml.load(f.read(), Loader=yaml.SafeLoader)

    for case in cases_info.keys():
        if ('coder' in model_path or 'CodeLlama' in model_path) and 'code' not in case:
            continue
        if is_pr_test and case != 'memory_test':
            continue
        case_info = cases_info.get(case)

        prompts = []
        response_list = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({'role': 'user', 'content': prompt})
            response = pipe([prompts], gen_config=gen_config, log_level='INFO', max_log_len=10)[0].text
            response_list.append({'prompt': prompt, 'response': response})
            prompts.append({'role': 'assistant', 'content': response})

        print(f'[caseresult {case} start]' + json.dumps(response_list, ensure_ascii=False) +
              f'[caseresult {case} end]\n')

    pipe.close()
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire()
