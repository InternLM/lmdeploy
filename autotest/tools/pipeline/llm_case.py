import json
import os

import fire
import yaml

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline

gen_config = GenerationConfig(max_new_tokens=500, min_new_tokens=10)


def run_pipeline_chat_test(model_path, run_config, cases_path, is_pr_test: bool = False):
    backend = run_config.get('backend')
    device = run_config.get('device', None)
    dtype = run_config.get('dtype', None)
    communicator = run_config.get('communicator')
    quant_policy = run_config.get('quant_policy')
    extra_params = run_config.get('extra_params', {})
    parallel_config = run_config.get('parallel_config', {})

    if backend == 'pytorch':
        backend_config = PytorchEngineConfig(quant_policy=quant_policy)
    else:
        backend_config = TurbomindEngineConfig(communicator=communicator, quant_policy=quant_policy)

    if device:
        backend_config.device_type = device
    if dtype:
        backend_config.dtype = dtype

    # quant format
    model_lower = model_path.lower()
    if 'w4' in model_lower or '4bits' in model_lower or 'awq' in model_lower:
        backend_config.model_format = 'awq'
    elif 'gptq' in model_lower:
        backend_config.model_format = 'gptq'

    # Parallel config
    for para_key in ('dp', 'ep', 'cp'):
        if para_key in parallel_config:
            backend_config[para_key] = parallel_config[para_key]
    if 'tp' in parallel_config and parallel_config['tp'] > 1:
        backend_config['tp'] = parallel_config['tp']

    # Extra params
    for key, value in extra_params.items():
        backend_config[key] = value

    print('backend_config config: ' + str(backend_config))
    pipe = pipeline(model_path, backend_config=backend_config)

    cases_path = os.path.join(cases_path)
    with open(cases_path) as f:
        cases_info = yaml.load(f.read(), Loader=yaml.SafeLoader)

    for case in cases_info.keys():
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


if __name__ == '__main__':
    fire.Fire()
