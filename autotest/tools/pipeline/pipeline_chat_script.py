import os

import fire
import yaml

from lmdeploy import pipeline
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)

cli_prompt_case_file = 'autotest/chat_prompt_case.yaml'
common_prompt_case_file = 'autotest/prompt_case.yaml'
config_file = 'autotest/config.yaml'


def main(type: str, model, tp: int = 1):
    config_path = os.path.join(config_file)
    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    case_path = os.path.join(common_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    run_pipeline_chat_test(env_config, case_config, model, tp, type)


def run_pipeline_chat_test(config, cases_info, model_case, tp, type):
    model_path = config.get('model_path')
    hf_path = model_path + '/' + model_case

    if 'pytorch' == type:
        backend_config = PytorchEngineConfig(tp=tp)
    else:
        if 'kvint8' in model_case and ('w4' in model_case
                                       or '4bits' in model_case
                                       or 'awq' in model_case.lower()):
            backend_config = TurbomindEngineConfig(tp=tp,
                                                   model_format='awq',
                                                   quant_policy=4)
        elif 'kvint8' in model_case:
            backend_config = TurbomindEngineConfig(tp=tp,
                                                   model_format='hf',
                                                   quant_policy=4)
        elif 'w4' in model_case or ('4bits' in model_case
                                    or 'awq' in model_case.lower()):
            backend_config = TurbomindEngineConfig(tp=tp, model_format='awq')
        else:
            backend_config = TurbomindEngineConfig(tp=tp)
    pipe = pipeline(hf_path, backend_config=backend_config)

    # run testcases
    gen_config = GenerationConfig(temperature=0.01)
    for case in cases_info.keys():
        case_info = cases_info.get(case)

        print('case:' + case)
        prompts = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({'role': 'user', 'content': prompt})
            print('prompt:' + prompt)

            response = pipe([prompts], gen_config=gen_config)[0].text

            prompts.append({'role': 'assistant', 'content': response})
            print('output:' + response)


if __name__ == '__main__':
    fire.Fire(main)
