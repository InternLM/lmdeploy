from lmdeploy.model import MODELS


# Deprecated function
def get_model_name(model):
    model_names = ['llama', 'llama2', 'llama3', 'internlm', 'internlm2', 'baichuan2', 'chatglm2', 'yi', 'qwen']
    model_names += list(MODELS.module_dict.keys())
    model_names.sort()
    model_name = _simple_model_name(model)
    model_name = model_name.lower()

    if model_name in model_names:
        return model_name
    if model_name in model_names:
        return model_name
    if ('llama-2' in model_name):
        return 'llama2'
    if ('llama-3-1' in model_name):
        return 'llama3_1'
    if ('llama-3' in model_name):
        return 'llama3'
    if 'vicuna' in model_name and 'llava' not in model_name:
        return 'vicuna'
    if 'llava' in model_name and 'v1' in model_name and 'v1.6-34b' not in model_name and 'mistral' not in model_name:
        return 'llava-v1'
    if 'llava' in model_name and 'v1.6-34b' in model_name:
        return 'llava-chatml'
    if 'internvl-chat' in model_name and 'v1-2' in model_name:
        return 'internvl-zh-hermes2'
    elif 'llava-1.5' in model_name:
        return 'llava-v1'
    if ('yi-vl' in model_name):
        return 'yi-vl'
    if ('qwen' in model_name):
        return 'qwen'
    if ('internvl') in model_name:
        return 'internvl-internlm2'
    if ('internlm2') in model_name:
        return 'internlm2'
    if ('internlm-xcomposer2d5') in model_name:
        return 'internlm-xcomposer2d5'
    if ('internlm-xcomposer2') in model_name:
        return 'internlm-xcomposer2'
    if ('glm-4') in model_name:
        return 'glm4'
    if len(model_name.split('-')) > 2 and '-'.join(model_name.split('-')[0:2]) in model_names:
        return '-'.join(model_name.split('-')[0:2])
    return model_name.split('-')[0]


def _simple_model_name(model):
    if '/' in model:
        model_name = model.split('/')[1]
    else:
        model_name = model
    model_name = model_name.replace('-inner-4bits', '')
    model_name = model_name.replace('-inner-w8a8', '')
    model_name = model_name.replace('-4bits', '')
    return model_name
