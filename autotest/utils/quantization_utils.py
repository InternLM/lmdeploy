import os
import subprocess
from subprocess import PIPE

from lmdeploy.utils import is_bf16_supported
from utils.config_utils import _is_bf16_supported_by_device


def quantization(config,
                 quantization_model_name,
                 origin_model_name,
                 quantization_type: str = 'awq',
                 cuda_prefix: str = 'CUDA_VISIBLE_DEVICES=0'):
    model_path = config.get('model_path')
    log_path = config.get('log_path')
    origin_model_path = config.get('model_path') + '/' + origin_model_name
    quantization_model_path = model_path + '/' + quantization_model_name
    quantization_log = os.path.join(
        log_path, '_'.join(['quantization', quantization_type,
                            quantization_model_name.split('/')[1]]) + '.log')

    if quantization_type == 'awq':
        quantization_cmd = ' '.join(
            ['lmdeploy lite auto_awq', origin_model_path, '--work-dir', quantization_model_path])
    elif quantization_type == 'gptq':
        quantization_cmd = ' '.join(
            ['lmdeploy lite auto_gptq', origin_model_path, '--work-dir', quantization_model_path])
    elif quantization_type == 'w8a8':
        quantization_cmd = ' '.join(
            ['lmdeploy lite smooth_quant', origin_model_path, '--work-dir', quantization_model_path])
    else:
        return False, 'quantization type should in [awq, gptq, w8a8], \
            now the type is ' + quantization_type
    
    # Add device option if specified in environment
    device = os.environ.get('DEVICE', '')
    if device:
        quantization_cmd += f' --device npu'

    if cuda_prefix is not None:
        quantization_cmd = ' '.join([cuda_prefix, quantization_cmd])

    if 'llama-3' in origin_model_name.lower():
        quantization_cmd += ' --search-scale'

    if not _is_bf16_supported_by_device() or quantization_type == 'gptq':
        quantization_cmd += ' --batch-size 8'
    elif str(config.get('env_tag')) == '3090':
        quantization_cmd += ' --batch-size 8'
    else:
        quantization_cmd += ' --batch-size 32'

    with open(quantization_log, 'w') as f:
        # remove existing folder
        subprocess.run([' '.join(['rm -rf', quantization_model_path])],
                       stdout=f,
                       stderr=f,
                       shell=True,
                       text=True,
                       encoding='utf-8')

        f.writelines('reproduce command quantization_cmd: ' + quantization_cmd + '\n')
        print('reproduce command quantization_cmd: ' + quantization_cmd)
        # quantization
        quantizationRes = subprocess.run([quantization_cmd],
                                         stdout=f,
                                         stderr=PIPE,
                                         shell=True,
                                         text=True,
                                         encoding='utf-8')
        f.writelines(quantizationRes.stderr)
        result = quantizationRes.returncode == 0

    return result, quantizationRes.stderr
