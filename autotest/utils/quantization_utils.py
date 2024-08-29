import os
import subprocess
from subprocess import PIPE


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
        log_path, '_'.join([
            'quantization', quantization_type,
            quantization_model_name.split('/')[1]
        ]) + '.log')

    if quantization_type == 'awq':
        quantization_cmd = ' '.join([
            cuda_prefix, 'lmdeploy lite auto_awq', origin_model_path,
            '--work-dir', quantization_model_path, '--batch-size 32'
        ])
    elif quantization_type == 'gptq':
        quantization_cmd = ' '.join([
            cuda_prefix, 'lmdeploy lite auto_gptq', origin_model_path,
            '--work-dir', quantization_model_path, '--batch-size 32'
        ])
    elif quantization_type == 'w8a8':
        quantization_cmd = ' '.join([
            cuda_prefix, 'lmdeploy lite smooth_quant', origin_model_path,
            '--work-dir', quantization_model_path, '--batch-size 32'
        ])
    else:
        return False, 'quantization type should in [awq, gptq, w8a8], \
            now the type is ' + quantization_type

    if 'llama-3' in origin_model_name.lower():
        quantization_cmd += ' --search-scale True'

    with open(quantization_log, 'w') as f:
        # remove existing folder
        subprocess.run([' '.join(['rm -rf', quantization_model_path])],
                       stdout=f,
                       stderr=f,
                       shell=True,
                       text=True,
                       encoding='utf-8')

        f.writelines('reproduce command quantization_cmd: ' +
                     quantization_cmd + '\n')
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
