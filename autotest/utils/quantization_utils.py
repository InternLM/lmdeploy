import os
import subprocess
from subprocess import PIPE


def quantization(config,
                 quantization_model_name,
                 origin_model_name,
                 quantization_type: str = 'w4a16',
                 cuda_prefix: str = 'CUDA_VISIBLE_DEVICES=0'):
    model_path = config.get('model_path')
    log_path = config.get('log_path')
    origin_model_path = config.get('model_path') + '/' + origin_model_name
    quantization_model_path = model_path + '/' + quantization_model_name
    quantization_log = os.path.join(
        log_path,
        '_'.join(['quantization', quantization_type, quantization_model_name
                  ]) + '.log')

    if quantization_type == 'w4a16':
        quantization_cmd = ' '.join([
            cuda_prefix, 'lmdeploy lite auto_awq', origin_model_path,
            '--work-dir', quantization_model_path
        ])
    elif quantization_type == 'w8a8':
        quantization_cmd = ' '.join([
            cuda_prefix, 'lmdeploy lite smooth_quant', origin_model_path,
            '--work-dir', quantization_model_path
        ])
    elif quantization_type == 'kvint8':
        quantization_cmd = ' '.join([
            cuda_prefix, 'lmdeploy lite calibrate', origin_model_path,
            '--work-dir', quantization_model_path
        ])
    else:
        return False, 'quantization type should in [w4a16, w8a8, kvint8], \
            now the type is ' + quantization_type

    with open(quantization_log, 'w') as f:
        # remove existing folder
        subprocess.run([' '.join(['rm -rf', quantization_model_path])],
                       stdout=f,
                       stderr=f,
                       shell=True,
                       text=True,
                       encoding='utf-8')

        if quantization_type == 'kvint8':
            cp_cmd = ' '.join(
                ['cp -r', origin_model_path, quantization_model_path])
            f.writelines('reproduce command quantization_cmd: ' + cp_cmd +
                         '\n')
            print('reproduce command quantization_cmd: ' + cp_cmd)
            subprocess.run([cp_cmd],
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
