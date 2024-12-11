# Copyright (c) OpenMMLab. All rights reserved.
import logging

import torch
from transformers import AutoTokenizer

from lmdeploy.lite.utils.calib_dataloader import get_calib_loaders


def auto_gptq(model: str,
              work_dir: str = './work_dir',
              w_bits: int = 4,
              w_group_size: int = 128,
              calib_dataset: str = 'ptb',
              calib_samples: int = 128,
              calib_seqlen: int = 2048,
              batch_size: int = 1,
              revision: str = None):
    """Perform weight quantization using AWQ algorithm.

    Args:
        model (str): The path of model in hf format.
        work_dir (str): The working directory to save results.
        calib_dataset (str): The calibration dataset name.
        calib_samples (int): The number of samples for calibration.
        batch_size (int): The batch size for running the calib samples.
            Low GPU mem requires small batch_size. Large batch_size
            reduces the calibration time while costs more VRAM.
        calib_seqlen (int): The sequence length for calibration.
        w_bits (int): Bit number for weight quantization.
        w_group_size (int): Group size for weight quantization statistics.
        search_scale (bool): Whether search scale ratio. Default to False,
            which means only smooth quant with 0.5 ratio will be applied.
        device (str): Device type of running.
        revision (str): The specific model version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified,
            will use the default version.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except Exception:
        raise ImportError('To use auto_gptq, please install auto-gptq by '
                          'pip install auto-gptq')
    logging.basicConfig(
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # support internlm2
    from auto_gptq.modeling import GPTQ_CAUSAL_LM_MODEL_MAP
    from auto_gptq.modeling._const import SUPPORTED_MODELS

    from ..modeling.internlm2_gptq import InternLM2GPTQForCausalLM
    SUPPORTED_MODELS.append('internlm2')
    GPTQ_CAUSAL_LM_MODEL_MAP.update(dict(internlm2=InternLM2GPTQForCausalLM))

    pretrained_model_dir = model
    quantized_model_dir = work_dir

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                              trust_remote_code=True)
    print('Loading calibrate dataset ...')
    calib_loader, _ = get_calib_loaders(calib_dataset,
                                        tokenizer,
                                        nsamples=calib_samples,
                                        seqlen=calib_seqlen)
    all_data = [
        data if isinstance(data, torch.Tensor) else data[0]
        for data in calib_loader
    ]
    attention_mask = [1] * calib_seqlen
    examples = [
        dict(input_ids=data.flatten().tolist(), attention_mask=attention_mask)
        for data in all_data
    ]

    quantize_config = BaseQuantizeConfig(
        bits=w_bits,  # quantize model to 4-bit
        group_size=w_group_size,  # it is recommended to set the value to 128
        desc_act=False,  # lmdeploy only supports False
        sym=True,  # lmdeploy only supports True
    )

    # load un-quantized model, by default,
    # the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir,
                                                quantize_config,
                                                revision=revision,
                                                trust_remote_code=True)

    # quantize model, the examples should be list of dict whose keys
    # can only be "input_ids" and "attention_mask"
    model.quantize(examples, batch_size=batch_size)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    tokenizer.save_pretrained(quantized_model_dir)


if __name__ == '__main__':
    import fire

    fire.Fire(auto_gptq)
