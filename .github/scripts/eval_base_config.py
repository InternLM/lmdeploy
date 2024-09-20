from mmengine.config import read_base
from opencompass.models import TurboMindModel

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.summarizers.medium import \
        summarizer  # noqa: F401, E501

tb_internlm2_5_7b = dict(
    type=TurboMindModel,
    abbr='internlm2_5-7b-turbomind',
    path='internlm/internlm2_5-7b',
    engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
    gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
    max_seq_len=7168,
    max_out_len=1024,
    batch_size=16,
    run_cfg=dict(num_gpus=1),
)

models = [tb_internlm2_5_7b]
datasets = [*gsm8k_datasets]
