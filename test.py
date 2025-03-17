import os
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.api_server import handle_torchrun
import torch.distributed as dist

def main(rank: int):
    # model_path ='/nvme2/huggingface_hub_137_llm/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
    model_path ='/nvme1/zhaochaoxing/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d'
    # model_path = '/nvme2/huggingface_hub_137_llm/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0'
    log_level = 'WARNING'
    prompts = [
        'hello world.',
        'fast fox jump over the lazy dog.',
        ]
    prompts = prompts[rank:rank+1]

    backend_config = PytorchEngineConfig(
        tp=1,
        dp=2,
        ep=2,
        dp_rank=rank,
        eager_mode=True,
    )
    gen_config = GenerationConfig(
        temperature=1.0,
        top_k=1,
        do_sample=True,
        max_new_tokens=32,
    )

    os.environ['LMDEPLOY_DP_MASTER_ADDR'] = '127.0.0.1'
    os.environ['LMDEPLOY_DP_MASTER_PORT'] = str(29555)
    with pipeline(model_path, backend_config=backend_config, log_level=log_level) as pipe:
        outputs = pipe(prompts, gen_config=gen_config)
        print(outputs)

        dist.barrier()

if __name__ == '__main__':
    handle_torchrun()
    rank = int(os.environ['RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    dist.init_process_group()
    try:
        main(rank)
    finally:
        dist.destroy_process_group()