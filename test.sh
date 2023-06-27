# python -m llmdeploy.serve.hf.client "/nvme/wangruohui/llama-7b-hf/"


deepspeed --module --num_gpus 2 llmdeploy.serve.hf.client \
    "/nvme/wangruohui/llama-13b-hf/" \
    --max_new_tokens 64 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 6
