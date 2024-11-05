## Support LLava-Interleave-Qwen-7B-hf

### generate gemm config (Optional)

`python3 lmdeploy/turbomind/generate_gemm_config.py --tensor-para-size 1 --max-batch-size 4 --model-path /models/llava-interleave-qwen-7b-hf`

### generate awq format model(Optional for awq format)

`lmdeploy lite auto_awq --work_dir models/llava-interleave-qwen-7b-hf/awq models/llava-interleave-qwen-7b-hf`

### start server

`python3 offline_vl.py models/llava-interleave-qwen-7b-hf`

`python3 offline_vl.py models/llava-interleave-qwen-7b-hf/awq --model-format awq`
