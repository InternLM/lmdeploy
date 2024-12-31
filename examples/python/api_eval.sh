
#!/bin/bash

export no_proxy=localhost,0.0.0.0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}

export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=${MASTER_PORT} \
    examples/python/evaluate_mmmu_prm.py --out-dir ./output \
      	--datasets MMMU_validation --infer-times 64 --cot \
	--url http://0.0.0.0:23333/v1 \
	--model-name InternVL2_5-78B
