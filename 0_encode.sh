model_path='/nvme3/interns1-mini-remote'


CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server \
    ${model_path} \
    --tp 1 \
    --role Encoder \
    --backend pytorch \
    --server-port 23334 \
    --proxy-url http://0.0.0.0:8001 \
    --log-level INFO
