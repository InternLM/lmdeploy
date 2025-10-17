model_path='/nvme3/interns1-mini-remote'


CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server \
    ${model_path} \
    --tp 1 \
    --role Hybrid \
    --backend pytorch \
    --server-port 23335 \
    --proxy-url http://0.0.0.0:8001 \
    --disable-vision-encoder \
    --log-level INFO
