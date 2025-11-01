lmdeploy serve proxy --server-name 0.0.0.0 --server-port 8001 --routing-strategy "min_expected_latency" --serving-strategy Hybrid --log-level DEBUG

curl -X POST http://0.0.0.0:8001/distserve/connection_warmup

curl http://0.0.0.0:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/nvme3/interns1-mini-remote",
    "messages": [
      {
        "role": "user",
        "content": "Hello! How are you?"
      }
    ]
  }'


curl http://0.0.0.0:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/nvme3/interns1-mini-remote",
    "messages": [
      {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
                }
            }
        ]
      }
    ],
    "max_tokens": 200
  }'
