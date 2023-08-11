# Restful API

## Launch Service

```shell
python lmdeploy/serve/openai/api_server.py ./workspace server_name server_port
```

## Test
```shell
curl -X 'POST'   'http://{server_name}:{server_port}/v1/chat/completions'   -H 'accept: application/json' -d '{"model": "internlm-chat-7b","messages":[{"role": "user", "content": "Write a poem."}],"stream": false}' --output -
```
