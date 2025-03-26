import argparse
import asyncio
import json

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass

from typing import List

import httpx

import requests
import uvicorn

import uvloop
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

app = FastAPI()


def get_url(endpoint, name):
    return f"{endpoint}/{name}"


@dataclass
class EngineSnapshot:
    prefill_endpoints: List[str]
    decode_endpoints: List[str]

    @property
    def endpoints(self):
        return self.prefill_endpoints + self.decode_endpoints


engine_snapshot: EngineSnapshot = None


"""
TODO: router relay v1/completions, v1/chat/completions, v1/encode, v1/embedding ...
"""


async def relay(endpoint: str, service_name: str, raw_requests: Request):
    raise NotImplementedError


"""
TODO: worker management
"""


@app.post("/router/add_worker")
async def add_worker(request: Request):
    raise NotImplementedError


@app.post("/router/remove_worker")
async def add_worker(request: Request):
    raise NotImplementedError


@app.post("/v1/completions")
async def generate(request: Request):
    client_data = await request.json()
    session_id = None

    async with httpx.AsyncClient() as client:
        try:
            # Prefill阶段
            prefill_client_data = deepcopy(client_data)
            prefill_client_data["max_tokens"] = 1
            prefill_url = get_url(
                engine_snapshot.prefill_endpoints[0], "v1/completions"
            )

            prefill_resp = await client.post(
                prefill_url, json=prefill_client_data, timeout=30.0
            )
            prefill_resp.raise_for_status()

            # 解析首行响应
            first_line = await prefill_resp.aiter_lines().__anext__()
            prefill_info = json.loads(first_line[5:])
            session_id = prefill_info["id"]
            block_ids = prefill_info["cache_block_ids"]
            remote_token_ids = prefill_info["choices"][0]["remote_token_ids"]

            decode_data = client_data.copy()
            decode_data.update(
                {
                    "block_ids": block_ids,
                    "remote_token_ids": remote_token_ids,
                    "session_id": session_id,
                    "max_tokens": decode_data.get(
                        "max_tokens", 16
                    ),  # 确保max_tokens存在
                }
            )

            # Decode阶段
            # TODO (CJF): directly relay
            decode_url = get_url(engine_snapshot.decode_endpoints[0], "v1/completions")
            decode_resp = await client.post(decode_url, json=decode_data, timeout=30.0)
            decode_resp.raise_for_status()

            # 流式响应生成器
            async def stream_generator():
                free_flags = False
                async for line in decode_resp.aiter_lines():
                    if not free_flags:
                        free_url = get_url(
                            engine_snapshot.prefill_endpoints[0], "distserve/free_cache"
                        )
                        requests.post(free_url, json={"session_id": session_id})
                        free_flags = True
                    if line:
                        yield line + "\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},  # 禁用代理缓冲
            )

        except Exception as e:
            # 异常时尝试释放资源
            if session_id:
                free_url = get_url(
                    engine_snapshot.prefill_endpoints[0], "distserve/free_cache"
                )
                await client.post(free_url, json={"session_id": session_id})
            raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="处理 prefill 和 decode 的 endpoint 参数"
    )

    parser.add_argument(
        "--host",
        type=str,
        required=True,  # 必须提供该参数
    )

    parser.add_argument(
        "--port",
        type=int,
        required=True,  # 必须提供该参数
    )

    # 添加 prefill-endpoint 参数，类型为字符串列表
    parser.add_argument(
        "--prefill-endpoint",
        nargs="+",  # 表示接受一个或多个值，形成列表
        type=str,
        required=True,  # 必须提供该参数
        help="指定 prefill 的 endpoint 列表，例如 --prefill-endpoint http://example1.com http://example2.com",
    )

    # 添加 decode-endpoint 参数，类型为字符串列表
    parser.add_argument(
        "--decode-endpoint",
        nargs="+",  # 同样表示接受一个或多个值，形成列表
        type=str,
        required=True,  # 必须提供该参数
        help="指定 decode 的 endpoint 列表，例如 --decode-endpoint http://example3.com http://example4.com",
    )

    args = parser.parse_args()
    return args


def init_migration(args):
    global engine_snapshot
    engine_snapshot = EngineSnapshot(
        prefill_endpoints=args.prefill_endpoint, decode_endpoints=args.decode_endpoint
    )

    # Step 1. get cache information
    total_blocks = []
    for endpoint in engine_snapshot.endpoints:
        cache_info = requests.get(get_url(endpoint, "distserve/get_engine_info")).json()
        total_blocks.append(cache_info["total"])

    with ThreadPoolExecutor(2) as executor:
        prefill_future = executor.submit(
            requests.post,
            get_url(engine_snapshot.prefill_endpoints[0], "distserve/init_rdma_link"),
            json={"remote_engine_id": 1},
        )
        decode_future = executor.submit(
            requests.post,
            get_url(engine_snapshot.decode_endpoints[0], "distserve/init_rdma_link"),
            json={"remote_engine_id": 0},
        )

    results = [prefill_future.result, decode_future.result]

    handler_config_prefill = {
        "total": total_blocks[1],
        "remote_engine_id": 1,
        "rdma_exchange_info": results[1],
    }

    handler_config_decode = {
        "total": total_blocks[0],
        "remote_engine_id": 0,
        "rdma_exchange_info": results[0],
    }

    with ThreadPoolExecutor(2) as executor:
        executor.submit(
            requests.post,
            get_url(engine_snapshot.prefill_endpoints[0], "distserve/rdma_connect"),
            json={"config": handler_config_prefill},
        )
        executor.submit(
            requests.post,
            get_url(engine_snapshot.decode_endpoints[0], "distserve/rdma_connect"),
            json={"config": handler_config_decode},
        )


if __name__ == "__main__":
    args = parse_args()
    init_migration(args)
    uvicorn.run(app, host=args.host, port=args.port)
