# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import httpx
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from lmdeploy.serve.openai.api_server import VariableInterface
from lmdeploy.disagg.conn_manager import pd_consolidation
from lmdeploy.disagg.messages import MigrationRequest

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


@app.post("/v1/completions")
async def generate(request: Request):
    client_data = await request.json()
    session_id = None

    async with httpx.AsyncClient() as client:
        try:
            # Prefill阶段
            prefill_data = deepcopy(client_data)
            if not prefill_data.get("id"):
                VariableInterface.session_id += 1
                prefill_data["session_id"] = VariableInterface.session_id
            prefill_data["max_tokens"] = 1
            prefill_data["stream"] = True
            prefill_data["with_cache"] = True
            prefill_url = get_url(
                engine_snapshot.prefill_endpoints[0], "v1/completions"
            )

            prefill_resp = await client.post(
                prefill_url, json=prefill_data, timeout=30.0
            )
            prefill_resp.raise_for_status()

            x = prefill_resp.aiter_lines()
            first_line = await x.__anext__()
            prefill_info = json.loads(first_line[5:])

            decode_data = client_data.copy()
            migration_request = MigrationRequest(
                remote_engine_id=0,
                remote_session_id=0,
                remote_block_ids=prefill_info["cache_block_ids"],
                remote_token_id=prefill_info["remote_token_ids"][-1],
            )
            decode_data["migration_request"] = migration_request.model_dump()

            # Decode阶段
            decode_url = get_url(engine_snapshot.decode_endpoints[0], "v1/completions")
            decode_resp = await client.post(decode_url, json=decode_data, timeout=30.0)
            decode_resp.raise_for_status()

            async def stream_generator():
                free_flags = False
                async for line in decode_resp.aiter_lines():
                    if not free_flags:
                        free_url = get_url(
                            engine_snapshot.prefill_endpoints[0], "distserve/free_cache"
                        )
                        requests.post(free_url, json={"session_id": prefill_data["session_id"]}, timeout=5)
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

    for p_idx, prefill_endpoint in enumerate(args.prefill_endpoint):
        for d_idx, decode_endpoint in enumerate(args.decode_endpoint):
            pd_consolidation(p_idx, prefill_endpoint, d_idx, decode_endpoint)


if __name__ == "__main__":
    args = parse_args()
    init_migration(args)
    uvicorn.run(app, host=args.host, port=args.port)
