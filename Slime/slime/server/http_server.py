import json
import time
from contextlib import asynccontextmanager

import asyncio

import torch

import uvicorn
import uvloop

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from slime.config import RDMAInfo
from slime.transport.engine import TransferEngine

from .server_args import ServerArgs

import argparse


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


session_id = 0

async def long_running_task():
    now = time.time()
    while True:
        await asyncio.sleep(1)  
        print(f"Heart Beat {time.time() - now}")
        now = time.time()

# 使用 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 应用启动时执行
    print("Application is starting up...")
    task = asyncio.create_task(long_running_task())
    
    yield  # 这里会暂停，直到应用关闭
    
    # Shutdown: 应用关闭时执行
    print("Application is shutting down...")
    task.cancel()  # 取消后台任务
    try:
        await task
    except asyncio.CancelledError:
        print("Periodic task has been cancelled.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware)


transfer_engine: TransferEngine = None

@app.get("/health")
async def health() -> Response:
    return JSONResponse({"health": True})

@app.post("/exchange_info")
async def exchange_info(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    id = raw_request["id"]
    info = RDMAInfo(**json.loads(raw_request["info"]))
    transfer_engine.construct(id, info)
    return JSONResponse({"status": True})


@app.get("/init_link")
async def create_link() -> Response:
    global session_id
    id = session_id
    session_id += 1
    transfer_engine.init_link(id)

    return JSONResponse({"status": "Success", "id": id})

@app.get("/stop_link")
async def register_mr(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    id = raw_request["id"]
    transfer_engine.stop_link(id)
    return JSONResponse({"status": True})

@app.post("/register_mr")
async def register_mr(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    id = raw_request["id"]
    mr_key = raw_request["mr_key"]
    length = raw_request["length"]
    transfer_engine.register_mr(id, mr_key, length)
    return JSONResponse({"status": True})

@app.post("/rdma_read")
async def rdma_read(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    id = raw_request["id"]
    mr_key = raw_request["mr_key"]
    length = raw_request["length"]
    rkey = raw_request["remote_rkey"]
    target_addr = raw_request["remote_addr"]
    offset = raw_request["offset"]

    begin = time.time()
    await transfer_engine.r_rdma_async(id, mr_key, target_addr, offset, length, rkey)
    end = time.time()
    print(f"latency: {end - begin}, bw: {(length) / (end - begin) / (1e9)} GBps")
    return JSONResponse({"psum": int(torch.sum(transfer_engine.links[id].memory_pool[mr_key]))})


@app.get("/get_local_info")
async def get_local_info(raw_request: Request) -> Response:
    raw_request = await raw_request.json()
    id = raw_request["id"]
    info = transfer_engine.get_local_info(id)
    return info.model_dump_json()


def launch_server(server_args, dev_name, ib_port, link_type):
    global transfer_engine
    transfer_engine = TransferEngine(dev_name, ib_port, link_type)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str,
                        help="--server-host", default="0.0.0.0")
    parser.add_argument("--server-port", type=int,
                        help="--server-port", default=4469)

    parser.add_argument("--dev-name", type=str, default="mlx5_bond_0")
    parser.add_argument("--ib-port", type=int, help="--ib-port", default=1)
    parser.add_argument("--link-type", type=str, help="--link-type", choices=["Ethernet", "Infiniband"], default="Ethernet")

    args = parser.parse_args()

    server_args = ServerArgs(host=args.server_host, port=args.server_port)

    launch_server(server_args=server_args, dev_name=args.dev_name, ib_port=args.ib_port, link_type=args.link_type)
