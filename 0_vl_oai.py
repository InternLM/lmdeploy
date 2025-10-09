import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from lmdeploy import Tokenizer
from lmdeploy.archs import get_model_arch
from lmdeploy.model import ChatTemplateConfig, best_match_model
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig, EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeCacheFreeRequest, DistServeConnectionRequest,
                                                   DistServeConnectionResponse, DistServeConnectionStatus,
                                                   DistServeEngineEndpointInfo, DistServeInitRequest,
                                                   DistServeInitResponse, MigrationProtocol)
from lmdeploy.pytorch.engine.encoder_cache_engine import EncoderCacheEngine
from lmdeploy.serve.openai.launch_server import get_host_ip
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.vl.model.internvl3_hf import InternVL3VisionModel
from lmdeploy.vl.utils import load_image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# --- 1. 全局模型变量 ---
model_instance: InternVL3VisionModel = None  # type: ignore
migration_backend_impl: Optional[MigrationBackendImpl] = None
model_path = '/mnt/137_nvme3/interns1-mini-remote'
SERVER_PORT = 8086
chat_template_name = best_match_model(model_path.lower())
# chat_template_config = ChatTemplateConfig(chat_template_name)
chat_template_config = ChatTemplateConfig(model_name=chat_template_name, model_path=model_path)
chat_template = chat_template_config.chat_template
tokenizer = Tokenizer(model_path)
# encoder_url = f"http://{get_host_ip()}:{SERVER_PORT}"
encoder_url = f'http://0.0.0.0:{SERVER_PORT}'
# 初始化 Cache Engine 相关变量
cache_engine_instance: EncoderCacheEngine = None  # type: ignore
NUM_GPU_BLOCKS = 128
free_blocks: List[int] = []
session_blocks: Dict[int, List[int]] = {}
session_counter = 0
block_manager_lock = Lock()  # 线程锁，用于安全地分配和释放块


def get_model_list():
    return [model_path]


# --- 2. 生命周期事件处理器 ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的事件
    global model_instance, cache_engine_instance, free_blocks
    logger = logging.getLogger('uvicorn.error')
    logger.setLevel(logging.INFO)
    logger.info('模型加载中，请稍候...')
    try:
        cfg = get_model_arch(model_path)[1]
        kwargs = dict(model_path=model_path, with_llm=False, max_memory=None, hf_config=cfg, backend='pytorch')
        model_instance = InternVL3VisionModel(**kwargs)
        model_instance.build_model()
        model_instance.build_preprocessor()
        logger.info('✅ 模型加载成功！服务器已准备就绪。')
    except Exception as e:
        logger.error(f'❌ 模型加载失败: {e}', exc_info=True)
        raise RuntimeError(f'模型初始化失败: {e}') from e

    # TODO MigrationBackendImpl ()

    # TODO 增加 memory 页表注册
    logger.info('正在初始化 Cache Engine...')
    try:
        # 实例化 CacheEngine
        cache_engine_instance = EncoderCacheEngine(NUM_GPU_BLOCKS)

        # 初始化空闲块列表
        free_blocks = list(range(NUM_GPU_BLOCKS))
        logger.info(f'✅ Cache Engine 初始化成功，总共 {NUM_GPU_BLOCKS} 个缓存块。')

    except Exception as e:
        logger.error(f'❌ Cache Engine 初始化失败: {e}', exc_info=True)
        raise RuntimeError(f'Cache Engine 初始化失败: {e}') from e

    # TODO 向 proxy 发送node add
    try:
        import requests
        engine_role = EngineRole.Encoder.value
        url = 'http://0.0.0.0:8001/nodes/add'
        data = {'url': f'http://0.0.0.0:{SERVER_PORT}', 'status': {'models': get_model_list(), 'role': engine_role}}
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        else:
            logger.info('✅ 服务注册成功！')
    except Exception as e:
        logger.error(f'Service registration failed: {e}')
    # TODO p2p initialize(warm up)
    # /nvme2/share/linbinbin1/src/lmdeploy-encoder/lmdeploy/serve/openai/api_server.py PD DIs

    # TODO p2p conn

    yield  # 应用运行期间

    # 关闭时的事件（如果需要清理资源）
    logger.info('🔄 正在关闭服务器...')
    del model_instance
    torch.cuda.empty_cache()
    logger.info('模型资源已释放。')


# --- 3. 初始化 FastAPI 应用 ---
app = FastAPI(title='InternVL Vision Model Server (Arrow Edition)',
              description='一个用于通过 InternVL3 模型为图片数组提取特征张量，并使用 Apache Arrow 高效返回结果的 API',
              version='1.2.0',
              lifespan=lifespan)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)


# --- 4. 辅助函数 ---
def find_forward_content(output: list) -> list:
    for item in output:
        if isinstance(item, dict) and item.get('role') == 'forward':
            return item.get('content', [])
    return []


async def async_convert_to_pil_images(messages: List[Dict]) -> List[Dict]:
    """Scan the provided messages to find image URLs or base64-encoded image
    data. Loads the images into Pillow image objects.

    Args:
        messages (List[Dict]): a user request of GPT4V message format
    """
    if isinstance(messages, Dict):
        messages = [messages]
    assert isinstance(messages, List)

    out_messages = [None] * len(messages)

    def _inner_call(i, in_messages, out_messages):
        role = in_messages[i]['role']
        content = in_messages[i]['content']
        assert role in ['system', 'user', 'assistant'], \
            f'unsupported role "{role}"'
        if role != 'user' or isinstance(content, str):
            # the content is a user's prompt or an assistant's prompt,
            # returning it directly
            out_messages[i] = in_messages[i]
            return
        # the role is a user and the content is a list, in which there
        # might be image_url or image_data
        assert isinstance(content, List)
        message = dict(role=role, content=[])
        for item in content:
            # image url or base64-encoded image data
            if item['type'] == 'image_url':
                """
                convert the following item:
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': 'image url or base64-encoded image data',
                        'key': 'value'  # parameters used in image processing
                        ...
                    }
                }
                to:
                {
                    'type': 'image',
                    'image': Pillow.Image,
                    'key': 'value'   # parameters used in image processing
                    ...
                }
                """  # noqa
                data = item['image_url'].copy()
                try:
                    url = data.pop('url')
                    image = load_image(url)
                    data.update(type='image', image=image)
                    message['content'].append(data)
                except KeyError:
                    logger.error(f'invalid format {message}')
            elif item['type'] == 'image_data':
                """
                convert the following item:
                {
                    'type': 'image_data',
                    'image_data': {
                        'data': Pillow.Image,
                        'key': 'value'  # parameters used in image processing
                        ...
                    }
                }
                to:
                {
                    'type': 'image',
                    'image': Pillow.Image,
                    'key': 'value'   # parameters used in image processing
                    ...
                }
                """  # noqa
                data = item['image_data'].copy()
                try:
                    image = data.pop('data')
                    data.update(type='image', image=image)
                    message['content'].append(data)
                except KeyError:
                    logger.error(f'invalid format {message}')
            elif item['type'] == 'text':
                message['content'].append(item)
            else:
                logger.error(f'unexpected content type {message}')
        out_messages[i] = message

    await asyncio.gather(*[
        asyncio.get_event_loop().run_in_executor(None, _inner_call, i, messages, out_messages)
        for i in range(len(messages))
    ])
    return out_messages


@app.get('/health')
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@dataclass
class EncoderResult:
    token_ids: List[int]
    image_mask: List[int]
    # MigrationRequest 中相似的字段
    protocol: MigrationProtocol  # RDMA
    remote_engine_id: str  # 标识 encode 引擎编号
    remote_session_id: int  # 用于 encode 引擎释放指定区域
    remote_block_ids: List[int]  # 从 encode 引擎读取指定区域内容


# --- 5. API 端点：处理图片并返回特征 ---


@app.post('/v1/chat/completion', summary='接收 open ai 格式的请求，并且返回给 proxy')
async def process_images(request_raw: ChatCompletionRequest = None):
    if model_instance is None:
        raise HTTPException(status_code=503, detail='模型正在加载或加载失败，请稍后再试。')

    request = request_raw.model_dump()
    messages = await async_convert_to_pil_images(request['messages'])
    results = model_instance.preprocess(messages)
    # print(results)
    # import pdb; pdb.set_trace()

    # prompt = chat_template.messages2prompt(messages)
    # input_ids = tokenizer.encode(prompt, add_bos=True) # 只包含了文本部分
    # prompt, input_ids（包含了图片 token 序列）, multi_modal
    # 这个是将要返回的内容
    to_pt = model_instance.to_pytorch(results, chat_template, tokenizer, True, None, None)
    image_mask = [1 if x == to_pt['multimodal'][0]['image_token_id'] else 0 for x in to_pt['input_ids']]
    # 这里用来获得 image embedding
    output = model_instance.forward(results)
    forward_content = find_forward_content(output)
    # tensor_shape = forward_content[0].shape
    if not forward_content:
        raise HTTPException(status_code=500, detail="无法在模型输出中找到 'forward' 内容。")
    # store the image embedding to gpu cache
    image_embedding = forward_content[0]
    image_embedding = image_embedding.to(
        torch.bfloat16
    )  # FIXME: forward() is used by turbomind, which returns float16 feature, but pytorch will return bfloat16
    print(f'image_embedding shape: {image_embedding.shape}')
    print(f'image_embedding: {image_embedding}')
    num_required_blocks = image_embedding.shape[0] // 256
    global session_counter
    allocated_block_ids = []
    session_id = -1
    with block_manager_lock:
        if len(free_blocks) < num_required_blocks:
            raise HTTPException(status_code=503, detail='GPU 缓存已满，请稍后再试。')

        allocated_block_ids = [free_blocks.pop() for _ in range(num_required_blocks)]
        session_counter += 1
        session_id = session_counter
        session_blocks[session_id] = allocated_block_ids
    print('in blocks')
    print(allocated_block_ids)
    print(cache_engine_instance.gpu_cache[allocated_block_ids].shape)
    print(cache_engine_instance.gpu_cache[allocated_block_ids])
    try:
        with torch.cuda.stream(cache_engine_instance.cache_stream):
            for i in range(num_required_blocks):
                src_chunk = image_embedding[i * 256:(i + 1) * 256, :]
                dst_block_id = allocated_block_ids[i]
                cache_engine_instance.gpu_cache[dst_block_id].copy_(src_chunk)
        cache_engine_instance.cache_stream.synchronize()
    except Exception as e:
        # 如果拷贝失败，必须归还申请的块，防止内存泄漏
        with block_manager_lock:
            free_blocks.extend(allocated_block_ids)
            del session_blocks[session_id]
        logger.error(f'拷贝 embedding 到缓存失败: {e}')
        raise HTTPException(status_code=500, detail='缓存图像 embedding 失败。')

    # 返回内容

    # FIXME, zhouxinyu this should not be empty
    # otherwise gen config related information are lost, for instance top_p, top_k, max_new_tokens
    # request['messages'] = []
    encoder_result_obj = EncoderResult(
        token_ids=to_pt['input_ids'],
        image_mask=image_mask,
        protocol=MigrationProtocol.RDMA,
        remote_engine_id=encoder_url,  # encode 引擎的 url
        remote_session_id=session_id,  # encode 阶段的 session id
        remote_block_ids=allocated_block_ids  # image embedding 的 memory block id
    )
    request['encoder_result'] = asdict(encoder_result_obj)

    return JSONResponse(jsonable_encoder(request))


@app.post('/distserve/p2p_initialize')
async def p2p_initialize(init_request: DistServeInitRequest):
    kv_eps = cache_engine_instance.p2p_initialize(init_request)
    # 目前 encoder 没有 zmq 通信；返回一个假地址
    zmq_addr = f'tcp://{get_host_ip()}:65001'
    resp = DistServeInitResponse(
        status=DistServeConnectionStatus.SUCCESS,
        engine_endpoint_info=DistServeEngineEndpointInfo(zmq_address=zmq_addr),
        kvtransfer_endpoint_info=kv_eps,
    )
    return JSONResponse(jsonable_encoder(resp.model_dump()))


@app.post('/distserve/p2p_connect')
async def p2p_connect(conn_request: DistServeConnectionRequest):
    cache_engine_instance.p2p_connect(
        conn_request.remote_engine_id,
        conn_request.remote_kvtransfer_endpoint_info,
    )
    resp = DistServeConnectionResponse(status=DistServeConnectionStatus.SUCCESS)
    return JSONResponse(jsonable_encoder(resp.model_dump()))


@app.post('/distserve/free_cache')
async def free_cache(free_req: DistServeCacheFreeRequest):
    # Free allocated GPU blocks for a given session id
    global free_blocks, session_blocks
    sid = free_req.remote_session_id
    with block_manager_lock:
        blocks = session_blocks.pop(sid, [])
        if blocks:
            free_blocks.extend(blocks)
    return JSONResponse({'success': True, 'freed_blocks': blocks if 'blocks' in locals() else []})


@app.get('/distserve/engine_info')
async def engine_info():

    response = DistServeEngineConfig(tp_size=1,
                                     dp_size=1,
                                     pp_size=1,
                                     ep_size=1,
                                     dp_rank=1,
                                     block_size=256 * 4096,
                                     num_cpu_blocks=0,
                                     num_gpu_blocks=NUM_GPU_BLOCKS)

    return response.model_dump_json()


# --- 6. 运行服务器 ---
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=SERVER_PORT)
