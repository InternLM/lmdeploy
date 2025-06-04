# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass, field
from itertools import count
from typing import List, Literal, Optional, Tuple, Union

import gradio as gr
from packaging.version import Version, parse
from PIL import Image

from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn
from lmdeploy.utils import get_logger

BATCH_SIZE = 32
logger = get_logger('lmdeploy')

if parse(gr.__version__) >= Version('4.0.0'):
    que_kwargs = {'default_concurrency_limit': BATCH_SIZE}
else:
    que_kwargs = {'concurrency_count': BATCH_SIZE}


@dataclass
class Session:
    """Chat session.

    Args:
        _session_id (int): session_id for internal use.
        _message (List[Tuple[Any, str]]): chat history for internal use.
        _step (int): the offset of the k/v cache for internal use.
    """

    _count = count()
    _session_id: int = None
    _message: List[Tuple[str, str]] = field(default_factory=list)

    def __init__(self):
        self._session_id = next(self._count)
        self._message = []

    @property
    def session_id(self):
        return self._session_id

    @property
    def message(self):
        return self._message

    def to_gpt4v(self):
        output = []
        for user, assistant in self._message:
            if isinstance(user, str):
                text, images = user, []
            else:
                text, images = user[0], user[1:]
            content = [dict(type='text', text=text)]
            for img in images:
                content.append(dict(type='image_url', image_url=dict(url=img)))
            user_input = dict(role='user', content=content)
            output.append(user_input)
            if assistant:
                output.append(dict(role='assistant', content=assistant))
        return output


def run_local(model_path: str,
              model_name: Optional[str] = None,
              backend: Literal['turbomind', 'pytorch'] = 'turbomind',
              backend_config: Optional[Union[PytorchEngineConfig, TurbomindEngineConfig]] = None,
              chat_template_config: Optional[ChatTemplateConfig] = None,
              server_name: str = '0.0.0.0',
              server_port: int = 6006,
              tp: int = 1,
              **kwargs):

    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
    engine = VLAsyncEngine(model_path=model_path,
                           model_name=model_name,
                           backend=backend,
                           backend_config=backend_config,
                           chat_template_config=chat_template_config,
                           tp=tp,
                           **kwargs)

    def add_image(chatbot, session, file):
        """Append image to query."""
        chatbot = chatbot + [((file.name, ), None)]
        history = session._message
        img = Image.open(file.name).convert('RGB')
        # [([user, img, img], assistant), ...]
        if len(history) == 0 or history[-1][-1] is not None:
            history.append([[img], None])
        else:
            history[-1][0].append(img)
        return chatbot, session

    async def chat(chatbot, session, query, max_new_tokens, top_p, top_k, temperature):
        """Chat with AI assistant."""
        chatbot = chatbot + [[query, None]]
        history = session._message
        if len(history) == 0 or history[-1][-1] is not None:
            history.append([query, None])
        else:
            history[-1][0].insert(0, query)
        yield chatbot, session, disable_btn, disable_btn, disable_btn

        prompt = session.to_gpt4v()
        gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens,
                                   top_p=top_p,
                                   top_k=top_k,
                                   temperature=temperature,
                                   stop_token_ids=engine.stop_words)
        r = dict(messages=prompt, gen_config=gen_cfg, stream_response=True, session_id=session.session_id)
        gen = engine.generate(**r)
        session.gen = gen
        async for outputs in gen:
            response = outputs.response
            if outputs.finish_reason == 'length' and \
                    outputs.generate_token_len == 0:
                gr.Warning('WARNING: exceed session max length.'
                           ' Please restart the session by reset button.')
            if outputs.generate_token_len < 0:
                gr.Warning('WARNING: running on the old session.'
                           ' Please restart the session by reset button.')

            if chatbot[-1][1] is None:
                chatbot[-1][1] = response
                history[-1][1] = response
            chatbot[-1][1] += response
            history[-1][1] += response
            yield chatbot, session, disable_btn, enable_btn, disable_btn
        yield chatbot, session, enable_btn, disable_btn, enable_btn

    async def stop(session):
        """Stop the session."""
        await engine.stop_session(session.session_id)

    async def cancel(session):
        """Stop the session and keep chat history."""
        await stop(session)

    async def reset(session):
        """Reset a new session."""
        if session is not None:
            await stop(session)
        session = Session()
        return [], session, enable_btn

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy VL Playground')

            chatbot = gr.Chatbot(elem_id='chatbot', label=engine.model_name)
            query = gr.Textbox(placeholder='Please input the instruction', label='Instruction')
            session = gr.State()

            with gr.Row():
                addimg_btn = gr.UploadButton('Upload Image', file_types=['image'])
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')
            with gr.Row():
                max_new_tokens = gr.Slider(1, 2048, value=512, step=1, label='Maximum new tokens')
                top_p = gr.Slider(0.01, 1, value=1.0, step=0.01, label='Top_p')
                top_k = gr.Slider(1, 100, value=50, step=1, label='Top_k')
                temperature = gr.Slider(0.01, 1.5, value=1.0, step=0.01, label='Temperature')

        addimg_btn.upload(add_image, [chatbot, session, addimg_btn], [chatbot, session], show_progress=True, queue=True)

        query.submit(chat, [chatbot, session, query, max_new_tokens, top_p, top_k, temperature],
                     [chatbot, session, query, cancel_btn, reset_btn])
        query.submit(lambda: gr.update(value=''), None, [query])

        cancel_btn.click(cancel, [session], None)

        reset_btn.click(reset, [session], [chatbot, session, query])

        demo.load(lambda: Session(), inputs=None, outputs=[session])

    demo.queue(api_open=True, **que_kwargs, max_size=100)
    demo.launch(
        share=True,
        server_port=server_port,
        server_name=server_name,
    )


if __name__ == '__main__':
    import fire
    fire.Fire(run_local)
