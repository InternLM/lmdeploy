# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
from dataclasses import dataclass, field
from itertools import count
from typing import List, Literal, Optional, Tuple, Union

import gradio as gr
from packaging.version import Version, parse
from PIL import Image

from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.pytorch.engine.request import _run_until_complete
from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn
from lmdeploy.tokenizer import DetokenizeState
from lmdeploy.utils import get_logger

BATCH_SIZE = 32
logger = get_logger('lmdeploy')

if parse(gr.__version__) >= Version('4.0.0'):
    que_kwargs = {'default_concurrency_limit': BATCH_SIZE}
else:
    que_kwargs = {'concurrency_count': BATCH_SIZE}


@dataclass
class Session:
    _count = count()
    _session_id: int = None
    _message: List[Tuple[str, str]] = field(default_factory=list)
    _step: int = 0

    def __init__(self):
        self._session_id = next(self._count)
        self._message = []
        self._step = 0

    @property
    def session_id(self):
        return self._session_id

    @property
    def message(self):
        return self._message

    @property
    def step(self):
        return self._step


def preprocess(engine, prompt, sequence_start: bool):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    inputs = loop.run_until_complete(
        engine._get_prompt_input(prompt, True, sequence_start=sequence_start))
    return inputs


def run_local(model_path: str,
              model_name: Optional[str] = None,
              backend: Literal['turbomind', 'pytorch'] = 'turbomind',
              backend_config: Optional[Union[PytorchEngineConfig,
                                             TurbomindEngineConfig]] = None,
              chat_template_config: Optional[ChatTemplateConfig] = None,
              server_name: str = '0.0.0.0',
              server_port: int = 7006,
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

    def add_text(chatbot, session, text):
        """User query."""
        chatbot = chatbot + [(text, None)]
        history = session._message
        if len(history) == 0 or history[-1][-1] is not None:
            history.append([text, None])
        else:
            history[-1][0].insert(0, text)
        return chatbot, session, disable_btn, enable_btn

    def chat(chatbot, session, max_new_tokens, top_p, top_k, temperature):
        """Chat with AI assistant."""
        generator = engine.engine.create_instance()
        history = session._message
        sequence_start = len(history) == 1

        if isinstance(history[-1][0], str):
            prompt = history[-1][0]
        else:
            prompt = history[-1][0][0]
            images = history[-1][0][1:]
            prompt = (prompt, images)

        logger.info('prompt: ' + str(prompt))
        prompt = engine.vl_prompt_template.prompt_to_messages(prompt)
        t0 = time.perf_counter()
        inputs = _run_until_complete(
            engine._get_prompt_input(prompt,
                                     True,
                                     sequence_start=sequence_start))
        t1 = time.perf_counter()
        logger.info('preprocess cost %.3fs' % (t1 - t0))

        input_ids = inputs['input_ids']
        logger.info('input_ids: ' + str(input_ids))
        if len(input_ids) + session.step + max_new_tokens > engine.session_len:
            gr.Warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
            yield chatbot, session, enable_btn, disable_btn, enable_btn
        else:
            gen_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                          top_p=top_p,
                                          top_k=top_k,
                                          temperature=temperature)
            step = session.step
            state = DetokenizeState()
            for outputs in generator.stream_infer(
                    session_id=session._session_id,
                    **inputs,
                    sequence_start=sequence_start,
                    step=step,
                    gen_config=gen_config,
                    stream_output=True):
                _, res, tokens = outputs
                response, state = engine.tokenizer.detokenize_incrementally(
                    res,
                    state,
                    skip_special_tokens=gen_config.skip_special_tokens)
                if chatbot[-1][1] is None:
                    chatbot[-1][1] = ''
                    history[-1][1] = ''
                chatbot[-1][1] += response
                history[-1][1] += response
                session._step = step + len(input_ids) + tokens
                yield chatbot, session, disable_btn, enable_btn, disable_btn
            yield chatbot, session, enable_btn, disable_btn, enable_btn

    def stop(session):
        """Stop the session."""
        generator = engine.engine.create_instance()
        for _ in generator.stream_infer(session_id=session.session_id,
                                        input_ids=[0],
                                        request_output_len=0,
                                        sequence_start=False,
                                        sequence_end=False,
                                        stop=True):
            pass

    def cancel(chatbot, session):
        """Stop the session and keey chat history."""
        stop(session)
        return chatbot, session, disable_btn, enable_btn, enable_btn

    def reset(session):
        """Reset a new session."""
        stop(session)
        session._step = 0
        session._message = []
        return [], session, enable_btn

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy VL Playground')

            chatbot = gr.Chatbot(elem_id='chatbot', label=engine.model_name)
            query = gr.Textbox(placeholder='Please input the instruction',
                               label='Instruction')
            session = gr.State()

            with gr.Row():
                addimg_btn = gr.UploadButton('Upload Image',
                                             file_types=['image'])
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')
            with gr.Row():
                max_new_tokens = gr.Slider(1,
                                           2048,
                                           value=512,
                                           step=1,
                                           label='Maximum new tokens')
                top_p = gr.Slider(0.01, 1, value=0.8, step=0.01, label='Top_p')
                top_k = gr.Slider(1, 100, value=50, step=1, label='Top_k')
                temperature = gr.Slider(0.01,
                                        1.5,
                                        value=0.7,
                                        step=0.01,
                                        label='Temperature')

        addimg_btn.upload(add_image, [chatbot, session, addimg_btn],
                          [chatbot, session],
                          show_progress=True,
                          queue=True)

        send_event = query.submit(
            add_text, [chatbot, session, query], [chatbot, session]).then(
                chat,
                [chatbot, session, max_new_tokens, top_p, top_k, temperature],
                [chatbot, session, query, cancel_btn, reset_btn])
        query.submit(lambda: gr.update(value=''), None, [query])

        cancel_btn.click(cancel, [chatbot, session],
                         [chatbot, session, cancel_btn, reset_btn, query],
                         cancels=[send_event])

        reset_btn.click(reset, [session], [chatbot, session, query],
                        cancels=[send_event])

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
