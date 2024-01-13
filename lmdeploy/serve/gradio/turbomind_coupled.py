# Copyright (c) OpenMMLab. All rights reserved.
from threading import Lock
from typing import Literal, Optional, Sequence, Union

import gradio as gr

from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn


class InterFace:
    async_engine: AsyncEngine = None
    global_session_id: int = 0
    lock = Lock()


async def chat_stream_local(instruction: str, state_chatbot: Sequence,
                            cancel_btn: gr.Button, reset_btn: gr.Button,
                            session_id: int, top_p: float, temperature: float,
                            request_output_len: int):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        cancel_btn (gr.Button): the cancel button
        reset_btn (gr.Button): the reset button
        session_id (int): the session id
    """
    state_chatbot = state_chatbot + [(instruction, None)]

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn)
    gen_config = GenerationConfig(max_new_tokens=request_output_len,
                                  top_p=top_p,
                                  temperature=temperature)

    gen_config = GenerationConfig(max_new_tokens=request_output_len,
                                  top_p=top_p,
                                  temperature=temperature)
    async for outputs in InterFace.async_engine.generate(
            instruction,
            session_id,
            gen_config=gen_config,
            stream_response=True,
            sequence_start=(len(state_chatbot) == 1),
            sequence_end=False):
        response = outputs.response
        if outputs.finish_reason == 'length':
            gr.Warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
        if outputs.generate_token_len < 0:
            gr.Warning('WARNING: running on the old session.'
                       ' Please restart the session by reset button.')
        if state_chatbot[-1][-1] is None:
            state_chatbot[-1] = (state_chatbot[-1][0], response)
        else:
            state_chatbot[-1] = (state_chatbot[-1][0],
                                 state_chatbot[-1][1] + response
                                 )  # piece by piece
        yield (state_chatbot, state_chatbot, enable_btn, disable_btn)

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn)


async def reset_local_func(instruction_txtbox: gr.Textbox,
                           state_chatbot: Sequence, session_id: int):
    """reset the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        session_id (int): the session id
    """
    state_chatbot = []
    # end the session
    InterFace.async_engine.end_session(session_id)
    return (state_chatbot, state_chatbot, gr.Textbox.update(value=''))


async def cancel_local_func(state_chatbot: Sequence, cancel_btn: gr.Button,
                            reset_btn: gr.Button, session_id: int):
    """stop the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        cancel_btn (gr.Button): the cancel button
        reset_btn (gr.Button): the reset button
        session_id (int): the session id
    """
    yield (state_chatbot, disable_btn, disable_btn)
    InterFace.async_engine.stop_session(session_id)
    InterFace.async_engine.end_session(session_id)
    messages = []
    for qa in state_chatbot:
        messages.append(dict(role='user', content=qa[0]))
        if qa[1] is not None:
            messages.append(dict(role='assistant', content=qa[1]))
    gen_config = GenerationConfig(max_new_tokens=0)
    async for out in InterFace.async_engine.generate(messages,
                                                     session_id,
                                                     gen_config=gen_config,
                                                     stream_response=True,
                                                     sequence_start=True,
                                                     sequence_end=False):
        pass
    yield (state_chatbot, disable_btn, enable_btn)


def run_local(model_path: str,
              model_name: Optional[str] = None,
              backend: Literal['turbomind', 'pytorch'] = 'turbomind',
              backend_config: Optional[Union[PytorchEngineConfig,
                                             TurbomindEngineConfig]] = None,
              chat_template_config: Optional[ChatTemplateConfig] = None,
              server_name: str = 'localhost',
              server_port: int = 6006,
              tp: int = 1,
              **kwargs):
    """chat with AI assistant through web ui.

    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm/internlm-chat-7b",
            "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat" and so on.
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        tp (int): tensor parallel for Turbomind
    """
    InterFace.async_engine = AsyncEngine(
        model_path=model_path,
        backend=backend,
        backend_config=backend_config,
        chat_template_config=chat_template_config,
        model_name=model_name,
        tp=tp,
        **kwargs)

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        state_chatbot = gr.State([])
        state_session_id = gr.State(0)

        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy Playground')

            chatbot = gr.Chatbot(
                elem_id='chatbot',
                label=InterFace.async_engine.engine.model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='Please input the instruction',
                label='Instruction')
            with gr.Row():
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')
            with gr.Row():
                request_output_len = gr.Slider(1,
                                               2048,
                                               value=512,
                                               step=1,
                                               label='Maximum new tokens')
                top_p = gr.Slider(0.01, 1, value=0.8, step=0.01, label='Top_p')
                temperature = gr.Slider(0.01,
                                        1.5,
                                        value=0.7,
                                        step=0.01,
                                        label='Temperature')

        send_event = instruction_txtbox.submit(chat_stream_local, [
            instruction_txtbox, state_chatbot, cancel_btn, reset_btn,
            state_session_id, top_p, temperature, request_output_len
        ], [state_chatbot, chatbot, cancel_btn, reset_btn])
        instruction_txtbox.submit(
            lambda: gr.Textbox.update(value=''),
            [],
            [instruction_txtbox],
        )
        cancel_btn.click(
            cancel_local_func,
            [state_chatbot, cancel_btn, reset_btn, state_session_id],
            [state_chatbot, cancel_btn, reset_btn],
            cancels=[send_event])

        reset_btn.click(reset_local_func,
                        [instruction_txtbox, state_chatbot, state_session_id],
                        [state_chatbot, chatbot, instruction_txtbox],
                        cancels=[send_event])

        def init():
            with InterFace.lock:
                InterFace.global_session_id += 1
            new_session_id = InterFace.global_session_id
            return new_session_id

        demo.load(init, inputs=None, outputs=[state_session_id])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=InterFace.async_engine.instance_num,
               max_size=100,
               api_open=True).launch(
                   max_threads=10,
                   share=True,
                   server_port=server_port,
                   server_name=server_name,
               )


if __name__ == '__main__':
    import fire
    fire.Fire(run_local)
