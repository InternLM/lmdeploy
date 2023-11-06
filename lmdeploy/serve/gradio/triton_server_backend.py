# Copyright (c) OpenMMLab. All rights reserved.
import os
from functools import partial
from threading import Lock
from typing import Sequence

import gradio as gr

from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn
from lmdeploy.serve.turbomind.chatbot import Chatbot


class InterFace:
    global_session_id: int = 0
    lock = Lock()


def chat_stream(state_chatbot: Sequence, llama_chatbot: Chatbot,
                cancel_btn: gr.Button, reset_btn: gr.Button, session_id: int):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        llama_chatbot (Chatbot): the instance of a chatbot
        cancel_btn (bool): enable the cancel button or not
        reset_btn (bool): enable the reset button or not
        session_id (int): the session id
    """
    instruction = state_chatbot[-1][0]

    bot_response = llama_chatbot.stream_infer(
        session_id, instruction, f'{session_id}-{len(state_chatbot)}')

    for status, tokens, _ in bot_response:
        state_chatbot[-1] = (state_chatbot[-1][0], tokens)
        yield (state_chatbot, state_chatbot, enable_btn, disable_btn)

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn)


def reset_all_func(instruction_txtbox: gr.Textbox, state_chatbot: gr.State,
                   llama_chatbot: gr.State, triton_server_addr: str,
                   model_name: str):
    """reset the session."""
    state_chatbot = []
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
    llama_chatbot = Chatbot(triton_server_addr,
                            model_name,
                            log_level=log_level,
                            display=True)

    return (
        llama_chatbot,
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=''),
    )


def cancel_func(
    state_chatbot: gr.State,
    llama_chatbot: gr.State,
    cancel_btn: gr.Button,
    reset_btn: gr.Button,
):
    """cancel the session."""
    yield (llama_chatbot, state_chatbot, disable_btn, disable_btn)
    session_id = llama_chatbot._session.session_id
    llama_chatbot.cancel(session_id)

    yield (llama_chatbot, state_chatbot, disable_btn, enable_btn)


def add_instruction(instruction, state_chatbot):
    state_chatbot = state_chatbot + [(instruction, None)]
    return ('', state_chatbot)


def run_triton_server(triton_server_addr: str,
                      server_name: str = 'localhost',
                      server_port: int = 6006):
    """chat with AI assistant through web ui.

    Args:
        triton_server_addr (str): the communication address of inference server
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
    """
    with gr.Blocks(css=CSS, theme=THEME) as demo:
        log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
        llama_chatbot = gr.State(
            Chatbot(triton_server_addr, log_level=log_level, display=True))
        state_chatbot = gr.State([])
        state_session_id = gr.State(0)
        model_name = llama_chatbot.value.model_name
        reset_all = partial(reset_all_func,
                            model_name=model_name,
                            triton_server_addr=triton_server_addr)

        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy Playground')

            chatbot = gr.Chatbot(elem_id='chatbot', label=model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='Please input the instruction',
                label='Instruction')
            with gr.Row():
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(
            add_instruction, [instruction_txtbox, state_chatbot],
            [instruction_txtbox, state_chatbot]).then(chat_stream, [
                state_chatbot, llama_chatbot, cancel_btn, reset_btn,
                state_session_id
            ], [state_chatbot, chatbot, cancel_btn, reset_btn])

        cancel_btn.click(cancel_func,
                         [state_chatbot, llama_chatbot, cancel_btn, reset_btn],
                         [llama_chatbot, chatbot, cancel_btn, reset_btn],
                         cancels=[send_event])

        reset_btn.click(
            reset_all, [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, state_chatbot, chatbot, instruction_txtbox],
            cancels=[send_event])

        def init():
            with InterFace.lock:
                InterFace.global_session_id += 1
            new_session_id = InterFace.global_session_id
            return new_session_id

        demo.load(init, inputs=None, outputs=[state_session_id])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=4, max_size=100, api_open=True).launch(
        max_threads=10,
        share=True,
        server_port=server_port,
        server_name=server_name,
    )
