# Copyright (c) OpenMMLab. All rights reserved.
import time
from threading import Lock
from typing import Sequence

import gradio as gr

from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn
from lmdeploy.serve.openai.api_client import (get_model_list,
                                              get_streaming_response)


class InterFace:
    api_server_url: str = None
    global_session_id: int = 0
    lock = Lock()


def chat_stream_restful(instruction: str, state_chatbot: Sequence,
                        cancel_btn: gr.Button, reset_btn: gr.Button,
                        session_id: int):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        session_id (int): the session id
    """
    state_chatbot = state_chatbot + [(instruction, None)]

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn)

    for response, tokens, finish_reason in get_streaming_response(
            instruction,
            f'{InterFace.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=512,
            interactive_mode=True):
        if finish_reason == 'length':
            gr.Warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
        if tokens < 0:
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


def reset_restful_func(instruction_txtbox: gr.Textbox, state_chatbot: gr.State,
                       session_id: int):
    """reset the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        session_id (int): the session id
    """
    state_chatbot = []
    # end the session
    for response, tokens, finish_reason in get_streaming_response(
            '',
            f'{InterFace.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=0,
            interactive_mode=False):
        pass

    return (
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=''),
    )


def cancel_restful_func(state_chatbot: gr.State, cancel_btn: gr.Button,
                        reset_btn: gr.Button, session_id: int):
    """stop the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        session_id (int): the session id
    """
    yield (state_chatbot, disable_btn, disable_btn)
    # end the session
    for out in get_streaming_response(
            '',
            f'{InterFace.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=0,
            stop=True):
        pass
    time.sleep(0.5)
    messages = []
    for qa in state_chatbot:
        messages.append(dict(role='user', content=qa[0]))
        if qa[1] is not None:
            messages.append(dict(role='assistant', content=qa[1]))
    for out in get_streaming_response(
            messages,
            f'{InterFace.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=0,
            interactive_mode=True):
        pass
    yield (state_chatbot, disable_btn, enable_btn)


def run_api_server(api_server_url: str,
                   server_name: str = 'localhost',
                   server_port: int = 6006,
                   batch_size: int = 32):
    """chat with AI assistant through web ui.

    Args:
        api_server_url (str): restufl api url
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        batch_size (int): batch size for running Turbomind directly
    """
    InterFace.api_server_url = api_server_url
    model_names = get_model_list(f'{api_server_url}/v1/models')
    model_name = ''
    if isinstance(model_names, list) and len(model_names) > 0:
        model_name = model_names[0]
    else:
        raise ValueError('gradio can find a suitable model from restful-api')

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        state_chatbot = gr.State([])
        state_session_id = gr.State(0)

        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy Playground')

            chatbot = gr.Chatbot(elem_id='chatbot', label=model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='Please input the instruction',
                label='Instruction')
            with gr.Row():
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(chat_stream_restful, [
            instruction_txtbox, state_chatbot, cancel_btn, reset_btn,
            state_session_id
        ], [state_chatbot, chatbot, cancel_btn, reset_btn])
        instruction_txtbox.submit(
            lambda: gr.Textbox.update(value=''),
            [],
            [instruction_txtbox],
        )
        cancel_btn.click(
            cancel_restful_func,
            [state_chatbot, cancel_btn, reset_btn, state_session_id],
            [state_chatbot, cancel_btn, reset_btn],
            cancels=[send_event])

        reset_btn.click(reset_restful_func,
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
    demo.queue(concurrency_count=batch_size, max_size=100,
               api_open=True).launch(
                   max_threads=10,
                   share=True,
                   server_port=server_port,
                   server_name=server_name,
               )
