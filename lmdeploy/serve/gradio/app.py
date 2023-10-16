# Copyright (c) OpenMMLab. All rights reserved.
import os
import threading
import time
from functools import partial
from typing import Sequence

import fire
import gradio as gr

from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.gradio.css import CSS
from lmdeploy.serve.openai.api_client import (get_model_list,
                                              get_streaming_response)
from lmdeploy.serve.openai.api_server import ip2id
from lmdeploy.serve.turbomind.chatbot import Chatbot

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    font=[gr.themes.GoogleFont('Inconsolata'), 'Arial', 'sans-serif'])

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def chat_stream(state_chatbot: Sequence, llama_chatbot: Chatbot,
                request: gr.Request):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        llama_chatbot (Chatbot): the instance of a chatbot
        request (gr.Request): the request from a user
        model_name (str): the name of deployed model
    """
    instruction = state_chatbot[-1][0]
    session_id = threading.current_thread().ident
    if request is not None:
        session_id = ip2id(request.kwargs['client']['host'])

    bot_response = llama_chatbot.stream_infer(
        session_id, instruction, f'{session_id}-{len(state_chatbot)}')

    for status, tokens, _ in bot_response:
        state_chatbot[-1] = (state_chatbot[-1][0], tokens)
        yield (state_chatbot, state_chatbot, '')

    return (state_chatbot, state_chatbot, '')


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
    instruction_txtbox: gr.Textbox,
    state_chatbot: gr.State,
    llama_chatbot: gr.State,
):
    """cancel the session."""
    session_id = llama_chatbot._session.session_id
    llama_chatbot.cancel(session_id)

    return (
        llama_chatbot,
        state_chatbot,
    )


def add_instruction(instruction, state_chatbot):
    state_chatbot = state_chatbot + [(instruction, None)]
    return ('', state_chatbot)


def run_server(triton_server_addr: str,
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
                cancel_btn = gr.Button(value='Cancel')
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(
            add_instruction, [instruction_txtbox, state_chatbot],
            [instruction_txtbox, state_chatbot]).then(
                chat_stream, [state_chatbot, llama_chatbot],
                [state_chatbot, chatbot])

        cancel_btn.click(cancel_func,
                         [instruction_txtbox, state_chatbot, llama_chatbot],
                         [llama_chatbot, chatbot],
                         cancels=[send_event])

        reset_btn.click(
            reset_all, [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, state_chatbot, chatbot, instruction_txtbox],
            cancels=[send_event])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=4, max_size=100, api_open=True).launch(
        max_threads=10,
        share=True,
        server_port=server_port,
        server_name=server_name,
    )


# a IO interface mananing variables
class InterFace:
    async_engine: AsyncEngine = None  # for run_local
    restful_api_url: str = None  # for run_restful


def chat_stream_restful(
    instruction: str,
    state_chatbot: Sequence,
    cancel_btn: gr.Button,
    reset_btn: gr.Button,
    request: gr.Request,
):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        request (gr.Request): the request from a user
    """
    session_id = threading.current_thread().ident
    if request is not None:
        session_id = ip2id(request.kwargs['client']['host'])
    bot_summarized_response = ''
    state_chatbot = state_chatbot + [(instruction, None)]

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn,
           f'{bot_summarized_response}'.strip())

    for response, tokens, finish_reason in get_streaming_response(
            instruction,
            f'{InterFace.restful_api_url}/generate',
            session_id=session_id,
            request_output_len=512,
            sequence_start=(len(state_chatbot) == 1),
            sequence_end=False):
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
        yield (state_chatbot, state_chatbot, enable_btn, disable_btn,
               f'{bot_summarized_response}'.strip())

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn,
           f'{bot_summarized_response}'.strip())


def reset_restful_func(instruction_txtbox: gr.Textbox, state_chatbot: gr.State,
                       request: gr.Request):
    """reset the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        request (gr.Request): the request from a user
    """
    state_chatbot = []

    session_id = threading.current_thread().ident
    if request is not None:
        session_id = ip2id(request.kwargs['client']['host'])
    # end the session
    for response, tokens, finish_reason in get_streaming_response(
            '',
            f'{InterFace.restful_api_url}/generate',
            session_id=session_id,
            request_output_len=0,
            sequence_start=False,
            sequence_end=True):
        pass

    return (
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=''),
    )


def cancel_restful_func(state_chatbot: gr.State, cancel_btn: gr.Button,
                        reset_btn: gr.Button, request: gr.Request):
    """stop the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        request (gr.Request): the request from a user
    """
    session_id = threading.current_thread().ident
    if request is not None:
        session_id = ip2id(request.kwargs['client']['host'])
    # end the session
    for out in get_streaming_response('',
                                      f'{InterFace.restful_api_url}/generate',
                                      session_id=session_id,
                                      request_output_len=0,
                                      sequence_start=False,
                                      sequence_end=False,
                                      stop=True):
        pass
    time.sleep(0.5)
    messages = []
    for qa in state_chatbot:
        messages.append(dict(role='user', content=qa[0]))
        if qa[1] is not None:
            messages.append(dict(role='assistant', content=qa[1]))
    for out in get_streaming_response(messages,
                                      f'{InterFace.restful_api_url}/generate',
                                      session_id=session_id,
                                      request_output_len=0,
                                      sequence_start=True,
                                      sequence_end=False):
        pass
    return (state_chatbot, disable_btn, enable_btn)


def run_restful(restful_api_url: str,
                server_name: str = 'localhost',
                server_port: int = 6006,
                batch_size: int = 32):
    """chat with AI assistant through web ui.

    Args:
        restful_api_url (str): restufl api url
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        batch_size (int): batch size for running Turbomind directly
    """
    InterFace.restful_api_url = restful_api_url
    model_names = get_model_list(f'{restful_api_url}/v1/models')
    model_name = ''
    if isinstance(model_names, list) and len(model_names) > 0:
        model_name = model_names[0]
    else:
        raise ValueError('gradio can find a suitable model from restful-api')

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        state_chatbot = gr.State([])

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
            chat_stream_restful,
            [instruction_txtbox, state_chatbot, cancel_btn, reset_btn],
            [state_chatbot, chatbot, cancel_btn, reset_btn])
        instruction_txtbox.submit(
            lambda: gr.Textbox.update(value=''),
            [],
            [instruction_txtbox],
        )
        cancel_btn.click(cancel_restful_func,
                         [state_chatbot, cancel_btn, reset_btn],
                         [state_chatbot, cancel_btn, reset_btn],
                         cancels=[send_event])

        reset_btn.click(reset_restful_func,
                        [instruction_txtbox, state_chatbot],
                        [state_chatbot, chatbot, instruction_txtbox],
                        cancels=[send_event])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=batch_size, max_size=100,
               api_open=True).launch(
                   max_threads=10,
                   share=True,
                   server_port=server_port,
                   server_name=server_name,
               )


async def chat_stream_local(
    instruction: str,
    state_chatbot: Sequence,
    cancel_btn: gr.Button,
    reset_btn: gr.Button,
    request: gr.Request,
):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        request (gr.Request): the request from a user
    """
    session_id = threading.current_thread().ident
    if request is not None:
        session_id = ip2id(request.kwargs['client']['host'])
    bot_summarized_response = ''
    state_chatbot = state_chatbot + [(instruction, None)]

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn,
           f'{bot_summarized_response}'.strip())

    async for outputs in InterFace.async_engine.generate(
            instruction,
            session_id,
            stream_response=True,
            sequence_start=(len(state_chatbot) == 1)):
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
        yield (state_chatbot, state_chatbot, enable_btn, disable_btn,
               f'{bot_summarized_response}'.strip())

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn,
           f'{bot_summarized_response}'.strip())


async def reset_local_func(instruction_txtbox: gr.Textbox,
                           state_chatbot: gr.State, request: gr.Request):
    """reset the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        request (gr.Request): the request from a user
    """
    state_chatbot = []

    session_id = threading.current_thread().ident
    if request is not None:
        session_id = ip2id(request.kwargs['client']['host'])
    # end the session
    async for out in InterFace.async_engine.generate('',
                                                     session_id,
                                                     request_output_len=1,
                                                     stream_response=True,
                                                     sequence_start=False,
                                                     sequence_end=True):
        pass

    return (
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=''),
    )


async def cancel_local_func(state_chatbot: gr.State, cancel_btn: gr.Button,
                            reset_btn: gr.Button, request: gr.Request):
    """stop the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        request (gr.Request): the request from a user
    """
    session_id = threading.current_thread().ident
    if request is not None:
        session_id = ip2id(request.kwargs['client']['host'])
    # end the session
    async for out in InterFace.async_engine.generate('',
                                                     session_id,
                                                     request_output_len=0,
                                                     stream_response=True,
                                                     sequence_start=False,
                                                     sequence_end=False,
                                                     stop=True):
        pass
    messages = []
    for qa in state_chatbot:
        messages.append(dict(role='user', content=qa[0]))
        if qa[1] is not None:
            messages.append(dict(role='assistant', content=qa[1]))
    async for out in InterFace.async_engine.generate(messages,
                                                     session_id,
                                                     request_output_len=0,
                                                     stream_response=True,
                                                     sequence_start=True,
                                                     sequence_end=False):
        pass
    return (state_chatbot, disable_btn, enable_btn)


def run_local(model_path: str,
              server_name: str = 'localhost',
              server_port: int = 6006,
              batch_size: int = 4,
              tp: int = 1):
    """chat with AI assistant through web ui.

    Args:
        model_path (str): the path of the deployed model
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        batch_size (int): batch size for running Turbomind directly
        tp (int): tensor parallel for Turbomind
    """
    InterFace.async_engine = AsyncEngine(model_path=model_path,
                                         instance_num=batch_size,
                                         tp=tp)

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        state_chatbot = gr.State([])

        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy Playground')

            chatbot = gr.Chatbot(
                elem_id='chatbot',
                label=InterFace.async_engine.tm_model.model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='Please input the instruction',
                label='Instruction')
            with gr.Row():
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(
            chat_stream_local,
            [instruction_txtbox, state_chatbot, cancel_btn, reset_btn],
            [state_chatbot, chatbot, cancel_btn, reset_btn])
        instruction_txtbox.submit(
            lambda: gr.Textbox.update(value=''),
            [],
            [instruction_txtbox],
        )
        cancel_btn.click(cancel_local_func,
                         [state_chatbot, cancel_btn, reset_btn],
                         [state_chatbot, cancel_btn, reset_btn],
                         cancels=[send_event])

        reset_btn.click(reset_local_func, [instruction_txtbox, state_chatbot],
                        [state_chatbot, chatbot, instruction_txtbox],
                        cancels=[send_event])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=batch_size, max_size=100,
               api_open=True).launch(
                   max_threads=10,
                   share=True,
                   server_port=server_port,
                   server_name=server_name,
               )


def run(model_path_or_server: str,
        server_name: str = 'localhost',
        server_port: int = 6006,
        batch_size: int = 32,
        tp: int = 1,
        restful_api: bool = False):
    """chat with AI assistant through web ui.

    Args:
        model_path_or_server (str): the path of the deployed model or the
            tritonserver URL or restful api URL. The former is for directly
            running service with gradio. The latter is for running with
            tritonserver by default. If the input URL is restful api. Please
            enable another flag `restful_api`.
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        batch_size (int): batch size for running Turbomind directly
        tp (int): tensor parallel for Turbomind
        restufl_api (bool): a flag for model_path_or_server
    """
    if ':' in model_path_or_server:
        if restful_api:
            run_restful(model_path_or_server, server_name, server_port,
                        batch_size)
        else:
            run_server(model_path_or_server, server_name, server_port)
    else:
        run_local(model_path_or_server, server_name, server_port, batch_size,
                  tp)


if __name__ == '__main__':
    fire.Fire(run)
