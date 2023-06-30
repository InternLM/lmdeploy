# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import os
import threading
from functools import partial

import fire
import gradio as gr
from strings import ABSTRACT, TITLE
from styles import PARENT_BLOCK_CSS

from llmdeploy.serve.fastertransformer.chatbot import Chatbot


def chat_stream(instruction,
                state_chatbot,
                llama_chatbot,
                model_name: str = None):
    bot_summarized_response = ''
    model_type = 'fastertransformer'
    state_chatbot = state_chatbot + [(instruction, None)]
    session_id = threading.current_thread().ident
    bot_response = llama_chatbot.stream_infer(
        session_id, instruction, f'{session_id}-{len(state_chatbot)}')

    yield (state_chatbot, state_chatbot, f'{bot_summarized_response}'.strip())

    for status, tokens, _ in bot_response:
        if state_chatbot[-1][-1] is None or model_type != 'fairscale':
            state_chatbot[-1] = (state_chatbot[-1][0], tokens)
        else:
            state_chatbot[-1] = (state_chatbot[-1][0],
                                 state_chatbot[-1][1] + tokens
                                 )  # piece by piece
        yield (state_chatbot, state_chatbot,
               f'{bot_summarized_response}'.strip())

    yield (state_chatbot, state_chatbot, f'{bot_summarized_response}'.strip())


def reset_textbox():
    return gr.Textbox.update(value='')


def reset_everything_func(instruction_txtbox, state_chatbot, llama_chatbot,
                          triton_server_addr, model_name):

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


def cancel_func(instruction_txtbox, state_chatbot, llama_chatbot):
    session_id = llama_chatbot._session.session_id
    llama_chatbot.cancel(session_id)

    return (
        llama_chatbot,
        state_chatbot,
    )


def run(triton_server_addr: str,
        model_name: str,
        server_name: str = 'localhost',
        server_port: int = 6006):
    with gr.Blocks(css=PARENT_BLOCK_CSS, theme='ParityError/Anime') as demo:
        chat_interface = partial(chat_stream, model_name=model_name)
        reset_everything = partial(reset_everything_func,
                                   model_name=model_name,
                                   triton_server_addr=triton_server_addr)
        log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
        llama_chatbot = gr.State(
            Chatbot(triton_server_addr,
                    model_name,
                    log_level=log_level,
                    display=True))
        state_chatbot = gr.State([])

        with gr.Column(elem_id='col_container'):
            gr.Markdown(f'## {TITLE}\n\n\n{ABSTRACT}')

            # with gr.Accordion('Context Setting', open=False):
            #     hidden_txtbox = gr.Textbox(
            #         placeholder='', label='Order', visible=False)

            chatbot = gr.Chatbot(elem_id='chatbot', label=model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='What do you want to say to AI?',
                label='Instruction')
            with gr.Row():
                cancel_btn = gr.Button(value='Cancel')
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(
            chat_interface,
            [instruction_txtbox, state_chatbot, llama_chatbot],
            [state_chatbot, chatbot],
            batch=False,
            max_batch_size=1,
        )
        reset_event = instruction_txtbox.submit(
            reset_textbox,
            [],
            [instruction_txtbox],
        )

        cancel_btn.click(cancel_func,
                         [instruction_txtbox, state_chatbot, llama_chatbot],
                         [llama_chatbot, chatbot],
                         cancels=[send_event])

        reset_btn.click(
            reset_everything,
            [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, state_chatbot, chatbot, instruction_txtbox],
            cancels=[send_event])

    demo.queue(concurrency_count=4, max_size=100, api_open=True).launch(
        max_threads=10,
        share=True,
        server_port=server_port,
        server_name=server_name,
    )


if __name__ == '__main__':
    fire.Fire(run)
