# flake8: noqa
from functools import partial

import fire
import gradio as gr
import os
from llama import chat_stream
from miscs.strings import ABSTRACT, TITLE
from miscs.styles import PARENT_BLOCK_CSS

from llmdeploy.serve.fastertransformer.chatbot import Chatbot


def reset_textbox():
    return gr.Textbox.update(value='')


def reset_everything_func(instruction_txtbox, state_chatbot, llama_chatbot,
                          triton_server_addr, model_name):

    state_chatbot = []
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
    llama_chatbot = Chatbot(
        triton_server_addr, model_name, log_level=log_level, display=True)

    return (
        llama_chatbot,
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=''),
    )


def run(model_name: str,
        triton_server_addr: str,
        server_name: str = 'localhost',
        server_port: int = 6006):
    with gr.Blocks(css=PARENT_BLOCK_CSS, theme='ParityError/Anime') as demo:
        chat_interface = partial(chat_stream, model_name=model_name)
        reset_everything = partial(
            reset_everything_func,
            model_name=model_name,
            triton_server_addr=triton_server_addr)
        log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
        llama_chatbot = gr.State(
            Chatbot(
                triton_server_addr,
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

        cancel_btn.click(None, None, None, cancels=[send_event])

        reset_btn.click(
            reset_everything,
            [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, state_chatbot, chatbot, instruction_txtbox],
            cancels=[send_event])

    demo.queue(
        concurrency_count=4, max_size=100, api_open=True).launch(
            max_threads=10,
            share=True,
            server_port=server_port,
            server_name=server_name,
        )


if __name__ == '__main__':
    fire.Fire(run)
