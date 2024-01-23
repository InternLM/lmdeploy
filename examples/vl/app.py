import argparse
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from threading import Lock
from typing import List, Tuple

import gradio as gr
from packaging.version import Version, parse
from qwen_model import QwenVLChat
from xcomposer_model import InternLMXComposer

from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn
from lmdeploy.turbomind import TurboMind
from lmdeploy.turbomind.chat import valid_str

BATCH_SIZE = 32
DEFAULT_MODEL_NAME = 'internlm-xcomposer-7b'
DEFAULT_HF_CKPT = 'internlm/internlm-xcomposer-7b'
# should use extract_xcomposer_llm.py to extract llm
# when use internlm-xcomposer-7b
DEFAULT_LLM_CKPT = None

SUPPORTED_MODELS = {
    'internlm-xcomposer-7b': InternLMXComposer,
    'qwen-vl-chat': QwenVLChat
}

if parse(gr.__version__) >= Version('4.0.0'):
    que_kwargs = {'default_concurrency_limit': BATCH_SIZE}
else:
    que_kwargs = {'concurrency_count': BATCH_SIZE}


@dataclass
class Session:
    _lock = Lock()
    _count = count()
    _session_id: int = None
    _message: List[Tuple[str, str]] = field(default_factory=list)
    _step: int = 0

    def __init__(self):
        with Session._lock:
            self._session_id = next(Session._count)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        default=DEFAULT_MODEL_NAME,
                        help='Model name, default to %(default)s')
    parser.add_argument(
        '--hf-ckpt',
        type=str,
        default=DEFAULT_HF_CKPT,
        help='hf checkpoint name or path, default to %(default)s')
    parser.add_argument(
        '--llm-ckpt',
        type=str,
        default=DEFAULT_LLM_CKPT,
        help='LLM checkpoint name or path, default to %(default)s')
    parser.add_argument('--server-port',
                        type=int,
                        default=9006,
                        help='Server port, default %(default)s')
    parser.add_argument('--server-name',
                        type=str,
                        default='0.0.0.0',
                        help='Server name, default %(default)s')
    args = parser.parse_args()
    return args


@contextmanager
def get_stop_words():
    from lmdeploy.tokenizer import Tokenizer
    old_func = Tokenizer.indexes_containing_token

    def new_func(self, token):
        indexes = self.encode(token, add_bos=False)
        return indexes

    Tokenizer.indexes_containing_token = new_func
    yield
    Tokenizer.indexes_containing_token = old_func


def load_preprocessor_model(args):
    """Load preprocessor and llm inference engine."""
    assert args.model_name in SUPPORTED_MODELS
    llm_ckpt = args.hf_ckpt if args.llm_ckpt is None else args.llm_ckpt
    preprocessor = SUPPORTED_MODELS[args.model_name](args.hf_ckpt)
    with get_stop_words():
        model = TurboMind.from_pretrained(llm_ckpt, model_name=args.model_name)
    return preprocessor, model


def launch_demo(args, preprocessor, model):

    def add_image(chatbot, session, file):
        """Append image to query."""
        chatbot = chatbot + [((file.name, ), None)]
        history = session._message
        # [([user, url, url], assistant), ...]
        if len(history) == 0 or history[-1][-1] is not None:
            history.append([[file.name], None])
        else:
            history[-1][0].append(file.name)
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

    def chat(
        chatbot,
        session,
        request_output_len=512,
    ):
        """Chat with AI assistant."""
        generator = model.create_instance()
        history = session._message
        sequence_start = len(history) == 1
        seed = random.getrandbits(64) if sequence_start else None
        input_ids, features, ranges = preprocessor.prepare_query(
            history[-1][0], sequence_start)

        if len(input_ids
               ) + session.step + request_output_len > model.model.session_len:
            gr.Warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
            yield chatbot, session, enable_btn, disable_btn, enable_btn
        else:
            response_size = 0
            step = session.step
            for outputs in generator.stream_infer(
                    session_id=session.session_id,
                    input_ids=input_ids,
                    input_embeddings=features,
                    input_embedding_ranges=ranges,
                    request_output_len=request_output_len,
                    stream_output=True,
                    sequence_start=sequence_start,
                    random_seed=seed,
                    step=step):
                res, tokens = outputs[0]
                # decode res
                response = model.tokenizer.decode(res.tolist(),
                                                  offset=response_size)
                if response.endswith('�'):
                    continue
                response = valid_str(response)
                response_size = tokens
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
        generator = model.create_instance()
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

            chatbot = gr.Chatbot(elem_id='chatbot', label=model.model_name)
            query = gr.Textbox(placeholder='Please input the instruction',
                               label='Instruction')
            session = gr.State()

            with gr.Row():
                addimg_btn = gr.UploadButton('Upload Image',
                                             file_types=['image'])
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')

        addimg_btn.upload(add_image, [chatbot, session, addimg_btn],
                          [chatbot, session],
                          show_progress=True,
                          queue=True)

        send_event = query.submit(
            add_text, [chatbot, session, query], [chatbot, session]).then(
                chat, [chatbot, session],
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
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = parse_args()

    cur_folder = Path(__file__).parent.as_posix()
    if cur_folder != os.getcwd():
        os.chdir(cur_folder)
        print(f'change working dir to {cur_folder}')

    preprocessor, model = load_preprocessor_model(args)
    launch_demo(args, preprocessor, model)


if __name__ == '__main__':
    main()
