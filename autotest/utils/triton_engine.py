from lmdeploy.serve.turbomind import chatbot
from lmdeploy.serve.turbomind.chatbot import Session


class Engine:

    def __init__(self, server_addr: str, log_level: str = 'ERROR', **kv_args):
        self.server_addr = server_addr
        self.log_level = log_level
        triton_obj = chatbot.Chatbot(self.server_addr,
                                     log_level=self.log_level,
                                     **kv_args)
        self.triton_obj = triton_obj

    def triton_cancel(self, seq_id):
        try:
            triton_obj = self.triton_obj
            end_rs = triton_obj.cancel(seq_id)
            return end_rs
        except Exception as e:
            print(f'Unknown error: {e}')
            return chatbot.StatusCode.TRITON_SERVER_ERR

    def triton_end(self, seq_id):
        try:
            triton_obj = self.triton_obj
            end_rs = triton_obj.end(seq_id)
            return end_rs
        except Exception as e:
            print(f'Unknown error: {e}')
            return chatbot.StatusCode.TRITON_SERVER_ERR

    def triton_resume(self, seq_id):
        try:
            triton_obj = self.triton_obj
            status = triton_obj.resume(seq_id)
            return status
        except Exception as e:
            print(f'Unknown error: {e}')
            return chatbot.StatusCode.TRITON_SERVER_ERR

    def triton_infer(self,
                     seq_id,
                     prompt: str,
                     req_id,
                     request_output_len,
                     sequence_start: bool = False,
                     sequence_end: bool = False,
                     **kv_args):
        try:
            status = None
            res = ''
            tokens = 0
            triton_obj = self.triton_obj
            for status, res, tokens in triton_obj.stream_infer(
                    seq_id,
                    prompt,
                    req_id,
                    request_output_len,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    **kv_args):
                continue

            return status, res, tokens
        except Exception as e:
            print(f'Unknown error: {e}')
        return chatbot.StatusCode.TRITON_SERVER_ERR, None, None

    def triton_set_session(self, session_id):
        try:
            triton_obj = self.triton_obj
            if session_id is None:
                triton_obj.session = None
            else:
                triton_obj.session = Session(session_id=session_id)
        except Exception as e:
            print(f'Unknown error: {e}')
        return