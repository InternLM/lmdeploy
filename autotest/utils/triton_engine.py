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
        print('init end')

    def triton_cancel(self, seq_id):
        try:
            print(f'cancel req start, seq_id:{seq_id}, type: {type(seq_id)}')
            triton_obj = self.triton_obj
            end_rs = triton_obj.cancel(seq_id)
            (f'cancel req end, seq_id:{seq_id}, end_rs: {end_rs}')
            return end_rs
        except Exception as e:
            print(f'Unknown error: {e}')
            return chatbot.StatusCode.TRITON_SERVER_ERR

    def triton_end(self, seq_id):
        try:
            print(f'stop start, seq_id:{seq_id}, type: {type(seq_id)}')
            triton_obj = self.triton_obj
            end_rs = triton_obj.end(seq_id)
            print(f'stop end, seq_id:{seq_id}, end_rs: {end_rs}')
            return end_rs
        except Exception as e:
            print(f'Unknown error: {e}')
            return chatbot.StatusCode.TRITON_SERVER_ERR

    def triton_resume(self, seq_id):
        try:
            print(f'resume start, seq_id:{seq_id}, type: {type(seq_id)}')
            triton_obj = self.triton_obj
            status = triton_obj.resume(seq_id)
            print(f'resume end, seq_id:{seq_id}, end_rs: {status}')
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
            print(f'triton_infer start, seq_id:{seq_id}')
            status = None
            res = ''
            tokens = 0
            triton_obj = self.triton_obj
            for status, res, tokens in triton_obj.stream_infer(
                    seq_id, prompt, req_id, request_output_len, **kv_args):
                tmp_status = status
                if status == chatbot.StatusCode.TRITON_STREAM_END:
                    tmp_status = chatbot.StatusCode.TRITON_STREAM_ING
            print(f'triton_infer end, seq_id:{seq_id}, status: {tmp_status}')
        except Exception as e:
            print(f'Unknown error: {e}')
        return

    def triton_set_session(self, session_id):
        try:
            triton_obj = self.triton_obj
            if session_id is None:
                triton_obj.session = None
            else:
                triton_obj.session = Session(session_id=session_id)
            print('triton set session_id: ' + str(session_id))
        except Exception as e:
            print(f'Unknown error: {e}')
        return
