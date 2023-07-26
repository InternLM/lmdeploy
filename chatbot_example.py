from lmdeploy.serve.turbomind.chatbot import Chatbot
from multiprocessing import Process

class SessionChatbot:
    def __init__(self, tritonserver_addr, session_id):
        self.chatbot = Chatbot(tritonserver_addr, log_level='WARNING', display=True)
        self.session_id = session_id
        self.round = 1

    def chat(self, prompt):
        request_id = f'{self.session_id}-{self.round}'
        for status, res, n_token in self.chatbot.stream_infer(
                self.session_id,
                prompt,
                request_id=request_id,
                request_output_len=512):
            # 返回的结果可能是一个生成器，所以我们只取最后一个响应
            pass
        self.round += 1
        return res

def chatbot_process(tritonserver_addr, session_id, prompts):
    bot = SessionChatbot(tritonserver_addr, session_id)
    for prompt in prompts:
        response = bot.chat(prompt)
        print(f'Session {session_id}: {response}')

tritonserver_addr = '0.0.0.0:33337'
sessions = [
    # 每个会话有一个唯一的ID和一系列的提示
    {'id': 1, 'prompts': ['hello', 'how are you?']},
    {'id': 2, 'prompts': ['hi', 'what is your name?']},
]

# 创建并启动多个进程
processes = []
for session in sessions:
    p = Process(target=chatbot_process, args=(tritonserver_addr, session['id'], session['prompts']))
    p.start()
    processes.append(p)

# 等待所有进程完成
for p in processes:
    p.join()





