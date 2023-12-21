import json
import threading
import time
import requests
from requests import exceptions as rex
from urllib3 import exceptions as uex

payload_template = {
    "model": "internlm-chat-7b",
    "messages": "string",
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "max_tokens": 128,
    "stop": False,
    "stream": False,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "user_id": "template",
    "repetition_penalty": 1,
    "session_id": -1,
    "ignore_eos": False
}

url = "http://localhost:64546/v1/chat/completions_qos"

def send_request(payload, stage, i):
    # print(f"send: {stage}#{i}, uid: {payload['user_id']}")
    try:
        with requests.post(url, json=payload) as response:
            rep_json = json.loads(response.content)
            print(f"{stage}#{i}, uid: {payload['user_id']}, {response.status_code}, {rep_json['usage']['completion_tokens']}")
    except (TimeoutError, uex.NewConnectionError, uex.MaxRetryError, rex.ConnectionError) as e:
        print(f"{stage}#{i}, uid: {payload['user_id']}, ERROR, {type(e).__name__}")


session_id = 0
def create_thread(uids, req_cnt, stage):
    """
    param
        uids: list of uids to send requests in one stage
        req_cnt: number of requests to send for each uid
        stage: stage number
    """
    global session_id
    for i in range(req_cnt):
        for uid in uids:
            payload = payload_template.copy()
            payload["user_id"] = uid
            payload["session_id"] = session_id
            session_id += 1
            th = threading.Thread(target=send_request, args=[payload, stage, i])
            th.start()
        time.sleep(0.03)


print("\nstage 1:\n")
create_thread(["user_id4"], 4000, 1)
print("\nstage 2:\n")
create_thread(["user_id4", "user_id5"], 1000, 2)
print("\nstage 3:\n")
create_thread(["user_id4"], 4000, 3)
print("\nstage 4:\n")
create_thread(["user_id4", "user_id5"], 2000, 4)
