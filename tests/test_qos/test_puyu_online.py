import csv
import os
import sys
import threading
import time
import requests
import json
from requests import exceptions as rex
from urllib3 import exceptions as uex


payload_template = {
    "model": "internlm-chat-7b",
    "messages": "string",
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "max_tokens": 512,
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


def send_request(payload, i):
    try:
        start_ts = time.time()
        with requests.post(url, json=payload) as response:
            rep_json = json.loads(response.content)
            end_ts = time.time()
            print(f"{i}, {payload['user_id']}, {rep_json['usage']['completion_tokens']}, {end_ts-start_ts}")
    except (TimeoutError, uex.NewConnectionError, uex.MaxRetryError, rex.ConnectionError) as e:
        print(f"{i}, {payload['user_id']}, ERROR, {type(e).__name__}")


def create_threads(uid, messages, intervals, session_ids):
    i = 0  
    # Set number of requests sent by each use case
    while True:
        payload = payload_template.copy()
        payload["user_id"] = uid
        payload["session_id"] = i % (session_ids[1]-session_ids[0]) + session_ids[0]

        payload["messages"] = messages[i % len(messages)]
        th = threading.Thread(target=send_request, args=[payload, i])
        th.start()
        time.sleep(intervals[i % len(intervals)])
        i += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_regular.py 0|1|2\n\t0: high priority user\n\t1: normal user\n\t2: pressure test")
        sys.exit(1)
    uid = ""
    session_ids = []
    pattern = sys.argv[1]
    if pattern == '0':
        uid = "user_id0"
        session_ids = [0, 6400]
    elif pattern == '1':
        uid = "user_id3"
        session_ids = [6422, 12800]
    elif pattern == '2':
        uid = "user_id4"
        session_ids = [12844, 21600]
    else:
        print("Usage: python test_regular.py 0|1|2\n\t0: high priority user\n\t1: normal user\n\t2: pressure test")
        sys.exit(1)
    
    # Load files
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)

    csv_file_name = script_directory + "/prompt_1115.csv"
    messages = []
    with open(csv_file_name, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # row[0]: prompt
            messages.append(row[0])
    
    itv_file_name = script_directory + "/" + [
        "interval_high_priority.csv", 
        "interval_normal_state.csv", 
        "interval_press_test.csv"][int(pattern)]
    intervals = []
    with open(itv_file_name, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # row[0]: sleep interval
            intervals.append(float(row[0]))

    # Send requests
    create_threads(uid, messages, intervals, session_ids)
