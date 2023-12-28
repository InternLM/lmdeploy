# Copyright (c) OpenMMLab. All rights reserved.
import time
from subprocess import PIPE, Popen

import fire


def parse_dialogue(inputs: str):
    sep = 'double enter to end input >>>'
    dialogues = inputs.strip()
    if dialogues.endswith(sep):
        dialogues = dialogues[:-len(sep)]
    dialogues = dialogues.strip()
    dialogues = dialogues.split(sep)
    dialogues = [d.strip() for d in dialogues]
    return dialogues[1:]


def test(port=33337):
    cmd = [f'lmdeploy serve triton_client localhost:{port}']

    test_cases = [
        dict(
            prompts='Hello! Please answer in English.',
            keywords=['Hello', 'hi'],
        ),
        dict(
            prompts='您好! 请用中文回答。',
            keywords=['您好', '你好'],
        ),
        dict(
            prompts='How many days does a week have? ',
            keywords=['seven', '7'],
        ),
        dict(
            prompts='一周有多少天?请用中文回答。',
            keywords=['七天', '7天'],
        ),
    ]

    sep = '\n\n'
    end = sep + 'exit\n\n\n'
    all_pass = True
    for cases in test_cases:
        quest = cases['prompts']
        keywords = [k.lower() for k in cases['keywords']]
        inputs = quest + end
        print(f'Test Input prompts: {quest}\nKey words: {keywords}')
        time.sleep(5)

        with Popen(cmd,
                   stdin=PIPE,
                   stdout=PIPE,
                   stderr=PIPE,
                   shell=True,
                   text=True,
                   encoding='utf-8') as proc:
            out, err = proc.communicate(input=inputs)
            print(f'Output: {out}')
            if proc.returncode == 0:
                out = parse_dialogue(out)[0].lower()
                success = any([k in out for k in keywords])
                if not success:
                    print(f'>>> Failed to output keywords: {out} {keywords}')
                    all_pass = False
            else:
                all_pass = False
                print(f'Failed to get outputs: {out} {err}')
    assert all_pass, 'Tests failed!'


if __name__ == '__main__':
    fire.Fire(test)
