#!/bin/bash
if  [ -z "$1" ]
then
    echo "please input model_name" >> "$GITHUB_ENV"
    exit 1
fi

if [[ $1 == *"w4a16" ]] || [[ $1 == *"4bit"* ]] || [[ $1 == *"awq"* ]] || [[ $1 == *"AWQ"* ]]
then
    echo "MODEL_FORMAT=--model-format awq" >> "$GITHUB_ENV"
else
    echo "MODEL_FORMAT=" >> "$GITHUB_ENV"
fi

if [[ $1 == *"llama2"* ]] || [[ $1 == *"Llama-2"* ]]
then
    echo "MAX_ENTRY_COUNT=--cache-max-entry-count 0.95" >> "$GITHUB_ENV"

else
    echo "MAX_ENTRY_COUNT=--cache-max-entry-count 0.90" >> "$GITHUB_ENV"
fi

if [[ $1 == *"llama2"* ]] || [[ $1 == *"Llama-2"* ]]
then
    echo "BATCHES=128" >> "$GITHUB_ENV"
    echo "MAX_BATCH_SIZE=" >> "$GITHUB_ENV"
else
    echo "BATCHES=128 256" >> "$GITHUB_ENV"
    echo "MAX_BATCH_SIZE=--max-batch-size 256" >> "$GITHUB_ENV"
fi

if [[ $1 == *"internlm2-chat-20b"* ]] || [[ $1 == *"Qwen1.5-32B-Chat"* ]]
then
  echo "TP_INFO=--tp 2" >> "$GITHUB_ENV"
fi
