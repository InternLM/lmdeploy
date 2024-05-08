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

if [[ $1 == *"llama"* ]] || [[ $1 == *"Llama"* ]]
then
    echo "MAX_ENTRY_COUNT=--cache-max-entry-count 0.95" >> "$GITHUB_ENV"
else
    echo "MAX_ENTRY_COUNT=--cache-max-entry-count 0.90" >> "$GITHUB_ENV"
fi


if [[ $1 == *"internlm2-chat-20b"* ]]
then
  echo "TP_INFO=--tp 2" >> "$GITHUB_ENV"
fi
