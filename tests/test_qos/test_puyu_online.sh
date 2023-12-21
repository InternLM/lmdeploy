#!/bin/sh
script_dir=$(dirname "$0")

python -u $script_dir/test_puyu_online.py 1 > $script_dir/normal_state.log &
normal_pid=$!
echo "normal state: $normal_pid"
sleep 20m

python -u $script_dir/test_puyu_online.py 2 > $script_dir/press_test.log &
press_pid=$!
echo "pressure test: $press_pid"
sleep 10m

python -u $script_dir/test_puyu_online.py 0 > $script_dir/high_priority.log &
high_pid=$!
echo "high priority: $high_pid"

sleep 30m
kill -9 $normal_pid
kill -9 $press_pid
kill -9 $high_pid
