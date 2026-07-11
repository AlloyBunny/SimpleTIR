#!/usr/bin/env bash

set -euo pipefail

pid_file="/data/L202500291/sandbox/local-sandbox.pid"

if [ ! -f "$pid_file" ]; then
  echo "Local sandbox is not running."
  exit 0
fi

pid=$(<"$pid_file")
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi

rm -f "$pid_file"
echo "Local sandbox stopped."
