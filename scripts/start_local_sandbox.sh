#!/usr/bin/env bash

set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$repository_root/scripts/activate_simpletir_env.sh"

runtime_dir="/data/L202500291/sandbox"
pid_file="$runtime_dir/local-sandbox.pid"
log_file="$runtime_dir/local-sandbox.log"
profile_source="$repository_root/sandbox/firejail/sandbox.profile"
profile_target="/etc/firejail/sandbox.profile"
host="${SANDBOX_HOST:-127.0.0.1}"
port="${SANDBOX_PORT:-12345}"
base_url="http://${host}:${port}"
endpoint="${base_url}/faas/sandbox/"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this script as root so it can install the Firejail profile and create sandboxer." >&2
  exit 1
fi

if [ "$host" != "127.0.0.1" ]; then
  echo "SANDBOX_HOST must remain 127.0.0.1 for the local sandbox." >&2
  exit 1
fi

if ! command -v firejail >/dev/null 2>&1; then
  echo "firejail is not installed. Install it with: apt-get update && apt-get install -y firejail" >&2
  exit 1
fi

if [ ! -f "$profile_source" ]; then
  echo "Missing repository Firejail profile: $profile_source" >&2
  exit 1
fi

if ! id sandboxer >/dev/null 2>&1; then
  useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin sandboxer
fi

install -d -m 0755 /etc/firejail
install -m 0644 "$profile_source" "$profile_target"
install -d -m 0750 "$runtime_dir"
install -d -m 0750 "$runtime_dir/home"
touch "$log_file"

if [ -f "$pid_file" ]; then
  existing_pid=$(<"$pid_file")
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Local sandbox is already running at $endpoint"
    exit 0
  fi
  rm -f "$pid_file"
fi

setsid nohup runuser -u sandboxer -- env \
  HOME="$runtime_dir/home" \
  PATH="$SIMPLETIR_ENV_ROOT/bin:/usr/bin:/bin" \
  PYTHONNOUSERSITE=1 \
  "$SIMPLETIR_ENV_ROOT/bin/python" -m uvicorn sandbox_api:app \
  --app-dir "$repository_root/sandbox" \
  --host 127.0.0.1 \
  --port "$port" \
  --workers 4 >>"$log_file" 2>&1 &
echo "$!" >"$pid_file"

for _ in $(seq 1 30); do
  if curl -fsS "$base_url/docs" >/dev/null; then
    readiness_response=$(curl -fsS -X POST "$endpoint" \
      -H 'Content-Type: application/json' \
      -d '{"code":"from pathlib import Path; assert not Path(\"/data/L202500291/SimpleTIR/README.md\").exists(); print(1 + 1)","language":"python","compile_timeout":1,"run_timeout":3}' || true)
    if printf '%s' "$readiness_response" | grep -Fq '"status":"success"'; then
      echo "SANDBOX_ENDPOINT=$endpoint"
      exit 0
    fi
    echo "Sandbox execution readiness check failed: $readiness_response" >>"$log_file"
    "$repository_root/scripts/stop_local_sandbox.sh"
    exit 1
  fi
  sleep 1
done

echo "Local sandbox did not become ready. See $log_file" >&2
"$repository_root/scripts/stop_local_sandbox.sh"
exit 1
