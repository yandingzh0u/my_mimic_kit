#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/run_serial_roll_2k.sh"
test_dir="$(mktemp -d)"
launcher_pid=""

cleanup() {
  if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -KILL "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
  fi
  local pid_file pid
  for pid_file in "$test_dir/leader.pid" "$test_dir/worker.pid"; do
    if [[ -f "$pid_file" ]]; then
      pid="$(< "$pid_file")"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  rm -rf -- "$test_dir"
}
trap cleanup EXIT

"$launcher" --signal-test-dir "$test_dir" \
  > "$test_dir/launcher.log" 2>&1 &
launcher_pid=$!

for _ in $(seq 1 100); do
  if [[ -s "$test_dir/leader.pid" && -s "$test_dir/worker.pid" ]]; then
    break
  fi
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    printf 'FAIL: launcher exited before creating test children\n' >&2
    exit 1
  fi
  sleep 0.05
done

if [[ ! -s "$test_dir/leader.pid" || ! -s "$test_dir/worker.pid" ]]; then
  printf 'FAIL: timed out waiting for managed child PIDs\n' >&2
  exit 1
fi

leader_pid="$(< "$test_dir/leader.pid")"
worker_pid="$(< "$test_dir/worker.pid")"
kill -TERM "$launcher_pid"
set +e
wait "$launcher_pid"
launcher_rc=$?
set -e
launcher_pid=""

if [[ "$launcher_rc" -ne 143 ]]; then
  printf 'FAIL: TERM returned %s, expected 143\n' "$launcher_rc" >&2
  exit 1
fi

for _ in $(seq 1 100); do
  if ! kill -0 "$leader_pid" 2>/dev/null \
      && ! kill -0 "$worker_pid" 2>/dev/null; then
    printf 'signal forwarding test: PASS\n'
    exit 0
  fi
  sleep 0.05
done

printf 'FAIL: managed process survived TERM (leader=%s worker=%s)\n' \
  "$leader_pid" "$worker_pid" >&2
exit 1
