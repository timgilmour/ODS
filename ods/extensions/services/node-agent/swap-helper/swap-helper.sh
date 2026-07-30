#!/bin/bash
# swap-helper.sh — privileged half of the spark model-swap split.
#
# The LAN-facing node-agent container never gets docker.sock; it only writes
# <ctl>/request.json. This helper runs on the host (tim, docker group),
# validates the requested profile against the real compose-*.yaml set, runs
# swap.sh, and reports through <ctl>/status.json. A compromised node-agent
# can therefore at worst swap between the operator's own approved profiles.
#
# Usage:
#   swap-helper.sh --once   <ctl-dir> <vllm-dir>   # process one pending request
#   swap-helper.sh --daemon <ctl-dir> <vllm-dir>   # poll loop (2s), for @reboot
set -u

MODE="${1:?mode}"; CTL="${2:?ctl dir}"; VLLM="${3:?vllm dir}"
REQ="$CTL/request.json"
STATUS="$CTL/status.json"
LOCK="$CTL/.lock"

write_status() { # state profile id message
  local tmp
  tmp=$(mktemp "$CTL/.status.XXXXXX")
  printf '{"state":"%s","profile":"%s","id":"%s","message":"%s","ts":"%s"}\n' \
    "$1" "$2" "$3" "$4" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
  mv "$tmp" "$STATUS"
}

process_one() {
  [ -f "$REQ" ] || return 0

  local parsed id profile
  parsed=$(python3 - "$REQ" <<'EOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("id", ""))
print(d.get("profile", ""))
EOF
  )
  rm -f "$REQ"   # consume first: a crash must not re-run a half-processed request
  if [ -z "$parsed" ]; then
    write_status error "" "" "malformed request json"
    return 0
  fi
  id=$(printf '%s' "$parsed" | sed -n 1p)
  profile=$(printf '%s' "$parsed" | sed -n 2p)

  if ! printf '%s' "$profile" | grep -qE '^[A-Za-z0-9_-]+$'; then
    write_status error "$profile" "$id" "invalid profile name"
    return 0
  fi
  if [ ! -f "$VLLM/compose-${profile}.yaml" ]; then
    write_status error "$profile" "$id" "unknown profile"
    return 0
  fi

  write_status swapping "$profile" "$id" "swap.sh running"
  if "$VLLM/swap.sh" "$profile" >> "$CTL/swap.log" 2>&1; then
    write_status done "$profile" "$id" "swap launched"
  else
    write_status error "$profile" "$id" "swap.sh failed (see swap.log)"
  fi
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "swap-helper: another instance holds the lock" >&2
  exit 1
fi

case "$MODE" in
  --once) process_one ;;
  --daemon)
    while true; do
      process_one
      sleep 2
    done
    ;;
  *) echo "swap-helper: unknown mode $MODE" >&2; exit 2 ;;
esac
