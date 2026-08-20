#!/bin/bash
# instances-helper.sh — privileged half of the Deck's engine-INSTANCES split (INST I1).
#
# The node-agent (LAN-facing, no docker) writes <ctl>/instance-req.json =
# {"verb": create|remove|move, "document": {...}}. This helper runs on the host
# (operator, docker group), renders the document through the repo-owned
# per-kind templates (render_instance.py; kind->template is templates/kinds.json)
# into <instances-dir>/<resource>.yaml and runs docker compose under the
# SEPARATE project `deck-instances` — never the ODS project, never
# --remove-orphans. A compromised node-agent can therefore at most ask for one
# of the operator's own templates on a GPU/port of its choosing.
#
# Usage:
#   instances-helper.sh --once   <ctl-dir> <templates-dir> <instances-dir> <ods-dir>
#   instances-helper.sh --daemon <ctl-dir> <templates-dir> <instances-dir> <ods-dir>
set -u
MODE="${1:?mode}"; CTL="${2:?ctl dir}"; TEMPLATES="${3:?templates dir}"
INSTANCES="${4:?instances dir}"; ODS_DIR="${5:?ods dir}"
HELPER_DIR=$(dirname "$0")
REQ="$CTL/instance-req.json"; LOCK="$CTL/.instances.lock"; LOG="$CTL/instances.log"
PROJECT=deck-instances
mkdir -p "$INSTANCES/data"

_write_result() { # resource verb ok(1|0) error result-path   (completion ONLY; JSON built in python)
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import datetime, json, os, sys
resource, verb, ok, error, path = sys.argv[1:6]
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump({"resource": resource, "verb": verb, "ok": ok == "1", "error": error or None,
               "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, fh)
os.replace(tmp, path)
PYEOF
}

_stage_route() { # verb document-json-path   — Task 9 fills this in; a no-op here, and it never affects the verb's outcome
  return 0
}

_compose() { # file verb...
  local f="$1"; shift
  docker compose -p "$PROJECT" -f "$f" "$@" >> "$LOG" 2>&1
}

process_one() {
  [ -f "$REQ" ] || return 0
  local docpath verb resource rc
  docpath=$(mktemp "$CTL/.doc.XXXXXX.json")
  verb=$(python3 - "$REQ" "$docpath" <<'PYEOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
doc = d.get("document")
if not isinstance(doc, dict): sys.exit(1)
json.dump(doc, open(sys.argv[2], "w"))
print(d.get("verb", ""))
PYEOF
  ); rc=$?
  rm -f "$REQ"   # consume first: a crash must not re-run a half-processed request
  if [ $rc -ne 0 ]; then
    echo "instances-helper: unparseable instance-req.json, no result written" >> "$LOG"; rm -f "$docpath"; return 0
  fi
  resource=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d.get("resource"); print(r if isinstance(r,str) else "")' "$docpath")
  if ! printf '%s' "$resource" | grep -qE '^[a-z0-9][a-z0-9-]*$'; then
    echo "instances-helper: document resource fails name validation, no result written" >> "$LOG"; rm -f "$docpath"; return 0
  fi
  local result="$CTL/instance-status-${resource}.json" file="$INSTANCES/${resource}.yaml"
  rm -f "$result"   # invalidate a stale answer BEFORE the slow part
  case "$verb" in
    create)
      local err
      if ! err=$(python3 "$HELPER_DIR/render_instance.py" "$TEMPLATES" "$docpath" "$file" "$INSTANCES" "$ODS_DIR" 2>&1); then
        _write_result "$resource" create 0 "$err" "$result"; rm -f "$docpath"; return 0
      fi
      if _compose "$file" up -d; then
        _stage_route create "$docpath"; _write_result "$resource" create 1 "" "$result"
      else
        _write_result "$resource" create 0 "docker compose up failed (see instances.log)" "$result"
      fi ;;
    remove)
      if [ ! -f "$file" ]; then
        _write_result "$resource" remove 0 "no rendered file for '$resource' (never created here, or already removed)" "$result"
      elif _compose "$file" down; then
        rm -f "$file"; _stage_route remove "$docpath"; _write_result "$resource" remove 1 "" "$result"
      else
        _write_result "$resource" remove 0 "docker compose down failed (see instances.log)" "$result"
      fi ;;
    move)
      local newfile="$file.next" err
      if [ ! -f "$file" ]; then
        _write_result "$resource" move 0 "no rendered file for '$resource' (never created here, or already removed)" "$result"
      elif ! err=$(python3 "$HELPER_DIR/render_instance.py" "$TEMPLATES" "$docpath" "$newfile" "$INSTANCES" "$ODS_DIR" 2>&1); then
        _write_result "$resource" move 0 "$err" "$result"          # render FIRST: a bad move never takes the instance down
      elif ! _compose "$file" down; then
        rm -f "$newfile"; _write_result "$resource" move 0 "docker compose down failed (see instances.log)" "$result"
      else
        mv -f "$newfile" "$file"
        if _compose "$file" up -d; then
          _stage_route move "$docpath"; _write_result "$resource" move 1 "" "$result"
        else
          _write_result "$resource" move 0 "docker compose up failed after down (see instances.log)" "$result"
        fi
      fi ;;
    *) _write_result "$resource" "$verb" 0 "invalid verb (want create, remove or move)" "$result" ;;
  esac
  rm -f "$docpath"
}

exec 9>"$LOCK"
if ! flock -n 9; then echo "instances-helper: another instance holds the lock" >&2; exit 1; fi
case "$MODE" in
  --once) process_one ;;
  --daemon) while true; do process_one; sleep 2; done ;;
  *) echo "instances-helper: unknown mode $MODE" >&2; exit 2 ;;
esac
