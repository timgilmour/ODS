#!/bin/bash
# swap-helper.sh — privileged half of the spark model-swap split.
#
# The LAN-facing node-agent container never gets docker.sock; it only writes
# <ctl>/request.json. This helper runs on the host (tim, docker group),
# validates the requested profile against the real compose-*.yaml set,
# launches it, and reports through <ctl>/status.json. A compromised node-agent
# can therefore at worst swap between the operator's own approved profiles.
#
# The OPTIONAL 4th argument is the settings dir shared with node-agent
# (NODE_SETTINGS_DIR). Given it, the helper renders <dir>/<profile>.json into
# a compose override and owns the launch itself, then harvests the engine's
# option catalog back into <dir>/catalog-<profile>.json. WITHOUT it — and
# with any missing or unusable document — the helper behaves exactly as it
# did before settings existed: it shells out to swap.sh. A settings bug must
# never be able to break a swap.
#
# Usage:
#   swap-helper.sh --once   <ctl-dir> <vllm-dir> [<settings-dir>]
#   swap-helper.sh --daemon <ctl-dir> <vllm-dir> [<settings-dir>]
set -u

MODE="${1:?mode}"; CTL="${2:?ctl dir}"; VLLM="${3:?vllm dir}"
SETTINGS="${4:-}"
HELPER_DIR=$(dirname "$0")   # harvest_probe.py ships next to this script
REQ="$CTL/request.json"
STATUS="$CTL/status.json"
LOCK="$CTL/.lock"
ENGINE_REQ="$CTL/engine-req.json"   # engine up/down protocol, see below

write_status() { # state profile id message
  local tmp
  tmp=$(mktemp "$CTL/.status.XXXXXX")
  printf '{"state":"%s","profile":"%s","id":"%s","message":"%s","ts":"%s"}\n' \
    "$1" "$2" "$3" "$4" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
  mv "$tmp" "$STATUS"
}

# Which branch _launch took. Read only by the failure message: on the
# settings-owned branch swap.sh is never invoked, so "swap.sh failed" points
# the operator at a script that did not run.
LAUNCH_BRANCH=swap.sh

# Launch: settings-owned when a valid document exists, else swap.sh verbatim.
# A missing or unparseable document MUST reproduce today's exact behaviour.
_launch() { # profile
  local profile="$1"
  # NEVER compose-<p>.override.yaml: swapctl.list_profiles globs compose-*.yaml
  # and would list the override as a ghost profile the node cannot serve.
  local doc="" override="$VLLM/settings-$profile.override.yaml"
  LAUNCH_BRANCH=swap.sh
  [ -n "$SETTINGS" ] && doc="$SETTINGS/$profile.json"
  if [ -n "$doc" ] && [ -f "$doc" ] && _write_override "$doc" "$override"; then
    LAUNCH_BRANCH=settings
    _teardown_all
    docker compose -f "$VLLM/compose-$profile.yaml" -f "$override" up -d \
      >> "$CTL/swap.log" 2>&1
  else
    rm -f "$override"   # stale settings must not haunt a settings-less swap
    "$VLLM/swap.sh" "$profile" >> "$CTL/swap.log" 2>&1
  fi
}

# Compose override from the document: JSON is valid YAML, so python3's stdlib
# is enough. Exits non-zero on any malformation, which drops the caller into
# the swap.sh branch above. Three outcomes, deliberately distinct:
#   0  override written
#   2  the document asserts nothing (settings_store.EMPTY) — fall back QUIETLY
#   1  the document is malformed — fall back and say so
_write_override() { # doc override
  python3 - "$1" "$2" <<'PYEOF'
import json, os, sys

doc_path, override = sys.argv[1], sys.argv[2]
# Compose's own rule for `environment` values: scalars only. A list or a
# nested object is a document this node cannot render, and rendering it
# anyway would hand `docker compose up` a file it rejects — i.e. a settings
# bug that breaks a swap. Refuse, and let the caller fall back.
scalar = (str, int, float, bool, type(None))
try:
    doc = json.load(open(doc_path))
    service, argv, env = doc["service"], doc["argv"], doc["env"]
except Exception as exc:
    # Falling back is safe, but it must not be silent: without this line the
    # only symptom of a bad document is settings that quietly stop applying.
    sys.stderr.write(
        "swap-helper: unusable settings document %s: %r\n" % (doc_path, exc))
    sys.exit(1)

# A document that asserts nothing is every profile's STARTING state, not a
# fault: node-agent hands out {"args": {}, "env": {}, "argv": [], "service":
# None} (settings_store.EMPTY) for any profile nobody has configured yet.
#
# It must not actuate. Rendering `command: []` would not be a no-op --
# compose REPLACES the base file's command outright, so an empty argv would
# launch the profile on the image's default CMD and report `done`: a settings
# document breaking a swap, the one outcome this split exists to prevent.
#
# And it must not warn. This shape reaches the helper on every swap of an
# unconfigured profile, so a diagnostic here would be noise that trains
# operators to ignore the line that DOES mean something.
if service is None or (isinstance(argv, list) and not argv):
    sys.exit(2)

try:
    assert isinstance(service, str) and service, "service must be a non-empty string"
    assert isinstance(argv, list), "argv must be a list"
    assert all(isinstance(t, str) for t in argv), "argv must be all strings"
    assert isinstance(env, dict), "env must be an object"
    assert all(isinstance(v, scalar) for v in env.values()), "env values must be scalars"
except Exception as exc:
    sys.stderr.write(
        "swap-helper: unusable settings document %s: %r\n" % (doc_path, exc))
    sys.exit(1)
try:
    tmp = override + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({"services": {service: {"command": argv,
                                          "environment": env}}}, handle)
    os.replace(tmp, override)   # atomic: compose never reads a partial file
except OSError as exc:
    sys.stderr.write(
        "swap-helper: cannot write override %s: %s\n" % (override, exc))
    sys.exit(1)
PYEOF
}

# The one place that knows how to read a container_name out of a compose
# file. A trailing inline comment, surrounding quotes and stray whitespace
# are stripped, so `container_name: "spark-foo"` and `container_name:
# spark-foo  # note` both extract cleanly instead of leaking quotes or
# comment into `docker rm -f` / `docker exec` (which would fail silently and
# never trip the zero-match warning below).
_container_name() { # compose-file
  sed -n 's/^[[:space:]]*container_name:[[:space:]]*//p' "$1" 2>/dev/null \
    | head -n1 \
    | sed -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\\(.*\\)'\$/\\1/"
}

# Ported from the LIVE sparky swap.sh (_profile_containers, archived
# 2026-08-07): container names are DERIVED from container_name: lines across
# every compose-*.yaml, never hard-coded — a hard-coded list is what let
# spark-ds4 keep holding :8000 across a swap on 2026-08-04.
_profile_containers() {
  local f name
  for f in "$VLLM"/compose-*.yaml; do
    [ -f "$f" ] || continue
    name=$(_container_name "$f")
    if [ -z "$name" ]; then
      # swap.log, not stderr: the status message already points operators
      # there ("see swap.log"), and this helper's stderr in --daemon mode
      # goes wherever @reboot cron sends it.
      echo "swap-helper: warning: ${f} has no container_name — its container will not be torn down" \
        >> "$CTL/swap.log"
      continue
    fi
    printf '%s\n' "$name"
  done | sort -u
}

_teardown_all() {
  local -a names=()
  mapfile -t names < <(_profile_containers)
  if [ "${#names[@]}" -gt 0 ]; then
    docker rm -f "${names[@]}" >/dev/null 2>&1 || true
  fi
}

# Post-launch catalog probe, vLLM profiles only. Argparse introspection needs
# the container to EXIST, not the server to be ready, so this runs straight
# after the launch. Every step is guarded: a probe failure is logged and
# swallowed, never touching the swap outcome the caller already published.
_harvest() { # profile
  [ -n "$SETTINGS" ] && [ -d "$SETTINGS" ] || return 0
  local profile="$1" engine container image probe out
  probe="$HELPER_DIR/harvest_probe.py"
  [ -f "$probe" ] || return 0
  engine=$(_profile_engine "$profile")
  [ "$engine" = "vllm" ] || return 0
  container=$(_container_name "$VLLM/compose-$profile.yaml")
  [ -n "$container" ] || return 0
  # The image content ID is the catalog's identity (app.harvest's
  # engine_version): no image, no honest catalog.
  image=$(docker inspect -f '{{.Image}}' "$container" 2>/dev/null) || return 0
  [ -n "$image" ] || return 0
  out=$(mktemp "$CTL/.harvest.XXXXXX") || return 0
  # -i is load-bearing: without it docker exec does not attach stdin and the
  # probe reads EOF. The fake-docker test asserts the flag is present.
  # python3, not python: the engine image has no `python` on PATH
  # (app.harvest.PROBE_INTERPRETER).
  # Probe stderr goes to swap.log, never /dev/null: the FIRST harvest of a
  # new engine image is exactly where a probe fails, and its reason (no
  # python3, an argparse shape the probe cannot walk) lives only there.
  if docker exec -i "$container" python3 - < "$probe" > "$out" 2>> "$CTL/swap.log"; then
    _write_catalog "$SETTINGS/catalog-$profile.json" "$image" "$out"
  else
    echo "harvest: probe failed for $profile" >&2
  fi
  rm -f "$out"
  return 0   # the swap already succeeded; nothing here may change that
}

# The probe output is handed over as a FILE PATH, never interpolated into the
# python source below: it is arbitrary engine stdout (JSON braces, quotes,
# backslashes, conceivably the heredoc terminator itself), and an
# interpolating heredoc would be a code-injection-shaped bug in a script that
# runs privileged on the host.
_write_catalog() { # path image out-file
  python3 - "$1" "$2" "$3" <<'PYEOF'
import datetime, json, os, sys

path, image, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    # errors="replace": a probe that emitted a stray non-UTF-8 byte should
    # cost a character, not the whole catalog.
    with open(out_path, encoding="utf-8", errors="replace") as handle:
        probe_output = handle.read()
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({
            "image_id": image,
            # Same wire format as write_status's `date -u`;
            # settings_store.read_newest_catalog orders catalogs by comparing
            # these as plain strings, so the shape is load-bearing.
            "harvested_ts": datetime.datetime.now(datetime.timezone.utc)
                                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "engine": "vllm",
            "probe_output": probe_output,
        }, handle)
    os.replace(tmp, path)   # atomic: a reader never sees a half catalog
except OSError as exc:
    sys.stderr.write("harvest: cannot write catalog %s: %s\n" % (path, exc))
    sys.exit(1)
PYEOF
}

# Engine from profiles.json; absent, malformed and non-object entries all
# default to vllm, matching node-agent swapctl._META_DEFAULTS / profile_meta.
_profile_engine() { # profile
  python3 - "$VLLM/profiles.json" "$1" <<'PYEOF'
import json, sys
try:
    entry = json.load(open(sys.argv[1])).get(sys.argv[2]) or {}
    print(entry.get("engine", "vllm") if isinstance(entry, dict) else "vllm")
except Exception:
    print("vllm")
PYEOF
}

# --- Engine up/down request protocol -----------------------------------
# <ctl>/engine-req.json = {"resource": str, "verb": "up"|"down", "ts": float},
# written by node-agent (LAN-facing, no docker rights). This helper alone
# has docker rights, so it validates `resource` against the host-owned
# $VLLM/engines.json allowlist (beside profiles.json) -- the compose file
# to act on is NEVER taken from the request, only looked up here -- and
# runs `docker compose -f <compose_file> up -d` / `down`, output appended
# to swap.log like a profile swap.
#
# Result: <ctl>/engine-status-<resource>.json, written ONLY at completion
# (tmp + atomic replace, like _write_catalog below). No "in-progress"
# marker is ever written to disk: a helper killed mid-verb must leave
# nothing that says "running" -- a stuck `swapping` status once stalled the
# deck ~20 minutes. Absence of a result file means "unknown"; the deck
# derives liveness from health probing, never from this file.

# The result document, JSON-built in python so an arbitrary verb or error
# string (request-controlled) can never break the file's shape -- same
# reasoning as _write_catalog's probe-output handling below.
_write_engine_status() { # resource verb ok(1|0) error result-path
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import datetime, json, os, sys

resource, verb, ok, error, path = sys.argv[1:6]
tmp = path + ".tmp"
with open(tmp, "w") as handle:
    json.dump({
        "resource": resource,
        "verb": verb,
        "ok": ok == "1",
        "error": error or None,
        "ts": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, handle)
os.replace(tmp, path)   # atomic: a reader never sees a partial result
PYEOF
}

# Resolve resource -> its declared compose_file, or refuse. Prints the
# compose path and exits 0 on success; prints a human error and exits 1
# otherwise. This lookup -- never the request -- is the whole security
# boundary: a compromised node-agent can at worst act on the operator's own
# declared engines.
_engine_compose_file() { # resource
  python3 - "$VLLM/engines.json" "$1" <<'PYEOF'
import json, os, sys

path, resource = sys.argv[1], sys.argv[2]
try:
    with open(path) as handle:
        data = json.load(handle)
except OSError:
    print("engines.json unavailable"); sys.exit(1)
except ValueError:
    print("engines.json malformed"); sys.exit(1)

if not isinstance(data, dict):
    print("engines.json malformed"); sys.exit(1)

entry = data.get(resource)
compose_file = entry.get("compose_file") if isinstance(entry, dict) else None
if not isinstance(compose_file, str) or not compose_file:
    # Never guess a compose path for a resource engines.json doesn't
    # actually declare -- same outcome whether the key is simply absent or
    # the entry is present but too broken to name a usable file.
    print("undeclared resource"); sys.exit(1)
if not os.path.isabs(compose_file):
    print("compose_file is not an absolute path"); sys.exit(1)
if not os.path.isfile(compose_file):
    print("compose file does not exist"); sys.exit(1)

print(compose_file)
PYEOF
}

_process_engine_req() {
  [ -f "$ENGINE_REQ" ] || return 0

  local parsed rc resource verb
  parsed=$(python3 - "$ENGINE_REQ" <<'EOF' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
resource = d.get("resource")
if not isinstance(resource, str) or not resource:
    sys.exit(1)
verb = d.get("verb")
print(resource)
print(verb if isinstance(verb, str) else "")
EOF
  )
  rc=$?
  rm -f "$ENGINE_REQ"   # consume first: a crash must not re-run a half-processed request

  if [ $rc -ne 0 ] || [ -z "$parsed" ]; then
    # No resource to key a result file by -- log and move on, same spirit
    # as the swap path's malformed-request handling, minus the status file.
    echo "swap-helper: unparseable engine-req.json, no result written" \
      >> "$CTL/swap.log"
    return 0
  fi

  resource=$(printf '%s' "$parsed" | sed -n 1p)
  verb=$(printf '%s' "$parsed" | sed -n 2p)

  if ! printf '%s' "$resource" | grep -qE '^[A-Za-z0-9_-]+$'; then
    # Same regex as the swap path's profile-name guard, checked BEFORE the
    # resource ever reaches a filename: an unsafe resource is refused like
    # an undeclared one, but with no result file -- naming one with it is
    # exactly the injection this guard exists to prevent.
    echo "swap-helper: engine-req.json resource fails name validation, no result written" \
      >> "$CTL/swap.log"
    return 0
  fi

  local result="$CTL/engine-status-${resource}.json"

  if [ "$verb" != up ] && [ "$verb" != down ]; then
    _write_engine_status "$resource" "$verb" 0 "invalid verb (want up or down)" "$result"
    return 0
  fi

  local compose
  compose=$(_engine_compose_file "$resource")
  if [ $? -ne 0 ]; then
    _write_engine_status "$resource" "$verb" 0 "$compose" "$result"
    return 0
  fi

  if [ "$verb" = up ]; then
    docker compose -f "$compose" up -d >> "$CTL/swap.log" 2>&1
  else
    docker compose -f "$compose" down >> "$CTL/swap.log" 2>&1
  fi
  if [ $? -eq 0 ]; then
    _write_engine_status "$resource" "$verb" 1 "" "$result"
  else
    _write_engine_status "$resource" "$verb" 0 "docker compose $verb failed (see swap.log)" "$result"
  fi
}

process_one() {
  _process_engine_req   # serialized by the same lock/loop as the swap below

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
  if _launch "$profile"; then
    write_status done "$profile" "$id" "swap launched"
    _harvest "$profile"   # guarded; never affects the swap outcome
  elif [ "$LAUNCH_BRANCH" = settings ]; then
    write_status error "$profile" "$id" "settings launch failed (see swap.log)"
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
