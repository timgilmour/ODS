#!/bin/sh
# Lemonade config-sync wrapper.
#
# Lemonade reads its settings from /root/.cache/lemonade/config.json. That file is
# written on Lemonade's first start from env — but on subsequent starts Lemonade
# ignores the env vars and trusts whatever is already in config.json. The cached
# config wins, so changing a value in docker-compose has no effect on an existing
# container. This wrapper reconciles config.json with env before Lemonade boots.
#
# Synced keys:
#
#   ctx_size          <- LEMONADE_CTX_SIZE
#       Qwen3.6-35B-A3B (native 256k context) was being served at 64k on strix-halo,
#       a 4x under-utilization with 7% of the memory budget in use.
#
#   llamacpp.rocm_bin <- LEMONADE_LLAMACPP_ROCM_BIN
#       Without this, rocm_bin stays "builtin" and Lemonade tries to DOWNLOAD a
#       llama-server matching the gfx arch it detects. Its arch detection maps GPU
#       marketing names to gfx targets and does not know the Radeon AI PRO R9700, so
#       on this host it resolved the *iGPU* and fetched
#           llama-b1231-ubuntu-rocm-gfx1036-x64.zip  -> HTTP 404
#       and inference failed to start. Pointing rocm_bin at the binary we build in
#       Dockerfile.amd (correct gfx target, self-contained ROCm stack) avoids the
#       download and the arch guess entirely.
#
#   llamacpp.backend  <- LEMONADE_LLAMACPP
#       "auto" makes Lemonade probe for ROCm; the same name-based arch detection
#       fails on unrecognised cards and it silently falls back to the Vulkan build
#       (radv). Forcing "rocm" keeps it on the ROCm path.
#
#   extra_models_dir  <- LEMONADE_EXTRA_MODELS_DIR
#   llamacpp.args     <- LEMONADE_LLAMACPP_ARGS
#       Both of these used to arrive as CLI flags (--extra-models-dir, --llamacpp-args).
#       v10.10.0 removed the `lemonade-server` CLI entirely (see the exec note at the
#       bottom): `lemond` accepts only --host/--port, and everything else must come
#       from config.json. They are synced here rather than dropped because config.json
#       PERSISTS on the lemonade-recipe volume — so on this box the old CLI values are
#       already baked in and losing the flags would look harmless, while a FRESH volume
#       would silently come up with no /models directory and no llama.cpp args at all.
#
# Idempotent: only writes when a value actually differs. If config.json does not yet
# exist, it is seeded from Lemonade's shipped defaults so that even a FIRST start on
# a fresh volume gets the right binary — otherwise the first run would attempt the
# bad download before we ever got a chance to fix the file.
set -e

# Defensive: ROCm's HSA runtime checks whether HSA_OVERRIDE_GFX_VERSION is *defined*,
# not whether it is non-empty. Defined-but-empty makes it try to parse "" as a gfx
# version, fail, and enumerate ZERO devices — inference then silently drops to CPU or
# Vulkan. Compose is supposed to omit the variable entirely (see docker-compose.amd.yml),
# but scrub it here too: a single stray "VAR=${VAR:-}" anywhere in the compose merge
# chain is enough to reintroduce it, and the failure mode is quiet.
if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then unset HSA_OVERRIDE_GFX_VERSION || true; fi
if [ -z "${HSA_XNACK:-}" ]; then unset HSA_XNACK || true; fi

CONFIG=/root/.cache/lemonade/config.json
DEFAULTS=/opt/lemonade/resources/defaults.json

if [ ! -f "$CONFIG" ] && [ -f "$DEFAULTS" ]; then
    mkdir -p "$(dirname "$CONFIG")"
    cp "$DEFAULTS" "$CONFIG"
    echo "[lemonade-entrypoint] seeded config.json from defaults.json" >&2
fi

if [ -f "$CONFIG" ]; then
    python3 - <<'PYEOF'
import json, os, sys

path = "/root/.cache/lemonade/config.json"
try:
    with open(path) as f:
        cfg = json.load(f)
except (OSError, ValueError) as e:
    print(f"[lemonade-entrypoint] cannot read config.json ({e}); leaving it alone", flush=True)
    sys.exit(0)

changed = []

ctx = os.environ.get("LEMONADE_CTX_SIZE", "")
if ctx.isdigit() and int(ctx) > 0 and cfg.get("ctx_size") != int(ctx):
    changed.append(f"ctx_size: {cfg.get('ctx_size')} -> {int(ctx)}")
    cfg["ctx_size"] = int(ctx)

for env_var, key in (("LEMONADE_LLAMACPP_ROCM_BIN", "rocm_bin"),
                     ("LEMONADE_LLAMACPP", "backend"),
                     ("LEMONADE_LLAMACPP_ARGS", "args")):
    want = os.environ.get(env_var, "")
    if not want:
        continue
    section = cfg.setdefault("llamacpp", {})
    if section.get(key) != want:
        changed.append(f"llamacpp.{key}: {section.get(key)} -> {want}")
        section[key] = want

# Top-level scalar, same shape as ctx_size. Was --extra-models-dir before v10.10.0.
models_dir = os.environ.get("LEMONADE_EXTRA_MODELS_DIR", "")
if models_dir and cfg.get("extra_models_dir") != models_dir:
    changed.append(f"extra_models_dir: {cfg.get('extra_models_dir')} -> {models_dir}")
    cfg["extra_models_dir"] = models_dir

# no_fetch_executables <- LEMONADE_NO_FETCH_EXECUTABLES  (v10.10.0+)
#     v10.10.0 added managed ROCm "channels" (rocm_channel: stable) and will DOWNLOAD a
#     TheRock ROCm runtime for whatever gfx target its name-based detection picks. On
#     this host that detection resolves the *iGPU* — observed live fetching 3.4 GB of
#     "TheRock ROCm 7.13.0 for gfx1036" on a box whose dGPU is gfx1201. That is the same
#     arch-detection bug rocm_bin was set to avoid; in v10.2.0 it 404'd and failed loudly,
#     in v10.10.0 it succeeds and installs the wrong stack. Blocking the fetch keeps
#     Lemonade on the self-contained ROCm build we ship at /opt/llama-custom.
no_fetch = os.environ.get("LEMONADE_NO_FETCH_EXECUTABLES", "")
if no_fetch:
    want_bool = no_fetch.strip().lower() in ("1", "true", "yes", "on")
    if cfg.get("no_fetch_executables") != want_bool:
        changed.append(f"no_fetch_executables: {cfg.get('no_fetch_executables')} -> {want_bool}")
        cfg["no_fetch_executables"] = want_bool

if not changed:
    sys.exit(0)

for line in changed:
    print(f"[lemonade-entrypoint] updating {line}", flush=True)

with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF
fi

# v10.10.0 REMOVED the `lemonade-server` binary. v10.2.0 shipped three executables in
# /opt/lemonade (lemonade, lemonade-server, lemond); v10.10.0 ships only `lemonade` and
# `lemond`, and the image's own CMD became ["./lemond","--host","0.0.0.0"]. `lemond`
# takes ONLY --host/--port (plus an optional cache_dir positional) — every other former
# CLI flag now has to live in config.json, which is why this wrapper grew two more
# synced keys above. Verified against the real images: v10.10.0 reads a config_version:1
# file written by v10.2.0 and migrates it in place to version 2, preserving rocm_bin,
# backend, args, prefer_system and extra_models_dir.
# ⚠ That migration is one-way as far as rollback is concerned — back up config.json
# before changing LEMONADE_SERVER_IMAGE.
exec /opt/lemonade/lemond "$@"
