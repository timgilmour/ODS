#!/usr/bin/env bash
# hipfire entrypoint for ODS.
#
# hipfire's server settings (host/port/idle_timeout/default_model) are CONFIG-FILE
# keys, not env vars — the CLI reads ~/.hipfire/config.toml and ignores the
# environment. So we translate ODS's env into `hipfire config set` on every start.
# (Same class of trap as Lemonade's cached config.json, which is what kept ODS's
# own llama-server on a Vulkan fallback.)
#
# The flat key spellings below (host, port, idle_timeout, max_seq, max_tokens,
# thinking_budget, default_model) are the post-rewrite CLI's documented legacy
# aliases for the namespaced schema (serve.host, serve.idle_timeout_seconds,
# memory.max_seq, ...) — hipfire-config lib.rs carries a legacy_key per field
# and docs/CONFIG.md commits to accepting them. If a `config set` here ever
# starts failing, that contract broke: check `hipfire config list` for the
# namespaced names.
set -euo pipefail

# Defence in depth against the empty-but-defined HSA bug: ROCm treats a *defined*
# HSA_OVERRIDE_GFX_VERSION as an override request, fails to parse "", and then
# enumerates ZERO devices. compose passes it bare so it should be absent here —
# but if anything reintroduces it empty, unset it rather than go dark.
if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
    unset HSA_OVERRIDE_GFX_VERSION || true
fi

HIPFIRE_PORT_INTERNAL="${HIPFIRE_PORT_INTERNAL:-11435}"

# idle_timeout defaults to 300s, which frees VRAM and forces a cold reload on the
# next request. Behind LiteLLM that is exactly wrong — default to never idling out.
hipfire config set host 0.0.0.0                                    >/dev/null
hipfire config set port "${HIPFIRE_PORT_INTERNAL}"                 >/dev/null
hipfire config set idle_timeout "${HIPFIRE_IDLE_TIMEOUT:-0}"       >/dev/null

# DELIBERATELY NOT SET HERE: max_seq, max_tokens, thinking_budget.
#
# hipfire resolves configuration as: global user override > registry model
# policy > built-in default. A `config set` here writes the GLOBAL layer, which
# outranks the per-model policy the registry ships for every tagged SKU — so
# setting these pinned one model's tuning onto every model the box ever served.
# `_do_hipfire_activate` only rewrites HIPFIRE_MODEL/HIPFIRE_ACTIVE on a swap,
# so the values stayed behind: the 262144/32768/xhigh trio tuned for
# Qwen3.6-35B-A3B on 2026-07-20 was still being applied to Qwen3.8-27B a month
# later, silently overriding that model's own card.
#
# Verified 2026-08-18 with `hipfire config <model> list`, which prints each
# key's source. For a tagged model the registry supplies kv_cache=q8,
# kv_backend=vmm, and the full generation.* sampling block on its own; these
# three keys were the ONLY ones the global layer was stomping. Leaving them
# unset is therefore both necessary and sufficient for a model to serve under
# its own policy.
#
# If a deliberate deviation from upstream is ever needed, the Model Deck
# settings store already resolves engine -> model -> engine_model most-specific
# -wins (app/routers/settings.py:_resolve_env, live for sparky's vllm
# profiles). That is the place to put it — NOT a global `config set` here.

if [ -n "${HIPFIRE_MODEL:-}" ]; then
    hipfire config set default_model "${HIPFIRE_MODEL}" >/dev/null
    # Models live on the mounted volume; only pull if it isn't already there.
    if ! hipfire list 2>/dev/null | grep -q -- "${HIPFIRE_MODEL}"; then
        echo "hipfire: ${HIPFIRE_MODEL} not present locally; pulling..."
        hipfire pull "${HIPFIRE_MODEL}"
    fi
fi

# Fail loudly and early if the GPU isn't visible, rather than silently serving
# from a broken state.
hipfire diag || true

exec hipfire serve "0.0.0.0:${HIPFIRE_PORT_INTERNAL}"
