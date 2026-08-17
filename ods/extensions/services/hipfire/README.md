# hipfire — RDNA-native LLM inference engine

An AMD-only, ROCm-direct inference engine ([upstream](https://github.com/Kaden-Schutt/hipfire))
that runs **alongside** `llama-server`/Lemonade rather than replacing it. LiteLLM routes text to
hipfire; Lemonade keeps serving ODS Talk vision and acts as the fallback.

Side-by-side is the blessed ODS pattern for a second engine — see `docs/VLLM-SETUP.md:42`.

## Why

Measured on this host (2× Radeon AI PRO R9700, gfx1201, ROCm 7.2.4), same weights
(Qwen3.6-35B-A3B), same prompt, same harness, 256 tokens:

| engine | GPUs | throughput |
|---|---|---|
| llama.cpp / Lemonade (Q4_K_M) | **2** | 64.7 tok/s |
| **hipfire (MQ4)** | **1** | **104.3 tok/s** |

~1.61× faster on half the hardware, which is what frees GPU1 to keep Lemonade resident.

## GPU split

hipfire owns **GPU0** (`HIPFIRE_GPU_INDEX=0`); Lemonade is pinned to **GPU1**
(`LLAMA_SERVER_GPU_INDICES=1`). This is required, not cosmetic: the 35B MQ4 needs ~30 GB and
must be resident on a *single* card, because hipfire has **no pipeline-parallel on master**
(upstream issue #58 — `--tp N` is expert-parallel for MoE routing, not a model split).

## The model is NOT interchangeable with ODS's GGUFs

hipfire uses its own MQ4 format, not GGUF. `ods model swap` and the dashboard's GGUF download
endpoints do **not** manage these files; the dashboard's *model activation* path does (it writes
`HIPFIRE_MODEL`, recreates the container, and lets the entrypoint pull).

One trap still stands, one has expired:

1. ~~**`hipfire pull qwen3.6:35b-a3b` 404s.**~~ **Fixed upstream.** As of the `8510ca5f` pin,
   `hipfire-models/qwen3.6-35b-a3b` is published with all seven SKUs (`.mq2` … `.mq6`, `.mfp4`,
   plus the `.mtp` sidecar), and `hipfire-models/qwen3.8-27b` carries `.mq4` + `.mq4r`. Pulling by
   tag or by registry filename works. Verified 2026-08-17 against the HF tree API.
2. **Converting an existing GGUF may still not work for MoE.** hipfire's GGUF→MQ4 path had no
   branch for 3-D MoE expert tensors and dumped all 256 experts/layer to F16: 7.1 % quantized,
   **66 GB out of a 22 GB input**, unloadable. Only the *safetensors* path splits and quantizes
   experts. Not re-verified against the post-restructure quantizer — pull the published SKU
   instead of converting. (Related: `run_gguf_pipeline` still maps only `"qwen3moe" => 6`, so a
   `qwen35moe` GGUF is misidentified. Fix exists locally at `~/projects/hipfire` `e6fb7ef9`,
   never upstreamed.)

### Model names must be registry-resolvable

`HIPFIRE_MODEL` is a **filename from the engine's bundled registry**, e.g. `qwen3.6-35b-a3b.mq4p`
or `qwen3.8-27b.mq4`. `hipfire list` prints the resolved tag in parentheses beside each local
file; **a file listed with no tag is not registry-resolvable** and will not receive its tag policy
(KV backend, native context, max-output allowance). Hand-built artifacts fall in this bucket — the
box that first ran this extension served `qwen36-35b-a3b.mq4`, a locally-quantized file whose name
matches nothing upstream, which is why the fresh-install default used to crashloop.

## Version pin

`HIPFIRE_REF` is pinned to a **master commit, deliberately not the v0.2.1 tag**: v0.2.1's
quantizer *silently* produces the broken 66 GB model above, and the gfx1201 kernel work landed
after the tag. Bump it consciously — **in `.env` too, whose pin beats the compose default**.

Bumped 2026-08-16 from `5d3683a7` (2026-07-11) to `8510ca5f` (2026-08-15), ~1,539 commits.
The repo moved to `warpfront/hipfire` (old URL redirects). Upstream restructured in between:

- The Bun/TS CLI (`cli/index.ts`) was rewritten as a Rust binary (`hipfire-cli`); the
  runtime image no longer contains Bun.
- Kernels are JIT-compiled by hipcc on first use (embedded sources) and cached in the
  mounted `data/hipfire/kernels` volume; the build-time "compile 528 kernels + count gate"
  stage is gone. Shipped-blob precompiles were rejected on purpose: their `toolchain_id=""`
  packaging hashes only validate on hipcc-free runtimes, and ours ships hipcc for JIT.
- **`finish-reason.patch` is retired.** It patched `cli/index.ts`, which no longer exists;
  the Rust CLI carries the terminal-finish_reason rule natively (its test suite asserts
  `finish_reason == "tool_calls"` on emitted tool calls). Verified against the live API at
  deploy time rather than assumed.
- The tool-call grammar that deadlocked omp (2026-08-08) now **defaults OFF** for plain
  Qwen3.5/3.6 models upstream; the native-XML path is first-class with matching history
  rendering. Our explicit `HIPFIRE_QWEN35_GRAMMAR=0` is kept and now agrees with upstream.
- Config keys went namespaced (`serve.host`, `memory.max_seq`, ...) but the entrypoint's
  flat spellings are documented legacy aliases and keep working.

## Enable / disable

```bash
./install-core.sh --hipfire      # or --no-hipfire (default)
```
Enable/disable is a file rename (`compose.yaml` ↔ `compose.yaml.disabled`) driven by
`_sync_extension_compose` in `installers/phases/03-features.sh`. Never hand-edit `.compose-flags`
— it is a cache the resolver regenerates.

## Model control (dashboard)

hipfire models are catalog entries in `config/model-library.json` with `"engine": "hipfire"`
and a `model_file` instead of a GGUF. They appear on the dashboard's Models page whenever
`ENABLE_HIPFIRE=true` (never as a download — hipfire pulls its own weights on container start).

Activating one drives the normal path (`POST /api/models/{id}/load` → host-agent
`/v1/model/activate`), which pins `HIPFIRE_MODEL` in `.env`, recreates the container,
health-gates until the model is resident, and re-renders `config/litellm/lemonade.yaml` so
`default` routes to hipfire. Activating a GGUF flips `default` back to llama-server/Lemonade.
Either way both engines stay reachable via their named LiteLLM routes (`hipfire` / `lemonade`)
— the routing is rendered from the template by `scripts/render-runtime-configs.py`, so a model
swap can no longer wipe it. State keys: `HIPFIRE_MODEL` (what hipfire serves) and
`HIPFIRE_ACTIVE` (whether it owns the `default` route).

## Notes

- No `/v1/embeddings` (ODS runs a separate TEI service — unaffected) and **no auth**. Safe only
  because the host port stays loopback-bound via `BIND_ADDRESS`.
- `HIPFIRE_IDLE_TIMEOUT=0` on purpose: upstream's 300 s default evicts the model and forces a
  30 s–2 min cold reload on the next request, which is wrong behind a router.
- 69 of 528 gfx1201 kernels fail to compile upstream (all `attention_dflash_wmma_*` spec-decode
  variants). This reproduces on the bare host toolchain and does not affect serving; the
  Dockerfile gates on a kernel **count**, not on the compile script's exit status.
