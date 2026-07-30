# ods-lan: manifest-driven, portable LAN gateway generation

Status: approved for planning
Date: 2026-07-30
Scope: sub-project 1 of 5 in the broader "make ods-lan hella better" effort (see Context below)

## Context

`ods-lan` (Caddy + Authelia LAN gateway, admin dashboard `ods-lan` extension) currently
exists only as hand-maintained files on one box (`~/ods/extensions/services/ods-lan/`) —
it is not tracked in the `~/projects/ODS` dev repo at all. The Caddyfile hardcodes the
box's LAN IP (`192.168.1.6`) and a static list of per-service site blocks that someone
must hand-edit (plus `setup-fw.sh`'s port list, plus the README) every time a service
is added, removed, or its tier changes. This has already caused real outages: newly
installed extensions (Immich, hipfire) silently had no LAN gateway entry until someone
noticed and hand-added one.

The end goal is a first-class, upstreamable ODS extension where:
- Any service can declare its LAN exposure tier in its own manifest.
- The gateway config (Caddyfile + firewall ports) is generated from those manifests,
  not hand-maintained.
- New/changed exposure is surfaced to the operator for approval, not silently applied.
- A dashboard settings screen lets the operator manage all of this visually.

This spec covers only the first, foundational piece: **manifest schema + generator +
approval plumbing**, scoped to get `ods-lan` itself working this way and landed in the
tracked repo. It intentionally excludes:
- The dedicated ods-lan dashboard settings page (a richer UI beyond the pending-change
  banner) — follow-on work once this plumbing exists.
- General auto-discovery UX polish beyond the initial approve/reject banner.
- Anything about `ods-proxy` (a separate, unrelated existing gateway extension already
  in the tracked repo — do not conflate the two).

## Manifest schema addition

New optional field on the existing `ods.services.v1` schema, `service.lan_tier`:

```yaml
lan_tier: none | users | admins   # default: none
```

- `none` (default): service is not exposed via ods-lan. No behavior change for any
  manifest that doesn't set this field — fully backward compatible.
- `users`: any authenticated LAN user reaches it (mirrors today's `users_site` Caddy
  snippet — forward_auth only, no group check).
- `admins`: authenticated AND member of the Authelia `admins` group (today's
  `admin_site` snippet — forward_auth + group check + 403 for non-admins).

A service with `lan_tier != none` MUST have a resolvable port (`external_port_default`
or `port`, same resolution `lib/service-registry.sh` already does) or the generator
skips it with a warning (not a hard failure — matches the registry's existing
lenient-skip pattern for malformed manifests).

Second new optional field, `service.lan_extra_directives`: a raw string of additional
Caddy directives injected into that service's generated site block, for the small
number of services that need more than a bare `reverse_proxy`. Today's one known case:
Model Deck's `header_up X-Deck-Proxy-Key {env.MODEL_DECK_PROXY_KEY}` (users-tier read
access plus a proxy-shared secret gating mutations). Kept as an escape hatch, not a
general templating system — if more than a couple of services ever need this, that's
a signal to revisit, not to keep growing this field's expressiveness.

`config/network-exposure-policy.json` stays as-is (risk/auth_required/notes
documentation, already required for anything with an external port) — `lan_tier` is
the new *actionable* counterpart, not a replacement.

## LAN IP source

`installers/phases/06-directories.sh` already detects and can populate `HOST_LAN_IP`
(`ip -4 route get 1.1.1.1` → `src` field, falling back to `hostname -I` / `ifconfig`),
but only writes it when `BIND_ADDRESS=0.0.0.0` was set (for openclaw's allowed-origins
use case). Change: populate `HOST_LAN_IP` unconditionally at install time (still
operator-overridable by hand-editing `.env`, same as today). ods-lan's generator reads
`HOST_LAN_IP` from `.env` as its one source of truth for the bind address, replacing
every hardcoded `192.168.1.6` in the current Caddyfile.

If `HOST_LAN_IP` is empty (e.g. pre-existing `.env` from before this change, detection
failed, multi-NIC box where auto-detection picked wrong), the generator refuses to
produce a Caddyfile and reports the missing value — no silent fallback to a guessed or
loopback address for a LAN-facing TLS gateway.

## Generator

New script `scripts/generate-lan-gateway.py`, following the existing convention set by
`scripts/generate-extensions-catalog.py` (manifest scanning) and
`scripts/validate-generated-configs.py` / `config/generated-config-contracts.json`
(validation harness):

- Scans `extensions/services/*/manifest.yaml` (built-in) and
  `data/user-extensions/*/manifest.yaml` (dashboard-installed), same two directories
  `lib/service-registry.sh` already scans.
- Filters to `lan_tier in (users, admins)` with a resolvable port.
- Renders two artifacts to a scratch directory (never touches live files directly):
  - `Caddyfile` — global `servers` block (one `listener_wrappers` entry per exposed
    port, mirroring today's structure) + one site block per service, using
    `import users_site <port>` / `import admin_site <port>` plus any
    `lan_extra_directives`, substituting `HOST_LAN_IP` throughout. The
    onboarding (`:8081`) and Authelia-portal (`:9091`) blocks are NOT
    manifest-driven — they're fixed infrastructure of the gateway itself, kept as a
    static template header/footer the generator always emits.
  - `setup-fw.sh` — regenerated `PORTS=(...)` array, one entry per exposed service
    port plus the fixed `8081`/`9091`.
- Validates its own output before returning success: pipes the generated Caddyfile
  through `caddy validate` (via a throwaway container or the running `ods-lan-caddy`
  container, matching how we hand-validated it in this session) and fails loudly if
  invalid — a generator that produces broken config is worse than no generator.

## Lifecycle + approval flow

dashboard-api already has install/enable/disable hooks per extension (`_run_hook` /
lifecycle hook map). Add a post-hook step: after any install/enable/disable, run the
generator into a scratch path and diff both generated files against the live
`extensions/services/ods-lan/{Caddyfile,setup-fw.sh}`. If they differ, write a pending-
change record (new `/api/lan-gateway/pending` endpoint: which services were
added/removed/changed tier, plus the actual diff text) instead of applying anything.

`Extensions.jsx` gets a banner when a pending record exists: "Gateway config has
pending changes (N services) — Review". Clicking it shows the diff (services added,
removed, or re-tiered) with Approve / Reject actions:
- **Approve**: dashboard-api copies the generated Caddyfile into place, restarts the
  `ods-lan-caddy` container (accepted downtime, per the "keep admin API disabled"
  decision), and — since firewall changes need `sudo` dashboard-api doesn't have —
  surfaces the exact `setup-fw.sh` command for the operator to run, same UX gap we hit
  live this session. Clears the pending record once the Caddyfile side is applied.
- **Reject**: discards the scratch-generated files, clears the pending record, changes
  nothing. (The underlying manifest's `lan_tier` is unchanged — rejecting just means
  "don't apply this to the gateway yet"; the banner will reappear on the next
  lifecycle event since the diff still exists. Acceptable for this increment — no
  persistent "ignore this diff" state yet.)

A manual trigger (`ods lan sync`, or a button once the settings screen exists) that
runs the same generate → diff → pending-record path on demand covers drift — e.g. an
operator hand-edits the live Caddyfile, or a manifest's `lan_tier` changes outside the
install/enable/disable path.

## Migration of the live box

Backfill `lan_tier` onto the ~14 services currently wired into the hand-maintained
Caddyfile, to reproduce current behavior exactly:

| Service | lan_tier | notes |
|---|---|---|
| open-webui | users | |
| comfyui | users | |
| model-deck | users | + `lan_extra_directives` for the proxy-key header |
| dashboard | admins | |
| perplexica | admins | |
| token-spy | admins | |
| langfuse | admins | |
| n8n | admins | |
| searxng | admins | |
| unsloth-studio | admins | |
| hermes-proxy | admins | (`hermes` itself stays `lan_tier: none` — internal only) |
| immich | admins | |
| hipfire | admins | |

After backfilling, run the generator and diff its output against the current live
Caddyfile/`setup-fw.sh` — the spec's acceptance bar for this migration is that diff
being empty (or explainable, e.g. comment-only differences) before cutover.

## Porting into the tracked repo

`ods-lan`'s existing files (`compose.yaml`, `manifest.yaml`, `Caddyfile` — now
generated instead of hand-maintained —, `README.md`, `setup-fw.sh` — now generated —,
`smoke.sh`, `userctl.sh`, `gen-secrets.sh`) move from `~/ods/extensions/services/ods-lan/`
into `~/projects/ODS/ods/extensions/services/ods-lan/`, becoming a real tracked
extension for the first time. `manifest.yaml`'s `host_network: true` /
`health_host: 192.168.1.6` (added earlier this session) gets its hardcoded IP replaced
with the same `HOST_LAN_IP`-driven mechanism.

## Testing

- New contract test alongside `tests/contracts/test-network-exposure-contracts.py`:
  every manifest with `lan_tier != none` has a resolvable, non-conflicting port
  (two services can't claim the same port), and appears in
  `network-exposure-policy.json` (existing rule already enforces the latter for
  anything with an external port).
- `scripts/validate-generated-configs.py` gets a new check: generate the Caddyfile,
  run `caddy validate` against it.
- `ods-lan/smoke.sh` continues to be the live-box integration check (unchanged
  role, generated ports flow into it same as manual ones do today).

## Explicitly out of scope for this spec

- The dedicated ods-lan settings page in the dashboard (richer than the pending-change
  banner: user management UI, live tier editing, cert/onboarding management surfaced
  visually). Follow-on sub-project once this plumbing exists to build on.
- Hot-reload via Caddy's admin API (rejected this session — stays `admin off`).
- Multi-NIC / multiple simultaneous LAN segments.
- A persistent "ignore this diff" / snooze state for rejected pending changes.
