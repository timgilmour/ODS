# Integrating an engine with the Model Deck

A checklist for wiring a new engine into the Model Deck, with **hipfire** as
the worked example throughout: every item names the file and symbol that
satisfies it for hipfire *today*, so this document can be checked against
the code and rots loudly instead of quietly.

## Read this before item 1

Items 1–4 below are what "looks integrated" means: a declared kind, an
observable adapter, load/unload verbs, and a provenance record. **hipfire
passed all four of these and was still being actuated behind the deck's
back.** `ods/bin/ods-host-agent.py` contained zero references to the Model
Deck; the dashboard's "activate model" path rewrote `.env` and recreated the
hipfire container directly, and the deck learned about the resulting outage
only by watching the engine vanish and firing a lifecycle-restore against an
operator's own deliberate change. That is the actuation bracket this branch
built (Tasks 1–4), and it is why items 5 and 6 exist. **A reader who
completes only items 1–4 has reproduced exactly the situation this branch
had to fix.** Do not stop at item 4.

## 1. Declare the kind

Add an entry to the kind table in `app/engine_kinds.py`'s `KNOWN_KINDS`
dict, naming the engine's connection schema (which fields a declaration
must/may carry) and whether it is `remote_capable` / `local_capable`. Pair
it with an adapter class implementing the per-kind protocol
(`observe`/`unknown`/`active`/`build_client`/... — see the module
docstring for the full surface).

*hipfire:* the `"hipfire": {"connection": {"container": True}, ...}` entry
in `KNOWN_KINDS` (`app/engine_kinds.py:182`), paired with `_HipfireAdapter`
(`app/engine_kinds.py:757`) and registered in the kind-to-adapter map as
`"hipfire": _HipfireAdapter()` (`app/engine_kinds.py:1298`).

## 2. Be observable

The adapter's `observe` must report reachable / loaded / model (and
whatever else its kind's state shape needs) so `derive_status` can classify
the resource, and `unknown()` must return the exact shape `observe` returns
on failure — the record a caller with no client at all still needs to
produce for a declared-but-unreachable engine.

*hipfire:* the deck dials the container directly over HTTP on port 11435
(`_HIPFIRE_PORT = 11435`, `app/engine_kinds.py:112`), used to build the
health/stats URLs in `_HipfireAdapter.build_client`
(`app/engine_kinds.py:866,870`). `_HipfireAdapter.observe`
(`app/engine_kinds.py:777`) reads `client.status()` and, while running,
`client.stats()`; `_HipfireAdapter.unknown()` (`app/engine_kinds.py:804`)
returns the matching failure shape.

## 3. Be actuable

The deck needs a way to change the engine's state, and a way to reconcile
it back to a previously-recorded intent after a restart. Two shapes exist
in the codebase today:

- **Arbiter-eligible kinds** (lemonade, comfyui, sglang-omni) implement
  `execute_load`/`execute_unload`/`execute_free` directly on the adapter —
  see `_LemonadeAdapter.execute_unload` (`app/engine_kinds.py:445`) and
  `.execute_load` (`app/engine_kinds.py:555`) for the shape. sglang-omni's
  `_SglangOmniAdapter` (`app/engine_kinds.py:889`) implements the same pair
  (`:1151,1242`) and its `arbiter_verbs()` (`:1009`) returns
  `frozenset({"unload"})` — the deck's own idle/contention arbiter can act
  on it unprompted, which is exactly the exposure that matters for item 5
  below.
- **Human-only kinds** expose their verbs through `human_verbs()` instead
  of an arbiter verb, dispatched by `(kind, verb)` in
  `app/routers/control.py`'s `_HANDLERS` table.

*hipfire:* `_HipfireAdapter.arbiter_verbs()` returns an empty frozenset
(`app/engine_kinds.py:830-831`) — park/resume are deliberately human-only,
never automatic, per the class's own docstring: "no arbiter verb — park
stays human-only (structural omission made explicit)"
(`app/engine_kinds.py:758-759`). The verbs
are dispatched as `_hipfire_park` (`app/routers/control.py:441`) /
`_hipfire_resume` (`app/routers/control.py:453`), wired into `_HANDLERS` as
`("hipfire", "park")` / `("hipfire", "resume")` (`app/routers/control.py:478-479`).

Every kind also needs `restore(client, model)` for post-restart
reconciliation. *hipfire:* `_HipfireAdapter.restore`
(`app/engine_kinds.py:878`) calls `client.resume()` — **dispatch-only, and
deliberately records nothing.** `restore` runs when the deck is
reconciling a resource back to an *already-recorded* intent; if it also
called `intent_store.record(...)`, a restore triggered by the deck's own
reconciliation loop would re-stamp the intent as the deck's doing, silently
overwriting whatever actor (`operator`, `deck`) actually caused the prior
state. Compare `_hipfire_park`/`_hipfire_resume`, which *do* call
`deck["intent_store"].record(...)` — those are actuation, not restore, and
recording is the entire point of an actuation path.

## 4. Record provenance

Add an artifact to the provenance store with an `origin`, at least one
`watch` source, and a verification mode, so the deck can classify what
image/build is actually running and detect drift from upstream.

*hipfire:* the artifact `oci:local:ods-hipfire` in the live provenance
store (`~/ods/data/model-deck/provenance.json`) carries
`role="engine"`, one `watch` source (`id="upstream"`, tracking
`github.com/warpfront/hipfire`'s `master` against the pinned
`HIPFIRE_REF`), and `current.verification="exact"`.

⚠ **`origin.build` is hand-written prose, and nothing enforces that it
stays true.** hipfire's `origin.build` field is a paragraph describing how
the image is built — which Dockerfile, which upstream repo, which local
patch is applied in the runtime stage. As of this writing (checked live
against `~/ods/data/model-deck/provenance.json`) that field still says the
build "applies the local `finish-reason.patch`" in the runtime stage.
`extensions/services/hipfire/Dockerfile.amd:23` — the actual build the
provenance record is describing — says the opposite in its own header
comment: **"The finish-reason.patch is RETIRED. It patched cli/index.ts,
which no longer exists; the Rust rewrite carries the terminal
finish_reason rule natively."** The provenance record and the Dockerfile
it describes have been contradicting each other since that rewrite, and
nothing flagged it. **Nothing in the codebase diffs `origin.build`
against the Dockerfile it describes.** Whoever changes `Dockerfile.amd`
must remember to also edit this field by hand, and until they do, the
record describes a build step that no longer exists. This is the same
failure mode as a stale repo copy sitting next to a fetch nobody exercises:
a hand-maintained description that nothing checks rots exactly like a
stale artifact, just more quietly, because nothing fails loudly when it
does.

## 5. Declare who else may actuate it

**This is the item that was missing, and it is what this branch (Tasks
1–4) exists to fix.** Passing items 1–4 makes an engine "look integrated"
while anything outside the deck can still stop, start, recreate, or
re-pin it without the deck's knowledge — the deck then reads the resulting
absence as a death and fires lifecycle-restore against an operator's own
deliberate change.

Any surface outside the deck (a dashboard route, a CLI command, an
installer script) that tears down or recreates an engine's container must
bracket that action: announce an `expect-absence` hold before the teardown
— every teardown, including a rollback's second one — and `adopt` after it,
**if and only if the new state is proved to be serving**. Ending any other
way (failure, rollback, a crash) releases the hold and records nothing.

*hipfire:* `ods-host-agent.py`'s dashboard "activate model" path,
`_do_hipfire_activate` (`ods/bin/ods-host-agent.py:8849`), wraps its `.env`
rewrite and container recreate in `_deck_bracket(env_pre, "local/hipfire")`
(`ods/bin/ods-host-agent.py:8894`, context manager defined at
`ods/bin/ods-host-agent.py:12574`, its handle at `:12538`). The bracket calls
`POST /api/lifecycle/expect-absence/{key}` before the teardown
(`app/routers/lifecycle.py:87`), and the yielded handle's `renew()` re-arms
that hold before the rollback's second recreate. `POST
/api/lifecycle/adopt/{key}` (`app/routers/lifecycle.py:117`) fires on exit
only when the body called `commit()`, which `_do_hipfire_activate` does at
exactly one point: after `/health` returned 200 *and* the LiteLLM restart
succeeded. Otherwise the bracket DELETEs the hold and adopts nothing.

⚠ **Adopt is not "record whatever is there" — it needs proof of serving.**
The tempting shortcut is to adopt unconditionally, reasoning that adopt
records what is actually running so a rollback simply records the old
model. That is false, and it was a real defect on this branch. The rollback
path is `restore_backups()` + `_compose_recreate_hipfire()`, which returns
as soon as `docker compose up -d` does — there is no health gate, and a
cold load takes minutes. The container is therefore *up but loading*, which
is REACHABLE, so adopt's unreachable guard does not fire; the record it
writes is `state="unloaded", actor="operator"` — the strongest possible
"the operator deliberately parked this". `derive_status` then reports
`parked`, `plan_reconcile` acts only on `down` (`app/reconcile.py:31,45`),
and the deck never restores hipfire again — not after a crash, not after a
reboot. One failed activation would permanently disable the machinery built
for the 2026-08-03 26-hour outage. The route now refuses a mid-transition
adopt as well (`app/routers/lifecycle.py`, beside the unreachable 409), but
the caller must not ask in the first place: after a rollback there is
nothing to adopt.

This is provably live: the deck's `intent.json` records
`"local/hipfire": {..., "actor": "operator", ...}` — that `actor` value is
what `adopt` writes, meaning a real dashboard activation went through the
bracket, not around it.

⚠ **The watcher's existing host-agent busy gate does NOT cover this.**
`Watcher.tick` already skips a whole tick while `hostagent.lifecycle()`
reports an active operation (`app/arbiter.py:850-857`), which looks like it
would suppress the restore on its own. It does not, for a one-line reason:
the probe is gated on `real_work` (`app/arbiter.py:848`), computed from
**arbitration** actions only, and `_reconcile_pass` runs *after* that gate
— its restores never set `real_work`, so a tick whose only work is a
lifecycle restore never asks the host-agent whether it is busy. The bracket
is what covers reconciliation; the busy gate covers arbitration.

⚠ **Hold keys are node-qualified — `local/hipfire`, never bare
`"hipfire"`.** `local_key()` (`app/observe.py:55`) produces the
`"local/<resource>"` form the bracket and the intent store both use. This
does not mean every store in the system is node-qualified: the live
`intent.json` keys on `local/hipfire` and `sparky/slot0`, while
`world.tenants` keys on the bare resource name. Both shapes coexist in the
running system today — check which one a given store uses before assuming
the other.

**`_do_model_activate`'s llama-server teardown is bracketed too** (the
follow-up wave after this branch merged). llama-server is not a
hypothetical future addition to the deck: it is *already* deck-managed
today, declared under the `lemonade` kind. The declared `lemonade`
resource's `connection.container` is literally `"ods-llama-server"` (live
`~/ods/data/model-deck/nodes.json`; `app/settings.py:88`'s
`lemonade_container: str = "ods-llama-server"` is only the one-time seed
default, per `ods/extensions/services/model-deck/README.md:50`). Every
runtime strategy that stops that container — compose stop+up
(`_compose_restart_llama_server`) and inspect-and-recreate
(`_recreate_llama_server`), on both the forward and rollback paths —
is reached only from inside `_do_model_activate`'s
`with _deck_bracket(persisted_env, "local/lemonade")` block, which wraps
the handler's whole transaction try/except. `rollback_and_prove()` renews
the hold as its first act; `deck_bracket.commit()` sits adjacent to the
handler's own `committed = True`, after final runtime proof. The wiring
is pinned structurally by the AST tests in
`ods/tests/host_agent/test_deck_bracket.py` (which also record why: the
handler's realistic states cannot be constructed in a unit test, so the
placement facts are pinned instead).

This gap was a **strictly larger exposure than hipfire's**, not an
equivalent one. `_LemonadeAdapter` implements `execute_load`/
`execute_unload` (item 3 above, `app/engine_kinds.py:445,555`), so
`lemonade` is arbiter-eligible: the deck's own automatic idle-release and
contention arbiter can act on this resource unprompted (the live
`intent.json` shows `local/lemonade` with `actor: "deck"` — the deck
already idle-releases it in practice). hipfire's park is human-only — the
exposure there was operator-vs-deck, a one-off race between a person and
the reconciler. Here the deck's own arbiter could fire *while* the
activation path was yanking the container out from under it, with nothing
announcing the teardown either way.

## 6. Own its durable state honestly

If anything other than the deck writes state that the deck *also* models,
say so explicitly in this document. Duplication is survivable — undeclared
duplication is not, because a reader has no way to know two things can
disagree.

*hipfire:* `.env`'s `HIPFIRE_MODEL` / `HIPFIRE_ACTIVE` keys are durable
model-selection state written by the dashboard's activation path
(`env_txn.update({"HIPFIRE_MODEL": model_file, "HIPFIRE_ACTIVE": "true"})`,
`ods/bin/ods-host-agent.py:8898`). The deck *also* models this resource's
load state, independently, in its own intent store (`IntentStore`,
`app/intent.py:55`), recorded under the `local_key("hipfire")` key by
`_hipfire_park`/`_hipfire_resume` (`app/routers/control.py:447,464`) and
by `_deck_bracket`'s `adopt` call (item 5). These are two records of the
same fact, written by two different actors, and nothing declares that
relationship or reconciles them against each other.
The actuation bracket (item 5) keeps the deck's *record* honest about what
happened; it does **not** collapse this duplication — `.env` and the
intent store can still independently disagree about which model is
active, and nothing here notices if they do. Collapsing it — e.g. making
the deck the single writer of `HIPFIRE_MODEL`, or making `.env` a read
projection of the intent store — is deliberately out of scope for this
branch. Say this plainly for any new engine, don't silently assume it away:
if a config file, an installer script, or another service also persists
model/engine state the deck models, name the field and the writer here,
even if resolving the duplication is future work.

## 7. Engine instances (a kind the deck can create, not just declare)

INST I1 adds a second way an engine's declaration comes to exist: instead
of an operator `POST`ing a declaration for a container they already
started, the deck creates the container itself. A kind opts in with
`"instance": True` in its `KNOWN_KINDS` entry (`app/engine_kinds.py`) plus
an `instance_env` allowlist and `instance_policy` defaults; three kinds do
this today (hipfire, lemonade, comfyui). Full operator-facing detail —
what a managed entry looks like, the ordering rule each verb follows
(D-I1-1), ports, consent, and the gateway-staging caveat — lives in
`README.md`'s **Engine instances** section; this item is the route/event
surface only, in this document's own inventory style.

`app/routers/instances.py`:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/nodes/{id}/instances` | Create (`{kind, gpu_indices, env}`) — declares then ships (D-I1-1); rolled back on a failed ship. **201** |
| `DELETE` | `/api/nodes/{id}/instances/{resource}` | Remove — holds, ships, forgets declaration/intent/policy |
| `POST` | `/api/nodes/{id}/instances/{resource}/move` | Move to a new `gpu_indices` claim — ships, updates the declaration, forgets intent |

Six events, `log_event`-written from those three handlers (`_ship`'s
`fail_kind` argument for the three failure cases):
`instance-created` · `instance-create-failed` · `instance-removed` ·
`instance-remove-failed` · `instance-move-requested` ·
`instance-move-failed`.

The actuation channel itself is a **third** party, not the deck calling
docker directly: `app/node_clients.py`'s `client_for` (control ==
`"instances"`) ships the wire document to the node-agent's
`POST /v1/node/instance/{resource}` (`node-agent/instances.py`,
`node-agent/app.py`), which only validates shape and queues a file for the
host-side instances-helper — see `node-agent/README.md` for that half.
This item therefore does not repeat items 1–3 above per-kind (hipfire's
own item 1–4 citations still hold for what it means to be declared,
observable, and actuable) — it is the ADDITIONAL surface a kind gets when
it also opts into `"instance": True`.

## Fleet status (not universal — check before assuming)

This checklist describes the shape of correct integration; it does not
claim every declared engine currently satisfies every item.

- Items 1–3 (kind declared, observable, actuable): satisfied by every
  engine with a `KNOWN_KINDS` entry (lemonade, comfyui, hipfire,
  sglang-omni) — that's what having an entry and an adapter means.
- Item 4 (provenance): present but *uneven* in the live provenance store
  (`~/ods/data/model-deck/provenance.json`). hipfire's artifact
  (`oci:local:ods-hipfire`) and lemonade's (`oci:local:ods-lemonade-server`)
  each carry one `watch` source and `verification="exact"`; comfyui's
  local artifact (`oci:local:ignatberesnev/comfyui-gfx1151`) carries zero
  watch sources — declared, but with no drift detection at all — and two
  of the sparky images (`oci:sparky:aeon-7/comfyui-aeon-spark`,
  `oci:sparky:ds4-spark`) read `verification="unknown"`, worse coverage
  than hipfire had going into this investigation. Having a `KNOWN_KINDS`
  entry does not imply a complete provenance record; check the artifact.
- Item 5 (actuation-bracket): satisfied for both dashboard activation
  paths — hipfire's (`_do_hipfire_activate`) and lemonade's
  (`_do_model_activate`, bracketed in the follow-up wave; see item 5).
  Other out-of-band actuators — installer scripts (`bootstrap-upgrade.sh`
  also stops llama-server), manual `docker compose` calls, a future
  extension's own control surface — have not been audited and should be
  assumed unbracketed until checked.
- Item 6 (durable-state declaration): written here for hipfire only. No
  other engine's out-of-deck durable state has been inventoried as part of
  this branch.
- The live gate (an end-to-end run proving the bracket holds under a real
  dashboard activation against the running deck) has not been run as part
  of this branch. The `intent.json` evidence cited under item 5 is from the
  live system's prior activation, not a gate this branch executed.
- Item 7 (instances): three kinds opt in today (hipfire, lemonade,
  comfyui) — sglang-omni does not (it is remote-only). The live
  create→load→serve→remove round trip is proven by
  `livetests/test_disruptive_instances.py`, but ONLY on a box where the
  local node has been switched to `control: "instances"` and the
  node-agent + instances-helper deployed (see `node-agent/README.md`) —
  that drill skips, naming the missing prerequisite, on a fresh install.
