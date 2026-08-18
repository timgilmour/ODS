# deck-gate — Model Deck browser-gate harness

Spec: `~/notes/designs/2026-08-17-model-deck-browser-gate-harness-design.md`

`deck-gate` runs headless Chrome (`playwright-core`, `channel: "chrome"` — the system
`google-chrome`, no vendored browser download) against the Model Deck UI and asserts real
DOM state after real clicks. It closes the **dispatch gap**: nothing else in this repo can
reach a component. The UI package has no component-test harness by convention (logic is
extracted into pure `ui/src/model/` modules and vitest-covered; components render and are
otherwise unverified) — mutating a handler back to a broken form has passed the full vitest
suite before, more than once. `deck-gate` is what actually clicks the button.

    ./deck-gate                # tier 1, fixture gate — DEFAULT, deterministic, runs anywhere
    ./deck-gate --live         # tier 2, fidelity gate — READ-ONLY, needs a reachable deck
    ./deck-gate --capture      # re-record the fixture's vocabulary from a live deck
    ./deck-gate -k engines     # selection passthrough (matches gate `name`, e.g. e1-engines)
    ./deck-gate --report-dir X # override the report directory (see "Where reports go" below)

There is no `--fixture` flag — the fixture tier is what runs when you type nothing else.
A peer session once documented `--fixture` as if it existed; it does not. If you see it
written down somewhere, that somewhere drifted.

Exit codes match `deck-drill` exactly: **0** pass, **1** one or more checks failed, **2**
refused before any check ran (Chrome missing, `ui/dist` stale or absent, deck unreachable
on `--live`, a `-k` pattern that matched nothing, a duplicate check name across gates).
Refused is not failed — nothing about the UI was found to be wrong, the run just couldn't
happen.

## The two tiers

**Tier 1 (fixture, default)** serves the **built bundle** (`ui/dist`) statically from the
same origin as a stub API (`stub-server.mjs`), rather than `npm run dev` behind a proxy —
production's container serves `dist/` and `/api` on one origin, so this is the artifact
that actually ships, and it removes the dev-server transform and its proxy as failure
modes. `npm run build` is a precondition; `deck-gate` refuses if `ui/dist` doesn't exist or
is older than `ui/src`, `ui/index.html`, `ui/vite.config.ts`, or `ui/public`. It is fully
deterministic and needs nothing running — no deck, no network, no GPU. This is where every
E1 gate item lives (`smoke.gate.mjs`, `e1-engines.gate.mjs`, `e1-board.gate.mjs`).

**Tier 2 (fidelity, `--live`)** is strictly **read-only** against a real deck and proves
the fixture tier's fixtures haven't quietly rotted. `readLive` (`capture.mjs`) re-reads
exactly five routes — `/api/state`, `/api/engine-kinds`, `/api/nodes`, `/api/events?n=500`,
and `/openapi.json` — extracts key sets (`shape`) and the observed enum-token inventory
(`tokens`), and checks that everything **live** now shows is already **known** to the
committed `fixtures/e1-seeded-triple/vocabulary.json` — one-directionally (see below). It
never diffs values (`/api/state` churns constantly — VRAM, queue depths,
`last_healthy_ts` — and a gate that reddens because VRAM moved is a gate people stop
reading), and it never writes anything. `fidelity.gate.mjs`'s `compare()` is a pure
function with no I/O, unit-tested with no deck required; `run()` is the only impure part.

Two things follow from that route list that are easy to miss, and matter more than the list
itself:

- **`/api/sets` and `/api/storage/state` are NOT re-read by tier 2 at all**, even though both
  are heavily pinned in the tier-1 fixtures (the cold-GGUF optgroup, the saved-set
  round-trip). Nothing checks those fixtures against reality — that is exactly the
  fixture-rot the two-tier design exists to prevent, and it is currently a real gap, not a
  covered one.
- **`/openapi.json` makes up roughly 95% of the committed shape fixture** and is otherwise
  unmentioned by name anywhere in this doc. One consequence follows directly: **adding any
  backend endpoint will redden tier 2** (a new path in the live schema that the committed
  fixture has never seen) **until `--capture` is re-run.** That is defensible behaviour —
  reality moved, the fixture hasn't caught up — but it is the most likely way this gate
  first goes red in ordinary use, and an operator hitting it cold needs to know that's what
  happened rather than suspect a real regression.

Why two tiers at all, in one sentence: a fixture-only harness is blind to the bug class
that has recurred five times on this project — a value copied from the mockup's vocabulary
instead of read out of the backend. A fixture can't catch that, because the fixture is
authored by the same person holding the wrong vocabulary. Tier 2's fixtures are *captured
from the live deck*, which is the only shape where they can't rot into the same trap — and
it makes a red result diagnosable: tier 1 red means the UI broke, tier 2 red means reality
moved.

## The stub NEVER derives state

`stub-server.mjs` is not a mock backend. **A scenario is a scripted sequence of responses**
per route; the stub's only intelligence is handing back the next one and recording which
requests arrived. It never accepts a POST and mutates some in-memory model of the world to
make a later GET agree with it.

This is deliberate, not a shortcut. A stub that derives state is reimplementing backend
logic, and a wrong stub makes the gate confidently lie — exactly the same failure as a
fixture authored out of the mockup's vocabulary, one level up. It would also be untested
code asserting on tested code. Where a flow needs an "after" state (add an engine, see a
4th row), the scenario pins that after-state explicitly, captured from a live deck or
hand-authored and pinned by tier 2 — the stub still just serves it as the next scripted
response on that route.

This buys two things beyond safety:

- The gate can assert **what the UI actually dispatched** (the request the stub recorded)
  — precisely the dispatch gap this harness exists to close, not just what got rendered.
- A scenario can express states the live box may not be in right now, and that would
  otherwise require *breaking production* to reach (a node down, `/api/engine-kinds`
  returning 500) — this is where the fixture tier earns its keep in later waves.

**When a route's scripted sequence runs out, the stub answers 599 and names the route.** It
never falls back to a default or re-serves a stale response. A stub that quietly kept
answering after the script ran dry would turn "the UI stopped making a request" into a
green gate — the exhaustion has to be as loud as any other failure.

### `repeat: true` — the one relief valve, and its one rule

Some routes are legitimately constant for the whole run (a route the UI polls, a route a
component re-fetches on every mount). Marking that route's **last** scenario entry
`"repeat": true` makes the stub keep re-serving it forever once every prior entry has been
consumed, instead of 599ing. `repeat` on any entry that is **not** the last one is refused
at startup — a non-terminal `repeat` would mean the stub never advances past it, silently
stranding every entry after it, so the harness catches that at scenario-load time rather
than at some confusing runtime symptom three items later.

## Timer-polled routes can only carry constant responses (R12)

`App.tsx`'s `POLL_MS = 3000` polling loop re-fetches `/api/state` and `/api/storage/state`
on a plain interval, unconditionally, regardless of what the operator clicked. `/api/facts`
and `/api/facts/drift` are one step removed from that: App only bumps a `refreshTrigger`
counter unconditionally on every tick; the actual `getFacts()`/`getFactsDrift()` fetch is
`ModelDetailDrawer`'s own effect, keyed on that counter (~:150-169) — so it fires on that
same clock, but only while the drawer happens to be mounted. `/api/nodes` and `/api/sets`
are different again: they only refetch inside an explicit `afterMutate` / on-mount reload
path, i.e. causally, in response to an action.

**A scripted state transition on a polled route advances on the clock, not the click.** If
you script `/api/state` to return a 3-row policy first and a 4-row policy second, hoping to
prove an Add dispatched, the transition will flip to the 4-row payload ~3 seconds after page
load whether or not the POST ever fired — because the poll ticks regardless. The check would
pass identically whether the feature works or is completely broken. This is not a hypothetical:
it happened during Task 7 (item 5) and was caught by review, not by the harness catching
itself.

The rule this forces: **assert transitions only on causal routes.** For anything the UI only
learns via `/api/state` (item 4's board-card re-derivation, for instance — see "Known
limitations" below), assert the **dispatched POST** and any **causally-refetched** evidence
instead (item 4 asserts the POST body and the new row in the engines list, fed by
`/api/nodes`, not a board card fed by `/api/state`). A polled route in a scenario carries one
entry with `repeat: true` and nothing else.

## How to add a gate item

1. **Read the component first.** Write a selector table into the gate file's header comment
   before writing any assertion — see `e1-engines.gate.mjs` and `e1-board.gate.mjs`'s own
   header comments for the format (verified against `NodesView.tsx` / `SetBuilder.tsx` /
   `PlacementActions.tsx` / `ResourcePanel.tsx` / `ModelDetailDrawer.tsx` at the commit that
   wrote them). Inventing a selector reproduces the exact mockup-vocabulary defect this
   harness exists to catch — it just moves the guess from a fixture into a selector string.
2. **Write the check**, positive form, against real DOM.
3. **Run it green** with `./deck-gate -k <name>`.
4. **Prove it RED.** Mutate the component so the item should genuinely fail, re-run the
   gate, and let it write a real report showing the FAIL row. Revert the mutation. Archive
   the RED report beside the PASS run in the evidence directory (see below) — it is the
   proof, not a step you can skip because "the code obviously does this."
5. Only then is the item done.

**Why this is a hard gate, not a nicety:** a gate item nobody has proven red is a gate that
is always green, indistinguishable from a check that silently never runs at all. This repo
has concrete instances of exactly that failure — 436 UI tests stayed green while a fix's own
handler was mutated back to its broken form; two headline E1 checklist items would have
passed *vacuously* because a fixture crash made the assertion a dead tick before this
harness existed.

**Proving RED is itself something that can be done wrong**, and this branch caught four
separate cases where the first attempted mutation proved nothing, none of them hypothetical:

- **A mutation that doesn't change the value under assertion.** The plan's own suggested
  item-1 mutation was "render `engine.kind` in place of `engine.resource`" — but in the
  seeded fixture every engine's `resource` happens to equal its `kind`, so that mutation
  changes nothing the check can see. The fix was a different field, not a weaker check.
- **A mutation that removes the thing the check counts, so a zero-count check stays green
  for the wrong reason.** An isolation check asserted "arming one row's Forget doesn't arm
  a sibling's" by counting confirm captions; the first mutation deleted the whole
  `ArmedButton` component, which removed **every** caption — the check's "no unexpected
  caption" condition held vacuously, proving nothing about isolation at all.
- **A mutation that reddens only half of a two-part claim.** Item 9 asserts both *which*
  cards render (membership) and the *order* they render in. A `.slice(0,1)` mutation only
  broke membership — the order half of the same check stayed green throughout, so the first
  RED run "proved" a check that was only half-tested. The order half needed its own
  mutation (reversing the sort comparator) with all three cards still present.
- **A uniform mutation a symmetric check can't see.** Item 15 asserts two GPU-1 cards read
  the *same* shared-GPU meter total. Halving GPU 1's declared capacity uniformly moved
  *both* cards' numbers together — the check (which compares the two cards to each other,
  not to a hardcoded value) never noticed. An asymmetric change (shrink one card's engine's
  share only) reddened it cleanly.

Per `defaults-that-hide-bugs`: a fixture (or mutation) left at a value that coincides with
the right answer cannot detect the defect it's meant to catch. When writing or mutating a
fixture, deliberately move it away from whatever value would pass by coincidence.

## No vacuous negations

`isTrulyDisabled(page, selector)` (`lib/dom.mjs`) returns `false` for a selector that
matches **nothing**, same as it does for an element that genuinely isn't disabled — because
`document.querySelector` returning `null` and an element with no `:disabled` match are
indistinguishable at that layer. That means `!(await isTrulyDisabled(...))` — "assert this
is NOT disabled" — passes silently if the selector is simply wrong. This exact vacuous-
negation shape was found and fixed **three separate times** on this branch (E1 items 6, 7,
and 11's absence check).

The house rule: **assert positive forms.** Where you genuinely need to assert an element's
*absence* (a control that must not render for some state), pair it with a positive assertion
that its **container** exists — so a broken selector for the container fails loudly instead
of making the absence check pass by finding nothing at all.

Playwright's `:has-text()` is a **substring** match, not exact — `:has-text("Load")` matches
a button literally labelled `Unload` (`"Unload".includes("Load")` is true). This bit a gate
on this branch during Task 9's own drafting (caught and fixed before commit), and again on a
selector that only stopped being unique once an ArmedButton with "Park" as a substring of its
own label (`"Force park"`) rendered alongside the plain button it was checking (T9, review fix
wave — `e1-board.gate.mjs`'s Force-park sequence). Prefer `:text-is()` for exact text, and
where a selector's uniqueness isn't obvious on sight, call `assertUnique` so a collision
throws loudly instead of silently matching the wrong element or an unintended extra one.
`assertUnique` itself is `makeAssertUnique(gateName)` (`lib/dom.mjs`), imported and bound to
the calling gate's own name — `e1-board.gate.mjs` and `e1-engines.gate.mjs` both do
`const assertUnique = makeAssertUnique("<gate name>");` near the top of the file. It is not
a bare importable `assertUnique`; a third gate should hoist through `makeAssertUnique`, not
copy the closure body a third time.

## `--capture` unions, it never silently shrinks (R17)

`./deck-gate --capture` re-reads a live deck's shape and vocabulary and merges it into
`fixtures/e1-seeded-triple/vocabulary.json`. As of the R17 fix, this is a **union** with
whatever is already committed — every shape key and every token, once known, survives a
routine re-capture, because `unionVocabulary` (`lib/vocab-merge.mjs`) only ever adds. A
single live snapshot only ever shows the vocabulary the box happens to be exercising *at
that instant*: a healthy box never shows `status: "unreachable"`, a quiet event tail can
easily miss `apply-vetoed`. The old raw-overwrite behavior meant a careless re-capture from
a healthy, quiet box could silently drop a hand-verified token family that took real work to
populate — a comment saying "hand-verify before committing" is not a guard against that; this
branch hit exactly that risk twice (the hand-supplemented `unreachable`/etc. status set, and
the 58-kind `eventKind` seed) before the union fix landed.

Retiring a genuinely dead token or key requires the explicit `--allow-shrink` flag, which
restores the old raw-overwrite behavior for that one run — and even then, the delta between
what was committed and what the raw snapshot would have kept is printed, so a deliberate
shrink is still a visible, reviewable line, never silent.

## Where reports go (R16)

Reports default to **`~/notes/evidence/deck-gates/<UTC stamp>.{md,json}`** — the durable
evidence directory, mirroring `deck-drill`'s `~/notes/evidence/deck-drills/`. Override with
`REPORT_DIR` or `--report-dir` only for a genuine reason, and never point it at a session
scratchpad. Two of this branch's own RED proofs were briefly written to
`/tmp/.../scratchpad/deck-gate-reports/` via a `REPORT_DIR` override; the proofs were real,
but scratchpad is session-temp and can be cleaned at any moment, while the code they claim to
prove stays in the repo forever. Evidence that can evaporate out from under a merged claim is
worthless as evidence — it was recovered by hand-copying into the durable dir this time,
which is not a repeatable plan.

## Known limitations (stated honestly, not solved)

1. **Tier 2 is one-directional.** Every token/key observed live must be known to the fixture;
   the fixture is never required to still be fully reproduced live. This is deliberate —
   sparky being down (or mid-boot) changes the payload wholesale, and a gate that reddens
   because a box rebooted is a gate people stop reading (the deck's own design makes this
   same `unreachable` != `down` distinction). The cost: a field **removed** from inside a
   node-dependent section is indistinguishable from "that node simply isn't present this
   run" — both just shrink live's key list at that path, which the one-directional rule lets
   through by design. `UNCONDITIONAL_SHAPE_PATHS` (`fidelity.gate.mjs`) claws part of this
   back for exactly two paths, both structurally present regardless of topology: the
   TOP-LEVEL `/api/state` key set (in which `lifecycle` is one key NAME among seven, not a
   contents check) and `/api/state.node`'s own key set. Nothing asserts the keys *inside* a
   lifecycle entry (`/api/state.lifecycle.<resource>`'s own shape) — that stays covered only
   by the general one-directional rule, i.e. not required-present at all; the rest is a known
   hole, not a solved problem.

2. **Item 4's board-card re-derivation is not asserted, by ruling R12.** The E1 checklist's
   own wording for item 4 says the new engine "appears as a board card" — but that card is
   fed by `/api/state`, a timer-polled route, and (per the R12 rule above) a scripted
   transition there would pass on the clock whether or not the Add dispatched at all. Item 4
   instead asserts what *is* honestly assertable in tier 1: the dispatched POST body, and the
   new row appearing in the engines list (fed causally by `/api/nodes`). The board's own
   re-derivation of a newly declared engine from `/api/state` stays ungated by this harness.
   Both gaps are named here rather than papered over with a check that would pass either way.

3. **Nothing verifies add/forget end-to-end against the real backend.** The design's own
   accepted gap (D3), restated here because it is the coverage boundary that actually
   matters: the live tier is strictly read-only (D3), so it never dispatches a POST/DELETE
   at all — every add/edit/forget flow this harness exercises (E1 items 4, 6, 7, 9-11) is
   proven only against the stub's scripted responses. That proves UI **dispatch** (the right
   request, with the right body); it proves nothing about backend **behaviour** — whether
   the real `POST /api/nodes` actually accepts that request or persists the result. A shape
   difference between the real route and the stub is what the tier-2 fidelity check is for;
   a purely behavioural difference is not caught anywhere in this harness. This is accepted,
   not overlooked, and end-to-end behavioural proof against a live box is `deck-drill`'s job,
   not this one's.

## Playwright and the production image

`playwright-core` is a **devDependency only** — it drives the system `google-chrome`
(`channel: "chrome"`), never downloads its own ~400 MB of vendored browser binaries, and must
never end up resolvable inside the shipped container. The Dockerfile's `npm ci && npm run
build` stage only carries `dist/` forward into the runtime image; `ui/gates/` and
`node_modules/` are never copied. See the extension README's packaging-probe command for how
to verify this directly against a built image.
