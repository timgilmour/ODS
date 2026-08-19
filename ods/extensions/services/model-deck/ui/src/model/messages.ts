/**
 * Every operator-visible string in the deck, in one place.
 *
 * Two reasons this exists rather than literals in components. First, tone is
 * attached to the message, so one condition cannot be red in one screen and
 * amber in another — the exact inconsistency the R1 drift card showed.
 * Second, backend detail strings ("busy — 2 requests in flight") arrive
 * lowercase and mid-sentence; Banner capitalizes at render, so nothing here
 * mutates a payload to make it presentable.
 *
 * Pure: no React, no formatting of live clocks. Callers pass in already
 * humanized ages.
 */

import type { Layer, SettingsKind, Widget } from "../api";

export type Tone = "neutral" | "warning" | "danger";

export interface Message {
  tone: Tone;
  title: string;
  body?: string;
  action?: { label: string };
}

export const messages = {
  nodeUnreachable: (label: string, age: string | null): Message => ({
    tone: "danger",
    title: "Node is unreachable",
    body: age
      ? `${label} last answered ${age} ago. What follows is the last thing it told us.`
      : `${label} is not answering. What follows is the last thing it told us.`,
    action: { label: "Retry" },
  }),

  // The node answered; what it should be running is not running. Distinct
  // from nodeUnreachable (we cannot see the box at all) and from warming (a
  // boot is legitimately in flight). `detail` is the backend's own sentence
  // — a swap helper error, or lifecycle's reason — because without it a
  // failed asynchronous swap renders as a red pill and nothing else.
  nodeDown: (label: string, detail: string | null): Message => ({
    tone: "danger",
    title: "Node is down",
    body: detail
      ? `${label} answered, but nothing is serving — ${detail}`
      : `${label} answered, but nothing is serving.`,
    action: { label: "Retry" },
  }),

  warmingFirstBoot: (): Message => ({
    tone: "neutral",
    title: "first boot can autotune for about 15 minutes — this is normal",
  }),

  // The node answered AND is serving, but its agent says its own config
  // makes serving detection blind (lifecycle-node-misconfigured,
  // app/arbiter.py's _node_observations — the body is node-agent
  // serving.py's PROBE_URL_WARNING, passed through verbatim). Warning, not
  // danger: nothing has failed yet; the node's env wants fixing before the
  // next blind swap — amber is for things wanting a decision.
  nodeMisconfigured: (label: string, warning: string): Message => ({
    tone: "warning",
    title: `${label} is misconfigured`,
    body: warning,
  }),

  // Distinct from nodeUnreachable, which reports what the BACKEND says about
  // a node. This one says THIS PAGE could not reach the deck's own endpoint
  // for that node — so everything below it may be minutes old even while the
  // backend's own view still calls the node reachable.
  nodeFetchFailed: (label: string, detail: string): Message => ({
    tone: "danger",
    title: `Cannot reach ${label} from this page`,
    body: `${detail} — what follows may be out of date.`,
  }),

  // "danger": the save was REFUSED and nothing was written. Naming the
  // offending locations matters — the modal edits several drives at once,
  // and "a watermark is invalid" would leave the operator hunting.
  invalidWatermark: (names: string[]): Message => ({
    tone: "danger",
    title: "Save refused",
    body:
      `Watermark for ${names.join(", ")} isn't a number — enter GB as a ` +
      "number, or leave it empty for no watermark.",
  }),

  // "Refused" is deliberately generic — the detail carries the reason.
  // Recurring producers, catalogued here so the exact wording stays
  // traceable even though they pass through VERBATIM rather than being
  // re-templated (the backend already names the exact resource/kind/verb
  // the operator needs — see PROBE_URL_WARNING's identical passthrough
  // reasoning above, nodeMisconfigured):
  // - 404 `unknown resource {resource!r}` (app/routers/control.py:144)
  // - 405 `{resource} ({kind}) does not support {verb}`
  //   (app/routers/control.py:148)
  // and, for a DECLARED REMOTE engine's verbs (Task 10b), the same family
  // from app/routers/serving.py's `engine_verb`, deliberately worded to
  // match its local sibling's:
  // - 404 `unknown engine {resource!r} on node {node_id!r}` (:243-244)
  // - 405 `{resource} ({kind}) does not support {verb}` (:295-296)
  // - 501 `{resource} ({kind}) declares {verb} but the node-agent engine
  //   channel has no call for it` (:302-305)
  // - 503 `node {node_id!r} cannot act on engine {resource!r} — ...` (:315-318)
  guardRefused: (detail: string): Message => ({
    tone: "danger",
    title: "Refused",
    body: detail,
  }),

  // The board's verb buttons for declared remote engines come from
  // `GET /api/engine-kinds` (app/routers/nodes.py:476-504) — never a UI
  // literal — so a failed fetch means those cards render status with no
  // controls. Said out loud rather than left as silently missing buttons.
  engineKindsFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Cannot read the engine catalog",
    body: `${detail} — remote engine verbs stay unavailable until it loads.`,
    action: { label: "Retry" },
  }),

  // Warning, not danger: this is asking the operator for a decision, which
  // is what warning means here. Nothing has failed or been refused — the
  // red-outlined button above it is what carries the danger.
  forceConfirm: (): Message => ({
    tone: "warning",
    title: "Click again to confirm",
  }),

  modelIsCold: (sizeGb: string): Message => ({
    tone: "warning",
    title: "Model is cold",
    body: `pull ${sizeGb} GB to hot storage, then load?`,
    action: { label: "Pull + load" },
  }),

  pullingFromCold: (): Message => ({
    tone: "neutral",
    title: "Pulling from cold storage",
    body: "the model will load once it lands on hot storage.",
  }),

  stateRefreshFailed: (detail: string): Message => ({
    tone: "danger",
    title: "State refresh failed",
    body: detail,
  }),

  eventsFetchFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Could not load events",
    body: detail,
  }),

  noEvents: (): Message => ({ tone: "neutral", title: "no events yet" }),

  lastKnown: (): Message => ({ tone: "neutral", title: "last known" }),

  // The state pill already reads "last known"; this answers the next
  // question, which is *how* stale — a different fact, not a redundant one.
  lastSeen: (age: string): Message => ({
    tone: "neutral",
    title: `last seen ${age} ago`,
  }),

  // MoveModal's terminal-phase notices.
  moveComplete: (): Message => ({ tone: "neutral", title: "Moved." }),
  moveCancelled: (): Message => ({ tone: "neutral", title: "Move cancelled." }),
  moveFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Move failed",
    body: detail,
  }),

  // MoveModal's confirm phase, when every other location is unregistered or
  // unavailable — nothing has failed, there is simply nowhere to send this
  // unit, so this stays neutral rather than danger/warning.
  noEligibleDestination: (): Message => ({
    tone: "neutral",
    title: "No eligible destination",
    body: "every other location is unregistered or unavailable.",
  }),

  // LocationCard: a location whose mount is missing. Unavailable, not
  // empty — the units it lists are still on disk, just unreachable right
  // now, and that distinction is load-bearing (an operator reading "empty"
  // could conclude the data is gone).
  mountMissing: (): Message => ({
    tone: "danger",
    title: "Mount missing",
    body: "files retained — the location is unavailable, not empty.",
  }),

  storageFetchFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Storage state failed",
    body: detail,
  }),

  // Set Builder. Neutral, and permanent: this screen composes a *draft*, and
  // the one place a draft becomes a deployment is Activate, which does not
  // live here. Deliberately not amber — nothing is asking for a decision, it
  // is stating what the screen is.
  draftNothingDeployed: (): Message => ({
    tone: "neutral",
    title: "DRAFT — nothing is deployed",
    body: "composition happens here; Activate is the only deploy verb.",
  }),

  // Amber, which is decision 5's exact meaning for amber: a save collided
  // with an existing slug and the operator has to say whether to replace it.
  // Banner's dismiss × is the cancel path.
  overwriteSet: (slug: string): Message => ({
    tone: "warning",
    title: `Overwrite set '${slug}'?`,
    action: { label: "Overwrite" },
  }),

  // Also amber, also a decision: the drafted footprint will not fit, so the
  // operator is asked to reconsider the draft. Nothing has failed — nothing
  // has even been attempted yet — so this is not danger.
  overBudget: (): Message => ({
    tone: "warning",
    title: "Over budget",
    body: "loads may fail.",
  }),

  // Settings panel (phase 3) ------------------------------------------------

  settingsLoadFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Could not load settings",
    body: detail,
  }),

  // Danger, and it names the one thing the operator cannot see from here:
  // Save walks one PUT per touched scope document sequentially, and the deck
  // has no transaction across them (app/routers/settings.py:put_settings
  // writes a single scope), so a failure part-way through leaves the earlier
  // scopes already written. Saying "nothing was saved" would be a guess, and
  // the wrong one.
  settingsSaveFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Save failed",
    body: `${detail} — earlier scopes in this save may already have been written; reopen this panel to see what landed.`,
  }),

  // The preview is server-rendered, so a failed render means the command
  // line on screen is the LAST GOOD one, not the current buffer. That
  // staleness is the fact worth reporting; the panel keeps showing the old
  // text rather than blanking it.
  settingsPreviewFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Command preview is out of date",
    body: detail,
  }),

  // app/argline.py's parse never raises, so this is a transport/HTTP failure
  // of POST /settings/preview rather than "your text is malformed" — the
  // copy must not blame the operator's typing for a server that answered 500.
  settingsImportFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Could not read that command line",
    body: detail,
  }),

  // All-options modal's manual Refresh (POST /api/settings/harvest/{node}/{engine}).
  // Neutral, not a Message-worthy failure: the probe ran, the catalog it
  // would have written is byte-identical to what is already on screen, so
  // there is nothing stale here to complain about.
  catalogCurrent: (): Message => ({
    tone: "neutral",
    title: "Catalog is already current",
  }),

  // Warning rather than danger: the panel itself is unaffected (it still
  // has whatever catalog it had before this click), so this reports "the
  // refresh did not land" rather than "something here is broken". `outcome`
  // carries either postHarvest's own "failed"/"empty" string or, when the
  // request itself never came back (ApiError/network failure), that error's
  // message — both land the operator in the same "try again" banner, since
  // neither case leaves them with a fresher catalog.
  harvestFailed: (outcome: string): Message => ({
    tone: "warning",
    title: outcome === "empty" ? "Harvest found no options" : "Harvest failed",
    body:
      outcome === "empty"
        ? "the engine answered but the probe found nothing to catalog."
        : `the engine could not be probed (${outcome}).`,
  }),

  // Model detail drawer (phase 3) ------------------------------------------

  factsLoadFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Could not load facts",
    body: detail,
  }),

  // "Nothing was written" is a claim this one can actually make, unlike
  // settingsSaveFailed's multi-scope walk: each declared edit is ONE PUT of
  // one field, and app/declared.py's store writes atomically — "a rejected
  // put leaves the file untouched" (its module docstring).
  declaredSaveFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Could not save",
    body: `${detail} — nothing was written.`,
  }),

  reloadFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Reload failed",
    body: detail,
  }),

  // The drawer's subject is no longer on the board: unloaded, parked, or
  // swapped away while the drawer was open. Neutral, because nothing failed
  // — polling never stopped, so this IS current information. What follows is
  // simply the last thing the board knew, and none of it can be acted on.
  placementGone: (): Message => ({
    tone: "neutral",
    title: "No longer on the deck",
    body: "this placement is not in the latest state — what follows is the last thing we knew.",
  }),

  // Nodes tab (Task 9) — the registry CRUD lives in app/routers/nodes.py.

  nodesEmptyTitle: (): Message => ({
    tone: "neutral",
    title: "No nodes yet",
    body: "Register the machines this deck should see. The local box is already here.",
  }),

  // The credential is write-only end to end (accepted on create/update,
  // never echoed back — app/routers/nodes.py's module docstring). This
  // caption is the one place the deck says so, next to the field itself.
  nodeCredentialCaption: (): Message => ({
    tone: "neutral",
    title: "stored server-side, never shown again",
  }),

  // POST /api/nodes/test's two outcomes (routers/nodes.py:106-141) — the
  // probe answered with a machine to describe, or it didn't.
  nodeTestOk: (name: string, gpuCount: number): Message => ({
    tone: "neutral",
    title: `Connected — ${name}, ${gpuCount} GPU${gpuCount === 1 ? "" : "s"}`,
  }),
  nodeTestFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Connection failed",
    body: detail,
  }),

  // Engines editor (E1 Task 12) — the local node's declared local-engine
  // CRUD (app/routers/nodes.py:200-321). Kind identity and per-kind
  // connection FIELD identity never come from here (spec §5: GET
  // /api/engine-kinds, model/engineForm.ts, is the one source) — these
  // four cover the section's own load/empty/forget/add-prerequisite states.

  enginesLoadFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Could not load engines",
    body: detail,
    action: { label: "Retry" },
  }),

  // Neutral: an empty declaration is not a failure — it is Task 3/11's
  // "zero, three, or five" starting point (spec §5), same posture
  // nodesEmptyTitle above takes for a fresh registry.
  enginesEmpty: (): Message => ({
    tone: "neutral",
    title: "no engines declared",
    body: "declare a local engine to give the deck something to schedule here.",
  }),

  // DELETE /api/nodes/local/engines/{resource} — forget_engine
  // (app/routers/nodes.py:273-321): bookkeeping only, never calls the
  // engine. Shown while a row's Forget button is ARMED (the confirm step)
  // rather than always-on, so the one click that actually asks "are you
  // sure" is also the one that explains what "sure" means — a two-click
  // delete that LOOKS as destructive as any other must not let the
  // operator assume it stops a running process.
  forgetEngineConfirm: (): Message => ({
    tone: "neutral",
    title: "removes the declaration only — a running engine keeps running, untouched",
  }),

  // Shown in the Add flow (README's "Declared Engines" section, Park-
  // allowlist prerequisite): declaring a resource is enough for the deck
  // to observe it, policy-manage it, and include it in a Set, but a
  // container-affecting verb (hipfire-kind park/resume; the storage
  // pull-through hook's automatic restart of a lemonade-kind resource)
  // additionally needs the resource's connection.container name in the
  // HOST's settings.park_allowlist (app/settings.py:95) — DockerCtl
  // refuses any other name with GuardError, checked before any HTTP call
  // (app/engines/docker_ctl.py:197-199). Neutral, not a warning: this is not a
  // defect in what the operator just typed, it's a heads-up about a
  // SEPARATE, host-level prerequisite the form has no way to check or set.
  engineParkAllowlistNote: (): Message => ({
    tone: "neutral",
    title: "container verbs need the host's park allowlist too",
    body: "load/unload/free reach the engine directly and work immediately; park, resume, and the storage-move restart hook go through Docker start/stop and refuse (409) until this container name is added to the host's park allowlist.",
  }),

  /** A duration for humans beside the raw seconds the API takes. 0 is a
   * MODE, not a duration: every kind's idle_action gates on
   * `policy["idle_ttl"] > 0` (app/engine_kinds.py per-kind idle rules), so
   * zero disables idle release outright. */
  ttlValue: (seconds: number): string => {
    if (seconds <= 0) return "off";
    if (seconds < 60) return `${seconds} s`;
    const minutes = Math.round(seconds / 60);
    return `${seconds} s — ${minutes >= 60 ? `${Math.round(minutes / 60)} h` : `${minutes} min`}`;
  },

  /** What this TTL will actually DO, which depends entirely on whether the
   * kind can reload itself. `demand` comes from GET /api/engine-kinds
   * (app/routers/nodes.py's list_engine_kinds), never from a kind name in
   * this file. null = the catalog did not load; say so rather than imply a
   * reversibility nobody verified. */
  ttlConsequence: (seconds: number, demand: boolean | null): string => {
    if (seconds <= 0) return "never released automatically";
    const duration = seconds < 60 ? `${seconds} s` : `${Math.round(seconds / 60) >= 60 ? `${Math.round(Math.round(seconds / 60) / 60)} h` : `${Math.round(seconds / 60)} min`}`;
    const when = `released after ${duration} idle`;
    if (demand === null) return `${when} — reload behaviour unknown (kind catalog unavailable)`;
    return demand
      ? `${when} — the next request reloads it automatically`
      : `${when} — reload is MANUAL, nothing brings it back`;
  },
};

/** Short labels — control text, badges, captions and the `title` tooltips
 * that explain them. Not notices, hence not `Message`s (there is no tone to
 * carry), but every one of them is operator-visible text and an ARIA name or
 * a tooltip is read out loud, so they are centralized here for the same
 * reason the notices are. Parameterized entries are plain pure functions;
 * they format a fact, they do not decide anything. */
export const labels = {
  dismiss: "Dismiss",
  close: "Close",
  cancel: "Cancel",
  confirm: "Confirm",
  applying: "Applying…",
  forceApply: "Force apply",
  forceApplyTitle:
    "override the hipfire conversation-guard — the live/recent conversation will lose its cache and its next turn will re-read the whole history",
  loadingPreview: "Loading preview…",
  forcePark: "Force park",
  forceSwap: "Force swap",
  filterEvents: "Filter events…",

  /** Caption for a resource with nothing on it. Deliberately says nothing
   * about WHAT the resource is — the panel above it is already titled
   * "GPU 0" or "Serving slot", and the previous copy ("Serving slot", taken
   * from a Message meant for the spark) rendered as "GPU 0 / Serving slot"
   * on the most common empty state the board has. A caption is a label, not
   * a notice: it carries no tone, so it is not a Message. */
  nothingPlaced: "nothing placed",

  // Top-level chrome and navigation. `events` was the only tab that ever
  // came from here, which is precisely how a half-applied rule dies: the
  // next author copies whichever neighbour they happened to read first.
  appTitle: "Model Deck",
  // Was "GPU/VRAM control for lemonade, ComfyUI, hipfire" — a fixed-triple
  // literal that predates E1's any-kind declared engines (Task 3/11's
  // "zero, three, or five", now genuinely open-ended once Task 12's
  // Engines editor lets an operator declare something other than the old
  // three). Flagged as a follow-up in Task 11's report; fixed here since
  // this task is exactly what makes the old wording wrong.
  appSubtitle: "GPU/VRAM control for declared local engines",
  deck: "On Deck",
  setBuilder: "Set Builder",
  storage: "Storage",
  events: "Events",
  policy: "Policy",
  loading: "loading…",

  // Placement controls.
  modelToLoad: "model to load",
  selectModel: "select a model…",
  noModels: "no models found",
  coldGroup: "❄ cold",
  coldOption: (name: string, sizeGb: string) => `❄ ${name} (${sizeGb} GB)`,
  load: "Load",
  unload: "Unload",
  /** A DECLARED REMOTE engine's verb button (Task 10b). The vocabulary is
   * the KIND's own `human_verbs` (model/engineVerbs.ts), so this formats a
   * verb rather than choosing one: the two the node-agent engine channel
   * carries today (app/routers/serving.py:228) get the same words the local
   * controls use, and anything else is shown verbatim rather than hidden —
   * a button that says what the backend was asked beats no button at all. */
  engineVerb: (verb: string) => ({ load: "Load", unload: "Unload" })[verb] ?? verb,
  /** Tooltip for the pill carrying a remote engine's OWN state word. Names
   * the source, because the pill sits beside a chip whose pill says the
   * lifecycle status — two different facts, one of which is the node's
   * live reading. */
  engineStateTitle: "what the node's agent reports about this engine right now",
  /** `HeaderEngineMenu`'s per-engine button (Task 5) — the node-header
   * INTERIM SURFACE for a DECLARED REMOTE engine that dropped off its GPU
   * card entirely (nothing loaded, so no chip, so no `RemoteEngineActions`
   * anywhere on screen). Names the resource, not the kind: the resource is
   * what the operator declared and what the verb route addresses. */
  loadEngine: (resource: string) => `Load ${resource}`,
  loadEngineTitle: (resource: string) => `load ${resource} on this node`,
  free: "Free",
  comfyuiBlockedTitle: "ComfyUI is busy or has a non-empty queue",
  park: "Park",
  resume: "Resume",

  // Spark's profile picker.
  swapTo: "swap to…",
  swap: "Swap",
  /** A profile whose engine is not the node's usual one says so in the
   * picker, the same fact the chip's engine badge carries. */
  swapOption: (profile: string, engine: string | null) =>
    engine ? `${profile} (${engine})` : profile,

  // Placement facts (see Placement's "operational facts" block in nodes.ts).
  pinned: "📌",
  pinnedTitle: "pinned — exempt from idle eviction",
  priority: (n: number) => `P${n}`,
  priorityTitle: "eviction priority",
  inUse: "in use",
  inUseTitle:
    "a conversation turn is being served right now — park/apply will refuse without force",
  queue: (n: number | null) => `queue ${n ?? "—"}`,
  queueTitle: "jobs waiting on this engine",
  idle: (seconds: number) => `idle ${Math.round(seconds)} s`,
  idleTitle: "time since last activity — counts towards the idle-TTL eviction",

  // Set-apply step vocabulary — one label per "step" kind app/sets.py's
  // plan_apply() emits. E1 Task 8 made every per-resource verb
  // GENERIC + resource-tagged (unload/load/free/park/resume, each
  // carrying its own "resource" field) — the old kind-suffixed names
  // (unload_lemonade, free_comfyui, park_hipfire, resume_hipfire) are
  // gone from the wire. Named, not line-cited: those numbers drifted three
  // times across this wave.
  stepUnload: (resource: string | null) => (resource ? `Unload — ${resource}` : "Unload"),
  stepLoad: (resource: string | null) => (resource ? `Load — ${resource}` : "Load"),
  stepFree: (resource: string | null) => (resource ? `Free — ${resource}` : "Free"),
  stepPark: (resource: string | null) => (resource ? `Park — ${resource}` : "Park"),
  stepResume: (resource: string | null) => (resource ? `Resume — ${resource}` : "Resume"),
  stepActivate: "Activate catalog model",
  stepPolicyPatch: "Apply policy overrides",
  // New in E1 Task 9 (app/sets.py:779's `{"step": "restore_settings", ...}`)
  // — not on the old vocabulary list above at all; added here so it never
  // falls through applySteps.ts's default (verbatim-kind) branch.
  stepRestoreSettings: "Restore settings",
  stepWarn: "Warning",
  /** A "warn" step's `reason` — a machine token (app/sets.py) — translated
   * to operator language; this is the one place that happens. Unknown
   * reasons render verbatim (same ugly-but-true rule applySteps.ts's
   * default step-kind branch follows), so a future backend reason degrades
   * gracefully instead of vanishing.
   *
   * "busy-skipped" (app/sets.py:695-710, resource-tagged, log site :708-710
   * — T8 review I3 renamed it from the kind-baked "comfyui-busy-skipped"):
   * the FREE-VERB branch (comfyui-kind is the only kind with "free" in
   * human_verbs() today) only frees when the resource's queue reading is
   * CONFIRMED zero (`tenant["queue"] == 0`) — a nonzero (busy) queue AND an
   * unreadable/unknown one (None) both fold into this ONE reason, on
   * purpose (the comment at :696-698: "never yank VRAM out from under a
   * running generation" applies equally to "we don't actually know"). The
   * copy below must not claim "busy" when the truth might just be
   * "unconfirmed" — that would be a fact the backend never asserted.
   * "no-model-to-load" (app/sets.py:758, resource-tagged): a "loaded" goal
   * on a load-verb resource named no model, live or in the set.
   * "durable-revert-unavailable" (app/sets.py:653, no resource — it is
   * about the DURABLE route, not a per-resource actuation): the previous-set
   * revert has no catalog id to re-activate the old default route with. */
  stepWarnReason: (reason: string, resource?: string | null): string => {
    switch (reason) {
      case "durable-revert-unavailable":
        return "durable revert unavailable — no catalog id to re-activate the previous model";
      case "busy-skipped":
        return resource
          ? `${resource} skipped — queue not confirmed empty`
          : "skipped — queue not confirmed empty";
      case "no-model-to-load":
        return resource ? `${resource} has no model to load` : "no model to load";
      default:
        return reason;
    }
  },
  estimate: (s: number) => `about ${s}s`,
  noChanges: "no changes — already matches this set",
  stepsCompleted: (n: number) => `${n} step(s) completed`,

  // PolicyModal.
  policyTitle: "Policy",
  save: "Save",
  saving: "Saving…",
  autoTiering: "Auto-tiering: archive to cold on watermark + silent pull-through",
  storageSection: "Storage",

  // MoveModal + the per-unit trigger LocationCard renders.
  moveModel: "Move model",
  moveTo: "Move to…",
  destination: "destination",
  startMove: "Move",
  starting: "Starting…",
  cancelMove: "Cancel move",

  // StorageView's role bands + LocationCard/JobsPanel/OnboardingPanel.
  hotBand: "HOT",
  coldBand: "COLD",
  rescan: "Rescan",
  rescanning: "Rescanning…",
  jobsPanel: "Moves",
  noActiveMoves: "no active moves",
  noUnits: "no units",
  pinTitlePinned: "pinned — click to unpin",
  pinTitleUnpinned: "click to pin (exempt from tiering)",
  readonly: "read-only",

  // Set Builder: the model library, the saved-sets list that replaced the
  // load dropdown, and the draft node card's own vocabulary.
  modelLibrary: "Model library",
  searchModels: "Search models…",
  savedSets: "Saved Sets",
  noSavedSets: "no saved sets",
  loadSet: "Load",
  duplicateSet: "Duplicate",
  deleteSet: "Delete",
  reallyDelete: "Really delete?",
  dropToAssign: "Drop to assign",
  draftPill: "DRAFT",
  saveDraft: "Save draft",
  previewSteps: "Preview steps",
  place: "Place",

  // Settings panel (phase 3) ------------------------------------------------

  /** One name per ladder layer — the five entries of app/ladder.py:48's
   * `LAYERS` tuple ("engine_defaults", "checkpoint_recommendations",
   * "engine", "model", "engine_model"), mirrored as `Layer` in api.ts. Each
   * name says both WHERE the value came from and, for the two derived
   * layers, that nothing declared it — which is the whole reason those two
   * are read-only and never shipped as flags
   * (app/routers/settings.py:_declared_only). */
  layerName: (layer: Layer): string =>
    ({
      engine_defaults: "engine default — harvested from the engine, not declared",
      checkpoint_recommendations:
        "checkpoint recommendation — from the model's generation_config.json, not declared",
      engine: "declared for this engine on this node",
      model: "declared for this model, on every node",
      engine_model: "declared for this model on this engine — the most specific scope",
    })[layer],

  /** The three write kinds — exactly app/settings_store.py:117's `KINDS`
   * ("engines", "models", "engine_models"), mirrored as `SettingsKind`.
   * Labelled by what each one SCOPES rather than by its store name, because
   * the store name is an implementation detail an operator never types. */
  kindName: (kind: SettingsKind): string =>
    ({
      engines: "ENGINE",
      models: "MODEL",
      engine_models: "ENGINE × MODEL",
    })[kind],

  settingsTitle: (engine: string, node: string) => `Settings — ${engine} on ${node}`,
  writeScope: "write scope",
  needsModelContext:
    "no model in context — open Settings from a model to write these scopes",

  /** Section headings. DECLARED is what the deck writes and ships; APPLIED
   * BY ENGINE is the two derived layers, which the deck shows but never
   * re-asserts back at the engine as flags
   * (app/routers/settings.py:get_effective's declared-only argline rule). */
  declaredSection: "DECLARED",
  declaredSectionHint: "the only values shipped to the engine",
  appliedSection: "APPLIED BY ENGINE",
  appliedSectionHint:
    "the engine's own defaults and the checkpoint's recommendations — shown for context, never re-asserted as flags",
  nothingDeclared: "nothing declared at any scope",
  nothingApplied: "no harvested defaults or checkpoint recommendations",

  editOption: "edit",
  removeOverride: "Remove",
  removeOverrideTitle: "remove this key from the selected scope",
  revertToInherited: "Revert to inherited",
  /** /effective resolves ONE winner per key (app/ladder.py:resolve_settings
   * keeps a single entry per key), so the panel can honestly show the
   * winning layer and nothing below it. */
  winningLayerOnly: "the winning layer — lower layers are not resolved server-side",
  provenance: "provenance",
  cannotUnsetHere:
    "not declared at the selected scope — switch scopes to remove it where it is set",
  unsavedBadge: "unsaved",
  // A brand-new add still sitting on the `""` placeholder. NOT "unsaved":
  // Save is disabled for it and `forSave` would drop it (editState), so
  // calling it unsaved would promise a write that is not going to happen.
  chooseValueBadge: "choose a value",
  willBeRemovedBadge: "will be removed",
  shadowedBadge: "shadowed by a more specific scope",
  positionalName: "positional",
  positionalTitle:
    "the command's leading positional tokens (app/argline.py's _positional) — not a flag",

  addOption: "+ Add option",
  addOptionTitle: "browse this engine's harvested option catalog",
  importArgline: "Import argline…",
  importArglineTitle: "Import argline",
  importArglinePlaceholder: "--max-model-len 131072 --enable-auto-tool-choice",
  importArglineHint:
    "parsed server-side and merged into the selected scope as pending edits — nothing is saved until Save",
  importApply: "Apply",

  commandPreview: "Command preview",
  appliesOnReload: "changes apply on reload — saving records intent, it does not restart anything",
  warningsHeading: "warnings",

  engineSettings: "Engine settings",
  engineSettingsTitle: "edit the engine-scope flags for this node",

  // All-options modal (phase 3) ---------------------------------------------

  allOptions: "All options",
  optionCount: (n: number) => `${n} option${n === 1 ? "" : "s"}`,
  /** Half of the provenance line — the other half is `optionCount`, composed
   * next to it as "harvested <age> · N options". Deliberately never takes
   * `catalog.engine_version`: it is an opaque image content id
   * (app/harvest.py:117-127, set from the engine container's own digest/tag),
   * not a fact an operator can read a version number out of, so there is no
   * label for it anywhere in this modal. */
  catalogAge: (age: string | null) => (age ? `harvested ${age}` : "never harvested"),
  catalogNeverHarvested: "not harvested yet — Refresh to probe the engine",
  noCatalogMatches: "no options match this search",
  searchOptions: "Search options…",
  setOnly: "Set only",
  refresh: "Refresh",
  refreshing: "Refreshing…",
  /** The five categories app/harvest.py:widget_for ever produces, mirrored
   * as `Widget` in api.ts — the modal's filter-chip labels. */
  widgetName: (w: Widget): string =>
    ({ toggle: "Toggle", list: "List", select: "Select", number: "Number", text: "Text" })[w],
  addOptionRowLabel: (flag: string) => `add ${flag}`,
  jumpToOptionLabel: (flag: string) => `${flag} already set — edit it`,

  // Model detail drawer (phase 3) -------------------------------------------

  modelDetail: "Model detail",

  factsSection: "FACTS",
  factsSectionHint: "what the deck read from the model's own artifact, plus what a human overrode",
  noFactsYet: "no facts recorded for this model yet",

  /** The dot's name in the facts table. Facts carry an ORIGIN, not a ladder
   * layer (app/facts.py:resolve_facts sets exactly "derived"/"declared"),
   * and `source` is the payload's own account of where the value came from
   * — "config.json", "compose import", "declared.json" — so it is quoted
   * rather than paraphrased. */
  factOrigin: (origin: "derived" | "declared", source: string) =>
    origin === "declared"
      ? `declared by a human — ${source}`
      : `derived — read from ${source}`,
  shadowedTitle: "the derived value this declaration overrides",

  /** One drift line: what the fact says, what is actually in force, and how
   * bad that is. `severity` is the backend's own word (app/facts.py's
   * DRIFT_RULES: "crash", "mismatch"), never re-graded here. */
  driftDetail: (expected: string, actual: string, severity: string) =>
    `expected ${expected} · actual ${actual} · ${severity}`,
  driftIn: (field: string) => `${field} —`,

  declaredFactsSection: "DECLARED",
  /** app/declared.py:32-38's allowlist is five fields and no more; the whole
   * point of the layer is that everything a machine can read, it reads. */
  declaredFactsHint: "the five fields a human owns — everything else is derived and read-only",
  tagsField: "tags",
  addTag: "add a tag…",
  removeTagTitle: (tag: string) => `remove ${tag}`,
  factLabel: "label",
  factNotes: "notes",
  factToolsVerified: "tools verified",
  factToolsVerifiedTitle:
    "true only after an end-to-end tool call actually worked — engines advertise tool support they cannot reliably parse",
  factEnginePreference: "engine preference",
  factEnginePreferenceTitle:
    "the tie-break when more than one engine could serve this model",

  actionsSection: "ACTIONS",
  reload: "Reload",
  reloading: "Reloading…",
  reloadTitle:
    "re-apply the declared settings on the node — the serving container is recreated",
  modelSettings: "Settings",
  modelSettingsTitle: "edit the flags this model is served with",
  /** Why the button is disabled while the facts fetch is outstanding — and,
   * on a failed fetch, indefinitely. The profile→checkpoint translation
   * (settingsIdentityFor) lives in those facts, and without it the panel
   * would open on the placement's PROFILE: a scope key nothing resolves,
   * with nothing on screen saying so. Refusing is the honest answer; a
   * silent wrong target is not. */
  modelSettingsNoFactsTitle:
    "facts have not loaded — without them Settings could open on the wrong scope",

  // Settings drift card (phase 3, Task 11) -----------------------------------
  // BLUE, not the facts table's amber above: build-design decision 5 treats
  // this as a decision pending — declared settings changed since the
  // placement last (re)launched (app/routers/__init__.py:116's
  // `settings_drift`, baselined on intent's `updated_ts`) — never a
  // disagreement the deck is flagging as wrong. The R1 mockup's amber
  // treatment for this exact card is superseded.

  settingsDrift: "Settings drift",
  keysChanged: (n: number) => `${n} key${n === 1 ? "" : "s"} changed`,
  /** entries[].new === null — settings_store.py:252's `ns.get(name)` reading
   * back empty after a `remove`. */
  removedValue: "(removed)",
  /** entries[].old === null — settings_store.py:252's `before.get(name)` for
   * a key that did not exist before this change. */
  addedValue: "(added)",
  notAppliedUntilReload: "not applied until this placement reloads",

  reloadToApply: "Reload to apply",
  reviewInSettings: "Review in Settings",
  openingSettings: "Opening…",

  // Nodes tab (Task 9). The rail's status dots read app/node_observer.py's
  // vocabulary directly (online/offline/error/unconfigured) rather than a
  // label from here — that module's own docstring is the one place that
  // vocabulary is declared, and re-labelling it here would be a second
  // place it could drift.
  nodes: "Nodes",
  addNode: "Add node",
  selectANode: "Select a node",
  /** Rail badge on the seeded local entry — app/node_store.py's one
   * `agent_kind: "local"` row, always present, never removable. */
  localBadge: "local",
  nodeId: "Id",
  nodeLabel: "Label",
  nodeAddress: "Address",
  nodeAddressPlaceholder: "http://host:7720",
  nodeServingAddress: "Serving address (optional)",
  nodeServingAddressPlaceholder: "http://host:8000",
  nodeCredential: "Credential",
  /** Shown only as the input's placeholder, and only once a credential is
   * already on file (`credential_set`) — never the operator's own typing,
   * which is real characters in a type="password" field, not this glyph. */
  nodeCredentialPlaceholder: "••••••••",
  testConnection: "Test connection",
  registerALocation: "Register a location",

  // control toggle (app/node_store.py _CONTROLS) — declared operability,
  // never inferred: "swap" gets serving verbs and a board card with
  // controls, "none" is observe-only regardless of what data the row
  // carries.
  nodeControlLabel: "Operate serving slot",
  nodeControlHint:
    "Off: observe only. On: the deck swaps/restores this node's serving slot.",

  // model/nodeForm.ts's validate() and testTarget() — pure decision logic
  // that must not own its own copy (the same "everything through here"
  // rule the rest of the deck follows). See applySteps.ts's stepRow for the
  // same idiom: a pure model function importing labels rather than
  // hardcoding strings, so a component rendering its output needs no
  // literal of its own either.
  nodeIdInvalid: "id must be a lowercase slug (a-z, 0-9, hyphens)",
  nodeLabelRequired: "label is required",
  nodeAddressRequired: "address is required",
  nodeCredentialRequiredForAdd: "credential is required for a new node",
  testBlockedNoAddress: "enter an address to test",
  testBlockedNoCredential: "type a credential to test",
  /** The one case Task 9's review caught: an edited address with no
   * retyped credential would otherwise silently test the OLD stored
   * address while the screen shows the new one. */
  testBlockedAddressChanged: "address changed — retype the credential to test the new pair",

  // Mirror of app/node_store.py:_require_swap_prereqs's three-prerequisite
  // rule for control: "swap" — address + serving_address + a credential,
  // all present, missing fields NAMED. The backend 422 is authoritative;
  // this only saves the round-trip.
  nodeSwapNeedsAddress: "operating a node requires an agent address",
  nodeSwapNeedsServingAddress: "operating a node requires a serving address",
  nodeSwapNeedsCredential: "operating a node requires a credential",
  /** Categorical, not a missing prerequisite: app/node_store.py:_validate
   * refuses control:"swap" for agent_kind:"local" unconditionally — local
   * actuation is docker-ctl, not the swap protocol (G1 revisits). */
  nodeControlLocalRefused: "the local node cannot operate a serving slot",

  // Engines editor (E1 Task 12) — the local node's declared local-engine
  // CRUD, nested under the Nodes tab's local-node form (app/routers/
  // nodes.py:200-339). Kind names and connection FIELD names are never
  // literals here — see model/engineForm.ts's docstring: the picker's
  // options and each field's label are both DERIVED from GET
  // /api/engine-kinds, not hardcoded per kind.
  engines: "Engines",
  addEngine: "+ Add engine",
  editEngine: "Edit",
  forgetEngine: "Forget",
  engineResource: "Resource name",
  engineKind: "Kind",
  engineGpu: "GPU",
  engineSelectGpu: "select a GPU…",
  enginePinnedLabel: "Pinned",
  enginePriority: "Priority",
  engineIdleTtl: "Idle TTL (seconds)",
  /** A connection field's label is DERIVED from the payload's own field key
   * (GET /api/engine-kinds' `connection` map, e.g. "metrics_url") rather
   * than a per-field literal here — spec §5's "never a UI literal" applies
   * to field IDENTITY as much as kind identity. This only makes an
   * underscore-joined key readable; it carries no knowledge of what any
   * specific field means. */
  engineFieldLabel: (field: string) => field.replace(/_/g, " "),

  // model/engineForm.ts's formErrors() — pure decision logic that must not
  // own its own copy, same rule nodeForm.ts's validate() follows (see
  // nodeIdInvalid et al. above). Requiredness-only, mirroring
  // app.engine_kinds.validate_engines (engine_kinds.py:116-140); the
  // backend 422 stays authoritative.
  engineResourceRequired: "resource name is required",
  engineGpuRequired: "select a GPU",
  engineConnectionFieldRequired: (field: string) =>
    `${field.replace(/_/g, " ")} is required for this kind`,

  // GpuStatsBlock (Task 4) — the GPU-Monitor-format stats row's own labels
  // (util's visible caption; temp/power's tooltip names, since their units
  // — °C, W — already say what the number is). model/gpuStats.ts owns the
  // tone/format decisions; this module owns only the words.
  gpuUtil: "Util",
  gpuTemp: "Temp",
  gpuPower: "Power",
};

/** Pill tone for an ENGINE'S OWN state word — a per-kind vocabulary
 * (`world.tenants[r].state` locally, `world.remote_tenants[k].state` off-box),
 * NOT `LifecycleStatus`, which is `ui/StatePill`'s closed enum. The two take
 * the same four tones so they can never disagree about what green means.
 *
 * One map, two call sites — `PlacementActions` (local declared resources)
 * and `RemoteEngineActions` (declared remote engines) — because a state
 * that is amber on one card and grey on the other is exactly the drift this
 * module exists to prevent. Anything unlisted is "off": a word this UI has
 * never seen is not a claim about health.
 *
 * "down" is grey deliberately (sglang-omni's not-serving word,
 * app/engine_kinds.py:899-900): the engine's own vocabulary cannot tell a
 * deliberate unload from a death, and the chip's lifecycle pill beside it —
 * `parked` vs `down` — is what carries that difference. */
export function stateTone(state: string): string {
  return ({
    loaded: "good",
    running: "good",
    loading: "busy",
    busy: "busy",
    unloaded: "off",
    parked: "off",
    idle: "off",
    down: "off",
    unknown: "bad",
  })[state] ?? "off";
}

/** "26h", "4m", "3d" — a compact age for a timestamp, or null when there is
 * no timestamp to age. NodeCard computes this once per node and threads it
 * into every place that fact is shown: the header's own age span, the
 * `nodeUnreachable` banner body, and `staleAge` on ResourcePanel (which
 * feeds the per-chip `lastSeen` caption) — one computed age, several
 * renderings, rather than each caller re-deriving it.
 *
 * `now` is injected so this stays pure and testable; production callers omit
 * it. Ages clamp at zero: a node whose clock runs ahead of ours must not be
 * reported as last seen in the future.
 *
 * Stays in hours through the second day (< 48h) before switching to days —
 * a 24h cutoff would print a 26-hour-old reading as "1d", which throws away
 * exactly the precision an operator needs right after a node drops. */
export function humanizeAge(iso: string | null, now: number = Date.now()): string | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;

  const seconds = Math.max(0, Math.floor((now - then) / 1000));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 2 * 86_400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86_400)}d`;
}
