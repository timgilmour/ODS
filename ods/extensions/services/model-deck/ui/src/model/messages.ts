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

  // Distinct from nodeUnreachable, which reports what the BACKEND says about
  // a node. This one says THIS PAGE could not reach the deck's own endpoint
  // for that node — so everything below it may be minutes old even while the
  // backend's own view still calls the node reachable.
  nodeFetchFailed: (label: string, detail: string): Message => ({
    tone: "danger",
    title: `Cannot reach ${label} from this page`,
    body: `${detail} — what follows may be out of date.`,
  }),

  guardRefused: (detail: string): Message => ({
    tone: "danger",
    title: "Refused",
    body: detail,
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
  appSubtitle: "GPU/VRAM control for lemonade, ComfyUI, hipfire",
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
  // plan_apply() emits: unload_lemonade (sets.py:237), free_comfyui (:244),
  // warn (:246,:265,:296), park_hipfire (:258), activate (:263),
  // resume_hipfire (:288), load_lemonade (:298), policy_patch (:302).
  stepUnload: "Unload",
  stepLoad: "Load",
  stepFreeComfyui: "Free ComfyUI VRAM",
  stepParkHipfire: "Park hipfire",
  stepResumeHipfire: "Resume hipfire",
  stepActivate: "Activate catalog model",
  stepPolicyPatch: "Apply policy overrides",
  stepWarn: "Warning",
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
};

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
