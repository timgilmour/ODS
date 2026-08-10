/**
 * The pure settings-buffer module — the seam between the ladder's read-only
 * resolution (app/ladder.py:resolve_settings, mirrored by `ResolvedEntry`)
 * and the Settings panel's three write kinds (`SettingsKind`: engines,
 * models, engine_models). Everything here is a function of its inputs: no
 * React, no fetch, no clock — mirrors src/model/nodes.ts's shape.
 *
 * The panel edits one `kind` at a time but the ladder resolves across all
 * five layers, so this module's whole job is reconciling "what did the
 * operator just type" (the buffer) against "what does the server currently
 * think wins" (`resolved`) without ever pretending to know more than the
 * client actually can — see buildChips's pendingRemove handling below.
 */

import type { ArgValue, FactsMap, Layer, ResolvedEntry, SettingsKind, Widget } from "../api";

// ---------------------------------------------------------------------------
// Value text — the one place an ArgValue becomes readable text and back
// ---------------------------------------------------------------------------

/** app/argline.py's `POSITIONAL_KEY` — the reserved key holding a command's
 * leading positional tokens ("serve /model" leads every adopted vLLM
 * profile). It is not a flag, and app/argline.py keeps its value
 * `list[str]`-shaped ALWAYS (the F1 fix: collapsing a one-element positional
 * to a scalar makes `render_argline` iterate the string character by
 * character), which is the one exception `parseValueText` below encodes. */
export const POSITIONAL_KEY = "_positional";

/** An `ArgValue` as an operator reads it.
 *
 * `true` is the bare-flag value end-to-end — app/argline.py renders `value is
 * True` as the flag alone, and `parse_argline` produces `True` for a flag
 * with no following token — so it renders as empty text: there is no `=` and
 * no value to show, only the flag's presence. A list joins on spaces because
 * that is exactly how `render_argline` emits a multi-value flag. */
export function displayValue(value: ArgValue): string {
  if (value === true) return "";
  if (value === false) return String(value);
  if (Array.isArray(value)) return value.join(" ");
  return value;
}

/** The inverse, for the panel's list editor: whitespace-separated tokens,
 * the same split `parse_argline` performs (via shlex) on a rendered line.
 *
 * A one-token list collapses to a scalar because app/argline.py's ruled
 * normalization (`normalize_args_map`, singleton-list axis) does exactly that
 * on write anyway — EXCEPT for `POSITIONAL_KEY`, which is exempt from that
 * axis backend-side and so is kept list-shaped here too, rather than relying
 * on the store to re-wrap what this module flattened. */
export function parseValueText(name: string, text: string): ArgValue {
  const parts = text.trim().split(/\s+/).filter((p) => p !== "");
  if (name === POSITIONAL_KEY) return parts;
  if (parts.length === 1) return parts[0];
  return parts;
}

/** Whether an edited value must be read back through `parseValueText` (as a
 * list) rather than committed as the typed scalar.
 *
 * The catalog's `widget` alone is NOT sufficient, and trusting it was a real
 * defect: `widget` degrades to "text" for any key the harvest has no option
 * for, and for EVERY key when the catalog is null (a pair that never
 * harvested — a supported state, app/routers/settings.py:get_catalog) or when
 * the catalog fetch failed. A live `--served-model-name` holding six values
 * would then open in a text editor and, on a blur with nothing typed, commit
 * back as ONE space-containing scalar — which `render_argline` quotes into a
 * single argument. That is a different command line than the one that was
 * there, produced by merely looking at a chip.
 *
 * So the value's own CURRENT shape is a second, independent signal: if the
 * resolution says this key holds a list, an edit of it stays a list whatever
 * the catalog does or does not know about the key. */
export function isListEdit(widget: Widget, current: ArgValue): boolean {
  return widget === "list" || Array.isArray(current);
}

// ---------------------------------------------------------------------------
// Scope keys
// ---------------------------------------------------------------------------

export interface ScopeKeys {
  engines: string;
  models: string | null;
  engine_models: string | null;
}

/** Mirrors app/routers/settings.py:_resolve's three store keys exactly:
 * `${node}/${engine}` (always present — the engine scope needs no model),
 * `model` verbatim (null with no model in scope), and the compound
 * `${node}/${engine}|${model}` (null under the same condition, since it
 * cannot be built without a model). */
export function scopeKeys(node: string, engine: string, model: string | null): ScopeKeys {
  const engineKey = `${node}/${engine}`;
  return {
    engines: engineKey,
    models: model,
    engine_models: model === null ? null : `${engineKey}|${model}`,
  };
}

/** Which model identity a Settings panel must be opened for, given a
 * placement's SETTINGS KEY (see `settingsKeyOf`).
 *
 * CORRECTION, max-review #8: an earlier version of this docstring said "a
 * spark placement is named by its PROFILE". It is not — `Placement.name` is
 * what the endpoint SERVES (`--served-model-name`; mm27b serves as "aeon"),
 * and callers that passed it here missed the lookup entirely whenever the
 * two differ. The profile rides separately on `Placement.profile`. The claim
 * mattered because this docstring is the authority the next implementer
 * reads, and it asserted the exact premise the fix refuted.
 *
 * The PROFILE is what /api/spark/swap takes and what intent records
 * (app/routers/spark.py deliberately records profiles). Settings, though,
 * live under the real
 * CHECKPOINT identity: `engine_models/<node>/<engine>|<identity>`. The
 * translation is the `profile_identities` characteristics fact, written by
 * the adopt sweep (app/routers/settings.py:319) and read exactly this way by
 * the backend itself — app/routers/spark.py:143 for reload, and
 * app/routers/__init__.py:160-215 for settings drift, whose docstring names
 * the failure this prevents: a PUT to `engine_models/sparky/vllm|<identity>`
 * never registers against the verbatim `sparky/spark|heretic` key intent
 * builds, so settings for the spark slot go silently dead (the D11 defect).
 *
 * Every miss falls back to the placement's own name, which is the honest
 * answer in each case it can happen: a LOCAL tenant's placement name IS its
 * model, and a spark profile with no adopted identity has nothing else to be
 * called. Never throws — `FactEntry.value` is `unknown` by contract, so each
 * step is checked rather than cast.
 */
/** The value `settingsIdentityFor` must be keyed on for a placement.
 *
 * One function rather than `placement.profile ?? placement.name` repeated at
 * each call site: both callers live inside components, where no test can
 * reach them (this UI has no component harness), so a fifth inline copy of
 * the rule would be a fifth place for it to drift — the recurrence this
 * codebase already answered structurally in model/editState.ts.
 *
 * `profile` is spark-only and absent for local placements, whose `name` IS
 * the model identity — so the fallback is the correct answer, not a guess. */
export function settingsKeyOf(placement: { name: string; profile?: string }): string {
  return placement.profile ?? placement.name;
}

export function settingsIdentityFor(
  facts: FactsMap,
  node: string,
  engine: string,
  profileOrName: string,
): string {
  // The engine-scoped facts key, `<kind>/<id>` like every other one
  // (app/routers/facts.py:64-71); the id for an engine is `<node>/<engine>`.
  const identities = facts[`engine/${node}/${engine}`]?.profile_identities?.value;
  if (typeof identities !== "object" || identities === null || Array.isArray(identities)) {
    return profileOrName;
  }
  const info = (identities as Record<string, unknown>)[profileOrName];
  if (typeof info !== "object" || info === null) return profileOrName;
  const identity = (info as Record<string, unknown>).identity;
  return typeof identity === "string" && identity !== "" ? identity : profileOrName;
}

/** Which declared layer each write kind targets — engines -> engine,
 * models -> model, engine_models -> engine_model (app/ladder.py:48's
 * LAYERS, the three declared entries). */
export const LAYER_FOR_KIND: Record<SettingsKind, Layer> = {
  engines: "engine",
  models: "model",
  engine_models: "engine_model",
};

const ALL_KINDS: SettingsKind[] = ["engines", "models", "engine_models"];

/** Layer specificity, most-specific LAST — mirrors app/ladder.py:48's
 * `LAYERS` tuple exactly: `resolve_settings` walks this same order, each
 * later layer overwriting an earlier one's value for the same key. RANK's
 * index is that same "more specific wins" ordering, used here to decide
 * whether a pending edit at one kind outranks the ladder's current winner. */
const RANK: Layer[] = ["engine_defaults", "checkpoint_recommendations", "engine", "model", "engine_model"];

function rankOf(layer: Layer): number {
  return RANK.indexOf(layer);
}

// ---------------------------------------------------------------------------
// Buffer — the operator's uncommitted edits, one sets/removes list per kind
// ---------------------------------------------------------------------------

export interface Buffer {
  sets: Partial<Record<SettingsKind, Record<string, ArgValue>>>;
  removes: Partial<Record<SettingsKind, string[]>>;
}

export const emptyBuffer: Buffer = { sets: {}, removes: {} };

/** Records a pending write. A fresh set on a key supersedes any pending
 * remove of that SAME key at the SAME kind — a key cannot be both
 * pending-set and pending-removed for one kind at once, so setting after
 * removing simply cancels the remove. */
export function bufferSet(b: Buffer, kind: SettingsKind, name: string, value: ArgValue): Buffer {
  const sets = { ...b.sets, [kind]: { ...b.sets[kind], [name]: value } };
  const removes = { ...b.removes };
  const kindRemoves = removes[kind];
  if (kindRemoves && kindRemoves.includes(name)) {
    removes[kind] = kindRemoves.filter((n) => n !== name);
  }
  return { sets, removes };
}

/** Records a pending removal. Guarding WHICH keys are legal to remove
 * (only ones whose winning value currently sits at `kind`, per buildChips's
 * `setAtKind`) is the caller's job — this function has no `resolved` to
 * judge that against, and stays a pure buffer-shape operation.
 *
 * Dropping a pending SET at the same key/kind is the one thing this
 * function always does: that edit never reached the server, so there is
 * nothing to tell it to remove, and no `removes` entry is added for it — a
 * true "undo". Only when there was no pending set to drop does the key land
 * in `removes[kind]`, which is what a real server-side removal needs. */
export function bufferRemove(b: Buffer, kind: SettingsKind, name: string): Buffer {
  const kindSets = b.sets[kind];
  if (kindSets && name in kindSets) {
    const nextKindSets = { ...kindSets };
    delete nextKindSets[name];
    return { sets: { ...b.sets, [kind]: nextKindSets }, removes: b.removes };
  }
  const kindRemoves = b.removes[kind] ?? [];
  if (kindRemoves.includes(name)) return b;
  return { sets: b.sets, removes: { ...b.removes, [kind]: [...kindRemoves, name] } };
}

/** A brand-new "+ Add option" whose editor has not committed anything yet —
 * the name, and the kind it was buffered at (the panel's write scope can be
 * switched out from under an open editor, so the kind cannot be re-read from
 * the panel's current state at cancel time). */
export interface PendingAdd {
  name: string;
  kind: SettingsKind;
}

/** Drops an abandoned brand-new add.
 *
 * "+ Add option" has to buffer a starting value immediately — the chip is
 * rendered FROM the buffer (buildChips's "pending set on a key `resolved`
 * doesn't have at all" branch), so there is no chip, and therefore no editor,
 * until the set exists. For the 100 live vLLM options that start at `""`
 * (catalogFilter's `startingValueFor` — see editState's `UncommittedAdd`),
 * and the editor's own empty-means-cancel rule (SettingsModal's `commit`)
 * deliberately does NOT write `""` back — so an operator who adds an option
 * and then Escapes, blurs, or walks away left the `""` sitting in the buffer,
 * and the next Save shipped `--flag ''`. This is the undo for that.
 *
 * `bufferRemove` is exactly the right operation and not an approximation of
 * one: on a key with a pending set at the same kind it DELETES the set and
 * records no removal at all (see its docstring) — which is what an edit the
 * server never saw deserves. Callers must therefore clear their `PendingAdd`
 * the moment the add is committed or removed by other means, or this would
 * turn into a real `removes` entry naming a key that scope never had. */
export function discardPendingAdd(b: Buffer, add: PendingAdd | null): Buffer {
  if (add === null) return b;
  return bufferRemove(b, add.kind, add.name);
}

export function isDirty(b: Buffer): boolean {
  return (
    Object.values(b.sets).some((m) => m !== undefined && Object.keys(m).length > 0) ||
    Object.values(b.removes).some((arr) => arr !== undefined && arr.length > 0)
  );
}

// ---------------------------------------------------------------------------
// Chips — the resolved ladder + buffer, viewed through one selected kind
// ---------------------------------------------------------------------------

export interface ChipView {
  name: string;
  value: ArgValue;
  layer: Layer;
  origin: "derived" | "declared";
  /** Winning value comes from the selected write kind, OR there is a
   * pending set at that kind for this key (even when it's shadowed by a
   * more specific declared layer) — the edit is real either way. */
  setAtKind: boolean;
  pendingRemove: boolean;
  pendingSet: boolean;
}

/** Builds the two chip lists for one selected `kind`.
 *
 * `applied` = every `origin === "derived"` entry (engine_defaults,
 * checkpoint_recommendations) — never editable, never touched by the
 * buffer. `declared` = everything else, sorted by name, with the buffer
 * folded in:
 *
 * - A pending set at `kind` overrides the shown value iff `kind`'s layer
 *   outranks or ties the resolved winner's layer (RANK above). Below that
 *   rank the shown value is untouched, but `pendingSet` still reads true —
 *   the edit is real, just shadowed by a more specific declared layer, and
 *   downstream UI can badge that.
 * - A pending set on a key `resolved` doesn't have at all creates a brand
 *   new chip at `kind`'s layer.
 * - A pending set on a key whose resolved entry is DERIVED (a harvested
 *   engine default or checkpoint recommendation — the ladder's core "I want
 *   to override what the engine already applies" case) does NOT touch that
 *   key's `applied` chip: the engine's own default keeps showing exactly as
 *   before. Instead it adds a SECOND, independent chip to `declared` — the
 *   override-in-progress — at `kind`'s layer with the pending value. The two
 *   coexist deliberately: one says "the engine applies this", the other
 *   says "you are about to declare this"; only Save reconciles them (into a
 *   single declared winner on the next `resolved`).
 * - A pending remove at the winning kind keeps the chip (never drops it)
 *   with `pendingRemove: true` and TODAY's resolved value. This is a
 *   deliberate approximation, not an oversight: `resolved` is already the
 *   ladder's single-winner-per-key view — there is no lower layer's value
 *   left in it to fall back to client-side. Exactness comes only from the
 *   server's refetch after Save; until then this is display-only.
 */
export function buildChips(
  resolved: Record<string, ResolvedEntry>,
  b: Buffer,
  kind: SettingsKind,
): { declared: ChipView[]; applied: ChipView[] } {
  const layer = LAYER_FOR_KIND[kind];
  const kindRank = rankOf(layer);
  const pendingSets = b.sets[kind] ?? {};
  const pendingRemoves = b.removes[kind] ?? [];

  const declared: ChipView[] = [];
  const applied: ChipView[] = [];
  const seen = new Set<string>();

  for (const [name, entry] of Object.entries(resolved)) {
    seen.add(name);

    if (entry.origin === "derived") {
      applied.push({
        name,
        value: entry.value,
        layer: entry.layer,
        origin: "derived",
        setAtKind: false,
        pendingRemove: false,
        pendingSet: false,
      });
      // A pending set at `kind` on a currently-derived key is a real,
      // in-progress override — the ladder's core use case — and gets its
      // OWN declared-section chip rather than vanishing with no feedback
      // while still shipping on Save (toPuts/mergedArgsForPreview read the
      // buffer directly and always carried this edit; only the chip was
      // missing).
      if (Object.hasOwn(pendingSets, name)) {
        declared.push({
          name,
          value: pendingSets[name],
          layer,
          origin: "declared",
          setAtKind: true,
          pendingRemove: false,
          pendingSet: true,
        });
      }
      continue;
    }

    const hasPendingSet = Object.hasOwn(pendingSets, name);
    const overrides = hasPendingSet && kindRank >= rankOf(entry.layer);
    const value = overrides ? pendingSets[name] : entry.value;
    const winningLayer = overrides ? layer : entry.layer;

    declared.push({
      name,
      value,
      layer: winningLayer,
      origin: "declared",
      setAtKind: winningLayer === layer || hasPendingSet,
      pendingRemove: pendingRemoves.includes(name),
      pendingSet: hasPendingSet,
    });
  }

  // Pending sets on keys `resolved` has no entry for at all.
  for (const [name, value] of Object.entries(pendingSets)) {
    if (seen.has(name)) continue;
    declared.push({
      name,
      value,
      layer,
      origin: "declared",
      setAtKind: true,
      pendingRemove: false,
      pendingSet: true,
    });
  }

  declared.sort((a, c) => a.name.localeCompare(c.name));
  return { declared, applied };
}

// ---------------------------------------------------------------------------
// Puts — the buffer, shaped for PUT /settings/{kind}/{key}
// ---------------------------------------------------------------------------

/** One PUT payload per kind that has anything pending. Kinds whose
 * `ScopeKeys` entry is null are refused (throw) — a model-less panel has no
 * store key to write model-scoped edits to in the first place, so buffering
 * them was already a bug the moment it happened; this is where it surfaces. */
export function toPuts(
  b: Buffer,
  keys: ScopeKeys,
): Array<{ kind: SettingsKind; key: string; values: Record<string, ArgValue>; remove: string[] }> {
  const puts: Array<{ kind: SettingsKind; key: string; values: Record<string, ArgValue>; remove: string[] }> = [];

  for (const kind of ALL_KINDS) {
    const values = b.sets[kind] ?? {};
    const remove = b.removes[kind] ?? [];
    if (Object.keys(values).length === 0 && remove.length === 0) continue;

    const key = keys[kind];
    if (key === null) {
      throw new Error(
        `toPuts: no scope key for "${kind}" — a model-less panel cannot buffer model-scoped edits`,
      );
    }
    puts.push({ kind, key, values, remove });
  }

  return puts;
}

// ---------------------------------------------------------------------------
// Merged preview args — for POST /settings/preview's live argline
// ---------------------------------------------------------------------------

/** The args map to ship to `previewRender` for the live argline: start from
 * the resolution's DECLARED-ONLY winners (origin === "declared" — derived
 * layers are never shipped, app/routers/settings.py's `_declared_only`
 * rule), fold in every kind's pending sets by rank (a higher-rank kind's
 * pending set beats both the resolved winner and a lower-rank kind's
 * pending set), then delete every pending-removed key regardless of kind.
 * Order matters: sets are folded in first, removals applied last, so a
 * remove always wins over a same-key pending set at a different kind. */
export function mergedArgsForPreview(
  resolved: Record<string, ResolvedEntry>,
  b: Buffer,
): Record<string, ArgValue> {
  const merged: Record<string, ArgValue> = {};
  const layerOf: Partial<Record<string, Layer>> = {};

  for (const [name, entry] of Object.entries(resolved)) {
    if (entry.origin !== "declared") continue;
    merged[name] = entry.value;
    layerOf[name] = entry.layer;
  }

  for (const kind of ALL_KINDS) {
    const layer = LAYER_FOR_KIND[kind];
    const pendingSets = b.sets[kind] ?? {};
    for (const [name, value] of Object.entries(pendingSets)) {
      const currentLayer = layerOf[name];
      if (currentLayer === undefined || rankOf(layer) >= rankOf(currentLayer)) {
        merged[name] = value;
        layerOf[name] = layer;
      }
    }
  }

  for (const kind of ALL_KINDS) {
    for (const name of b.removes[kind] ?? []) {
      delete merged[name];
      delete layerOf[name];
    }
  }

  return merged;
}
