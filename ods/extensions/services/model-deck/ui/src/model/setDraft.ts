import type {
  ConfigSet,
  Durable,
  ResourceDesired,
  ResourceTenant,
  TenantPolicy,
} from "../api";
import { sortedResourceEntries } from "./nodes";

/** The lemonade default-route prefix for library files — mirrors
 * SetBuilder's original constant; app/sets.py stores the route string
 * verbatim, the UI derives library membership from this prefix alone. */
export const EXTRA_PREFIX = "extra.";

/** Which `desired` values a resource's declared KIND actually accepts, and
 * whether it may carry `model` — mirrors app/sets.py's `_DESIRED_VERBS`
 * table (:129-134) crossed with each kind's `human_verbs()`
 * (app/engine_kinds.py — lemonade :234, comfyui :494, hipfire :627), the
 * SAME verb-generic rule `Ephemeral`'s validator enforces server-side
 * (app/sets.py:194-204). `KNOWN_KINDS` (app/engine_kinds.py:177-192) is a
 * closed backend enum today (lemonade/comfyui/hipfire) — a real fourth kind
 * needs a new entry here, same as `nodes.ts`'s `tenantPlacement` and
 * `PlacementActions.tsx` branch on the same three.
 *
 * `supportsModel` is true only for a load-verb kind (today: lemonade
 * alone, app/sets.py:203-204) — the one case a "loaded" goal carries a
 * specific file, driven by the Set Builder's model-library drop target
 * rather than the plain choice buttons every other kind/desired
 * combination uses.
 *
 * `optionLabels` is PER KIND, not a single global table keyed on `desired`
 * alone — "loaded" means a different VERB depending on which kind accepts
 * it (app/sets.py's `_DESIRED_VERBS`: "loaded" -> {load, resume}), so it
 * reads "Load" for a load-verb kind but "Running" for a resume-verb one
 * (hipfire). Fix-loop finding: a single `desired -> label` table read
 * "Running" for EVERY kind offering "loaded", including a SECOND
 * lemonade-kind resource that isn't picked as the model-library target
 * (`pickLoadResource` below) and therefore falls to the generic
 * choice-button row too — that resource would have misread "Running"
 * (hipfire's verb) on what is actually a Load button. */
export interface KindDraftSpec {
  options: readonly ResourceDesired["desired"][];
  supportsModel: boolean;
  optionLabels: Partial<Record<ResourceDesired["desired"], string>>;
}

export const KIND_DRAFT_SPEC: Record<string, KindDraftSpec> = {
  lemonade: {
    options: ["loaded", "unloaded"], supportsModel: true,
    optionLabels: { loaded: "Load", unloaded: "Unload" },
  },
  hipfire: {
    options: ["loaded", "parked"], supportsModel: false,
    optionLabels: { loaded: "Running", parked: "Parked" },
  },
  comfyui: {
    options: ["freed"], supportsModel: false,
    optionLabels: { freed: "Free" },
  },
};

/** The declared resource whose kind carries a model (E1: any number of
 * declared resources now — `supportsModel` is true only for a load-verb
 * kind, lemonade today) — the Set Builder's model-library drop
 * target/picker is keyed to it. Sorted the SAME way the board orders cards
 * (`sortedResourceEntries`: gpu_index then resource name) before picking,
 * so the choice is deterministic — a plain unsorted `Object.entries().find`
 * would pick whichever key JS's engine happened to enumerate first, which
 * for a non-numeric-string-keyed object is insertion order and therefore
 * whichever resource the operator (or a loaded set) happened to declare
 * first, not a stable, predictable one. `null` when nothing declared
 * supports a model (the library panel then does not render — SetBuilder's
 * job, not this function's). */
export function pickLoadResource(tenants: Record<string, ResourceTenant>): string | null {
  const sorted = sortedResourceEntries(tenants);
  return sorted.find(([, tenant]) => KIND_DRAFT_SPEC[tenant.engine]?.supportsModel)?.[0] ?? null;
}

/** The draft's current choice for one resource: "none" (absent — the
 * "don't touch" default every resource starts at) or one of the
 * backend-verb-generic desired values (app/sets.py's `ResourceDesired`).
 * Never invented here: reads exactly what is IN the map, nothing derived. */
export type DesiredChoice = "none" | ResourceDesired["desired"];

export function draftChoiceFor(
  resources: Record<string, ResourceDesired>,
  resource: string,
): DesiredChoice {
  return resources[resource]?.desired ?? "none";
}

/** Sets (or clears, for "none") one resource's entry immutably. "Don't
 * touch" is modeled as an ABSENT key (app/sets.py's `Ephemeral.resources`
 * docstring: "An ABSENT resource means don't touch it"), never a
 * present-but-null value, so a round trip through `buildDraft` can never
 * accidentally touch a resource the operator never chose. */
export function setResourceChoice(
  resources: Record<string, ResourceDesired>,
  resource: string,
  choice: DesiredChoice,
): Record<string, ResourceDesired> {
  if (choice === "none") {
    const next = { ...resources };
    delete next[resource];
    return next;
  }
  return { ...resources, [resource]: { desired: choice } };
}

/** The editable fields of the Set Builder, as one value. Pure data — the
 * component holds ONE of these in state instead of nine separate hooks
 * deciding anything. */
export interface DraftFields {
  name: string;
  notes: string;
  durable: Durable | null;
  /** resource -> desired goal. E1 Task 8/11: replaces the old fixed
   * lemonade/comfyui/hipfire legs — any number of any-kind DECLARED
   * resources now, keyed by name. An absent resource means "don't touch
   * it" (CRITICAL 2's invariant, generalized: a set that never mentioned a
   * resource must never silently gain a concrete directive on the next
   * save). */
  resources: Record<string, ResourceDesired>;
  policyOverrides: Record<string, TenantPolicy> | null;
}

export function buildDraft(f: DraftFields): ConfigSet {
  return {
    name: f.name.trim(),
    notes: f.notes,
    durable: f.durable,
    ephemeral: { resources: f.resources },
    policy_overrides: f.policyOverrides,
  };
}

/** Load a saved set into editable fields. CRITICAL 2 lives here: a resource
 * ABSENT from the saved set's `ephemeral.resources` stays absent from the
 * draft — never defaulted to a concrete "none" placeholder that a save
 * could turn into a real directive. policy_overrides carries verbatim for
 * the same reason. */
export function fieldsFromSet(cfgset: ConfigSet, clearName: boolean): DraftFields {
  return {
    name: clearName ? "" : cfgset.name,
    notes: cfgset.notes,
    durable: cfgset.durable,
    resources: cfgset.ephemeral?.resources ?? {},
    policyOverrides: cfgset.policy_overrides ?? null,
  };
}

/** Resource names present in the draft's `resources` map that are NOT among
 * the CURRENTLY declared engines — e.g. a set saved while "gguf-a" was
 * declared, loaded again after "gguf-a" was forgotten (DELETE
 * /api/nodes/local/engines/{resource}, Task 10). `fieldsFromSet` keeps such
 * an entry verbatim (CRITICAL 2 says an untouched resource must never be
 * silently dropped either), but `SetBuilder`'s card list only ever iterates
 * DECLARED resources (`sortedResourceEntries(world.tenants)`), so with
 * nothing else, the entry would be invisible on screen AND unremovable —
 * and still ride along on Save, where `app/sets.py:197`'s Ephemeral
 * validator refuses it ("is not a currently declared engine", 422) with no
 * hint anywhere in the UI why. This is the pure DETECTION half of that fix;
 * the component renders one row per name this returns, each with a remove
 * control that calls `setResourceChoice(resources, name, "none")` — the
 * same removal primitive every other resource's "don't touch" choice
 * already uses. Sorted for deterministic rendering. */
export function undeclaredResources(
  resources: Record<string, ResourceDesired>,
  declaredResourceNames: readonly string[],
): string[] {
  const declared = new Set(declaredResourceNames);
  return Object.keys(resources).filter((r) => !declared.has(r)).sort();
}

/** Best-effort: only durable.default_route_model with the "extra." prefix
 * AND an active "loaded" goal on SOME resource can be traced back to a
 * library model file. A set naming some other litellm route can't be — the
 * chip is left blank and the user re-drops to pin one down.
 *
 * Generalized past the old lemonade-only check: with any number of
 * declared resources, this function alone (no kind context) cannot say
 * WHICH resource's "loaded" goal the durable route belongs to — it only
 * asks whether ANY resource in the set is drafted loaded, which is exactly
 * what the pre-Task-8 lemonade-only check asked too, since there was only
 * ever one candidate. */
export function derivePlacedModel(cfgset: ConfigSet): string | null {
  const route = cfgset.durable?.default_route_model;
  const derived =
    route && route.startsWith(EXTRA_PREFIX)
      ? route.slice(EXTRA_PREFIX.length)
      : null;
  const anyLoaded = Object.values(cfgset.ephemeral?.resources ?? {}).some(
    (rd) => rd.desired === "loaded",
  );
  return anyLoaded ? derived : null;
}

/** `ephemeral.resources`' key order sorted — a dynamic Record (E1 Task 8/11)
 * can accumulate keys in a different insertion order between a loaded set
 * and a freshly edited draft holding the SAME content, which the old fixed
 * lemonade/comfyui/hipfire shape could never do (three literal fields,
 * always the same order). `draftEquals` needs this so an equivalent set
 * never reads as "changed" and blocks Preview for no reason. */
function withSortedResources(cfgset: ConfigSet): ConfigSet {
  const resources = cfgset.ephemeral?.resources;
  if (!resources) return cfgset;
  const sorted = Object.fromEntries(
    Object.entries(resources).sort(([a], [b]) => a.localeCompare(b)),
  );
  return { ...cfgset, ephemeral: { ...cfgset.ephemeral, resources: sorted } };
}

/** A rough VRAM estimate for one resource under the CURRENT draft choice —
 * Set Builder budget feedback only (a Meter fill + the over-budget banner),
 * NEVER sent to the backend (the wire payload is `ResourceDesired`, which
 * has no footprint field). "don't touch" (`choice === "none"`) shows the
 * LIVE reading (what's there right now, same as the pre-Task-8 null-leg
 * rule); a load-verb kind's "loaded" choice shows the PICKED file's size
 * (the caller supplies it — only the component knows which file is
 * selected); hipfire-kind's "loaded" (resume) uses the fixed budget figure
 * the caller passes (app/registry.py's HIPFIRE_FOOTPRINT — hipfire has no
 * live per-model footprint reading the way lemonade does). Every other
 * combination (parked/unloaded/freed, or a kind with no footprint concept
 * at all — comfyui) has no drafted delta modeled: this stays at the live
 * reading, same as "don't touch". */
export function draftedFootprintBytes(
  tenant: ResourceTenant,
  choice: DesiredChoice,
  pickedFileBytes: number,
  hipfireFootprintBytes: number,
): number {
  // Kind literals: same closed backend enum as KIND_DRAFT_SPEC above
  // (KNOWN_KINDS, app/engine_kinds.py:177-192).
  if (tenant.engine === "lemonade") {
    if (choice === "loaded") return pickedFileBytes;
    if (choice === "unloaded") return 0;
    return tenant.footprint ?? 0;
  }
  if (tenant.engine === "hipfire") {
    if (choice === "loaded") return hipfireFootprintBytes;
    if (choice === "parked") return 0;
    return tenant.footprint ?? 0;
  }
  return tenant.footprint ?? 0;
}

/** Structural equality for the save-gating ("Preview steps" is only
 * reachable while the saved copy matches the live draft). JSON stringify is
 * sufficient once resource-key order is normalized: both sides are built by
 * buildDraft from the same field order otherwise. */
export function draftEquals(a: ConfigSet, b: ConfigSet): boolean {
  return JSON.stringify(withSortedResources(a)) === JSON.stringify(withSortedResources(b));
}
