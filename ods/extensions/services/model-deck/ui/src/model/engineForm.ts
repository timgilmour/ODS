/**
 * Local-engine add/edit form logic — pure, componentless (the same "logic
 * inline in a component is logic no test can reach" rule nodeForm.ts's
 * docstring names).
 *
 * Every per-kind field — which connection keys exist, and which of them are
 * required — comes from `GET /api/engine-kinds` (app/routers/nodes.py:476-504's
 * `list_engine_kinds`, sourced from `KNOWN_KINDS`, app/engine_kinds.py:177-192),
 * NEVER a UI literal (spec §5). `EngineFormState` bakes the matching kind's
 * required-field LIST in at construction time (`emptyForm`/`formForEntry`/
 * `withKind`) precisely so `canSave` needs no second lookup against the
 * kinds payload — the requiredness check travels with the form buffer it
 * gates.
 *
 * Requiredness here mirrors the backend for SAVE-GATING ONLY — the backend's
 * `app.engine_kinds.validate_engines` stays the sole authority; a 422 from
 * it always wins however permissive this module's own gate is (resource
 * shape, duplicate names, and gpu_index sanity beyond "something is picked"
 * are deliberately NOT re-implemented here — see `formErrors`'s docstring).
 */

import type {
  DeclaredEngine,
  EngineKindDef,
  EngineKindsResponse,
  EnginePolicyDefaults,
  RemoteTenant,
  ResourceTenant,
} from "../api";
import { labels } from "./messages";

export interface EngineFormState {
  resource: string;
  kind: string;
  connection: Record<string, string>;
  /** The selected kind's REQUIRED connection field names, baked in at
   * construction (emptyForm/formForEntry) or on a kind switch (withKind) —
   * see this module's docstring for why canSave() takes no `kinds` argument
   * of its own. */
  requiredConnectionFields: string[];
  gpuIndex: number | null;
  priority: number;
  pinned: boolean;
  idleTtl: number;
}

/** A freshly-declared engine's starting policy — priority 0 (no eviction
 * preference over anything else), not pinned, idle_ttl 0 (app/engine_kinds.py's
 * per-kind `idle_action`s all gate on `policy["idle_ttl"] > 0`, so 0 means
 * "never idle-release" — the conservative default for something the
 * operator just declared, rather than risking a surprise eviction seconds
 * after Save). */
const DEFAULT_POLICY: EnginePolicyDefaults = { priority: 0, pinned: false, idle_ttl: 0 };

function schemaFor(
  kinds: EngineKindsResponse,
  kind: string,
): Record<string, { required: boolean }> {
  return kinds.kinds.find((k) => k.kind === kind)?.connection ?? {};
}

function requiredFieldsOf(schema: Record<string, { required: boolean }>): string[] {
  return Object.entries(schema)
    .filter(([, spec]) => spec.required)
    .map(([field]) => field);
}

/** A blank connection buffer for `kind` (every field ""), plus its required
 * list baked in — shared by emptyForm and withKind so switching kind on an
 * in-progress form rebuilds both together and never lets one lag the
 * other. Deliberately drops any OTHER kind's typed values: two kinds'
 * fields sharing a name (none do today) would still not be the same
 * fact, so nothing here tries to carry values across a kind switch. */
function connectionFor(
  kinds: EngineKindsResponse,
  kind: string,
): Pick<EngineFormState, "connection" | "requiredConnectionFields"> {
  const schema = schemaFor(kinds, kind);
  return {
    connection: Object.fromEntries(Object.keys(schema).map((field) => [field, ""])),
    requiredConnectionFields: requiredFieldsOf(schema),
  };
}

/** Which of `kinds.kinds` may be declared on the node currently being
 * edited — `isRemote` true for a node-agent target, false for the local
 * one. Mirrors `app.engine_kinds.validate_engines`'s two capability
 * refusals (app/engine_kinds.py:234-235 for a node-agent target's
 * `remote_capable` check, :236-238 for the local mirror) at the picker
 * itself (sglang-omni Task 10, Task 7's review fallout): offering a kind
 * here that the write gate would refuse is exactly the 422-after-the-fact
 * `list_engine_kinds`' own docstring (app/routers/nodes.py:485-492)
 * describes serving both flags to avoid. The Add/Edit kind `<select>` maps
 * over this function's OUTPUT rather than `kinds.kinds` directly
 * (no-logic-inline-in-components) — the capability check itself lives here,
 * once, not in the component's JSX. */
export function kindsFor(kinds: EngineKindsResponse, isRemote: boolean): EngineKindDef[] {
  return kinds.kinds.filter((k) => (isRemote ? k.remote_capable : k.local_capable));
}

export function emptyForm(kinds: EngineKindsResponse, kind: string): EngineFormState {
  return {
    resource: "",
    kind,
    ...connectionFor(kinds, kind),
    gpuIndex: null,
    priority: DEFAULT_POLICY.priority,
    pinned: DEFAULT_POLICY.pinned,
    idleTtl: DEFAULT_POLICY.idle_ttl,
  };
}

/** Edit mode's seed. `resource` rides along but is treated as immutable by
 * every caller (app/routers/nodes.py's `update_engine`, :244-259, 422s a
 * body `resource` that disagrees with the path — rename is refused, "forget
 * and re-add instead") — the form UI disables the field rather than ever
 * constructing a payload that could disagree with it. `kind`, unlike
 * `resource`, IS editable in edit mode: app/state.py's snapshot (:141-150)
 * explicitly accommodates "a resource re-declared under a DIFFERENT kind in
 * place", so this form does not forbid it either. */
export function formForEntry(entry: DeclaredEngine, kinds: EngineKindsResponse): EngineFormState {
  return {
    resource: entry.resource,
    kind: entry.kind,
    connection: { ...entry.connection },
    requiredConnectionFields: requiredFieldsOf(schemaFor(kinds, entry.kind)),
    gpuIndex: entry.gpu_index,
    priority: entry.policy_defaults.priority,
    pinned: entry.policy_defaults.pinned,
    idleTtl: entry.policy_defaults.idle_ttl,
  };
}

/** Kind switch: keeps resource/gpu/policy, rebuilds `connection` for the new
 * kind's own schema (see connectionFor's docstring for why nothing carries
 * across). Used identically by the Add form's kind picker and the Edit
 * form's (state.py's re-declare-under-a-different-kind case). */
export function withKind(
  form: EngineFormState,
  kinds: EngineKindsResponse,
  kind: string,
): EngineFormState {
  return { ...form, kind, ...connectionFor(kinds, kind) };
}

/** Sets ONE connection field. Named exactly this (not e.g.
 * `setConnectionField`) because it is the one field-level mutator this form
 * needs a pure function for — every other field (resource, gpuIndex,
 * priority/pinned/idleTtl) is a trivial top-level `{...form, x}` spread, the
 * same "not logic, just render wiring" line nodeForm.ts's NodeForm already
 * draws for its own label/address/credential fields. */
export function setField(form: EngineFormState, field: string, value: string): EngineFormState {
  return { ...form, connection: { ...form.connection, [field]: value } };
}

/** Whether every REQUIRED connection field (the selected kind's own schema,
 * baked in at construction) is filled — mirrors
 * `app.engine_kinds.validate_engines`'s per-field check
 * (engine_kinds.py:247-249) for save-gating only. Deliberately does NOT
 * also require `resource`/`gpuIndex` to be set — see `formErrors` below for
 * the full gate; this narrower check is its own pinned unit, reused by
 * `formErrors` rather than duplicated. */
export function canSave(form: EngineFormState): boolean {
  return form.requiredConnectionFields.every((field) => Boolean(form.connection[field]?.trim()));
}

/** The full Save-gate + tooltip text: `canSave` above, plus the two
 * structural fields every kind needs regardless of its own connection
 * schema — `app.engine_kinds.validate_engines` requires a non-empty
 * `resource` (engine_kinds.py:224-227) and an integer `gpu_index`
 * (:251-253) unconditionally. Still requiredness-only (no format/duplicate
 * checks — the backend 422 stays authoritative for those, per this
 * module's docstring): an operator can still hit Save with e.g. a
 * duplicate resource name and see the server's own refusal, same as
 * nodeForm.ts's `validate` leaves id-collision entirely to the 409. */
export function formErrors(form: EngineFormState): string[] {
  const errors: string[] = [];
  if (!form.resource.trim()) errors.push(labels.engineResourceRequired);
  if (form.gpuIndex === null) errors.push(labels.engineGpuRequired);
  for (const field of form.requiredConnectionFields) {
    if (!form.connection[field]?.trim()) {
      errors.push(labels.engineConnectionFieldRequired(field));
    }
  }
  return errors;
}

/** The POST/PUT body — exactly `app.engine_kinds.validate_engines`'s
 * accepted shape (resource, kind, connection, gpu_index, policy_defaults;
 * engine_kinds.py:220-223's extra-field check refuses anything else). Only
 * meaningful once `formErrors(form)` is empty (`gpuIndex` is asserted
 * non-null here on that assumption, same posture nodeForm.ts's
 * `toCreatePayload`/`toPatchPayload` take: the Save button stays disabled
 * until then, so this is never called on a form the gate has not cleared). */
export function toPayload(form: EngineFormState): DeclaredEngine {
  return {
    resource: form.resource,
    kind: form.kind,
    connection: { ...form.connection },
    gpu_index: form.gpuIndex as number,
    policy_defaults: { priority: form.priority, pinned: form.pinned, idle_ttl: form.idleTtl },
  };
}

/** Canonical declared-engine ordering: gpu_index then resource name — the
 * same tie-break rule `nodes.ts`'s `sortedResourceEntries` uses for the
 * board (gpu_index alone cannot break a tie between two resources sharing a
 * GPU). Kept as its own small sort rather than routed through
 * `sortedResourceEntries` because that function's input is `world.tenants`
 * (a `Record<string, ResourceTenant>`, the LIVE observation map) while this
 * one sorts the RAW declaration list (`DeclaredEngine[]`, from
 * `listNodeRegistry()`) — different producer, different shape, same
 * ordering rule stated once here instead of forcing a conversion between
 * the two just to reuse the other's comparator. */
export function sortedEngines(engines: DeclaredEngine[]): DeclaredEngine[] {
  return [...engines].sort(
    (a, b) => a.gpu_index - b.gpu_index || a.resource.localeCompare(b.resource),
  );
}

/** The `demand` flag for `kind`, or null when it cannot be known (catalog
 * absent, or a kind it does not carry). Null is deliberately not `false`:
 * messages.ttlConsequence renders unknown as unknown rather than promising
 * a manual-reload consequence nobody verified. */
export function demandFor(
  catalog: EngineKindDef[] | null,
  kind: string,
): boolean | null {
  const entry = catalog?.find((k) => k.kind === kind);
  return entry ? entry.demand : null;
}

/** The `idle_release` flag for `kind` — whether a nonzero idle_ttl on this
 * kind does anything at all — or null when it cannot be known (catalog
 * absent, or a kind it does not carry). Same null-is-not-false posture as
 * `demandFor` above, and the same catalog shape; kept as its own small
 * selector rather than folded into `demandFor`'s return so every existing
 * `demandFor` call site (and its tests) stays untouched. */
export function idleReleaseFor(
  catalog: EngineKindDef[] | null,
  kind: string,
): boolean | null {
  const entry = catalog?.find((k) => k.kind === kind);
  return entry ? entry.idle_release : null;
}

/** resource -> declared kind, from the world snapshot the app already
 * holds. `engine` is stamped on every tenant regardless of kind
 * (app/state.py's World.snapshot, mirrored in api.ts's ResourceTenant), so
 * the LOCAL half needs no extra request.
 *
 * `remoteTenants` folds in `world.remote_tenants` (FINDING 1): app/state.py
 * builds `world.tenants` from the LOCAL node alone, but a policy row can
 * name a REMOTE-declared engine (sglang-omni is local_capable false — it
 * can never appear in `tenants` at all, only here). Without this half, a
 * remote engine's row always missed the join and fell to the `?? ""`
 * fallback at every call site, rendering the false "kind catalog
 * unavailable" text instead of the exact idle-release warning this join
 * exists to produce. Each RemoteTenant record carries its own `resource`
 * (api.ts), so no second parameter shape is needed beyond the map itself. */
export function resourceKindMap(
  tenants: Record<string, ResourceTenant> | undefined,
  remoteTenants?: Record<string, RemoteTenant>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [resource, tenant] of Object.entries(tenants ?? {})) {
    out[resource] = tenant.engine;
  }
  for (const t of Object.values(remoteTenants ?? {})) {
    out[t.resource] = t.engine;
  }
  return out;
}
