/**
 * Node add/edit form logic — pure, componentless (the "logic inline in a
 * component is logic no test can reach" rule, 4 recurrences in this repo).
 *
 * The credential field is WRITE-ONLY end to end: formForEntry always seeds
 * it "", the backend never echoes it (app/routers/nodes.py), and
 * toPatchPayload only ships it when the operator actually typed one.
 *
 * validate()'s error strings and testTarget()'s blocked reasons are
 * sourced from model/messages.ts's `labels`, not written here — both are
 * operator-visible text, so the one-catalog rule that governs the rest of
 * the deck governs this module too. applySteps.ts's stepRow is the same
 * idiom already in this codebase: a pure model function that imports
 * `labels` and returns its values, so nothing downstream needs a literal
 * of its own either.
 */

import type { DeckNodeEntry } from "../api";
import { labels } from "./messages";

/** Mirrors app/node_store.py _ID_RE — the backend refuses what this allows
 * through, so the two must agree; the backend is authoritative. */
export const ID_RE = /^[a-z0-9][a-z0-9-]*$/;

export interface NodeFormState {
  id: string;
  label: string;
  address: string;
  servingAddress: string;
  credential: string;
  /** Declared operability (app/node_store.py _CONTROLS) — see
   * DeckNodeEntry.control in api.ts. "instances" (INST I1) now has its own
   * UI here — the port-range fields immediately below — built in Task 11. */
  control: "none" | "swap" | "instances";
  /** String buffers for `control: "instances"`' port-range prerequisite
   * (app/node_store.py's `_require_instances_prereqs`, :311-322) — plain
   * text inputs, not numbers, so an in-progress or invalid edit ("11" typed
   * so far) can sit on screen without being silently coerced to 0 the way a
   * bare `<input type="number">` value would be. Empty means "unset",
   * exactly like `NodeFormState`'s other optional text fields
   * (servingAddress). `portRangeOf`/`parsedPort` below are the only readers
   * that turn these into the wire's `{start, end}` ints. */
  instancePortStart: string;
  instancePortEnd: string;
}

export function emptyForm(): NodeFormState {
  return {
    id: "", label: "", address: "", servingAddress: "", credential: "", control: "none",
    instancePortStart: "", instancePortEnd: "",
  };
}

export function formForEntry(entry: DeckNodeEntry): NodeFormState {
  return {
    id: entry.id,
    label: entry.label,
    address: entry.address ?? "",
    servingAddress: entry.serving_address ?? "",
    credential: "",
    control: entry.control,
    instancePortStart: entry.instance_port_range ? String(entry.instance_port_range.start) : "",
    instancePortEnd: entry.instance_port_range ? String(entry.instance_port_range.end) : "",
  };
}

/** Parses one port-range field: empty means "not typed yet" (`null`),
 * exactly like `testTarget`'s own "nothing to test" reading of a blank
 * field — anything else must be an integer inside the backend's
 * 1024-65535 window (`app/node_store.py`'s `_validate_port_range`,
 * :60-70) to count as a usable value at all. An out-of-range or
 * non-numeric value collapses to `null` the same as empty, so `validate`'s
 * gate and `portRangeOf` below always agree on what "nothing entered" vs
 * "a bad entry still needs fixing" means — there is no third value one of
 * them could read as valid and the other as unset. */
function parsedPort(raw: string): number | null {
  if (!raw.trim()) return null;
  const n = Number(raw);
  return Number.isInteger(n) && n >= 1024 && n <= 65535 ? n : null;
}

/** The wire's `instance_port_range` shape, or `null` when either field is
 * unset/invalid (`parsedPort`) — the one place both fields turn into the
 * `{start, end}` object `toCreatePayload`/`toPatchPayload` ship. Only
 * meaningful once `validate(form)` is empty for `control: "instances"`,
 * same posture `engineForm.ts`'s `toPayload` takes for its own gate: the
 * Save button stays disabled until then, so this never has to decide what
 * an out-of-order or out-of-range pair should mean on the wire. */
function portRangeOf(form: NodeFormState): { start: number; end: number } | null {
  const start = parsedPort(form.instancePortStart);
  const end = parsedPort(form.instancePortEnd);
  return start !== null && end !== null ? { start, end } : null;
}

/** Whether two port ranges (or their `null` absence) are the SAME fact —
 * `toPatchPayload`'s own "only when changed" rule, field-wise rather than a
 * JSON.stringify comparison so key order can never make an unchanged range
 * look changed. */
function sameRange(
  a: { start: number; end: number } | null,
  b: { start: number; end: number } | null,
): boolean {
  if (a === null || b === null) return a === b;
  return a.start === b.start && a.end === b.end;
}

/** `agentKind` decides whether an address is required. The seeded local
 * node (app/node_store.py:179's seed spec) is stored with no address at
 * all and never gets one — it is a loopback identity, not a location to
 * dial — so its edit form must still be able to save a label-only change.
 * Every node-agent entry keeps requiring one: `add()` only ever creates
 * node-agent rows (node_store.py:107-110 refuses `agent_kind: "local"`
 * outside the one seed), so callers pass "node-agent" for add
 * unconditionally and the target entry's own kind for edit. */
export function validate(
  form: NodeFormState,
  mode: "add" | "edit",
  agentKind: "local" | "node-agent",
  entry: DeckNodeEntry | null = null,
): string[] {
  const errors: string[] = [];
  if (mode === "add" && !ID_RE.test(form.id)) {
    errors.push(labels.nodeIdInvalid);
  }
  if (!form.label.trim()) errors.push(labels.nodeLabelRequired);
  if (agentKind === "node-agent" && !form.address.trim()) {
    errors.push(labels.nodeAddressRequired);
  }
  if (mode === "add" && !form.credential) {
    errors.push(labels.nodeCredentialRequiredForAdd);
  }
  if (form.control === "swap") {
    if (agentKind === "local") {
      // Categorical, not prerequisite-shaped: app/node_store.py:_validate
      // refuses control:"swap" for agent_kind:"local" unconditionally —
      // local actuation is docker-ctl, not the swap protocol (G1 revisits).
      // The three prerequisites below are moot for a refused kind, so they
      // are skipped rather than piled on top of a refusal that already
      // makes them irrelevant.
      errors.push(labels.nodeControlLocalRefused);
    } else {
      // Mirror of app/node_store.py:_require_swap_prereqs — same three
      // prerequisites, same missing-field naming; the backend 422 is
      // authoritative, this just saves the round-trip.
      if (!form.address.trim()) errors.push(labels.nodeSwapNeedsAddress);
      if (!form.servingAddress.trim()) errors.push(labels.nodeSwapNeedsServingAddress);
      const credentialPresent = Boolean(form.credential) || Boolean(entry?.credential_set);
      if (!credentialPresent) errors.push(labels.nodeSwapNeedsCredential);
    }
  } else if (form.control === "instances") {
    // Mirror of app/node_store.py:_require_instances_prereqs, :311-322 —
    // address + a credential + a port range, all present, missing fields
    // NAMED. UNLIKE control:"swap" this has no categorical local refusal:
    // that same function's own docstring says "Local or remote alike; the
    // protocol is node-generic" — the local box can declare control:
    // "instances" same as any node-agent entry. Address/credential reuse
    // the SWAP labels rather than a second pair of near-identical strings:
    // both read as generic "operating a node requires ..." sentences, and
    // `_require_instances_prereqs` asks for exactly the same two fields
    // (never servingAddress, which instances has no use for).
    if (!form.address.trim()) errors.push(labels.nodeSwapNeedsAddress);
    const credentialPresent = Boolean(form.credential) || Boolean(entry?.credential_set);
    if (!credentialPresent) errors.push(labels.nodeSwapNeedsCredential);
    const start = parsedPort(form.instancePortStart);
    const end = parsedPort(form.instancePortEnd);
    if (start === null || end === null || start > end) {
      errors.push(labels.nodeInstancesNeedsPortRange);
    }
  }
  return errors;
}

export function toCreatePayload(form: NodeFormState) {
  const range = portRangeOf(form);
  return {
    id: form.id,
    label: form.label,
    address: form.address,
    serving_address: form.servingAddress.trim() || null,
    credential: form.credential,
    control: form.control,
    // Only when both fields parse (portRangeOf's own gate) — an add form
    // with control other than "instances" never has both filled in
    // practice, so this key is simply absent rather than shipping a
    // half-typed range the backend would 422 on anyway.
    ...(range ? { instance_port_range: range } : {}),
  };
}

export function toPatchPayload(form: NodeFormState, entry: DeckNodeEntry) {
  const patch: Record<string, string | null | { start: number; end: number }> = {};
  if (form.label !== entry.label) patch.label = form.label;
  if (form.address !== (entry.address ?? "")) patch.address = form.address;
  const serving = form.servingAddress.trim() || null;
  if (serving !== (entry.serving_address ?? null)) patch.serving_address = serving;
  if (form.credential) patch.credential = form.credential;
  if (form.control !== entry.control) patch.control = form.control;
  const range = portRangeOf(form);
  if (!sameRange(range, entry.instance_port_range ?? null)) patch.instance_port_range = range;
  return patch;
}

/** What "Test connection" would actually probe, given the buffer on
 * screen — kept out of NodesView because "would this test lie about what's
 * displayed" is a decision, not render wiring.
 *
 * - "stored": nothing typed that would change what's tested — probe
 *   `entry`'s on-file address+credential.
 * - "typed": a credential is on the screen right now — probe exactly the
 *   address+credential the operator typed, address included regardless of
 *   whether it also changed, so a typed pair is always tested as typed.
 * - "blocked": there is nothing honest to test. In particular, an edited
 *   address with no retyped credential would otherwise silently probe the
 *   OLD stored address while the screen shows the new one — the bug this
 *   type exists to make unrepresentable. `reason` is pre-resolved catalog
 *   text (model/messages.ts), ready for a tooltip with no lookup in the
 *   caller. */
export type TestTarget =
  | { kind: "stored" }
  | { kind: "typed"; address: string; credential: string }
  | { kind: "blocked"; reason: string };

export function testTarget(form: NodeFormState, entry: DeckNodeEntry | null): TestTarget {
  if (!form.address.trim()) {
    return { kind: "blocked", reason: labels.testBlockedNoAddress };
  }
  if (form.credential) {
    return { kind: "typed", address: form.address, credential: form.credential };
  }
  if (!entry) {
    return { kind: "blocked", reason: labels.testBlockedNoCredential };
  }
  if (form.address !== (entry.address ?? "")) {
    return { kind: "blocked", reason: labels.testBlockedAddressChanged };
  }
  return { kind: "stored" };
}
