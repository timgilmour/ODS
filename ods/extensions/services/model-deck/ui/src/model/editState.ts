import type { ArgValue, SettingsKind } from "../api";
import {
  bufferRemove,
  bufferSet,
  discardPendingAdd,
  emptyBuffer,
  isDirty,
  type Buffer,
  type PendingAdd,
} from "./settingsView";

/**
 * The Settings panel's uncommitted-edit state, as a state machine.
 *
 * WHY this is a module and not two `useState`s in SettingsModal: the panel's
 * buffer and its "this add hasn't been chosen yet" flag only make sense
 * together, and every bug this feature has had was one exit path that updated
 * one without the other. That defect (F1) escaped two fix waves — each wave
 * patched the call sites it could see, and the tests, which only ever reached
 * the buffer primitives, stayed green either way. The exits are enumerable, so
 * they are enumerated HERE, where a test can walk them.
 *
 * The transitions below are the complete set of ways the buffer changes. If a
 * new one is needed, it belongs in this file — a `bufferSet` reached directly
 * from a component is how the last three leaks happened.
 */
export interface EditState {
  buffer: Buffer;
  /**
   * The brand-new "+ Add option" whose editor has not committed a real value
   * yet, or null.
   *
   * "+ Add option" cannot wait for the editor: the chip is rendered FROM the
   * buffer (buildChips's "pending set on a key `resolved` doesn't have at
   * all" branch), so there is no chip, and therefore no editor, until the set
   * exists. That set is scaffolding, not an operator decision, and this is
   * what remembers the difference. It carries the kind it was buffered at
   * because the panel's write scope can be switched out from under an open
   * editor.
   */
  pendingAdd: PendingAdd | null;
}

export const emptyEdit: EditState = { buffer: emptyBuffer, pendingAdd: null };

/**
 * Is a starting value scaffolding rather than a choice?
 *
 * Only the empty string. Measured against the live sparky/vllm catalog
 * (274 options, 2026-08-08): 168 have no catalog default, but 73 of those are
 * toggles, which `startingValueFor` answers `true` for before it ever looks at
 * the default — leaving 95 that genuinely start at `""`. Five are `select`
 * widgets (collect-detailed-traces, gdn-prefill-backend,
 * mm-encoder-attn-dtype, safetensors-load-strategy, spec-method), and those
 * are the ones with no onBlur to cancel them on the way out.
 *
 * `""` is not a value a declared flag can hold:
 * app/argline.py would render it as `--flag ''`, which is what the text
 * editor's own commit path already refuses to write.
 *
 * Everything else it produces IS a value and is kept on the spot:
 * - a toggle's `true` — a checked box is the whole meaning of a bare flag, so
 *   "add --enforce-eager, click away" must declare it, not silently drop it;
 * - a real catalog default — declaring the current default explicitly pins it
 *   against a future upstream change, which is a legitimate thing to want.
 */
function isProvisional(value: ArgValue): boolean {
  return value === "";
}

/**
 * The buffer with any provisional add the incoming action is NOT itself
 * resolving taken back out.
 *
 * Standing aside for the add's own key is the subtle half. `commitEdit`
 * writes the value the add was waiting for, and `removeChip`'s `bufferRemove`
 * IS the undo — it deletes a pending set and records no removal — so
 * discarding first would leave that call nothing to delete and turn it into a
 * genuine `removes` entry naming a key the scope never had.
 *
 * `kind` is checked when the caller has one, because the write scope can move
 * under an open editor: an add buffered at `engine_models` is NOT the thing a
 * remove at `engines` resolves, even under the same name. `beginEdit` passes
 * none — reopening an editor is about the key, whatever scope the panel has
 * since switched to.
 */
function supersede(s: EditState, name: string, kind?: SettingsKind): Buffer {
  const add = s.pendingAdd;
  if (add === null) return s.buffer;
  const resolvesTheAdd = add.name === name && (kind === undefined || add.kind === kind);
  return resolvesTheAdd ? s.buffer : discardPendingAdd(s.buffer, add);
}

/** "+ Add option" for a key not already declared: buffer the starting value
 * so the chip can render, and remember whether that value was a choice. Any
 * earlier provisional add is abandoned — the operator moved on from it. */
export function addOption(
  s: EditState,
  kind: SettingsKind,
  name: string,
  value: ArgValue,
): EditState {
  const buffer = bufferSet(supersede(s, name, kind), kind, name, value);
  return { buffer, pendingAdd: isProvisional(value) ? { name, kind } : null };
}

/** Opening a chip's editor. Reopening the provisional add's OWN editor
 * commits nothing, so it stays provisional; opening any other chip's is the
 * operator moving on, and nothing else would ever take that add back out —
 * `select` and `toggle` have no onBlur to route through a cancel. */
export function beginEdit(s: EditState, name: string): EditState {
  if (s.pendingAdd?.name === name) return s;
  return { buffer: supersede(s, name), pendingAdd: null };
}

/** An editor committed a value. On the pending add's own key that is the
 * moment it stops being scaffolding and becomes the operator's. */
export function commitEdit(
  s: EditState,
  kind: SettingsKind,
  name: string,
  value: ArgValue,
): EditState {
  return {
    buffer: bufferSet(supersede(s, name, kind), kind, name, value),
    pendingAdd: null,
  };
}

/** The chip ✕ / "revert to inherited".
 *
 * On the pending add's own key `bufferRemove` IS the undo — it deletes a
 * pending set and records no removal — which is why `supersede` deliberately
 * stands aside for the same key at the same kind: discarding first would leave
 * this call nothing to delete and turn it into a genuine `removes` entry
 * naming a key the scope never had. */
export function removeChip(s: EditState, kind: SettingsKind, name: string): EditState {
  return {
    buffer: bufferRemove(supersede(s, name, kind), kind, name),
    pendingAdd: null,
  };
}

/** The editor closed with nothing chosen — Escape, an emptied field blurring,
 * a write-scope switch, or a modal about to steal focus. */
export function abandonEdit(s: EditState): EditState {
  if (s.pendingAdd === null) return s;
  return { buffer: discardPendingAdd(s.buffer, s.pendingAdd), pendingAdd: null };
}

/** A parsed argline import: real operator values for many keys at once.
 *
 * Clearing `pendingAdd` is the point, not bookkeeping. An import that names
 * the key an add is still provisional on replaces scaffolding with a real
 * value, and leaving the flag armed would let the next Escape — or the next
 * click on any other chip — silently delete what the operator just imported.
 * An add the import did NOT name was abandoned by opening the import box. */
export function applyParsed(
  s: EditState,
  kind: SettingsKind,
  parsed: Record<string, ArgValue>,
): EditState {
  let buffer = discardPendingAdd(s.buffer, s.pendingAdd);
  for (const [name, value] of Object.entries(parsed)) {
    buffer = bufferSet(buffer, kind, name, value);
  }
  return { buffer, pendingAdd: null };
}

/** The buffer to actually PUT.
 *
 * Save is an exit from an open editor exactly as Escape is, and it is the one
 * no cancel path covers: the operator can click it with a never-committed add
 * still on screen. Routing the save through here rather than reading
 * `state.buffer` is what stops that from shipping `--flag ''`. */
export function forSave(s: EditState): Buffer {
  return discardPendingAdd(s.buffer, s.pendingAdd);
}

/** Is there anything to save? Deliberately asks of `forSave`'s buffer, not
 * the raw one: an add with no value chosen is not a change, and an enabled
 * Save that ships an empty PUT list would close the panel as if it had
 * written something. */
export function hasChanges(s: EditState): boolean {
  return isDirty(forSave(s));
}
