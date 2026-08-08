import type { ArgValue, SettingsKind } from "../api";
import {
  bufferRemove,
  bufferSet,
  discardPendingAdd,
  emptyBuffer,
  isDirty,
  type Buffer,
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
 * The transitions below are the complete set of ways the buffer changes. The
 * buffer itself is behind a module-private key so that stays true by
 * compilation rather than by good intentions: `forChips` and `forSave` are the
 * only ways out, and they answer deliberately different questions.
 */
const BUFFER = Symbol("editState.buffer");

export interface EditState {
  readonly [BUFFER]: Buffer;
  /**
   * The "+ Add option" whose editor has not committed anything yet, or null.
   *
   * "+ Add option" cannot wait for the editor: the chip is rendered FROM the
   * buffer (buildChips's "pending set on a key `resolved` doesn't have at
   * all" branch), so there is no chip, and therefore no editor, until the set
   * exists. This is what remembers that the set arrived that way. It carries
   * the kind it was buffered at because the panel's write scope can be
   * switched out from under an open editor.
   */
  readonly pendingAdd: UncommittedAdd | null;
}

export interface UncommittedAdd {
  name: string;
  kind: SettingsKind;
  /**
   * Whether the starting value is scaffolding rather than a choice — which
   * decides what happens when the operator simply moves on, as opposed to
   * explicitly cancelling.
   *
   * True only for the empty string. Measured against the live sparky/vllm
   * catalog (274 options, 2026-08-08): 100 options start at `""` — 95 with no
   * catalog default at all plus 5 whose default is literally `''` — and five
   * of those are `select` widgets (collect-detailed-traces,
   * gdn-prefill-backend, mm-encoder-attn-dtype, safetensors-load-strategy,
   * spec-method). `""` is not a value a declared flag can hold: app/argline.py
   * renders it as `--flag ''`, which the text editor's own commit path already
   * refuses to write.
   *
   * The other 174 arrive with a real value and keep it when the operator moves
   * on, because picking the option out of the catalog IS the decision:
   * - 78 toggles, whose starting value is `true` (catalogFilter's
   *   `startingValueFor` answers toggles before it ever looks at the default).
   *   These have no choice in the matter — the checkbox renders already
   *   checked and its `onChange` only fires on a click, so no commit event
   *   exists for a fresh toggle add. Dropping it would make "add
   *   --enforce-eager, click away" silently do nothing.
   * - 96 number/select/text/list options with a real catalog default.
   *   Declaring the current default explicitly pins it against a future
   *   upstream change, which is a legitimate thing to want.
   */
  provisional: boolean;
}

export const emptyEdit: EditState = { [BUFFER]: emptyBuffer, pendingAdd: null };

function next(buffer: Buffer, pendingAdd: UncommittedAdd | null): EditState {
  return { [BUFFER]: buffer, pendingAdd };
}

/** The buffer once an add the operator never gave a value to is taken back
 * out. Applies when they simply moved on; an add that arrived with a real
 * value stays, having been chosen. */
function dropProvisional(s: EditState): Buffer {
  const add = s.pendingAdd;
  return add?.provisional ? discardPendingAdd(s[BUFFER], add) : s[BUFFER];
}

/** `dropProvisional`, unless the incoming write at (`kind`, `name`) IS the
 * thing that resolves the add.
 *
 * Standing aside for the add's own key is the subtle half. `commitEdit` writes
 * the value the add was waiting for, and `removeChip`'s `bufferRemove` is
 * itself the undo — it deletes a pending set and records no removal — so
 * discarding first would leave that call nothing to delete and turn it into a
 * genuine `removes` entry naming a key the scope never had.
 *
 * `kind` counts as much as `name`: the write scope can move under an open
 * editor, and an add buffered at `engine_models` is not what a remove at
 * `engines` resolves, however the two are spelled. */
function supersede(s: EditState, kind: SettingsKind, name: string): Buffer {
  const add = s.pendingAdd;
  if (add && add.name === name && add.kind === kind) return s[BUFFER];
  return dropProvisional(s);
}

/** "+ Add option" for a key not already declared: buffer the starting value so
 * the chip can render, and record whether that value was a choice. */
export function addOption(
  s: EditState,
  kind: SettingsKind,
  name: string,
  value: ArgValue,
): EditState {
  return next(bufferSet(supersede(s, kind, name), kind, name, value), {
    name,
    kind,
    provisional: value === "",
  });
}

/** Opening a chip's editor. Reopening the add's OWN editor commits nothing, so
 * it stays exactly as uncommitted as it was; opening any other chip's is the
 * operator moving on, and for a provisional add nothing else would ever take
 * its `""` back out — `select` and `toggle` have no onBlur to route through a
 * cancel. */
export function beginEdit(s: EditState, name: string): EditState {
  if (s.pendingAdd?.name === name) return s;
  return leaveEdit(s);
}

/** The operator moved on without saying no — the write scope changed under the
 * editor, or a modal is about to take focus away from it. Settles an add that
 * arrived with a value; retracts one that never had one. */
export function leaveEdit(s: EditState): EditState {
  return next(dropProvisional(s), null);
}

/** An editor committed a value. On the add's own key that is the moment it
 * stops being this panel's scaffolding and becomes the operator's. */
export function commitEdit(
  s: EditState,
  kind: SettingsKind,
  name: string,
  value: ArgValue,
): EditState {
  return next(bufferSet(supersede(s, kind, name), kind, name, value), null);
}

/** The chip ✕ / "revert to inherited". */
export function removeChip(s: EditState, kind: SettingsKind, name: string): EditState {
  return next(bufferRemove(supersede(s, kind, name), kind, name), null);
}

/** The operator explicitly said no — Escape, or emptying the field and
 * blurring (SettingsModal's `commit` reads an emptied box as a cancel, since
 * committing `""` would ship `--flag ''`).
 *
 * Unlike simply moving on, this retracts an add even when it arrived with a
 * real value: "Escape means cancel" does not get to depend on whether the
 * catalog happened to have a default for the option. */
export function abandonEdit(s: EditState): EditState {
  const add = s.pendingAdd;
  if (add === null) return s;
  return next(discardPendingAdd(s[BUFFER], add), null);
}

/** A parsed argline import: real operator values for many keys at once.
 *
 * The add is settled first, not cleared as bookkeeping. An import that names
 * the key an add is still uncommitted on replaces scaffolding with a real
 * value, and leaving the flag armed would let the next Escape delete what the
 * operator just imported. Belt and braces with `openImport`, which cancels the
 * editor before the modal can steal focus — either one alone is sufficient,
 * and this one keeps the module correct on its own terms. */
export function applyParsed(
  s: EditState,
  kind: SettingsKind,
  parsed: Record<string, ArgValue>,
): EditState {
  let buffer = dropProvisional(s);
  for (const [name, value] of Object.entries(parsed)) {
    buffer = bufferSet(buffer, kind, name, value);
  }
  return next(buffer, null);
}

/** The buffer the chips are drawn from — the raw one, deliberately. A
 * provisional add's set exists for no reason other than to put its chip and
 * editor on screen, so hiding it here would close the editor the operator is
 * standing in. */
export function forChips(s: EditState): Buffer {
  return s[BUFFER];
}

/** The buffer to actually PUT, and to render the argline preview from.
 *
 * Save is an exit from an open editor exactly as Escape is, and it is the one
 * no cancel path covers: the operator can click it with a never-committed add
 * still on screen. Asking here rather than reading the raw buffer is what
 * stops that shipping `--flag ''`, and what stops the preview advertising a
 * flag the save would drop. */
export function forSave(s: EditState): Buffer {
  return dropProvisional(s);
}

/** Is there anything to save? Asks of `forSave`'s buffer: an add with no value
 * chosen is not a change, and an enabled Save that shipped an empty PUT list
 * would close the panel as if it had written something. */
export function hasChanges(s: EditState): boolean {
  return isDirty(forSave(s));
}
