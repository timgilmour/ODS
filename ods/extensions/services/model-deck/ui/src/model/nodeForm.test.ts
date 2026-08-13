import { describe, expect, test } from "vitest";
import type { DeckNodeEntry } from "../api";
import { labels } from "./messages";
import {
  emptyForm,
  formForEntry,
  testTarget,
  toCreatePayload,
  toPatchPayload,
  validate,
  type NodeFormState,
} from "./nodeForm";

// N1 T14 minimal fixture fix (Task 15 owns the rest of this file): swapped
// the deleted `actuation_stale` field for the `control` field DeckNodeEntry
// carries now (app/node_store.py _CONTROLS) — fixture-only, no logic here.
const entry: DeckNodeEntry = {
  id: "hera", label: "Hera Box", agent_kind: "node-agent",
  address: "http://hera:7720", serving_address: "http://hera:8000",
  credential_set: true, status: "online", last_seen: null,
  gpus: null, serving: null, error: null, control: "none",
};

// The seeded local node (app/node_store.py:179's seed spec) — no address,
// no credential, agent_kind "local". Its edit form is the one place
// validate() must NOT require an address.
const localEntry: DeckNodeEntry = {
  id: "local", label: "This box", agent_kind: "local",
  address: null, serving_address: null,
  credential_set: false, status: "online", last_seen: null,
  gpus: null, serving: null, error: null, control: "none",
};

test("formForEntry never carries the credential", () => {
  const form = formForEntry(entry);
  expect(form.credential).toBe("");   // write-only: nothing to show, ever
  expect(form.label).toBe("Hera Box");
});

test("formForEntry seeds control from the entry", () => {
  expect(formForEntry(entry).control).toBe("none");
  expect(formForEntry({ ...entry, control: "swap" }).control).toBe("swap");
});

describe("validate", () => {
  test("add mode requires slug id, label, address", () => {
    expect(validate(emptyForm(), "add", "node-agent").length).toBeGreaterThan(0);
    expect(validate({ ...formForEntry(entry), id: "Not A Slug", credential: "k" },
                    "add", "node-agent").length).toBeGreaterThan(0);
    expect(validate({ ...formForEntry(entry), credential: "k" }, "add", "node-agent")).toEqual([]);
  });
  test("edit mode does not require a credential", () => {
    expect(validate(formForEntry(entry), "edit", "node-agent")).toEqual([]);
  });
  // Task 9 review, Important 1: renaming the local node's label was
  // impossible because validate() required an address unconditionally, and
  // the seeded local node never has one.
  test("edit mode on the local node does not require an address", () => {
    expect(validate(formForEntry(localEntry), "edit", "local")).toEqual([]);
  });
  test("a node-agent entry still requires an address in edit mode", () => {
    const form = { ...formForEntry(entry), address: "" };
    expect(validate(form, "edit", "node-agent")).toContain(
      "address is required",
    );
  });
});

// Mirror of app/node_store.py:_require_swap_prereqs — control: "swap"
// requires address + serving_address + a credential, all present, with
// missing fields NAMED. The backend 422 is authoritative; this only saves
// the round-trip.
describe("validate — control: swap prerequisites", () => {
  test("swap with no serving address and no credential names both, in add mode", () => {
    const form: NodeFormState = {
      ...formForEntry(entry),
      control: "swap",
      servingAddress: "",
      credential: "",
    };
    const errors = validate(form, "add", "node-agent");
    expect(errors).toContain(labels.nodeSwapNeedsServingAddress);
    expect(errors).toContain(labels.nodeSwapNeedsCredential);
  });

  test("swap with address, serving address, and a typed credential has no control errors", () => {
    const form: NodeFormState = { ...formForEntry(entry), control: "swap", credential: "k" };
    const errors = validate(form, "add", "node-agent");
    expect(errors).not.toContain(labels.nodeSwapNeedsAddress);
    expect(errors).not.toContain(labels.nodeSwapNeedsServingAddress);
    expect(errors).not.toContain(labels.nodeSwapNeedsCredential);
  });

  // Mirror of _require_swap_prereqs's credential_present = bool(credential)
  // or self.credential_set(node_id) — a stored credential satisfies the
  // rule with nothing retyped, so an edit that touches only the label
  // doesn't force the operator to retype a credential that's already on
  // file.
  test("edit mode: entry.credential_set true satisfies the credential prerequisite with nothing retyped", () => {
    const form: NodeFormState = { ...formForEntry(entry), control: "swap" };
    expect(validate(form, "edit", "node-agent", entry)).toEqual([]);
  });

  // Categorical, not prerequisite-shaped: app/node_store.py:_validate
  // refuses control:"swap" for agent_kind:"local" unconditionally, distinct
  // from the three-prerequisite rule above. Add mode is impossible for
  // local (add() only ever creates node-agent rows), so this is edit-mode
  // only, on the seeded local entry, with every prerequisite otherwise
  // satisfiable.
  test("local entry: control swap is refused categorically, even with every prerequisite fillable", () => {
    const form: NodeFormState = {
      ...formForEntry(localEntry),
      control: "swap",
      address: "http://local:7720",
      servingAddress: "http://local:8000",
      credential: "k",
    };
    expect(validate(form, "edit", "local", localEntry)).toContain(labels.nodeControlLocalRefused);
  });

  test("a node-agent entry with the same fields is not refused categorically", () => {
    const form: NodeFormState = { ...formForEntry(entry), control: "swap", credential: "k" };
    expect(validate(form, "edit", "node-agent", entry)).not.toContain(
      labels.nodeControlLocalRefused,
    );
  });
});

describe("toPatchPayload", () => {
  test("sends only changed fields", () => {
    const form = { ...formForEntry(entry), label: "Renamed" };
    expect(toPatchPayload(form, entry)).toEqual({ label: "Renamed" });
  });
  test("credential rides only when typed", () => {
    const form = { ...formForEntry(entry), credential: "new-key" };
    expect(toPatchPayload(form, entry)).toEqual({ credential: "new-key" });
    expect(toPatchPayload(formForEntry(entry), entry)).toEqual({});
  });
  test("clearing serving_address sends explicit null", () => {
    const form = { ...formForEntry(entry), servingAddress: "" };
    expect(toPatchPayload(form, entry)).toEqual({ serving_address: null });
  });
  test("control rides only when it differs from entry.control", () => {
    expect(toPatchPayload({ ...formForEntry(entry), control: entry.control }, entry)).toEqual({});
    const changed: NodeFormState = { ...formForEntry(entry), control: "swap" };
    expect(toPatchPayload(changed, entry)).toEqual({ control: "swap" });
  });
});

test("toCreatePayload maps camelCase to the wire", () => {
  expect(toCreatePayload({ id: "hera", label: "Hera Box",
    address: "http://hera:7720", servingAddress: "", credential: "k",
    control: "none" })).toEqual({
      id: "hera", label: "Hera Box", address: "http://hera:7720",
      serving_address: null, credential: "k", control: "none" });
});

// Task 9 review, Important 2: the Test button used to branch only on
// whether a credential was typed, so an edited-but-unsaved address with no
// retyped credential silently tested the OLD stored address while the
// screen showed the new one.
describe("testTarget", () => {
  test("a typed credential tests exactly the typed pair, even with an edited address", () => {
    const form = { ...formForEntry(entry), address: "http://hera-new:7720", credential: "new-key" };
    expect(testTarget(form, entry)).toEqual({
      kind: "typed", address: "http://hera-new:7720", credential: "new-key",
    });
  });

  test("unchanged address, no typed credential, an entry on file: tests the stored pair", () => {
    expect(testTarget(formForEntry(entry), entry)).toEqual({ kind: "stored" });
  });

  test("changed address with no retyped credential is blocked, not silently stale", () => {
    const form = { ...formForEntry(entry), address: "http://hera-new:7720" };
    const result = testTarget(form, entry);
    expect(result.kind).toBe("blocked");
  });

  test("add mode (no entry) with nothing typed is blocked", () => {
    expect(testTarget(emptyForm(), null).kind).toBe("blocked");
  });

  test("a blank address is blocked even with a typed credential", () => {
    const form = { ...emptyForm(), credential: "k" };
    expect(testTarget(form, null).kind).toBe("blocked");
  });
});
