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

// Mirror of app/node_store.py:_require_instances_prereqs (:311-322) —
// control: "instances" requires address + a credential + a port range,
// all present, missing fields NAMED. address/credential reuse the swap
// labels (nodeForm.ts's own comment on this branch says why); the port
// range gets its own label since neither swap prerequisite names it.
describe("validate — control: instances prerequisites", () => {
  test("missing address, credential, and port range are each named", () => {
    const form: NodeFormState = {
      ...formForEntry(entry), control: "instances",
      address: "", credential: "", instancePortStart: "", instancePortEnd: "",
    };
    const errors = validate(form, "add", "node-agent");
    expect(errors).toContain(labels.nodeSwapNeedsAddress);
    expect(errors).toContain(labels.nodeSwapNeedsCredential);
    expect(errors).toContain(labels.nodeInstancesNeedsPortRange);
  });

  test("a start > end range is refused by name", () => {
    const form: NodeFormState = {
      ...formForEntry(entry), control: "instances", credential: "k",
      instancePortStart: "11510", instancePortEnd: "11500",
    };
    expect(validate(form, "add", "node-agent")).toContain(labels.nodeInstancesNeedsPortRange);
  });

  test("an out-of-window port (below 1024) is refused by name", () => {
    const form: NodeFormState = {
      ...formForEntry(entry), control: "instances", credential: "k",
      instancePortStart: "80", instancePortEnd: "11509",
    };
    expect(validate(form, "add", "node-agent")).toContain(labels.nodeInstancesNeedsPortRange);
  });

  test("address, credential, and a valid port range together have no control errors", () => {
    const form: NodeFormState = {
      ...formForEntry(entry), control: "instances", credential: "k",
      instancePortStart: "11500", instancePortEnd: "11509",
    };
    const errors = validate(form, "add", "node-agent");
    expect(errors).not.toContain(labels.nodeSwapNeedsAddress);
    expect(errors).not.toContain(labels.nodeSwapNeedsCredential);
    expect(errors).not.toContain(labels.nodeInstancesNeedsPortRange);
  });

  // Same credential_present rule swap's own equivalent test proves above —
  // a stored credential satisfies the prerequisite with nothing retyped.
  test("edit mode: entry.credential_set true satisfies the credential prerequisite with nothing retyped", () => {
    const form: NodeFormState = {
      ...formForEntry(entry), control: "instances",
      instancePortStart: "11500", instancePortEnd: "11509",
    };
    expect(validate(form, "edit", "node-agent", entry)).toEqual([]);
  });

  // UNLIKE control: "swap", control: "instances" has no categorical local
  // refusal (_require_instances_prereqs' own docstring: "Local or remote
  // alike; the protocol is node-generic").
  test("the local entry is not refused categorically for control: instances", () => {
    const form: NodeFormState = {
      ...formForEntry(localEntry), control: "instances",
      address: "http://local:7720", credential: "k",
      instancePortStart: "11500", instancePortEnd: "11509",
    };
    expect(validate(form, "edit", "local", localEntry)).toEqual([]);
  });
});

describe("formForEntry — instance port range", () => {
  test("seeds both fields as strings from the entry's instance_port_range", () => {
    const withRange: DeckNodeEntry = {
      ...entry, control: "instances",
      instance_port_range: { start: 11500, end: 11509 },
    };
    const form = formForEntry(withRange);
    expect(form.instancePortStart).toBe("11500");
    expect(form.instancePortEnd).toBe("11509");
  });

  test("blank when the entry carries none", () => {
    const form = formForEntry(entry);
    expect(form.instancePortStart).toBe("");
    expect(form.instancePortEnd).toBe("");
  });
});

describe("toCreatePayload / toPatchPayload — instance_port_range", () => {
  test("toCreatePayload carries the range when both fields are set", () => {
    const form: NodeFormState = {
      ...formForEntry(entry), id: "hera", credential: "k", control: "instances",
      instancePortStart: "11500", instancePortEnd: "11509",
    };
    expect(toCreatePayload(form)).toEqual({
      id: "hera", label: "Hera Box", address: "http://hera:7720",
      serving_address: "http://hera:8000", credential: "k", control: "instances",
      instance_port_range: { start: 11500, end: 11509 },
    });
  });

  test("toCreatePayload omits the key when either field is unset", () => {
    const form = { ...formForEntry(entry), control: "instances" as const, credential: "k" };
    expect(toCreatePayload(form)).not.toHaveProperty("instance_port_range");
  });

  test("toPatchPayload emits the range when it changes vs the entry", () => {
    const form = {
      ...formForEntry(entry), instancePortStart: "11500", instancePortEnd: "11509",
    };
    expect(toPatchPayload(form, entry)).toEqual({
      instance_port_range: { start: 11500, end: 11509 },
    });
  });

  test("toPatchPayload emits nothing when the range is unchanged", () => {
    const withRange: DeckNodeEntry = {
      ...entry, instance_port_range: { start: 11500, end: 11509 },
    };
    const form = formForEntry(withRange);
    expect(toPatchPayload(form, withRange)).toEqual({});
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
    control: "none", instancePortStart: "", instancePortEnd: "" })).toEqual({
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
