import { describe, expect, test } from "vitest";
import type { DeckNodeEntry } from "../api";
import {
  emptyForm,
  formForEntry,
  testTarget,
  toCreatePayload,
  toPatchPayload,
  validate,
} from "./nodeForm";

const entry: DeckNodeEntry = {
  id: "hera", label: "Hera Box", agent_kind: "node-agent",
  address: "http://hera:7720", serving_address: "http://hera:8000",
  credential_set: true, status: "online", last_seen: null,
  gpus: null, serving: null, error: null, actuation_stale: false,
};

// The seeded local node (app/node_store.py:179's seed spec) — no address,
// no credential, agent_kind "local". Its edit form is the one place
// validate() must NOT require an address.
const localEntry: DeckNodeEntry = {
  id: "local", label: "This box", agent_kind: "local",
  address: null, serving_address: null,
  credential_set: false, status: "online", last_seen: null,
  gpus: null, serving: null, error: null, actuation_stale: false,
};

test("formForEntry never carries the credential", () => {
  const form = formForEntry(entry);
  expect(form.credential).toBe("");   // write-only: nothing to show, ever
  expect(form.label).toBe("Hera Box");
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
});

test("toCreatePayload maps camelCase to the wire", () => {
  expect(toCreatePayload({ id: "hera", label: "Hera Box",
    address: "http://hera:7720", servingAddress: "", credential: "k" })).toEqual({
      id: "hera", label: "Hera Box", address: "http://hera:7720",
      serving_address: null, credential: "k" });
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
