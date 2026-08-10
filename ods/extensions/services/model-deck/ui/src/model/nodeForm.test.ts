import { describe, expect, test } from "vitest";
import type { DeckNodeEntry } from "../api";
import { emptyForm, formForEntry, toCreatePayload, toPatchPayload, validate } from "./nodeForm";

const entry: DeckNodeEntry = {
  id: "hera", label: "Hera Box", agent_kind: "node-agent",
  address: "http://hera:7720", serving_address: "http://hera:8000",
  credential_set: true, status: "online", last_seen: null,
  gpus: null, serving: null, error: null,
};

test("formForEntry never carries the credential", () => {
  const form = formForEntry(entry);
  expect(form.credential).toBe("");   // write-only: nothing to show, ever
  expect(form.label).toBe("Hera Box");
});

describe("validate", () => {
  test("add mode requires slug id, label, address", () => {
    expect(validate(emptyForm(), "add").length).toBeGreaterThan(0);
    expect(validate({ ...formForEntry(entry), id: "Not A Slug", credential: "k" },
                    "add").length).toBeGreaterThan(0);
    expect(validate({ ...formForEntry(entry), credential: "k" }, "add")).toEqual([]);
  });
  test("edit mode does not require a credential", () => {
    expect(validate(formForEntry(entry), "edit")).toEqual([]);
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
