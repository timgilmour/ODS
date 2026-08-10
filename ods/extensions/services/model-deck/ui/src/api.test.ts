import { afterEach, describe, expect, it, vi } from "vitest";
import { createNode, updateNode } from "./api";
import type { NodeRegistryEntry } from "./api";

// The registry-CRUD wire shape, exactly as app/routers/nodes.py::_public
// produces it over NodeStore.add()/update() (app/node_store.py:104-139):
// {id, label, agent_kind, address, serving_address, added_ts,
// credential_set}. NO observer fields (status/last_seen/gpus/serving/error)
// — those belong to a DIFFERENT producer, status.py's _nodes_block, and
// DeckNodeEntry mirrors that one. createNode/updateNode used to be typed
// Promise<DeckNodeEntry>, which let `.status` compile against a payload
// that was never actually there.
const registryResponse = {
  id: "hera",
  label: "Hera Box",
  agent_kind: "node-agent",
  address: "http://hera:7720",
  serving_address: null,
  added_ts: "2026-08-10T00:00:00+00:00",
  credential_set: true,
};

function mockFetchOnce(body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createNode / updateNode — registry CRUD wire shape", () => {
  it("resolves exactly what app/routers/nodes.py::_public returns, never an observer field", async () => {
    mockFetchOnce(registryResponse);
    const result: NodeRegistryEntry = await createNode({
      id: "hera",
      label: "Hera Box",
      address: "http://hera:7720",
    });
    expect(result).toEqual(registryResponse);
    // The bug this guards: DeckNodeEntry's observer-only fields must never
    // be assumed present on a CRUD response, at the type level or here.
    expect("status" in result).toBe(false);
    expect("last_seen" in result).toBe(false);
    expect("gpus" in result).toBe(false);
    expect("serving" in result).toBe(false);
    expect("error" in result).toBe(false);
    // What the next task (NodesView) actually relies on.
    expect(result.id).toBe("hera");
  });

  it("updateNode resolves the same registry shape", async () => {
    mockFetchOnce({ ...registryResponse, label: "Hera Box 2" });
    const result = await updateNode("hera", { label: "Hera Box 2" });
    expect(result.label).toBe("Hera Box 2");
    expect("status" in result).toBe(false);
  });
});
