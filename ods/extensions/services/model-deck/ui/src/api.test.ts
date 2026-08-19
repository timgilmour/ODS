import { afterEach, describe, expect, it, vi } from "vitest";
import {
  addEngine,
  ApiError,
  createNode,
  errorMessage,
  forgetEngine,
  getEngineKinds,
  getNodeServingStatus,
  listNodeRegistry,
  postEngineVerb,
  postNodeReload,
  postNodeSwap,
  putUnitPinned,
  updateEngine,
  updateNode,
} from "./api";
import type { DeclaredEngine, NodeRegistryEntry, SparkStatus } from "./api";

// The registry-CRUD wire shape, exactly as app/routers/nodes.py::_public
// produces it over NodeStore.add()/update() (app/node_store.py:104-139):
// {id, label, agent_kind, address, serving_address, added_ts,
// credential_set, control}. NO observer fields
// (status/last_seen/gpus/serving/error)
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
  // Declared operability (app/node_store.py _CONTROLS) — present on every
  // registry row, healed to "none" when missing/invalid
  // (NodeStore._heal_control), never inferred from spark-ness.
  control: "none",
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

  it("updateNode on the seeded local node omits address/serving_address entirely — the ?? access pattern still works", async () => {
    // node_store.py:179 seeds the local node with NO address or
    // serving_address key at all (agent_kind "local" needs neither), and
    // _public() spreads the stored dict as-is — so PUT /api/nodes/local
    // {"label": ...} resolves a response missing both keys, not `null`.
    const localShaped = {
      id: "local", label: "autarch", agent_kind: "local",
      added_ts: "2026-08-01T00:00:00+00:00", credential_set: false,
      control: "none",
    };
    mockFetchOnce(localShaped);
    const result = await updateNode("local", { label: "autarch" });
    expect("address" in result).toBe(false);
    expect("serving_address" in result).toBe(false);
    // The typed access pattern a caller must use, now that both fields are
    // optional (not just nullable): `??` collapses missing and null alike.
    expect(result.address ?? "").toBe("");
    expect(result.serving_address ?? "").toBe("");
  });
});

describe("errorMessage", () => {
  it("renders a validation-error LIST as readable messages", () => {
    // app/main.py's RequestValidationError handler emits `detail` as
    // pydantic's LIST of {type, loc, msg} dicts (with `input` stripped).
    // Typed as a string, that array reached the ApiError message as-is and
    // the operator saw "[object Object]" [max-review c32].
    const message = errorMessage(
      { detail: [{ type: "missing", loc: ["body", "address"], msg: "Field required" }] },
      422,
      "Unprocessable Entity",
    );
    expect(message).toContain("Field required");
    expect(message).not.toContain("[object Object]");
  });

  it("joins several validation errors", () => {
    const message = errorMessage(
      { detail: [{ msg: "Field required" }, { msg: "Input should be a valid string" }] },
      422,
      "Unprocessable Entity",
    );
    expect(message).toBe("Field required; Input should be a valid string");
  });

  it("passes a plain string detail through unchanged", () => {
    expect(errorMessage({ detail: "node has no address to test" }, 422, "x"))
      .toBe("node has no address to test");
  });

  it("falls back to status + statusText when there is no usable detail", () => {
    expect(errorMessage(null, 500, "Internal Server Error"))
      .toBe("500 Internal Server Error");
    expect(errorMessage({}, 500, "Internal Server Error"))
      .toBe("500 Internal Server Error");
    expect(errorMessage({ detail: [] }, 500, "Internal Server Error"))
      .toBe("500 Internal Server Error");
  });

  it("does not lose an item that has no msg", () => {
    // Better a JSON blob than a silently dropped error.
    const message = errorMessage({ detail: [{ type: "weird" }] }, 422, "x");
    expect(message).toContain("weird");
  });
});

describe("putUnitPinned", () => {
  it("encodes the unit id into the path", async () => {
    // Unit ids come from catalog scans of real filenames; a "#" truncated
    // the path silently [max-review c34].
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push(url);
      return { ok: true, status: 200, json: async () => ({}) } as Response;
    }));
    await putUnitPinned("hot:weird#name?.gguf", true);
    expect(calls[0]).toBe("/api/storage/units/hot%3Aweird%23name%3F.gguf");
  });
});

// The GET /api/nodes/{id}/serving/status body, exactly as
// app/routers/serving.py's serving_status route relays it from the
// engine client's .status() (SparkStatus's own shape, unchanged).
const servingStatus: SparkStatus = {
  profiles: [{ name: "vllm-70b", engine: "vllm", health_url: null, container: null }],
  swap_status: null,
  serving: { model: null, endpoint_ok: false, container_status: null },
};

describe("getNodeServingStatus", () => {
  it("resolves the body on 200", async () => {
    mockFetchOnce(servingStatus);
    const result = await getNodeServingStatus("boxa");
    expect(result).toEqual(servingStatus);
  });

  it("resolves null on 503 — the node is known but not operable (app/routers/serving.py:_client)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: "Service Unavailable",
        json: () => Promise.resolve({ detail: 'node "boxa" is not operable' }),
      }),
    );
    const result = await getNodeServingStatus("boxa");
    expect(result).toBeNull();
  });

  it("throws ApiError on every other failure — a declared-operable node the deck cannot reach is a real error, not an absent card", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        statusText: "Bad Gateway",
        json: () => Promise.resolve({ detail: "upstream error" }),
      }),
    );
    await expect(getNodeServingStatus("boxa")).rejects.toBeInstanceOf(ApiError);
  });

  it("encodes the node id into the path", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push(url);
      return { ok: true, status: 200, json: async () => servingStatus } as Response;
    }));
    await getNodeServingStatus("box a/weird#name");
    expect(calls[0]).toBe("/api/nodes/box%20a%2Fweird%23name/serving/status");
  });
});

describe("postNodeSwap", () => {
  it("encodes {profile, force} and hits /api/nodes/boxa/serving/swap", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return { ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response;
    }));
    await postNodeSwap("boxa", "vllm-70b", true);
    expect(calls[0].url).toBe("/api/nodes/boxa/serving/swap");
    expect(calls[0].init?.method).toBe("POST");
    expect(JSON.parse(calls[0].init?.body as string)).toEqual({ profile: "vllm-70b", force: true });
  });

  it("defaults force to false", async () => {
    const calls: { init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      calls.push({ init });
      return { ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response;
    }));
    await postNodeSwap("boxa", "vllm-70b");
    expect(JSON.parse(calls[0].init?.body as string)).toEqual({ profile: "vllm-70b", force: false });
  });

  it("encodes the node id into the path", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push(url);
      return { ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response;
    }));
    await postNodeSwap("box a/weird#name", "vllm-70b");
    expect(calls[0]).toBe("/api/nodes/box%20a%2Fweird%23name/serving/swap");
  });
});

describe("postNodeReload", () => {
  it("posts {} to /api/nodes/boxa/serving/reload", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return { ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response;
    }));
    await postNodeReload("boxa");
    expect(calls[0].url).toBe("/api/nodes/boxa/serving/reload");
    expect(calls[0].init?.method).toBe("POST");
    expect(JSON.parse(calls[0].init?.body as string)).toEqual({});
  });

  it("encodes the node id into the path", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push(url);
      return { ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response;
    }));
    await postNodeReload("box a/weird#name");
    expect(calls[0]).toBe("/api/nodes/box%20a%2Fweird%23name/serving/reload");
  });
});

// E1 Task 12 — declared-engines CRUD (app/routers/nodes.py:200-339).

describe("listNodeRegistry", () => {
  it("resolves {nodes} verbatim, including the local entry's engines[]", async () => {
    // The one producer whose local row carries `engines[]` — _public
    // (app/routers/nodes.py:88-89) spreads the stored NodeStore entry
    // as-is, unlike /api/state's `nodes` block (DeckNodeEntry).
    const body = {
      nodes: [
        { ...registryResponse, id: "local", agent_kind: "local", address: null,
          serving_address: null, engines: [] },
      ],
    };
    mockFetchOnce(body);
    const result = await listNodeRegistry();
    expect(result).toEqual(body);
    expect(result.nodes[0].engines).toEqual([]);
  });
});

const widgetEngine: DeclaredEngine = {
  resource: "widget-a",
  kind: "widget",
  connection: { host: "http://widget-a:9000" },
  gpu_index: 2,
  policy_defaults: { priority: 0, pinned: false, idle_ttl: 0 },
  container_consent: false,
};

describe("getEngineKinds", () => {
  it("resolves the {kinds} body verbatim", async () => {
    const body = {
      kinds: [{
        kind: "widget", connection: { host: { required: true } },
        remote_capable: false, local_capable: true, human_verbs: ["free"],
      }],
    };
    mockFetchOnce(body);
    expect(await getEngineKinds()).toEqual(body);
  });
});

describe("addEngine", () => {
  it("POSTs the body verbatim to /api/nodes/{node_id}/engines", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return { ok: true, status: 200, json: async () => widgetEngine } as Response;
    }));
    // A non-"local" node id (sglang-omni Task 10) — proves the route is
    // genuinely parametrized rather than "local" being the only string
    // this function can ever produce.
    const result = await addEngine("sparky", widgetEngine);
    expect(calls[0].url).toBe("/api/nodes/sparky/engines");
    expect(calls[0].init?.method).toBe("POST");
    expect(JSON.parse(calls[0].init?.body as string)).toEqual(widgetEngine);
    expect(result).toEqual(widgetEngine);
  });

  it("'local' is an id like any other — the pre-sglang-omni route, byte for byte", async () => {
    const calls: { url: string }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push({ url });
      return { ok: true, status: 200, json: async () => widgetEngine } as Response;
    }));
    await addEngine("local", widgetEngine);
    expect(calls[0].url).toBe("/api/nodes/local/engines");
  });
});

describe("updateEngine", () => {
  it("PUTs to the node- and resource-encoded path", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return { ok: true, status: 200, json: async () => widgetEngine } as Response;
    }));
    await updateEngine(
      "sparky", "widget a/weird#name", { ...widgetEngine, resource: "widget a/weird#name" },
    );
    expect(calls[0].url).toBe("/api/nodes/sparky/engines/widget%20a%2Fweird%23name");
    expect(calls[0].init?.method).toBe("PUT");
  });

  it("'local' is an id like any other — the pre-sglang-omni route, byte for byte", async () => {
    const calls: { url: string }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push({ url });
      return { ok: true, status: 200, json: async () => widgetEngine } as Response;
    }));
    await updateEngine("local", "widget-a", widgetEngine);
    expect(calls[0].url).toBe("/api/nodes/local/engines/widget-a");
  });
});

describe("forgetEngine", () => {
  it("DELETEs the node- and resource-encoded path", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return { ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response;
    }));
    const result = await forgetEngine("sparky", "widget-a");
    expect(calls[0].url).toBe("/api/nodes/sparky/engines/widget-a");
    expect(calls[0].init?.method).toBe("DELETE");
    expect(result).toEqual({ status: "ok" });
  });

  it("'local' is an id like any other — the pre-sglang-omni route, byte for byte", async () => {
    const calls: { url: string }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push({ url });
      return { ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response;
    }));
    await forgetEngine("local", "widget-a");
    expect(calls[0].url).toBe("/api/nodes/local/engines/widget-a");
  });
});

describe("postEngineVerb", () => {
  it("POSTs to the node/resource/verb path, every segment encoded", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return {
        ok: true, status: 202,
        json: async () => ({ status: "accepted", node_id: "box a",
                             resource: "song/lab", verb: "unload" }),
      } as Response;
    }));
    // Ids and resource names that would break an unencoded path — a
    // registry is hand-editable and a resource name is operator-typed.
    await postEngineVerb("box a", "song/lab", "unload");
    expect(calls[0].url).toBe("/api/nodes/box%20a/engines/song%2Flab/unload");
    expect(calls[0].init?.method).toBe("POST");
    // No body: the route takes none (app/routers/serving.py's engine_verb —
    // "No `force`, no body").
    expect(calls[0].init?.body).toBeUndefined();
  });

  it("surfaces the route's own refusal sentence, with its status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: false, status: 405, statusText: "Method Not Allowed",
      json: async () => ({ detail: "song-lab (sglang-omni) does not support park" }),
    } as Response)));
    await expect(postEngineVerb("zeta", "song-lab", "park")).rejects.toMatchObject({
      status: 405,
      message: "song-lab (sglang-omni) does not support park",
    });
  });
});
