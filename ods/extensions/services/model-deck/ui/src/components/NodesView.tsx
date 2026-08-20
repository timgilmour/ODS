import { useEffect, useState, type ChangeEvent } from "react";
import {
  addEngine,
  bytesToGB,
  createInstance,
  createNode,
  deleteNode,
  forgetEngine,
  getEngineKinds,
  listNodeRegistry,
  moveInstance,
  removeInstance,
  testNode,
  updateEngine,
  updateNode,
  type DeckNodeEntry,
  type DeclaredEngine,
  type EngineKindsResponse,
  type Gpu,
  type NodeTestResult,
  type PolicyMap,
} from "../api";
import {
  demandFor,
  emptyForm as emptyEngineForm,
  engineGpus,
  formErrors,
  formForEntry as engineFormForEntry,
  idleReleaseFor,
  kindsFor,
  maxGpusFor,
  setField as setEngineField,
  sortedEngines,
  toggleGpu,
  toggleIndex,
  toPayload as toEnginePayload,
  withKind,
  type EngineFormState,
} from "../model/engineForm";
import {
  emptyInstanceForm,
  instanceFormErrors,
  instanceKindsFor,
  sameClaim,
  setInstanceEnv,
  toggleInstanceGpu,
  toInstancePayload,
  withInstanceKind,
  type InstanceFormState,
} from "../model/instanceForm";
import { labels, messages } from "../model/messages";
import {
  emptyForm,
  formForEntry,
  testTarget,
  toCreatePayload,
  toPatchPayload,
  validate,
  type NodeFormState,
} from "../model/nodeForm";
import { isArmedFor } from "../model/armed";
import ArmedButton from "../ui/ArmedButton";
import Banner from "../ui/Banner";
import Panel from "../ui/Panel";

/** Rail dot per app/node_observer.py's status vocabulary (online / offline /
 * error / unconfigured), plus null for "never observed" (a node the observer
 * hasn't ticked yet). `error` and `offline` share a dot: both mean "the deck
 * cannot currently rely on this node", and the row's own `title` carries the
 * exact word for whichever it was. The local node is always "online" —
 * app/routers/status.py's `_nodes_block` hardcodes it, never asks the
 * observer (which only watches `agent_kind: "node-agent"` entries). */
const DOT: Record<string, string> = {
  online: "node-dot--online",
  offline: "node-dot--offline",
  error: "node-dot--offline",
  unconfigured: "node-dot--dim",
};

export default function NodesView({
  nodes,
  gpus,
  policy,
  onChanged,
  instanceSeed,
  onSeedConsumed,
}: {
  nodes: DeckNodeEntry[];
  /** The Engines editor's GPU-picker source (spec §5: `/api/state`'s
   * `world.gpus`, never a UI literal) — threaded down to whichever node's
   * form is open, local or node-agent either one (sglang-omni Task 10; used
   * to be local-only). NOTE this list is always the LOCAL box's own GPUs —
   * `world.gpus` has no remote half — so a node-agent target's picker shows
   * the wrong machine's indices/capacities today. Deliberate: the
   * sglang-omni design (§4, "Increment-4-minimal") scopes remote
   * GPU-index semantics beyond a single unified pool as out of scope
   * "until a real node needs it"; the operator picks the index THEIR node
   * actually has by its own knowledge, same as declaring any other field
   * this form cannot independently verify. */
  gpus: Gpu[];
  /** The Engines editor's pinned-badge source. LIVE policy
   * (`/api/state`'s `policy` map, app/policy.py's `PolicyStore.get()`),
   * not each engine's own frozen `policy_defaults` — PolicyStore always
   * returns one row per currently-declared resource (declared default
   * overlaid by any stored override), so this stays correct after an
   * operator repins a resource from the Policy modal without re-editing
   * its engine declaration. */
  policy: PolicyMap;
  onChanged: () => void;
  /** The board's "+ add engine here" click (App.tsx's `onAddEngineHere`),
   * naming which node and GPU to open the create-instance form on. Selects
   * that node in the rail below (the effect right after `selected`'s own
   * declaration) so the operator lands on the right form without a second
   * click, and is threaded down to that node's `EnginesSection`, which
   * consumes it once and calls `onSeedConsumed` — see NodeForm's own prop
   * doc for why "once per mount" already means "once per seed" here. */
  instanceSeed: { nodeId: string; gpuIndex: number } | null;
  onSeedConsumed: () => void;
}) {
  const [selected, setSelected] = useState<string | "add" | null>(null);
  const remotes = nodes.filter((n) => n.agent_kind !== "local");
  const entry = nodes.find((n) => n.id === selected) ?? null;

  // Fires whenever a fresh seed arrives (App sets `instanceSeed` and calls
  // `showView("nodes")` in the same click) — selects the seed's node in the
  // rail so EnginesSection, below, mounts already showing that node's form
  // and can consume the seed itself.
  useEffect(() => {
    if (instanceSeed) setSelected(instanceSeed.nodeId);
  }, [instanceSeed]);

  return (
    <Panel className="nodes-view" title={labels.nodes}>
      {/* Panel renders its title as a flow sibling of these children (the
          head bar, then whatever this passes), so the rail/divider/pane grid
          lives one level down on .nodes-body rather than on the Panel's own
          section — putting display:grid there would pull the title bar into
          the rail's 280px column alongside everything else. */}
      <div className="nodes-body">
        <div className="nodes-rail">
          {nodes.map((n) => (
            <button
              key={n.id}
              type="button"
              className={`nodes-row ${selected === n.id ? "nodes-row--selected" : ""}`}
              onClick={() => setSelected(n.id)}
            >
              <span
                className={`node-dot ${DOT[n.status ?? ""] ?? "node-dot--dim"}`}
                // Empty, not "unconfigured", when status is null: null means
                // the observer hasn't ticked this node yet (a fresh add, or
                // the pass hasn't run), which is a different fact from the
                // backend's actual "unconfigured" (no credential stored) —
                // conflating them would misreport a just-added node as
                // having no credential when it may well have one.
                title={n.status ?? ""}
              />
              <span className="nodes-row-label">{n.label}</span>
              {n.agent_kind === "local" && (
                <span className="nodes-row-badge">{labels.localBadge}</span>
              )}
              <span className="nodes-row-address">{n.address ?? ""}</span>
            </button>
          ))}
          <button
            type="button"
            className="nodes-row nodes-row--add"
            onClick={() => setSelected("add")}
          >
            {labels.addNode}
          </button>
        </div>
        <div className="nodes-rail-divider" aria-hidden="true" />
        <div className="nodes-form-pane">
          {selected === null && remotes.length === 0 ? (
            <EmptyState onAdd={() => setSelected("add")} />
          ) : selected === null ? (
            <div className="nodes-hint">{labels.selectANode}</div>
          ) : selected === "add" ? (
            <NodeForm
              key="add"
              mode="add"
              entry={null}
              gpus={gpus}
              policy={policy}
              onDone={(id) => {
                setSelected(id);
                onChanged();
              }}
              onEnginesChanged={onChanged}
              instanceSeed={instanceSeed}
              onSeedConsumed={onSeedConsumed}
            />
          ) : entry ? (
            <>
              {/* Keyed on the entry's own id: switching rail rows must START
                  a new form, not carry the previous row's buffers (label
                  edits, a typed credential, an armed delete) into this one.
                  A useState initializer and a ref-held arming flag only run
                  once per mount, so the key is what makes "once per
                  selection" true. */}
              <NodeForm
                key={entry.id}
                mode="edit"
                entry={entry}
                gpus={gpus}
                policy={policy}
                onDone={() => onChanged()}
                onDeleted={() => {
                  setSelected(null);
                  onChanged();
                }}
                onEnginesChanged={onChanged}
                instanceSeed={instanceSeed}
                onSeedConsumed={onSeedConsumed}
              />
            </>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}

/** Mockup 15: the screen an operator sees before they have registered
 * anything beyond the local box. */
function EmptyState({ onAdd }: { onAdd: () => void }) {
  const m = messages.nodesEmptyTitle();
  return (
    <div className="nodes-empty">
      <h2>{m.title}</h2>
      <p>{m.body}</p>
      <ol className="nodes-empty-steps">
        <li>
          <span className="nodes-step nodes-step--active">1</span>
          {labels.addNode}
        </li>
        <li>
          <span className="nodes-step">2</span>
          {labels.registerALocation}
        </li>
      </ol>
      <button type="button" className="primary" onClick={onAdd}>
        {labels.addNode}
      </button>
    </div>
  );
}

function NodeForm({
  mode,
  entry,
  gpus,
  policy,
  onDone,
  onDeleted,
  onEnginesChanged,
  instanceSeed,
  onSeedConsumed,
}: {
  mode: "add" | "edit";
  entry: DeckNodeEntry | null;
  gpus: Gpu[];
  policy: PolicyMap;
  onDone: (id: string) => void;
  onDeleted?: () => void;
  /** Bumps the board-level refresh after any engine add/edit/forget, so the
   * board's resource cards and the live policy map (this form's own pinned
   * badges included) catch up immediately rather than waiting for the next
   * poll tick. */
  onEnginesChanged: () => void;
  /** Threaded straight through to `EnginesSection` below — see NodesView's
   * own doc for the seed's origin and lifecycle. Unused in "add" mode
   * (EnginesSection never mounts for a node that does not exist yet), but
   * accepted unconditionally so both NodeForm call sites in NodesView stay
   * symmetrical. */
  instanceSeed: { nodeId: string; gpuIndex: number } | null;
  onSeedConsumed: () => void;
}) {
  // Seeded ONCE per mount. The parent keys this component by selection (see
  // NodesView above), so switching rows unmounts/remounts rather than
  // re-rendering this instance with new props — this initializer running
  // exactly once per node is what makes that safe.
  const [form, setForm] = useState<NodeFormState>(entry ? formForEntry(entry) : emptyForm());
  const [testResult, setTestResult] = useState<NodeTestResult | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  // Two-click delete, armed with plain local state. model/armed.ts exists
  // because ArmedButton used to keep `armed` across a re-render of the SAME
  // instance (a retry re-arming a force override the operator never
  // clicked). That failure mode needs the component to survive the state
  // change it's guarding against. Here it can't: any change of `selected`
  // — including the one this button itself causes via onDeleted — remounts
  // this component under a new key, so a stale `true` can never carry into
  // another node's form. The refusal-identity machinery has nothing to do.
  const [deleteArmed, setDeleteArmed] = useState(false);

  // Add always creates a node-agent row (node_store.py's add() refuses
  // agent_kind "local" outside its one seed); edit carries the target's own
  // kind, so the seeded local entry's address stops being required — see
  // nodeForm.ts's validate() doc comment. Also gates control: "swap" itself
  // categorically (app/node_store.py:_validate refuses it for "local"
  // unconditionally), which is why the checkbox below is disabled for it.
  const agentKind = entry?.agent_kind ?? "node-agent";
  // `entry` is threaded through so a stored credential (entry.credential_set)
  // can satisfy control: "swap"'s credential prerequisite without forcing a
  // retype — mirrors app/node_store.py:_require_swap_prereqs's own
  // credential_present check.
  const errors = validate(form, mode, agentKind, entry);
  // What Test would actually probe, recomputed every render from the
  // current buffer — see nodeForm.ts's testTarget() doc comment for why an
  // edited-but-unretyped address is "blocked" rather than silently testing
  // the old stored one.
  const target = testTarget(form, entry);
  const set = (field: keyof NodeFormState) => (e: ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [field]: e.target.value });
    setTestResult(null);
  };

  async function save() {
    setSaveError(null);
    try {
      if (mode === "add") {
        const created = await createNode(toCreatePayload(form));
        onDone(created.id);
      } else if (entry) {
        await updateNode(entry.id, toPatchPayload(form, entry));
        onDone(entry.id);
      }
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    }
  }

  async function test() {
    if (target.kind === "blocked") return; // the button is disabled for this
    setTestResult(null);
    try {
      const result =
        target.kind === "typed"
          ? await testNode({ address: target.address, credential: target.credential })
          : await testNode({ node_id: entry!.id }); // "stored" only when entry exists
      setTestResult(result);
    } catch (err) {
      setTestResult({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  }

  async function handleDelete() {
    if (!entry || !onDeleted) return;
    if (!deleteArmed) {
      setDeleteArmed(true);
      return;
    }
    setSaveError(null);
    try {
      await deleteNode(entry.id);
      onDeleted();
    } catch (err) {
      // node_store.py refuses to remove "local" (GuardError) — unreachable
      // here since the button is hidden for it, but any other refusal (e.g.
      // a concurrent delete) lands here rather than an unhandled rejection.
      setSaveError(err instanceof Error ? err.message : String(err));
      setDeleteArmed(false);
    }
  }

  return (
    <div className="nodes-form">
      <label>
        {labels.nodeId}
        <input type="text" value={form.id} onChange={set("id")} disabled={mode === "edit"} />
      </label>
      <label>
        {labels.nodeLabel}
        <input type="text" value={form.label} onChange={set("label")} />
      </label>
      <label>
        {labels.nodeAddress}
        <input
          type="text"
          value={form.address}
          onChange={set("address")}
          placeholder={labels.nodeAddressPlaceholder}
        />
      </label>
      <label>
        {labels.nodeServingAddress}
        <input
          type="text"
          value={form.servingAddress}
          onChange={set("servingAddress")}
          placeholder={labels.nodeServingAddressPlaceholder}
        />
      </label>
      <label>
        {labels.nodeCredential}
        <input
          type="password"
          value={form.credential}
          onChange={set("credential")}
          placeholder={entry?.credential_set ? labels.nodeCredentialPlaceholder : ""}
        />
        <span className="nodes-caption">{messages.nodeCredentialCaption().title}</span>
      </label>
      <label>
        {labels.nodeControl}
        <select
          value={form.control}
          onChange={(e) => {
            setForm({ ...form, control: e.target.value as NodeFormState["control"] });
            setTestResult(null);
          }}
        >
          <option value="none">{labels.nodeControlNone}</option>
          {/* Never removed from the list — app/node_store.py:_validate
              refuses control:"swap" for agent_kind:"local" unconditionally
              (G1 revisits), so this option is disabled rather than absent
              here, the same "never offer a kind the write gate would
              refuse" posture kindsFor's own doc states, applied via the
              option's own `disabled` rather than a second picker. */}
          <option value="swap" disabled={agentKind === "local"}>{labels.nodeControlSwap}</option>
          {/* UNLIKE "swap", control: "instances" has no categorical local
              refusal (app/node_store.py's _require_instances_prereqs' own
              docstring: "Local or remote alike; the protocol is
              node-generic") — offered unconditionally. */}
          <option value="instances">{labels.nodeControlInstances}</option>
        </select>
        <span className="nodes-caption">
          {agentKind === "local" && form.control === "swap"
            ? labels.nodeControlLocalRefused
            : labels.nodeControlHint}
        </span>
      </label>
      {form.control === "instances" && (
        <>
          <label>
            {labels.nodeInstancePortStart}
            <input
              type="number"
              min={1024}
              max={65535}
              value={form.instancePortStart}
              onChange={(e) => {
                setForm({ ...form, instancePortStart: e.target.value });
                setTestResult(null);
              }}
            />
          </label>
          <label>
            {labels.nodeInstancePortEnd}
            <input
              type="number"
              min={1024}
              max={65535}
              value={form.instancePortEnd}
              onChange={(e) => {
                setForm({ ...form, instancePortEnd: e.target.value });
                setTestResult(null);
              }}
            />
          </label>
        </>
      )}

      <div className="nodes-form-actions">
        <button
          type="button"
          onClick={test}
          disabled={target.kind === "blocked"}
          title={target.kind === "blocked" ? target.reason : undefined}
        >
          {labels.testConnection}
        </button>
        {testResult && (
          <Banner
            message={
              testResult.ok
                ? messages.nodeTestOk(testResult.name ?? "", testResult.gpu_count ?? 0)
                : messages.nodeTestFailed(testResult.error ?? "")
            }
            onDismiss={() => setTestResult(null)}
          />
        )}
      </div>

      {saveError && (
        <Banner message={messages.guardRefused(saveError)} onDismiss={() => setSaveError(null)} />
      )}

      <div className="nodes-form-actions">
        {/* node_store.py:142-143 refuses to remove "local" unconditionally
            — offering the button would only ever end in that refusal. */}
        {mode === "edit" && entry && entry.agent_kind !== "local" && onDeleted && (
          <button
            type="button"
            className={deleteArmed ? "danger" : undefined}
            onClick={handleDelete}
          >
            {/* SavedSets already cataloged this exact two-click pair as
                deleteSet/reallyDelete ("Delete"/"Really delete?") — reused
                verbatim rather than adding a duplicate delete/confirmDelete
                pair with identical text. */}
            {deleteArmed ? labels.reallyDelete : labels.deleteSet}
          </button>
        )}
        <button
          type="button"
          className="primary"
          disabled={errors.length > 0}
          onClick={save}
          title={errors.join("; ")}
        >
          {labels.save}
        </button>
      </div>

      {/* Declared engines (E1 Task 12) live on whichever node's own card is
          selected — local AND node-agent (sglang-omni Task 10:
          `app/node_store.py`'s `_validate`, :204-214, and `_heal_engines`,
          :112-138, relaxed the write gate off "local entry only" in E1
          Task 5 — a node-agent entry may carry `engines[]` too, gated on
          each declared kind's own `remote_capable` flag rather than on
          which node it lives on). */}
      {mode === "edit" && entry && (entry.agent_kind === "local" || entry.agent_kind === "node-agent") && (
        <EnginesSection
          nodeId={entry.id}
          isRemote={entry.agent_kind === "node-agent"}
          gpus={gpus}
          policy={policy}
          onChanged={onEnginesChanged}
          instanceSeed={instanceSeed}
          onSeedConsumed={onSeedConsumed}
        />
      )}
    </div>
  );
}

/** One node's declared-engine list + add/edit sub-form (E1 Task 12; opened
 * up to node-agent entries by sglang-omni Task 10). Reads its own data
 * independently of the outer `nodes` prop: neither `DeckNodeEntry` (this
 * screen's own `nodes` prop, `/api/state`'s `_nodes_block`) nor
 * `world.tenants` (the LIVE observation map, local-only in any case) carries
 * the raw declaration's `connection`/`policy_defaults` fields this editor
 * needs to prefill an edit — only `listNodeRegistry()`'s matching entry does
 * (api.ts's `NodeRegistryEntry.engines` doc comment). */
function EnginesSection({
  nodeId,
  isRemote,
  gpus,
  policy,
  onChanged,
  instanceSeed,
  onSeedConsumed,
}: {
  /** Which registry entry this instance edits — "local" or any node-agent
   * id, the same `id` the outer rail selected (NodesView's own `entry.id`).
   * Threaded through every CRUD call below instead of a hardcoded "local"
   * literal (sglang-omni Task 10) — the lookup shape itself
   * (`registry.find((n) => n.id === X)`) was already generic before this
   * task; only `X` was ever fixed. */
  nodeId: string;
  /** Whether `nodeId` is a node-agent entry — decides which capability
   * direction the kind picker filters on (`kindsFor`, model/engineForm.ts).
   * Passed rather than re-derived from the registry fetch below so the
   * picker's filter is correct from the very first render, before that
   * fetch resolves — it is exactly `entry.agent_kind === "node-agent"` off
   * the SAME entry this section was mounted for. */
  isRemote: boolean;
  gpus: Gpu[];
  policy: PolicyMap;
  onChanged: () => void;
  /** The board's "+ add engine here" seed (NodesView's own doc), consumed
   * exactly once below when it names THIS node — a stale seed surviving a
   * rail switch to a DIFFERENT node must never open an instance form here. */
  instanceSeed: { nodeId: string; gpuIndex: number } | null;
  onSeedConsumed: () => void;
}) {
  const [kinds, setKinds] = useState<EngineKindsResponse | null>(null);
  const [engines, setEngines] = useState<DeclaredEngine[] | null>(null);
  // The registry entry's own declared `control` (app/node_store.py
  // _CONTROLS) — read off the SAME `listNodeRegistry()` fetch `engines`
  // comes from, so "declared control: instances" and "this node's own
  // engines[]" can never describe two different polls. Gates the
  // "+ Create instance" button below: instances are a free-standing verb
  // this node's control must actually support, not merely "some engine
  // exists to edit".
  const [control, setControl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [editing, setEditing] =
    useState<string | "add" | { mode: "instance"; gpuIndex: number | null } | null>(null);
  // messages.instanceRequested()'s banner (D-I1-1: create writes a
  // declaration, no intent — the operator's first load/adopt is a SEPARATE
  // click) — shown once after a successful create, since InstanceFormPanel
  // itself unmounts via afterCreate's `setEditing(null)` and cannot keep
  // showing its own banner. Cleared on the next edit rather than left to
  // stick around describing a request that already resolved one way or
  // the other.
  const [notice, setNotice] = useState<string | null>(null);

  function reload() {
    Promise.all([getEngineKinds(), listNodeRegistry()]).then(
      ([k, { nodes: registry }]) => {
        const own = registry.find((n) => n.id === nodeId);
        setKinds(k);
        setEngines(own?.engines ?? []);
        setControl(own?.control ?? null);
        setLoadError(null);
      },
      (err) => setLoadError(err instanceof Error ? err.message : String(err)),
    );
  }

  // Fires once per mount, and again if `nodeId` ever changed under an
  // instance that stayed mounted. Today it never does — the outer NodeForm
  // is keyed on `entry.id` (NodesView above), so switching rail rows already
  // remounts NodeForm and, with it, this section — but naming `nodeId` as
  // the dependency (rather than an empty array) keeps that true by the
  // effect's OWN contract instead of by a coincidence one level up.
  useEffect(reload, [nodeId]);

  // Consumes the board's "+ add engine here" seed exactly once per mount.
  // An EMPTY dependency array is deliberate, same "once per mount already
  // means once per seed" reasoning `reload`'s own comment states just
  // above: NodeForm remounts this section on every rail switch, so a seed
  // naming a DIFFERENT node never reaches an already-mounted instance of
  // this one, and a seed naming THIS node is consumed the one time this
  // effect can ever see it.
  // Empty deps deliberately, not `[instanceSeed, nodeId, onSeedConsumed]` —
  // see the comment above: this must run exactly once per mount, not
  // re-fire on every later `instanceSeed` change (App clears it right back
  // to null via `onSeedConsumed`, which would otherwise immediately
  // re-trigger this effect's own body against a now-null seed).
  // oxlint-disable react-hooks/exhaustive-deps
  useEffect(() => {
    if (instanceSeed?.nodeId === nodeId) {
      setEditing({ mode: "instance", gpuIndex: instanceSeed.gpuIndex });
      onSeedConsumed();
    }
  }, []);
  // oxlint-enable react-hooks/exhaustive-deps

  if (loadError) {
    return (
      <div className="engines-section">
        <h3>{labels.engines}</h3>
        <Banner message={messages.enginesLoadFailed(loadError)} onAction={reload} />
      </div>
    );
  }
  if (!kinds || !engines) {
    return (
      <div className="engines-section">
        <h3>{labels.engines}</h3>
        <div className="nodes-caption">{labels.loading}</div>
      </div>
    );
  }

  const sorted = sortedEngines(engines);

  function afterMutate() {
    setEditing(null);
    setNotice(null);
    reload();
    onChanged();
  }

  function afterCreate() {
    setEditing(null);
    setNotice(messages.instanceRequested().title);
    reload();
    onChanged();
  }

  return (
    <div className="engines-section">
      <h3>{labels.engines}</h3>
      {notice && <Banner message={messages.instanceRequested()} onDismiss={() => setNotice(null)} />}
      {sorted.length === 0 && editing === null && (
        <div className="engines-empty">
          <p>{messages.enginesEmpty().title}</p>
        </div>
      )}
      {sorted.length > 0 && (
        <ul className="engines-list">
          {sorted.map((e) => (
            <EngineRow
              key={e.resource}
              nodeId={nodeId}
              engine={e}
              kinds={kinds}
              gpus={gpus}
              pinned={policy[e.resource]?.pinned ?? e.policy_defaults.pinned}
              onEdit={() => setEditing(e.resource)}
              onForgotten={afterMutate}
            />
          ))}
        </ul>
      )}
      {editing === null && (
        <div className="engines-section-actions">
          <button type="button" onClick={() => setEditing("add")}>
            {labels.addEngine}
          </button>
          {/* Only when this node's own registry entry declares control:
              "instances" — a node that merely happens to have engines
              declared (control "none"/"swap") gets no free-standing
              instance-create surface at all, same as the board's own
              "+ add engine here" gate (NodeCard's `node.instancesCapable`). */}
          {control === "instances" && (
            <button
              type="button"
              onClick={() => setEditing({ mode: "instance", gpuIndex: null })}
            >
              {labels.createInstance}
            </button>
          )}
        </div>
      )}
      {editing !== null && (
        typeof editing === "string" ? (
          <EngineFormPanel
            // Keyed on what's being edited — same "start a new buffer, don't
            // inherit the last one" rule NodeForm's own key={entry.id} states
            // above, applied one level down.
            key={editing}
            mode={editing === "add" ? "add" : "edit"}
            entry={editing === "add" ? null : (sorted.find((e) => e.resource === editing) ?? null)}
            nodeId={nodeId}
            isRemote={isRemote}
            kinds={kinds}
            gpus={gpus}
            onDone={afterMutate}
            onCancel={() => setEditing(null)}
          />
        ) : (
          <InstanceFormPanel
            key={`instance:${editing.gpuIndex}`}
            nodeId={nodeId}
            isRemote={isRemote}
            kinds={kinds}
            gpus={gpus}
            seedGpu={editing.gpuIndex}
            onDone={afterCreate}
            onCancel={() => setEditing(null)}
          />
        )
      )}
    </div>
  );
}

/** One declared engine's row: identity + a pinned badge + either Edit/Forget
 * (a plain declared engine) or Move/Remove (`engine.managed` — an INST I1
 * instance). Every guarded action reuses the deck's one armed-action
 * machinery (`isArmedFor`/`ArmedButton`, model/armed.ts) rather than
 * inventing an inline arm/confirm — same pattern PlacementActions.tsx's
 * Force park and SparkSwap.tsx's Force swap use. Move and Remove are two
 * INDEPENDENT guarded verbs that can both sit on screen at once (Remove is
 * always visible; Move's picker opens beside it), so each gets its OWN
 * refusalSeq/armedForSeq identity pair — sharing one would let arming
 * Remove read as Move already being armed too, the exact cross-button
 * version of the stale-arming bug armed.ts's own docstring describes. */
function EngineRow({
  nodeId,
  engine,
  pinned,
  kinds,
  gpus,
  onEdit,
  onForgotten,
}: {
  nodeId: string;
  engine: DeclaredEngine;
  pinned: boolean;
  /** For `maxGpusFor` — bounds the Move picker's checkbox group to this
   * engine's own kind, same source EngineFormPanel's GPU picker reads. */
  kinds: EngineKindsResponse;
  /** The Move picker's own checkbox source — the same board GPU list
   * EngineFormPanel/InstanceFormPanel already read from `EnginesSection`. */
  gpus: Gpu[];
  onEdit: () => void;
  onForgotten: () => void;
}) {
  const [error, setError] = useState<string | null>(null);

  const [forgetRefusalSeq, setForgetRefusalSeq] = useState(0);
  const [forgetArmedForSeq, setForgetArmedForSeq] = useState<number | null>(null);
  const forgetArmed = isArmedFor(forgetArmedForSeq, forgetRefusalSeq);

  const [removeRefusalSeq, setRemoveRefusalSeq] = useState(0);
  const [removeArmedForSeq, setRemoveArmedForSeq] = useState<number | null>(null);
  const removeArmed = isArmedFor(removeArmedForSeq, removeRefusalSeq);

  const [moving, setMoving] = useState(false);
  const [target, setTarget] = useState<number[]>(() => engineGpus(engine));
  const [moveRefusalSeq, setMoveRefusalSeq] = useState(0);
  const [moveArmedForSeq, setMoveArmedForSeq] = useState<number | null>(null);
  const moveArmed = isArmedFor(moveArmedForSeq, moveRefusalSeq);

  const max = maxGpusFor(kinds.kinds, engine.kind);

  async function doForget() {
    try {
      // DELETE /api/nodes/{node_id}/engines/{resource} = forget_engine
      // (app/routers/nodes.py:376-473): bookkeeping only, never calls the
      // engine — see messages.ts's forgetEngineConfirm, shown below while
      // armed.
      await forgetEngine(nodeId, engine.resource);
      onForgotten();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setForgetRefusalSeq((n) => n + 1);
    }
  }

  async function doRemove() {
    try {
      // DELETE /api/nodes/{node_id}/instances/{resource} (app/routers/
      // instances.py): UNLIKE forgetEngine above, this STOPS AND DELETES
      // the running container (D-I1-1 — an instance is deck-managed end to
      // end) — messages.removeInstanceConfirm() says so while armed, below.
      await removeInstance(nodeId, engine.resource);
      onForgotten();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setRemoveRefusalSeq((n) => n + 1);
    }
  }

  async function doMove() {
    try {
      // POST /api/nodes/{node_id}/instances/{resource}/move (app/routers/
      // instances.py): the honest two-phase (D-I1-1) — the container is
      // removed, then re-created on `target`; nothing migrates live.
      await moveInstance(nodeId, engine.resource, target);
      // Fix round 1 (INST I1 T12 controller ruling): disarm BEFORE
      // onForgotten(), not after — a real defect, found by the gate this
      // task added. UNLIKE doForget/doRemove above, a successful move does
      // NOT unmount this EngineRow instance (the resource keeps its name
      // and key across a move, D-I1-1) — reload()'s refetch just re-renders
      // it with a new GPU claim. `moveArmedForSeq`/`moveRefusalSeq` only
      // ever change on a REFUSAL (the catch branch below), so without this,
      // a successful move left the Move ArmedButton — and its "Click again
      // to confirm" caption/hint — stuck in the armed state indefinitely,
      // with the picker still open, even though nothing was left to
      // confirm. Closing the picker (`setMoving(false)`) and clearing the
      // armed identity (`setMoveArmedForSeq(null)`, `isArmedFor`'s own
      // model/armed.ts contract: null never matches any refusalSeq) mirrors
      // how a successful Forget/Remove ends up "unarmed" by virtue of the
      // row disappearing entirely.
      setMoving(false);
      setMoveArmedForSeq(null);
      onForgotten();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setMoveRefusalSeq((n) => n + 1);
    }
  }

  return (
    <li className="engine-row">
      <span className="engine-row-resource">{engine.resource}</span>
      <span className="engine-row-kind">{engine.kind}</span>
      <span className="engine-row-gpu">{`GPU ${engineGpus(engine).join("+")}`}</span>
      {engine.managed && (
        <span className="ui-pill instance-tag">{labels.instanceTag}</span>
      )}
      {engine.managed && engine.port !== undefined && (
        <span className="engine-row-port">{labels.instancePort(engine.port)}</span>
      )}
      {pinned && (
        <span className="ui-pill" title={labels.pinnedTitle}>
          {labels.pinned}
        </span>
      )}
      {error && (
        <Banner message={messages.guardRefused(error)} onDismiss={() => setError(null)} />
      )}
      <div className="engine-row-actions">
        {engine.managed ? (
          <>
            <button
              type="button"
              onClick={() => {
                setMoving((m) => !m);
                setTarget(engineGpus(engine));
              }}
            >
              {labels.moveInstance}
            </button>
            {moving && (
              <>
                <fieldset className="engine-gpu-picker">
                  <legend>{labels.engineGpus}</legend>
                  {gpus.map((g) => (
                    <label key={g.index}>
                      <input
                        type="checkbox"
                        checked={target.includes(g.index)}
                        onChange={() => setTarget(toggleIndex(target, g.index, max))}
                      />
                      {`GPU ${g.index} — ${bytesToGB(g.free)} / ${bytesToGB(g.total)} GB free`}
                    </label>
                  ))}
                </fieldset>
                <ArmedButton
                  label={labels.moveInstance}
                  armed={moveArmed}
                  disabled={target.length === 0 || sameClaim(target, engineGpus(engine))}
                  onArm={() => setMoveArmedForSeq(moveRefusalSeq)}
                  onConfirm={doMove}
                />
                {moveArmed && (
                  <span className="engine-caption">{messages.moveInstanceConfirm(target).title}</span>
                )}
              </>
            )}
            <ArmedButton
              label={labels.removeInstance}
              armed={removeArmed}
              onArm={() => setRemoveArmedForSeq(removeRefusalSeq)}
              onConfirm={doRemove}
            />
            {removeArmed && (
              <span className="engine-caption">{messages.removeInstanceConfirm().title}</span>
            )}
          </>
        ) : (
          <>
            <button type="button" onClick={onEdit}>
              {labels.editEngine}
            </button>
            <ArmedButton
              label={labels.forgetEngine}
              armed={forgetArmed}
              onArm={() => setForgetArmedForSeq(forgetRefusalSeq)}
              onConfirm={doForget}
            />
            {forgetArmed && (
              <span className="engine-caption">{messages.forgetEngineConfirm().title}</span>
            )}
          </>
        )}
      </div>
    </li>
  );
}

/** The Create-instance sub-form (INST I1) — sibling to `EngineFormPanel`
 * below, over the free-standing instances route rather than the
 * declared-engines CRUD: its own kind filter (`instanceKindsFor`, only
 * kinds `instance: true` marks), its own env buffer (`EngineKindDef.
 * instance_env`, not `connection`), and no `resource` field at all — the
 * node's instances-helper names the container, the operator never types a
 * resource name for one. */
function InstanceFormPanel({
  nodeId,
  isRemote,
  kinds,
  gpus,
  seedGpu,
  onDone,
  onCancel,
}: {
  nodeId: string;
  isRemote: boolean;
  kinds: EngineKindsResponse;
  gpus: Gpu[];
  /** The board card the operator clicked "+ Create instance"/"+ add engine
   * here" from, or `null` for the section's own "+ Create instance" button
   * (no GPU pre-picked). Seeds the form's starting claim only — the
   * operator can still change it before saving. */
  seedGpu: number | null;
  onDone: () => void;
  onCancel: () => void;
}) {
  const available = instanceKindsFor(kinds, isRemote);
  // Seeded once per mount, same rationale as EngineFormPanel's own `form`
  // state below — the parent keys this by `editing`, so a different target
  // remounts rather than reusing this instance's buffer.
  const [form, setForm] = useState<InstanceFormState>(
    emptyInstanceForm(kinds, available[0]?.kind ?? "", seedGpu),
  );
  const [saveError, setSaveError] = useState<string | null>(null);
  const errors = instanceFormErrors(form);
  const max = maxGpusFor(kinds.kinds, form.kind);

  async function save() {
    setSaveError(null);
    try {
      // POST /api/nodes/{node_id}/instances (app/routers/instances.py's
      // create_instance): 201 = declared AND queued at the node's
      // instances-helper; the container appears on the board once it
      // answers (messages.instanceRequested(), shown by EnginesSection
      // after `onDone` below resolves).
      await createInstance(nodeId, toInstancePayload(form));
      onDone();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="engine-form instance-form">
      <label>
        {labels.engineKind}
        <select
          value={form.kind}
          onChange={(e) => setForm(withInstanceKind(form, kinds, e.target.value))}
        >
          {available.map((k) => (
            <option key={k.kind} value={k.kind}>
              {k.kind}
            </option>
          ))}
        </select>
      </label>
      <fieldset className="engine-gpu-picker">
        <legend>{labels.engineGpus}</legend>
        {gpus.map((g) => (
          <label key={g.index}>
            <input
              type="checkbox"
              checked={form.gpuIndices.includes(g.index)}
              onChange={() => setForm(toggleInstanceGpu(form, g.index, max))}
            />
            {`GPU ${g.index} — ${bytesToGB(g.free)} / ${bytesToGB(g.total)} GB free`}
          </label>
        ))}
      </fieldset>
      {Object.keys(form.env).map((name) => (
        <label key={name}>
          {labels.instanceEnvLabel(name)}
          {form.requiredEnv.includes(name) ? " *" : ""}
          <input
            value={form.env[name]}
            onChange={(e) => setForm(setInstanceEnv(form, name, e.target.value))}
          />
        </label>
      ))}
      {saveError && (
        <Banner message={messages.guardRefused(saveError)} onDismiss={() => setSaveError(null)} />
      )}
      <div className="engine-form-actions">
        <button type="button" onClick={onCancel}>
          {labels.cancel}
        </button>
        <button
          type="button"
          className="primary"
          disabled={errors.length > 0}
          onClick={save}
          title={errors.join("; ")}
        >
          {labels.createInstance}
        </button>
      </div>
    </div>
  );
}

/** The Add/Edit sub-form for one declared engine, on `nodeId` (local or
 * node-agent — sglang-omni Task 10). `entry === null` is Add; otherwise
 * Edit, pre-filled and with `resource` locked (rename is refused
 * server-side — app/routers/nodes.py's `update_engine`, :332-336 — so this
 * never offers it). `kind` stays editable in BOTH modes: app/state.py's
 * snapshot (:164-174) explicitly accommodates a resource re-declared under
 * a different kind, so an operator fixing a misdeclared kind does not need
 * to forget-and-re-add.
 *
 * The kind `<select>` below maps over `kindsFor(kinds, isRemote)`, never
 * `kinds.kinds` directly — offering a kind this target's write gate would
 * refuse (a local-only kind on a node-agent entry, or the reverse) is
 * exactly the 422-after-the-fact `kindsFor`'s own doc describes filtering
 * out (Task 7's review fallout). Add mode's default kind is that filtered
 * list's first entry, for the same reason: `kinds.kinds[0]` alone (the
 * catalog's own alphabetical order — "comfyui" today) could default Add to
 * a kind this node cannot run. */
function EngineFormPanel({
  mode,
  entry,
  nodeId,
  isRemote,
  kinds,
  gpus,
  onDone,
  onCancel,
}: {
  mode: "add" | "edit";
  entry: DeclaredEngine | null;
  nodeId: string;
  isRemote: boolean;
  kinds: EngineKindsResponse;
  gpus: Gpu[];
  onDone: () => void;
  onCancel: () => void;
}) {
  const availableKinds = kindsFor(kinds, isRemote);
  // Seeded once per mount, same rationale as NodeForm's own `form` state
  // above — the parent keys this by `editing`, so a different target
  // remounts rather than reusing this instance's buffer.
  const [form, setForm] = useState<EngineFormState>(
    entry ? engineFormForEntry(entry, kinds) : emptyEngineForm(kinds, availableKinds[0]?.kind ?? ""),
  );
  const [saveError, setSaveError] = useState<string | null>(null);
  const errors = formErrors(form);

  async function save() {
    setSaveError(null);
    try {
      const payload = toEnginePayload(form);
      if (mode === "add") {
        await addEngine(nodeId, payload);
      } else if (entry) {
        await updateEngine(nodeId, entry.resource, payload);
      }
      onDone();
    } catch (err) {
      // The buffer (`form`) is deliberately left untouched on failure — a
      // 422/409 must not cost the operator what they already typed.
      setSaveError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="engine-form">
      {/* Both modes, not just add: an EDIT can repoint connection.container
          at a container outside the consent setting just as easily. */}
      <Banner message={messages.engineConsentNote()} />
      <label>
        {labels.engineResource}
        <input
          type="text"
          value={form.resource}
          onChange={(e) => setForm({ ...form, resource: e.target.value })}
          disabled={mode === "edit"}
        />
      </label>
      <label>
        {labels.engineKind}
        <select
          value={form.kind}
          onChange={(e) => setForm(withKind(form, kinds, e.target.value))}
        >
          {availableKinds.map((k) => (
            <option key={k.kind} value={k.kind}>
              {k.kind}
            </option>
          ))}
        </select>
      </label>
      {Object.keys(form.connection).map((field) => (
        <label key={field}>
          {labels.engineFieldLabel(field)}
          {form.requiredConnectionFields.includes(field) ? " *" : ""}
          <input
            type="text"
            value={form.connection[field]}
            onChange={(e) => setForm(setEngineField(form, field, e.target.value))}
          />
        </label>
      ))}
      <fieldset className="engine-gpu-picker">
        <legend>{labels.engineGpus}</legend>
        {/* CONTROLLER RULING (Task 10 review landmine): the single-GPU
            <select> this replaces wrote ONLY `gpuIndices[0]` on save,
            silently truncating a multi-GPU declared engine's claim (edit an
            entry with gpu_indices [2,3], and Save would ship [2] alone).
            `toggleGpu` with `max === 1` REPLACES the whole claim (its own
            doc in engineForm.ts), so a single-GPU kind's picker still
            behaves like a radio group — the cap comes from the descriptor,
            never a literal here. */}
        {gpus.map((g) => (
          <label key={g.index}>
            <input
              type="checkbox"
              checked={form.gpuIndices.includes(g.index)}
              onChange={() =>
                setForm(toggleGpu(form, g.index, maxGpusFor(kinds.kinds, form.kind)))
              }
            />
            {`GPU ${g.index} — ${bytesToGB(g.free)} / ${bytesToGB(g.total)} GB free`}
          </label>
        ))}
      </fieldset>
      <label>
        <input
          type="checkbox"
          checked={form.pinned}
          onChange={(e) => setForm({ ...form, pinned: e.target.checked })}
        />
        {labels.enginePinnedLabel}
      </label>
      <label>
        <input
          type="checkbox"
          checked={form.containerConsent}
          onChange={(e) => setForm({ ...form, containerConsent: e.target.checked })}
        />
        {labels.engineConsentLabel}
      </label>
      <label>
        {labels.enginePriority}
        <input
          type="number"
          value={form.priority}
          onChange={(e) => setForm({ ...form, priority: Number(e.target.value) })}
        />
      </label>
      <label>
        {labels.engineIdleTtl}
        <input
          type="number"
          min={0}
          step={1}
          value={form.idleTtl}
          onChange={(e) => setForm({ ...form, idleTtl: Number(e.target.value) })}
        />
        <span className="field-value">{messages.ttlValue(form.idleTtl)}</span>
        {/* The consequence, not the setting: 900 on lemonade is free and
            saves ~70 W of idle burn; 900 on sglang-omni means a ~4 min
            rebuild by hand (GF4). 900 on hipfire does NOTHING (no idle rule
            at all — idle_release false). Same number, up to three different
            meanings — the distinction is the backend's `demand()`/
            `idle_release`, never a kind literal here. */}
        <span className="field-hint">
          {messages.ttlConsequence(
            form.idleTtl,
            demandFor(kinds.kinds, form.kind),
            idleReleaseFor(kinds.kinds, form.kind),
          )}
        </span>
      </label>

      {saveError && (
        <Banner message={messages.guardRefused(saveError)} onDismiss={() => setSaveError(null)} />
      )}

      <div className="engine-form-actions">
        <button type="button" onClick={onCancel}>
          {labels.cancel}
        </button>
        <button
          type="button"
          className="primary"
          disabled={errors.length > 0}
          onClick={save}
          title={errors.join("; ")}
        >
          {labels.save}
        </button>
      </div>
    </div>
  );
}
