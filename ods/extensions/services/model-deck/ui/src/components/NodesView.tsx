import { useState, type ChangeEvent } from "react";
import {
  createNode,
  deleteNode,
  testNode,
  updateNode,
  type DeckNodeEntry,
  type NodeTestResult,
} from "../api";
import { labels, messages } from "../model/messages";
import {
  emptyForm,
  formForEntry,
  toCreatePayload,
  toPatchPayload,
  validate,
  type NodeFormState,
} from "../model/nodeForm";
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
  onChanged,
}: {
  nodes: DeckNodeEntry[];
  onChanged: () => void;
}) {
  const [selected, setSelected] = useState<string | "add" | null>(null);
  const remotes = nodes.filter((n) => n.agent_kind !== "local");
  const entry = nodes.find((n) => n.id === selected) ?? null;

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
              onDone={(id) => {
                setSelected(id);
                onChanged();
              }}
            />
          ) : entry ? (
            // Keyed on the entry's own id: switching rail rows must START a
            // new form, not carry the previous row's buffers (label edits,
            // a typed credential, an armed delete) into this one. A
            // useState initializer and a ref-held arming flag only run once
            // per mount, so the key is what makes "once per selection" true.
            <NodeForm
              key={entry.id}
              mode="edit"
              entry={entry}
              onDone={() => onChanged()}
              onDeleted={() => {
                setSelected(null);
                onChanged();
              }}
            />
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
  onDone,
  onDeleted,
}: {
  mode: "add" | "edit";
  entry: DeckNodeEntry | null;
  onDone: (id: string) => void;
  onDeleted?: () => void;
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

  const errors = validate(form, mode);
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
    // The Test button is disabled exactly when neither branch below has
    // anything to test against (see its `disabled` prop) — this guard is
    // defensive, not a third user-facing outcome.
    if (!form.credential && !entry) return;
    setTestResult(null);
    try {
      // A typed credential tests that typed pair; otherwise the stored one.
      const result = form.credential
        ? await testNode({ address: form.address, credential: form.credential })
        : await testNode({ node_id: entry!.id });
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

      <div className="nodes-form-actions">
        <button
          type="button"
          onClick={test}
          disabled={!form.address || (!form.credential && !entry)}
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
    </div>
  );
}
