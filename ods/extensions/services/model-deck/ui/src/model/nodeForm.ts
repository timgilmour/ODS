/**
 * Node add/edit form logic — pure, componentless (the "logic inline in a
 * component is logic no test can reach" rule, 4 recurrences in this repo).
 *
 * The credential field is WRITE-ONLY end to end: formForEntry always seeds
 * it "", the backend never echoes it (app/routers/nodes.py), and
 * toPatchPayload only ships it when the operator actually typed one.
 */

import type { DeckNodeEntry } from "../api";

/** Mirrors app/node_store.py _ID_RE — the backend refuses what this allows
 * through, so the two must agree; the backend is authoritative. */
export const ID_RE = /^[a-z0-9][a-z0-9-]*$/;

export interface NodeFormState {
  id: string;
  label: string;
  address: string;
  servingAddress: string;
  credential: string;
}

export function emptyForm(): NodeFormState {
  return { id: "", label: "", address: "", servingAddress: "", credential: "" };
}

export function formForEntry(entry: DeckNodeEntry): NodeFormState {
  return {
    id: entry.id,
    label: entry.label,
    address: entry.address ?? "",
    servingAddress: entry.serving_address ?? "",
    credential: "",
  };
}

export function validate(form: NodeFormState, mode: "add" | "edit"): string[] {
  const errors: string[] = [];
  if (mode === "add" && !ID_RE.test(form.id)) {
    errors.push("id must be a lowercase slug (a-z, 0-9, hyphens)");
  }
  if (!form.label.trim()) errors.push("label is required");
  if (!form.address.trim()) errors.push("address is required");
  if (mode === "add" && !form.credential) {
    errors.push("credential is required for a new node");
  }
  return errors;
}

export function toCreatePayload(form: NodeFormState) {
  return {
    id: form.id,
    label: form.label,
    address: form.address,
    serving_address: form.servingAddress.trim() || null,
    credential: form.credential,
  };
}

export function toPatchPayload(form: NodeFormState, entry: DeckNodeEntry) {
  const patch: Record<string, string | null> = {};
  if (form.label !== entry.label) patch.label = form.label;
  if (form.address !== (entry.address ?? "")) patch.address = form.address;
  const serving = form.servingAddress.trim() || null;
  if (serving !== (entry.serving_address ?? null)) patch.serving_address = serving;
  if (form.credential) patch.credential = form.credential;
  return patch;
}
