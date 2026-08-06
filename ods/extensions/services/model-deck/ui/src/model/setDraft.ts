import type {
  ComfyuiEphemeral,
  ConfigSet,
  Durable,
  HipfireEphemeral,
  LemonadeEphemeral,
  TenantPolicy,
} from "../api";

/** The lemonade default-route prefix for library files — mirrors
 * SetBuilder's original constant; app/sets.py stores the route string
 * verbatim, the UI derives library membership from this prefix alone. */
export const EXTRA_PREFIX = "extra.";

/** The editable fields of the Set Builder, as one value. Pure data — the
 * component holds ONE of these in state instead of nine separate hooks
 * deciding anything. */
export interface DraftFields {
  name: string;
  notes: string;
  durable: Durable | null;
  lemonade: LemonadeEphemeral | null;
  comfyui: ComfyuiEphemeral | null;
  hipfire: HipfireEphemeral | null;
  policyOverrides: Record<string, TenantPolicy> | null;
}

export function buildDraft(f: DraftFields): ConfigSet {
  return {
    name: f.name.trim(),
    notes: f.notes,
    durable: f.durable,
    ephemeral: { lemonade: f.lemonade, comfyui: f.comfyui, hipfire: f.hipfire },
    policy_overrides: f.policyOverrides,
  };
}

/** Load a saved set into editable fields. CRITICAL 2 lives here: ephemeral
 * legs that are null STAY null ("don't touch") — a set that never mentioned
 * hipfire must never silently gain a concrete directive on the next save.
 * policy_overrides carries verbatim for the same reason. */
export function fieldsFromSet(cfgset: ConfigSet, clearName: boolean): DraftFields {
  return {
    name: clearName ? "" : cfgset.name,
    notes: cfgset.notes,
    durable: cfgset.durable,
    lemonade: cfgset.ephemeral?.lemonade ?? null,
    comfyui: cfgset.ephemeral?.comfyui ?? null,
    hipfire: cfgset.ephemeral?.hipfire ?? null,
    policyOverrides: cfgset.policy_overrides ?? null,
  };
}

/** Best-effort: only durable.default_route_model with the "extra." prefix
 * AND an active "loaded" lemonade intent can be traced back to a library
 * model file. A set naming some other litellm route can't be — the chip is
 * left blank and the user re-drops to pin one down. */
export function derivePlacedModel(cfgset: ConfigSet): string | null {
  const route = cfgset.durable?.default_route_model;
  const derived =
    route && route.startsWith(EXTRA_PREFIX)
      ? route.slice(EXTRA_PREFIX.length)
      : null;
  return cfgset.ephemeral?.lemonade?.state === "loaded" ? derived : null;
}

/** Structural equality for the save-gating ("Preview steps" is only
 * reachable while the saved copy matches the live draft). JSON stringify is
 * sufficient: both sides are built by buildDraft from the same field order. */
export function draftEquals(a: ConfigSet, b: ConfigSet): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
