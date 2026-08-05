/**
 * Whether a destructive-override button is armed.
 *
 * This module exists because of a Critical found late in this project. The
 * `armed` flag used to live inside ArmedButton, and the call sites only
 * unmount that component when `error && offer*` goes FALSY. A retry of the
 * same guarded action sets `error` truthy → truthy, so React updated the
 * existing instance instead of remounting it, and `armed === true` carried
 * across into a refusal the operator had never clicked. The next single
 * click fired the force override. Reachable by ordinary retry behaviour.
 *
 * The first fix — a resetToken prop plus a disarming effect — was correct,
 * but it could only ever be checked by reasoning: "the component instance
 * survived a new refusal" is not something a pure function can say, so
 * nothing tested it, and the safety property rested on an effect firing.
 *
 * So `armed` stops being a flag and becomes an IDENTITY. The owner records
 * which refusal it armed against; `refusalSeq` increments on every refusal
 * (see runAction/doSwap). The stale state is then not merely prevented, it
 * is unsayable: two numbers either match or they do not, and a refusal the
 * operator did not arm has a number nothing is holding.
 *
 * Pure, and deliberately trivial. Its value is that it is reachable by a
 * test at all — the previous version's whole failure was living where no
 * test could get to it.
 */

/** `armedForSeq` is the refusal the operator armed against, or null if they
 * have armed nothing. `refusalSeq` is the refusal currently on screen. */
export function isArmedFor(armedForSeq: number | null, refusalSeq: number): boolean {
  return armedForSeq !== null && armedForSeq === refusalSeq;
}
