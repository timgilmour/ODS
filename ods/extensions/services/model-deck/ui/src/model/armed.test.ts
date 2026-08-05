import { describe, expect, it } from "vitest";
import { isArmedFor } from "./armed";

describe("isArmedFor", () => {
  it("is not armed before the operator has armed anything", () => {
    expect(isArmedFor(null, 0)).toBe(false);
    expect(isArmedFor(null, 7)).toBe(false);
  });

  it("is armed for the refusal the operator actually clicked", () => {
    expect(isArmedFor(4, 4)).toBe(true);
  });

  it("DISARMS the moment a new refusal arrives", () => {
    // The Critical this whole module exists for. The operator armed against
    // refusal 4; the same guarded action was retried and refused again, so
    // the refusal on screen is now 5. The old flag survived that transition
    // because the component was updated rather than remounted — an identity
    // cannot.
    expect(isArmedFor(4, 5)).toBe(false);
  });

  it("stays disarmed however many refusals go by", () => {
    for (let seq = 5; seq < 20; seq++) expect(isArmedFor(4, seq)).toBe(false);
  });

  it("cannot be re-armed by a counter that happens to go backwards", () => {
    // Not reachable today (the counter only increments), but the property
    // worth having is "armed means THIS refusal", not "armed means some
    // refusal with a smaller number".
    expect(isArmedFor(9, 4)).toBe(false);
  });

  it("treats the very first refusal, numbered zero, as armable", () => {
    // A `!armedForSeq` truthiness check would get this wrong: 0 is a real
    // refusal id, and the first refusal of a session is exactly the one an
    // operator is most likely to force.
    expect(isArmedFor(0, 0)).toBe(true);
  });
});
