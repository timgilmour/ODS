# deck-drill — Model Deck live capability suite

Spec: `~/notes/designs/2026-07-30-model-deck-live-capability-suite-design.md`
Architecture background: `~/notes/model-deck-architecture.md`

    ./deck-drill                 # safe tier: reversible drills, runs anytime
    ./deck-drill --disruptive    # + hipfire park/resume, route flips, heal replay
                                 #   ONLY in a declared window (pre-flight enforces)
    ./deck-drill --force ...     # override the hipfire-activity pre-flight
    ./deck-drill -k s11 -v       # pytest selection passthrough

Exit codes: 0 pass, 1 failures, 2 pre-flight refused.
Reports: `~/notes/evidence/deck-drills/<UTC stamp>.{md,json}` (+ `bookend.json` drift proof).

Safety model: every mutating fixture restores in a finalizer; the session fails if the
box's end state (loaded model, hipfire state, policy) differs from its start state.
Safe tier is structurally unable to park hipfire or activate routes. Never send hipfire
completions outside the disruptive window (single-slot conversation-cache eviction).
