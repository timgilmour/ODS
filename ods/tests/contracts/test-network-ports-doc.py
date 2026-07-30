#!/usr/bin/env python3
"""docs/NETWORK-PORTS.md must match what the manifests and config say.

A hand-maintained port table is wrong the first time someone changes a port and
forgets the doc, and a table that is quietly wrong about the firewall is worse
than no table. This regenerates it from the sources and compares, so the doc is
either correct or the build is red.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
GENERATOR = ROOT_DIR / "scripts" / "generate-network-ports-doc.py"
DOC_FILE = ROOT_DIR / "docs" / "NETWORK-PORTS.md"
REGENERATE = "python3 scripts/generate-network-ports-doc.py --write"


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_ports_doc", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_doc_exists() -> None:
    if not DOC_FILE.exists():
        fail(f"{DOC_FILE.relative_to(ROOT_DIR)} is missing. Run: {REGENERATE}")
    print("[PASS] docs/NETWORK-PORTS.md exists")


def test_doc_matches_sources(generator) -> None:
    expected = generator.render()
    actual = DOC_FILE.read_text(encoding="utf-8")
    if expected == actual:
        print("[PASS] docs/NETWORK-PORTS.md matches manifests and config")
        return
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for index, (want, got) in enumerate(zip(expected_lines, actual_lines), 1):
        if want != got:
            fail(f"docs/NETWORK-PORTS.md is stale at line {index}.\n"
                 f"  expected: {want}\n  found:    {got}\n  Run: {REGENERATE}")
    fail(f"docs/NETWORK-PORTS.md is stale: expected {len(expected_lines)} "
         f"lines, found {len(actual_lines)}. Run: {REGENERATE}")


def test_every_exposed_service_has_a_row(generator) -> None:
    """The generator's own filter is the thing most likely to drop a service
    silently -- host-networked services carry `port`, not
    `external_port_default`, so a filter written around the latter alone omits
    them and the table looks complete."""
    manifests = generator.load_manifests()
    exposed = {sid for sid, svc in manifests.items() if generator.is_exposed(svc)}
    doc = DOC_FILE.read_text(encoding="utf-8")
    missing = sorted(sid for sid in exposed if f"`{sid}`" not in doc)
    if missing:
        fail(f"Exposed services absent from the table: {', '.join(missing)}")
    host_networked = sorted(sid for sid, svc in manifests.items()
                            if svc.get("host_network"))
    if not host_networked:
        fail("No host-networked service found; this guard has stopped guarding"
             " anything and needs rewriting against a service that has one.")
    print(f"[PASS] All {len(exposed)} exposed services are listed "
          f"(including host-networked: {', '.join(host_networked)})")


def test_policy_fields_are_rendered(generator) -> None:
    """Guards against the join silently degrading to em-dashes if a config key
    is renamed: the table would still build, just uninformative."""
    doc = DOC_FILE.read_text(encoding="utf-8")
    policy = generator.load_policy()
    exposures = {entry.get("lan_exposure") for entry in policy.values()}
    unrendered = sorted(value for value in exposures
                        if value and value not in doc)
    if unrendered:
        fail(f"lan_exposure values missing from the table: {unrendered}. "
             f"Did a key in network-exposure-policy.json get renamed?")
    print("[PASS] Policy exposure values reach the table")


def main() -> int:
    if not GENERATOR.exists():
        fail(f"{GENERATOR.relative_to(ROOT_DIR)} is missing")
    generator = load_generator()
    test_doc_exists()
    test_doc_matches_sources(generator)
    test_every_exposed_service_has_a_row(generator)
    test_policy_fields_are_rendered(generator)
    print("All network-ports doc contracts passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
