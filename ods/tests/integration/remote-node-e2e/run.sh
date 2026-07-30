#!/usr/bin/env bash
# Run the remote-node end-to-end harness. Requires Docker; needs no GPU.
#
#   ./run.sh            build and run, exit with the test-runner's status
#   ./run.sh --keep     leave the containers up afterwards for poking at
set -euo pipefail

cd "$(dirname "$0")"
COMPOSE=(docker compose -f docker-compose.test.yml)
KEEP=false
[ "${1:-}" = "--keep" ] && KEEP=true

teardown() {
    if [ "$KEEP" = true ]; then
        echo "Left running (--keep). Tear down with:"
        echo "  ${COMPOSE[*]} down -v --remove-orphans"
        return
    fi
    "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
}

# A previous interrupted run leaves the network and its static IPs claimed.
"${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
trap teardown EXIT

if [ "$KEEP" = true ]; then
    # --abort-on-container-exit stops EVERY container when the runner exits,
    # so it and --keep contradict each other: detach instead, stream the
    # runner's output, and leave the agent containers up for poking at.
    "${COMPOSE[@]}" up --build -d
    "${COMPOSE[@]}" logs -f test-runner &
    code="$(docker wait ods-e2e-test-runner)"
    wait %% 2>/dev/null || true
    exit "$code"
fi

"${COMPOSE[@]}" up --build --abort-on-container-exit --exit-code-from test-runner
