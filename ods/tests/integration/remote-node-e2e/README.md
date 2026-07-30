# Remote-node end-to-end harness

Runs dashboard-api's remote-node poller against a **real `node-agent`
container**, over a real network hop, on a machine with no GPU.

```bash
./run.sh            # build, run, exit with the test suite's status
./run.sh --keep     # leave the containers up to poke at
```

Requires Docker and compose v2. Takes about a minute cold, seconds warm. Also
runs in CI (`.github/workflows/dashboard.yml`, job `remote-node-e2e`).

## Why it exists

`dashboard-api/tests/test_remote_nodes_poller.py` drives the poller through an
`httpx.MockTransport`. That mock is written in the same file as the assertions,
so it encodes *our* belief about what node-agent returns. If the two
independently-written sides ever disagree — a renamed field, a changed status
code, a header the agent stopped accepting — those tests keep passing.
Structurally, they cannot catch it.

This harness closes that gap. The bytes come from the actual service image
built from the actual `Dockerfile`, over an actual TCP connection, parsed by
the actual poller.

Two classes of bug it has already been shown to catch, neither of which the
unit tests can see:

- **`assigned_services` leaking across nodes.** The vendored collector infers
  service names from local compute processes; on a remote node those names
  belong to the *dashboard* host. node-agent scrubs the field, and
  `test_assigned_services_never_cross_the_wire` fails if that scrub is removed.
  Before this harness, only the metal deploy caught it.
- **Collector-level parsing.** Rounding, unit handling and argv construction in
  `gpu.py` all execute for real.

## Layout

| Path | What it is |
|---|---|
| `docker-compose.test.yml` | Three containers on a private bridge with static IPs |
| `Dockerfile.test-runner` | pytest + dashboard-api's import-time deps |
| `stubs/nvidia-smi` | Fake vendor CLI, so no GPU is needed |
| `stubs/serving_stub.py` | Minimal OpenAI-shaped `/v1/models` endpoint |
| `tests/` | The suite, run inside `test-runner` |

| Container | Address | Role |
|---|---|---|
| `ods-e2e-node-agent` | 172.30.20.21 | The real service image |
| `ods-e2e-serving-stub` | 172.30.20.22 | What the node is "serving" |
| `ods-e2e-test-runner` | 172.30.20.100 | Imports `remote_nodes.py`, polls the agent |
| *(nothing)* | 172.30.20.99 | The dead peer, for offline/isolation cases |

dashboard-api's source is bind-mounted read-only rather than copied, so the
harness always tests the working tree. Ordering is handled by healthcheck-gated
`depends_on`; nothing sleep-polls.

## Stubbing the CLI, not the Python

`stubs/nvidia-smi` replaces the vendor binary on `PATH` inside the node-agent
container and answers both queries the collector makes (`--query-gpu` and
`--query-compute-apps`). Patching `subprocess.run` in Python instead would skip
the collector's own argv construction and CSV parsing — the parts most likely
to break. The values it reports are deliberately odd (12345 MB used, 122880 MB
total, 30.5 W) so an assertion cannot pass by coincidence.

To exercise a different scenario, edit the stub: an empty response makes the
collector return nothing, which should surface as an `online` node carrying a
collector `error` rather than an offline one.

## Known limits

- **Going offline is simulated by repointing the node at an address nothing
  answers on, not by stopping the container.** The poller's in-process `_STATE`
  has to survive both cycles, and handing the test-runner the Docker socket to
  stop a sibling is the exact grant node-agent's own README tells operators to
  refuse. What the poller observes — a refused connection from a node it had
  reached a moment ago — is the same either way.
- The subnet `172.30.20.0/24` is fixed, chosen clear of ODS's own networks
  (172.17–172.22). Change it here if it collides with something on your host.
- An interrupted run can leave the network claimed; `run.sh` tears down before
  it starts for that reason.
