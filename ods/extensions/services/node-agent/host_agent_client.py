"""Shim satisfying gpu.py imports; windows-host paths are unsupported here."""


class AgentClientError(RuntimeError):
    pass


def request_json(*args, **kwargs):
    raise AgentClientError("host-agent not available on a remote node")
