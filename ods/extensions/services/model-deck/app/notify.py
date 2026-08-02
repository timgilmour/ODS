"""
Post-move engine notification hooks — make the engine SEE a file that just
arrived in its store.

lemonade (verified live, v10.2.0): registers store GGUFs only at startup;
no rescan endpoint exists. Hook = container restart via DockerCtl
(`ods-llama-server` is already on the park allowlist). DEFERRED with a
returned warning when a model is loaded — we never yank a loaded model to
register a file; the operator retries after unload or the idle TTL clears
the way.

comfyui (verified live): /api/models/{type} lists files per request — no
action needed. engine "none": nothing to notify.
"""


def notify_engine(location: dict, deck: dict) -> str | None:
    if location["engine"] != "lemonade":
        return None
    if deck["lemonade"].status()["loaded"]:
        return ("lemonade has a model loaded — restart deferred; the new file "
                "registers after the next lemonade restart")
    container = deck["settings"].lemonade_container
    deck["dockerctl"].stop(container)
    deck["dockerctl"].start(container)
    return None
