"""Helpers for downstream tests: an in-thread HTTP service and a tiny JSON client.

    clock = VirtualClock()
    sim = SimulatedExecutor(clock, kind="numeric.objective", duration=("uniform", 5, 15))
    with running_service([sim], pools={"default": 2}, clock=clock) as es:
        status, _, body = request(es.base_url, "PUT", "/v1/experiments/c1.d1.s0", {...})
        es.service.advance()          # deterministic: dispatch, move the virtual clock, settle
"""
from __future__ import annotations

import json
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .server import make_server
from .service import ExperimentService

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route loopback through a proxy


@dataclass
class ServiceHandle:
    base_url: str
    service: ExperimentService
    database: Path

    @property
    def clock(self):
        return self.service.clock


@contextmanager
def running_service(executors, *, pools=None, clock=None, database=None, autorun=False, poll_interval=0.02,
                    token=None, **options):
    """Run an ES with an HTTP server on 127.0.0.1:<free port> in a daemon thread.

    ``autorun=False`` (default) leaves scheduling to explicit ``handle.service.tick()`` /
    ``run_until_idle()`` / ``advance()`` calls, which is what deterministic tests want; with
    ``autorun=True`` a background ticker runs every ``poll_interval`` seconds and on each request.
    Pass the same ``database`` path to a second ``running_service`` to simulate an ES restart.
    """
    scratch = tempfile.TemporaryDirectory(prefix="es-") if database is None else None
    database = Path(scratch.name) / "es.sqlite3" if scratch else Path(database)
    service = ExperimentService(database, executors, pools=pools or {"default": 4}, clock=clock, **options)
    server = make_server(service, "127.0.0.1", 0, token=token)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    if autorun:
        service.start_background(poll_interval)
    try:
        yield ServiceHandle(server.url, service, database)
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=10)
        if scratch:
            scratch.cleanup()


def request(base_url, method, path, body=None, *, token=None, timeout=30.0) -> tuple[int, dict, Any]:
    """Send one JSON request; returns ``(status, headers, decoded body)`` for 2xx and errors alike."""
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url + path, data=data, method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as response:
            return response.status, dict(response.headers), _decode(response)
    except urllib.error.HTTPError as error:
        with error:
            return error.code, dict(error.headers), _decode(error)


def _decode(response):
    raw = response.read()
    content_type = response.headers.get("Content-Type", "")
    return json.loads(raw) if raw and "json" in content_type else raw
