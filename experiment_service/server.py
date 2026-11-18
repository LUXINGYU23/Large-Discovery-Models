"""HTTP binding of the ES contract (design 3.2) on the standard-library threading server."""
from __future__ import annotations

import hmac
import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from . import contract
from .service import PROBLEM_ROUTE_NOT_FOUND, Problem

log = logging.getLogger(__name__)
MAX_BODY = 1 << 20
ROUTES = (
    ("PUT", re.compile(r"/v1/experiments/([^/:]+)"), "put"),
    ("GET", re.compile(r"/v1/experiments/([^/:]+)"), "get"),
    ("GET", re.compile(r"/v1/experiments"), "changes"),
    ("POST", re.compile(r"/v1/experiments/([^/:]+):cancel"), "cancel"),
    ("GET", re.compile(r"/v1/experiments/([^/:]+)/artifacts/([^/]+)"), "artifact"),
    ("GET", re.compile(r"/v1/kinds"), "kinds"),
    ("GET", re.compile(r"/v1/contract"), "contract"),
    ("GET", re.compile(r"/v1/kinds/([^/]+)"), "kind"),
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ExperimentService/1"

    def do_GET(self):
        self._handle("GET")

    def do_PUT(self):
        self._handle("PUT")

    def do_POST(self):
        self._handle("POST")

    def log_message(self, format, *args):  # noqa: A002 - signature fixed by the base class
        log.debug("%s %s", self.address_string(), format % args)

    def _handle(self, method):
        url = urlsplit(self.path)
        try:
            body = self._body() if method in ("PUT", "POST") else None
            token = self.server.token
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                raise Problem(401, "Missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
            path_matched = False
            for route_method, pattern, name in ROUTES:
                match = pattern.fullmatch(url.path)
                if match:
                    path_matched = True
                    if route_method == method:
                        args = [unquote(g) for g in match.groups()]
                        return getattr(self, "_" + name)(*args, query=parse_qs(url.query), body=body)
            if path_matched:
                raise Problem(405, "Method not allowed")
            raise Problem(404, "Not found", problem_type=PROBLEM_ROUTE_NOT_FOUND)
        except Problem as problem:
            self._send_json(problem.status, problem.to_json(), problem.headers, "application/problem+json")
        except Exception:  # noqa: BLE001 - never leak a traceback to the client
            log.exception("experiment service request failed")
            self._send_json(500, Problem(500, "Internal server error").to_json(), None, "application/problem+json")

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            raise Problem(413, "Request body too large")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Problem(400, "Request body is not valid JSON", str(exc)) from None

    def _send_json(self, status, payload, headers=None, content_type="application/json"):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _param(query, name, default, convert):
        values = query.get(name)
        if not values:
            return default
        try:
            return convert(values[-1])
        except ValueError:
            raise Problem(400, f"Invalid query parameter {name!r}") from None

    def _put(self, experiment_id, query, body):
        status, resource = self.server.service.submit(experiment_id, body)
        self._send_json(status, resource, {"Location": f"/v1/experiments/{experiment_id}"} if status == 201 else None)

    def _get(self, experiment_id, query, body):
        self._send_json(200, self.server.service.get(experiment_id))

    def _cancel(self, experiment_id, query, body):
        status, resource = self.server.service.cancel(experiment_id, body)
        self._send_json(status, resource)

    def _changes(self, query, body):
        self._send_json(200, self.server.service.changes(
            self._param(query, "campaign_id", None, str), self._param(query, "changed_after", "", str),
            self._param(query, "limit", 100, int), self._param(query, "wait_seconds", 0.0, float)))

    def _kinds(self, query, body):
        service = self.server.service
        self._send_json(200, {"kinds": [service.kind_info(kind) for kind in sorted(service.executors)]})

    def _kind(self, kind, query, body):
        self._send_json(200, self.server.service.kind_info(kind))

    def _contract(self, query, body):
        self._send_json(200, contract.load(), None, "application/schema+json")

    def _artifact(self, experiment_id, name, query, body):
        media_type, path, sha256 = self.server.service.artifact(experiment_id, name)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", f'"sha256:{sha256}"')
        self.end_headers()
        self.wfile.write(data)


class ExperimentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128  # the socketserver default of 5 resets bursts of concurrent clients

    def __init__(self, service, host="127.0.0.1", port=0, token=None):
        self.service, self.token = service, token
        super().__init__((host, port), _Handler)

    @property
    def url(self):
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


def make_server(service, host="127.0.0.1", port=0, token=None) -> ExperimentHTTPServer:
    """Bind (port 0 picks a free port); call ``serve_forever`` to run."""
    return ExperimentHTTPServer(service, host, port, token)
