"""HTTP client for the experiment service contract (design 3.2): urllib, timeouts, error classes.

Errors are classified by what the caller may do next:

* ``ESUnavailable`` - outcome unknown (timeout, connection error, 5xx): retry with the same id.
* ``ESBackpressure`` - 429: nothing was created; retry after ``retry_after`` seconds.
* ``ESRejected`` - a definite 4xx answer (400/401/404/409/422, ...): a caller or task defect; never retried.
"""
from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.parse import quote, urlencode


class TransportError(OSError):
    """The request may or may not have reached the service."""


class ESError(Exception):
    def __init__(self, message, *, status=None, problem=None):
        super().__init__(message)
        self.status, self.problem = status, problem


class ESUnavailable(ESError):
    pass


class ESBackpressure(ESUnavailable):
    def __init__(self, message, *, retry_after: float, status=429, problem=None):
        super().__init__(message, status=status, problem=problem)
        self.retry_after = retry_after


class ESRejected(ESError):
    pass


Transport = Callable[[str, str, "bytes | None", dict, float], "tuple[int, dict, bytes]"]


class UrllibTransport:
    """``(method, url, body, headers, timeout) -> (status, headers, body)``; 4xx/5xx are returned, not raised."""

    def __init__(self, use_env_proxy: bool = False):
        handlers = [] if use_env_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    def __call__(self, method, url, body, headers, timeout):
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return response.status, _lower(response.headers), response.read()
        except urllib.error.HTTPError as error:
            with error:
                try:
                    data = error.read()
                except (OSError, http.client.HTTPException):
                    data = b""
                return error.code, _lower(error.headers), data
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException,
                OSError) as exc:
            raise TransportError(f"{method} {url}: {type(exc).__name__}: {exc}") from exc


def _lower(headers) -> dict:
    return {str(k).lower(): v for k, v in (headers.items() if headers is not None else ())}


def parse_retry_after(value, default: float, now: float | None = None) -> float:
    if value is None:
        return default
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(str(value)).timestamp()
    except (TypeError, ValueError, IndexError, OverflowError):
        return default
    return max(0.0, when - (time.time() if now is None else now))


# The ES's typed problem for an unknown experiment id (by contract; the controller never imports the ES).
EXPERIMENT_NOT_FOUND = "urn:ldm-es:problem:experiment-not-found"


class ESClient:
    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 10.0,
                 transport: Transport | None = None, default_retry_after: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.token, self.timeout, self.default_retry_after = token, timeout, default_retry_after
        self.transport = transport or UrllibTransport()

    # ---- contract operations --------------------------------------------------------------------

    def put_experiment(self, experiment_id: str, body: bytes) -> tuple[int, dict]:
        """Send stored request bytes verbatim; returns ``(200|201, resource)``."""
        status, resource = self._call("PUT", self._path(experiment_id), body, ok=(200, 201))
        return status, self._resource(resource)

    def get_experiment(self, experiment_id: str) -> dict | None:
        """``None`` on 404 (the caller decides whether that is normal or an anomaly)."""
        status, resource = self._call("GET", self._path(experiment_id), ok=(200, 404))
        if status == 404:
            # Only the ES's own problem document means "no such experiment"; a bare 404 (wrong base_url, a proxy)
            # must not mark every experiment lost (controller-core review item 7).
            if not (isinstance(resource, dict) and resource.get("status") == 404
                    and resource.get("type") == EXPERIMENT_NOT_FOUND):
                raise ESUnavailable(f"GET {self._path(experiment_id)}: 404 that is not {EXPERIMENT_NOT_FOUND}",
                                    status=404, problem=resource if isinstance(resource, dict) else None)
            return None
        return self._resource(resource)

    def cancel_experiment(self, experiment_id: str, campaign_id: str) -> tuple[int, dict]:
        """Always names the campaign so a tombstone shows up in that campaign's change feed."""
        body = json.dumps({"campaign_id": campaign_id}).encode()
        status, resource = self._call("POST", self._path(experiment_id) + ":cancel", body, ok=(200, 202))
        return status, self._resource(resource)

    def changes(self, campaign_id: str, cursor: str = "", *, limit: int = 100, wait_seconds: float = 0.0) -> dict:
        query = urlencode({"campaign_id": campaign_id, "changed_after": cursor or "", "limit": int(limit),
                           "wait_seconds": wait_seconds})
        timeout = self.timeout + max(0.0, float(wait_seconds))
        _, page = self._call("GET", f"/v1/experiments?{query}", ok=(200,), timeout=timeout)
        if not isinstance(page, dict) or not isinstance(page.get("items"), list) or not isinstance(page.get("cursor"), str):
            raise ESUnavailable("malformed change-feed page")
        for item in page["items"]:
            self._resource(item)
        return page

    def kind(self, kind: str) -> dict:
        return self._call("GET", f"/v1/kinds/{quote(kind, safe='')}", ok=(200,))[1]

    # ---- plumbing ---------------------------------------------------------------------------------

    @staticmethod
    def _path(experiment_id: str) -> str:
        return f"/v1/experiments/{quote(experiment_id, safe='')}"

    @staticmethod
    def _resource(resource: Any) -> dict:
        if not isinstance(resource, dict) or not isinstance(resource.get("experiment_id"), str) \
                or not isinstance(resource.get("state"), str) or "revision" not in resource:
            raise ESUnavailable("malformed experiment resource")
        return resource

    def _call(self, method, path, body=None, *, ok, timeout=None):
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            status, response_headers, raw = self.transport(method, self.base_url + path, body, headers,
                                                           timeout or self.timeout)
        except TransportError as exc:
            raise ESUnavailable(str(exc)) from exc
        payload = None
        if raw:
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
        if status in ok:
            if payload is None and status != 404:
                raise ESUnavailable(f"{method} {path}: undecodable {status} body", status=status)
            return status, payload
        title = payload.get("title") if isinstance(payload, dict) else None
        message = f"{method} {path}: {status} {title or ''}".rstrip()
        if status == 429:
            retry = parse_retry_after(response_headers.get("retry-after"), self.default_retry_after)
            raise ESBackpressure(message, retry_after=retry, problem=payload)
        if status >= 500 or status == 408:
            raise ESUnavailable(message, status=status, problem=payload)
        raise ESRejected(message, status=status, problem=payload)
