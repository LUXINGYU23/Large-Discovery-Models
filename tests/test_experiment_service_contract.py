"""The machine-readable wire contract (experiment_service/contract) against the real HTTP service."""
import json
import re
from datetime import datetime, timedelta

import pytest

from experiment_service import Artifact, SimulatedExecutor, VirtualClock, contract
from experiment_service.contract import SchemaError, check
from experiment_service.contract.validator import ANNOTATIONS, KEYWORDS
from experiment_service.testing import request, running_service

SCHEMA = contract.load()
OPERATIONS = SCHEMA["x-operations"]
COMMON_ERRORS = set(SCHEMA["x-common-errors"])
EXPERIMENT_NOT_FOUND = "urn:ldm-es:problem:experiment-not-found"
ROUTE_NOT_FOUND = "urn:ldm-es:problem:route-not-found"


# ---- the validator itself -----------------------------------------------------------------------------------------

def test_validator_keywords():
    assert check(1, {"type": "integer"}) == [] and check(True, {"type": "integer"}) != []
    assert check(1.5, {"type": "number"}) == [] and check(True, {"type": "number"}) != []
    assert check(None, {"type": ["string", "null"]}) == [] and check(3, {"type": ["string", "null"]}) != []
    assert check("a", {"enum": ["a", "b"]}) == [] and check("c", {"enum": ["a", "b"]}) != []
    assert check(None, {"enum": [None, "x"]}) == [] and check(1, {"enum": [True]}) != []  # 1 is not True
    assert check("ab12", {"pattern": "^[a-z]+[0-9]+$"}) == [] and check("12ab", {"pattern": "^[a-z]+[0-9]+$"}) != []
    assert check("", {"minLength": 1}) != [] and check(-1, {"minimum": 0}) != []
    obj = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}, "additionalProperties": False}
    assert check({"a": "x"}, obj) == []
    assert check({}, obj) != [] and check({"a": 1}, obj) != [] and check({"a": "x", "b": 1}, obj) != []
    assert check({"a": "x", "b": 1}, {**obj, "additionalProperties": {"type": "integer"}}) == []
    assert check({"a": "x", "b": "s"}, {**obj, "additionalProperties": {"type": "integer"}}) != []
    assert check({"b": 1}, {"additionalProperties": {"type": "integer"}}) == []
    assert check({"b": "s"}, {"additionalProperties": {"type": "integer"}}) != []
    items = {"type": "array", "items": {"type": "integer"}}
    assert check([1, 2], items) == [] and "$[1]" in check([1, "x"], items)[0]
    root = {"$defs": {"n": {"type": "integer"}}}
    assert check(1, {"$ref": "#/$defs/n"}, root) == [] and check("x", {"$ref": "#/$defs/n"}, root) != []


def test_validator_rejects_what_it_cannot_enforce():
    for schema in ({"oneOf": []}, {"properties": {"a": {"format": "date-time"}}}, {"type": "date"},
                   {"$ref": "http://example.invalid/x"}):
        with pytest.raises(SchemaError):
            check({"a": 1}, schema)


def walk_schema(node, where="#"):
    """Static check of the whole contract: only supported keywords, resolvable references."""
    assert isinstance(node, dict), where
    for key, value in node.items():
        assert key in KEYWORDS or key in ANNOTATIONS or key.startswith("x-"), f"{where}: unsupported keyword {key}"
        if key in ("properties", "$defs"):
            for name, sub in value.items():
                walk_schema(sub, f"{where}/{key}/{name}")
        elif key == "items" or (key == "additionalProperties" and isinstance(value, dict)):
            walk_schema(value, f"{where}/{key}")
        elif key == "$ref":
            assert value.startswith("#/$defs/") and value[len("#/$defs/"):] in SCHEMA["$defs"], f"{where}: {value}"
        elif key == "pattern":
            re.compile(value)
        elif key == "required":
            assert set(value) <= set(node.get("properties", {})), f"{where}: required names an unlisted property"


def test_contract_document_is_self_consistent():
    walk_schema(SCHEMA)
    assert SCHEMA["x-contract-version"] == "1" and SCHEMA["$schema"].endswith("2020-12/schema")
    for name, op in OPERATIONS.items():
        for definition in [op.get("request")] + [r["schema"] for r in op["responses"].values()]:
            assert definition is None or definition in SCHEMA["$defs"], (name, definition)
    assert {m["problemType"] for op in OPERATIONS.values() for m in op["responses"].values() if "problemType" in m} \
        <= set(SCHEMA["$defs"]["Problem"]["properties"]["type"]["enum"])


# ---- helpers: every exchange with the service is validated against its operation ---------------------------------

class Recorder:
    def __init__(self, es, token=None):
        self.es, self.token, self.resources, self.statuses = es, token, [], set()

    def call(self, operation, path, body=None, *, expect=None, valid=True):
        op = OPERATIONS[operation]
        method = op["method"]
        if valid and "request" in op and body is not None:
            assert check(body, {"$ref": f"#/$defs/{op['request']}"}, SCHEMA) == [], (operation, body)
        status, headers, payload = request(self.es.base_url, method, path, body, token=self.token)
        self.statuses.add(status)
        if expect is not None:
            assert status == expect, (operation, path, status, payload)
        declared = op["responses"].get(str(status))
        if declared is None:
            assert str(status) in COMMON_ERRORS, (operation, status, payload)
            declared = {"schema": "Problem"}
        for header in declared.get("headers", ()):
            assert header in headers, (operation, status, header)
        if declared["schema"] == "Problem":
            assert headers["Content-Type"] == "application/problem+json", (operation, status)
            assert payload["status"] == status
            if "problemType" in declared:
                assert payload["type"] == declared["problemType"]
        elif declared["schema"] is not None:
            assert headers["Content-Type"] == "application/json"
        if declared["schema"] is not None:
            assert check(payload, {"$ref": f"#/$defs/{declared['schema']}"}, SCHEMA) == [], (operation, status, payload)
        if declared["schema"] in ("Experiment", "ChangePage"):
            self.resources += payload["items"] if declared["schema"] == "ChangePage" else [payload]
        return status, headers, payload

    def put(self, eid, body, **kw):
        return self.call("putExperiment", f"/v1/experiments/{eid}", body, **kw)

    def get(self, eid, **kw):
        return self.call("getExperiment", f"/v1/experiments/{eid}", **kw)[2]

    def cancel(self, eid, body=None, **kw):
        return self.call("cancelExperiment", f"/v1/experiments/{eid}:cancel", body, **kw)

    def feed(self, campaign, after="", **params):
        query = "&".join([f"campaign_id={campaign}", f"changed_after={after}"] + [f"{k}={v}" for k, v in params.items()])
        return self.call("listChanges", f"/v1/experiments?{query}", expect=200)[2]


def spec_body(kind, x, campaign="c1", **extra):
    return {"campaign_id": campaign, "kind": kind, "kind_version": "1", "spec": {"x": x}, **extra}


def artifacts(spec, result):
    return [Artifact("log.txt", "text/plain", data=f"x={spec['x']}".encode()), Artifact("blob.bin", data=b"\x00\x01")]


def failure_by_x(spec, attempt):
    return {2: "infeasible", 3: "execution_error", 4: "timeout", 5: "infrastructure"}.get(spec["x"])


def test_every_resource_state_validates_against_the_contract():
    clock = VirtualClock()
    sim = SimulatedExecutor(clock, kind="sim.main", duration=10.0, artifacts=artifacts, failures=failure_by_x,
                            cancel_delay=1.0, objective=lambda spec: {"value": spec["x"] / 2, "nested": {"ok": True}})
    unsure = SimulatedExecutor(clock, kind="sim.unsure", duration=10.0, dispatch_uncertain=1.0, launch_on_uncertain=0.0)
    flaky = SimulatedExecutor(clock, kind="sim.flaky", duration=10.0)
    retry = SimulatedExecutor(clock, kind="sim.retry", duration=3.0, retry_safe=True,
                              failures=lambda spec, attempt: "infrastructure" if attempt == 1 else None)
    with running_service([sim, unsure, flaky, retry], pools={"default": 12}, clock=clock) as es:
        rec = Recorder(es)
        assert rec.call("listKinds", "/v1/kinds", expect=200)[2]["kinds"][0]["capabilities"]
        for x in range(1, 9):
            limits = {"limits": {"max_runtime_seconds": 5}} if x == 6 else {}
            assert rec.put(f"m{x}", spec_body("sim.main", x, client={"decision_id": f"d{x}"}, **limits), expect=201)[1]["Location"]
        rec.put("u1", spec_body("sim.unsure", 1), expect=201)
        rec.put("f1", spec_body("sim.flaky", 1), expect=201)
        rec.put("r1", spec_body("sim.retry", 1), expect=201)
        for x in range(9, 12):  # beyond the pool capacity of 12: these stay QUEUED
            rec.put(f"m{x}", spec_body("sim.main", x), expect=201)
        assert rec.get("m11")["state"] == "QUEUED"
        rec.cancel("m10", {"campaign_id": "c1"}, expect=200)  # QUEUED -> CANCELED
        es.service.run_until_idle()
        flaky.set_unreachable()
        es.service.tick()
        assert rec.get("f1")["condition"] == "unreachable" and rec.get("u1")["condition"] == "dispatch_unconfirmed"
        flaky.set_unreachable(unreachable=False)
        rec.cancel("m8", {"campaign_id": "c1"}, expect=202)  # RUNNING with a stop requested
        rec.cancel("ghost", {"campaign_id": "c1"}, expect=200)  # tombstones, with and without a campaign
        rec.cancel("ghost2", expect=200)
        rec.cancel("m1", {"campaign_id": "c1"}, expect=202)  # canceled while running, then completes first
        rec.put("ghost", spec_body("sim.main", 1), expect=200)
        rec.feed("c1")
        es.service.run_simulation()
        for eid in [f"m{x}" for x in range(1, 12)] + ["u1", "f1", "r1", "ghost", "ghost2"]:
            rec.get(eid)
        pages, after = [], ""
        while True:
            page = rec.feed("c1", after, limit=3)
            pages.append(page)
            after = page["cursor"]
            if not page["more"]:
                break
        assert len(pages) > 1 and all(len(p["items"]) <= 3 for p in pages)
        assert rec.feed("c1", after)["items"] == [] and rec.feed("c1", after)["cursor"] == after
        assert rec.feed("no-such-campaign")["cursor"] == f"{0:016d}"
        # artifacts: manifest entries are exact and the bytes behind each href match
        done = rec.get("m7")
        assert done["state"] == "SUCCEEDED" and {a["name"] for a in done["artifacts"]} == {"log.txt", "blob.bin"}
        for entry in done["artifacts"]:
            status, headers, data = request(es.base_url, "GET", entry["href"])
            assert status == 200 and headers["Content-Type"] == entry["media_type"]
            assert headers["ETag"] == f'"sha256:{entry["sha256"]}"' and len(data) == entry["size"]

    seen = {(r["state"], r["condition"]) for r in rec.resources}
    assert {"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED"} == {s for s, _ in seen}
    assert {c for _, c in seen} >= {None, "unreachable", "dispatch_unconfirmed"}
    assert {r["failure"]["class"] for r in rec.resources if r["failure"]} == {
        "infeasible", "execution_error", "timeout", "infrastructure"}
    final = {r["experiment_id"]: r for r in rec.resources if r["state"] in ("SUCCEEDED", "FAILED", "CANCELED")}
    assert final["m1"]["state"] == "CANCELED"
    assert final["m6"]["failure"] == {"class": "timeout", "message": "max_runtime_seconds exceeded"}
    assert final["m8"]["state"] == "CANCELED" and final["m10"]["state"] == "CANCELED" and final["m10"]["started_at"] is None
    assert [a["outcome"]["state"] for a in final["r1"]["attempts"]] == ["failed", "succeeded"] and final["r1"]["state"] == "SUCCEEDED"
    assert final["ghost"]["tombstone"] and final["ghost"]["campaign_id"] == "c1" and final["ghost"]["spec"] is None
    assert final["ghost2"]["tombstone"] and final["ghost2"]["campaign_id"] is None
    assert final["m7"]["usage"]["pool"] == "default" and final["m7"]["usage"]["reported"]
    assert any(r["attempts"] and r["attempts"][0]["outcome"] is None for r in rec.resources)  # an attempt in flight
    assert any(r["client"] == {"decision_id": "d3"} for r in rec.resources) and any(r["client"] is None for r in rec.resources)
    assert any(r["limits"] == {"max_runtime_seconds": 5} for r in rec.resources)


def test_executor_unavailable_condition_validates(tmp_path):
    clock, database = VirtualClock(), tmp_path / "es.sqlite3"
    executor = SimulatedExecutor(clock, kind="sim.main", duration=10.0)
    with running_service([executor], clock=clock, database=database) as es:
        Recorder(es).put("m1", spec_body("sim.main", 1), expect=201)
        es.service.run_until_idle()
    with running_service([], clock=clock, database=database) as es:
        rec = Recorder(es)
        es.service.run_until_idle()
        assert rec.get("m1")["condition"] == "executor_unavailable"
        assert rec.call("listKinds", "/v1/kinds", expect=200)[2] == {"kinds": []}


def test_format_rules_revisions_cursors_timestamps():
    clock = VirtualClock(start=1_800_000_000.0)
    with running_service([SimulatedExecutor(clock, kind="sim.main", duration=2.5)], clock=clock) as es:
        rec = Recorder(es)
        for x in range(4):
            rec.put(f"t{x}", spec_body("sim.main", x), expect=201)
        es.service.run_simulation()
        page = rec.feed("c1")
        revisions = [r["revision"] for r in page["items"]]
        assert revisions == sorted(revisions) and len(set(revisions)) == 4
        assert all(len(r) == 16 and r.isdigit() for r in revisions + [page["cursor"]]) and page["cursor"] == revisions[-1]
        for resource in page["items"]:
            stamps = [resource[k] for k in ("accepted_at", "started_at", "finished_at")]
            parsed = [datetime.fromisoformat(s) for s in stamps]
            assert all(p.utcoffset() == timedelta(0) for p in parsed) and parsed == sorted(parsed)
            assert resource["accepted_at"].startswith("2027-") and resource["usage"]["runtime_seconds"] == 2.5
        # the schema is not vacuous: drifted resources are rejected
        sample = page["items"][0]
        for field, value in (("revision", "12"), ("revision", 12), ("accepted_at", "2027-01-15T08:00:00Z"),
                             ("finished_at", "2027-01-15T08:00:00+01:00"), ("state", "DONE"), ("condition", "stale"),
                             ("tombstone", 0), ("spec_digest", "sha256:xyz"), ("extra", 1)):
            assert contract.validate({**sample, field: value}, "Experiment") != [], (field, value)
        assert contract.validate({k: v for k, v in sample.items() if k != "usage"}, "Experiment") != []
        assert contract.validate({**page, "cursor": ""}, "ChangePage") != []


def test_error_bodies_are_problem_json_and_declared():
    clock = VirtualClock()
    sim = SimulatedExecutor(clock, kind="sim.main", validate=lambda s: [] if isinstance(s.get("x"), int) else ["x must be int"])
    with running_service([sim], clock=clock, max_queued_per_campaign=1, retry_after_seconds=3) as es:
        rec = Recorder(es)
        rec.put("a", spec_body("sim.main", 1), expect=201)
        rec.put("a", spec_body("sim.main", 2), expect=409)
        rec.put("b", spec_body("sim.main", 2), expect=429)
        problem = rec.put("c", spec_body("sim.main", "bad", campaign="c2"), expect=422)[2]
        assert problem["problems"] == ["x must be int"]
        assert rec.put("c", spec_body("sim.nope", 1, campaign="c2"), expect=422)[2]["type"] == "about:blank"
        rec.put("c", {"campaign_id": "c2"}, expect=400, valid=False)
        rec.put("bad%20id", spec_body("sim.main", 1), expect=400)
        rec.call("getExperiment", "/v1/experiments/-leading", expect=400)
        rec.call("listChanges", "/v1/experiments", expect=400)
        rec.call("listChanges", "/v1/experiments?campaign_id=c1&changed_after=abc", expect=400)
        rec.call("getKind", "/v1/kinds/sim.nope", expect=404)
        rec.call("getArtifact", "/v1/experiments/a/artifacts/missing", expect=404)
        rec.call("getContract", "/v1/nowhere", expect=404)
        rec.call("getExperiment", "/v1/experiments/a:cancel", expect=405)
        status, _, payload = request(es.base_url, "PUT", "/v1/experiments/z", "[1]")  # a JSON string, not an object
        assert status == 400 and check(payload, {"$ref": "#/$defs/Problem"}, SCHEMA) == []
    with running_service([sim], clock=clock, token="s3cret") as es:
        status, headers, payload = request(es.base_url, "GET", "/v1/kinds")
        assert status == 401 and headers["WWW-Authenticate"] == "Bearer" and contract.validate(payload, "Problem") == []


def test_request_schemas_accept_good_and_reject_bad_bodies():
    good = spec_body("k", 1, limits={"max_runtime_seconds": 5}, client={"a": 1}, spec_digest="sha256:" + "0" * 64)
    assert contract.validate(good, "PutRequest") == []
    assert contract.validate({**good, "limits": None, "client": None}, "PutRequest") == []
    for bad in ({k: v for k, v in good.items() if k != "campaign_id"}, {**good, "spec": []}, {**good, "campaign_id": ""},
                {**good, "spec_digest": "md5:abc"}, {**good, "unknown": 1}, {**good, "limits": {"cpu": 2}}):
        assert contract.validate(bad, "PutRequest") != [], bad
    assert contract.validate({}, "CancelRequest") == [] and contract.validate({"campaign_id": "c"}, "CancelRequest") == []
    assert contract.validate({"campaign_id": 1}, "CancelRequest") != []


# ---- the contract endpoint and the 404 distinction ----------------------------------------------------------------

def test_contract_endpoint_serves_the_packaged_document():
    clock = VirtualClock()
    with running_service([SimulatedExecutor(clock, kind="sim.main")], clock=clock, token="s3cret") as es:
        assert request(es.base_url, "GET", "/v1/contract")[0] == 401
        status, headers, document = request(es.base_url, "GET", "/v1/contract", token="s3cret")
    assert status == 200 and headers["Content-Type"] == "application/schema+json"
    assert document == SCHEMA == json.loads(contract.SCHEMA_PATH.read_text())
    assert contract.validate({"items": [], "cursor": "0" * 16, "more": False}, "ChangePage") == []


def test_unknown_experiment_404_is_distinguishable_from_unknown_route():
    clock = VirtualClock()
    with running_service([SimulatedExecutor(clock, kind="sim.main")], clock=clock) as es:
        status, headers, missing = request(es.base_url, "GET", "/v1/experiments/never.submitted")
        assert status == 404 and headers["Content-Type"] == "application/problem+json"
        assert missing["type"] == EXPERIMENT_NOT_FOUND and missing["title"] == "Experiment not found"
        assert contract.validate(missing, "Problem") == []
        status, headers, route = request(es.base_url, "GET", "/v1/not/a/route")
        assert status == 404 and headers["Content-Type"] == "application/problem+json" and route["type"] == ROUTE_NOT_FOUND
        assert route["type"] != missing["type"]
        kind = request(es.base_url, "GET", "/v1/kinds/nope")[2]
        artifact = request(es.base_url, "GET", "/v1/experiments/never.submitted/artifacts/x")[2]
        assert len({missing["type"], route["type"], kind["type"], artifact["type"]}) == 4
        # a malformed id is a 400 (never an "experiment missing" signal); a canceled-before-submit id is a 200 tombstone
        assert request(es.base_url, "GET", "/v1/experiments/.hidden")[0] == 400
        assert request(es.base_url, "POST", "/v1/experiments/never.submitted:cancel", {"campaign_id": "c1"})[0] == 200
        status, _, tomb = request(es.base_url, "GET", "/v1/experiments/never.submitted")
        assert status == 200 and tomb["tombstone"] is True
