"""Black-box HTTP contract tests for the experiment service (design 8.1)."""
import hashlib
import sqlite3
import threading
import time

import pytest

from experiment_service import Artifact, SimulatedExecutor, VirtualClock, digest
from experiment_service.testing import request, running_service

KIND = "sim.objective"


def body(x, campaign="c1", **extra):
    return {"campaign_id": campaign, "kind": KIND, "kind_version": "1", "spec": {"x": x},
            "client": {"decision_id": f"d{x}"}, **extra}


def put(es, eid, payload):
    return request(es.base_url, "PUT", f"/v1/experiments/{eid}", payload)


def get(es, eid):
    return request(es.base_url, "GET", f"/v1/experiments/{eid}")[2]


def cancel(es, eid, payload=None):
    return request(es.base_url, "POST", f"/v1/experiments/{eid}:cancel", payload)


def feed(es, campaign="c1", after="", **params):
    query = "&".join([f"campaign_id={campaign}", f"changed_after={after}"] + [f"{k}={v}" for k, v in params.items()])
    return request(es.base_url, "GET", f"/v1/experiments?{query}")


def sim(clock, **options):
    options.setdefault("duration", 10.0)
    return SimulatedExecutor(clock, validate=lambda spec: [] if isinstance(spec.get("x"), int) else ["x must be int"],
                             **options)


@pytest.fixture
def clock():
    return VirtualClock()


def test_idempotent_put_cases(clock):
    with running_service([sim(clock)], clock=clock) as es:
        status, headers, created = put(es, "c1.d1.s0", body(1))
        assert status == 201 and created["state"] == "QUEUED" and headers["Location"] == "/v1/experiments/c1.d1.s0"
        assert created["spec_digest"] == digest({"x": 1}) and created["client"] == {"decision_id": "d1"}
        status, _, again = put(es, "c1.d1.s0", body(1))
        assert status == 200 and again["revision"] == created["revision"]
        es.service.run_until_idle()
        status, _, running = put(es, "c1.d1.s0", body(1, spec_digest=digest({"x": 1})))
        assert status == 200 and running["state"] == "RUNNING"
        for conflicting in (body(2), body(1, campaign="c2"), body(1, limits={"max_runtime_seconds": 5})):
            status, headers, problem = put(es, "c1.d1.s0", conflicting)
            assert status == 409 and headers["Content-Type"] == "application/problem+json"
        assert get(es, "c1.d1.s0")["spec"] == {"x": 1}
        assert put(es, "c1.d2.s0", body(2, spec_digest="sha256:00"))[0] == 422
        assert put(es, "c1.d3.s0", body("bad"))[0] == 422
        assert put(es, "c1.d4.s0", dict(body(4), kind="other"))[0] == 422
        assert put(es, "c1.d5.s0", dict(body(5), kind_version="2"))[0] == 422
        assert put(es, "c1.d6.s0", body(6, limits={"max_runtime_seconds": -1}))[0] == 422
        assert put(es, "c1.d7.s0", body(7, limits={"walltime": 3}))[0] == 422
        assert put(es, "bad%2Fid", body(8))[0] == 400
        assert put(es, "c1.d9.s0", {"campaign_id": "c1"})[0] == 400
        for eid in ("c1.d2.s0", "c1.d3.s0", "c1.d4.s0", "c1.d5.s0", "c1.d6.s0", "c1.d7.s0", "c1.d9.s0"):
            assert request(es.base_url, "GET", f"/v1/experiments/{eid}")[0] == 404


def test_concurrent_duplicate_puts_create_one_experiment(clock):
    with running_service([sim(clock)], clock=clock) as es:
        barrier, statuses = threading.Barrier(16), []

        def send():
            barrier.wait()
            statuses.append(put(es, "c1.dup.s0", body(3))[0])

        threads = [threading.Thread(target=send) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(statuses) == [200] * 15 + [201]
        assert [item["experiment_id"] for item in feed(es)[2]["items"]] == ["c1.dup.s0"]
        es.service.run_simulation()
        assert get(es, "c1.dup.s0")["state"] == "SUCCEEDED"


def test_backlog_limit_returns_429_without_creating(clock):
    with running_service([sim(clock)], clock=clock, max_queued_per_campaign=2, retry_after_seconds=7) as es:
        assert put(es, "a", body(1))[0] == 201 and put(es, "b", body(2))[0] == 201
        status, headers, _ = put(es, "c", body(3))
        assert status == 429 and headers["Retry-After"] == "7"
        assert request(es.base_url, "GET", "/v1/experiments/c")[0] == 404
        assert put(es, "a", body(1))[0] == 200  # a retry of an accepted request is never throttled
        assert put(es, "other", body(3, campaign="c2"))[0] == 201
        es.service.run_until_idle()  # both start (pool capacity 4), which empties the backlog
        assert put(es, "c", body(3))[0] == 201


def test_cancel_unknown_writes_tombstone_and_late_put_sees_it(clock):
    executor = sim(clock)
    with running_service([executor], clock=clock) as es:
        status, _, tomb = cancel(es, "c1.late.s0", {"campaign_id": "c1"})
        assert status == 200 and tomb["state"] == "CANCELED" and tomb["tombstone"] and tomb["started_at"] is None
        status, _, late = put(es, "c1.late.s0", body(5))
        assert status == 200 and late["state"] == "CANCELED" and late["revision"] == tomb["revision"]
        es.service.run_simulation()
        assert executor.launches == [] and get(es, "c1.late.s0")["state"] == "CANCELED"
        assert [i["experiment_id"] for i in feed(es)[2]["items"]] == ["c1.late.s0"]
        assert cancel(es, "c1.late.s0")[0] == 200


def test_cancel_queued_never_starts(clock):
    executor = sim(clock)
    with running_service([executor], pools={"default": 1}, clock=clock) as es:
        put(es, "first", body(1))
        put(es, "second", body(2))
        es.service.run_until_idle()
        status, _, canceled = cancel(es, "second")
        assert status == 200 and canceled["state"] == "CANCELED" and canceled["started_at"] is None
        assert canceled["cancel_requested_at"] is not None and canceled["attempts"] == []
        es.service.run_simulation()
        assert executor.launches == [("first", 1)]


@pytest.mark.parametrize("mode", ["cooperative", "immediate"])
def test_cancel_running_is_confirmed_by_the_executor(clock, mode):
    with running_service([sim(clock, cancel_running=mode, cancel_delay=2.0)], clock=clock) as es:
        put(es, "r", body(1))
        es.service.run_until_idle()
        clock.advance(3.0)
        status, _, pending = cancel(es, "r")
        assert status == 202 and pending["state"] == "RUNNING" and pending["cancel_requested_at"] is not None
        status, _, repeated = cancel(es, "r")
        assert status == 202 and repeated["revision"] == pending["revision"]
        es.service.run_until_idle()
        if mode == "cooperative":
            assert get(es, "r")["state"] == "RUNNING"
            es.service.advance()
        final = get(es, "r")
        assert final["state"] == "CANCELED" and final["started_at"] is not None and final["result"] is None
        assert final["finished_at"] == ("1970-01-01T00:00:05.000000+00:00" if mode == "cooperative"
                                        else "1970-01-01T00:00:03.000000+00:00")


def test_uncancellable_kind_and_cancel_losing_to_completion(clock):
    executors = [sim(clock, kind="sim.stubborn", cancel_running="none"),
                 sim(clock, cancel_running="cooperative", cancel_delay=5.0)]
    with running_service(executors, clock=clock) as es:
        put(es, "stubborn", dict(body(1), kind="sim.stubborn"))
        put(es, "racing", body(2))
        es.service.run_until_idle()
        clock.advance(8.0)
        assert cancel(es, "stubborn")[0] == 202 and cancel(es, "racing")[0] == 202
        es.service.run_simulation()
        for eid in ("stubborn", "racing"):
            final = get(es, eid)
            assert final["state"] == "SUCCEEDED" and final["cancel_requested_at"] is not None
            assert final["result"] is not None
        assert request(es.base_url, "GET", "/v1/kinds/sim.stubborn")[2]["capabilities"]["cancel_running"] == "none"


def test_terminal_state_is_immutable_and_result_digest_matches(clock):
    def outputs(spec, result):
        return [Artifact("trace.txt", "text/plain", data=f"x={spec['x']}".encode())]

    executor = sim(clock, artifacts=outputs, objective=lambda spec: {"objective": -spec["x"] ** 2})
    with running_service([executor], clock=clock) as es:
        put(es, "t", body(3))
        es.service.run_simulation()
        final = get(es, "t")
        assert final["state"] == "SUCCEEDED" and final["result"] == {"objective": -9}
        assert final["result_digest"] == digest(final["result"])
        [artifact] = final["artifacts"]
        status, headers, data = request(es.base_url, "GET", artifact["href"])
        assert status == 200 and data == b"x=3" and headers["Content-Type"] == "text/plain"
        assert artifact["sha256"] == hashlib.sha256(data).hexdigest() and artifact["size"] == len(data)
        assert request(es.base_url, "GET", "/v1/experiments/t/artifacts/missing")[0] == 404
        assert cancel(es, "t")[0] == 200 and put(es, "t", body(3))[0] == 200
        es.service.run_simulation()
        assert get(es, "t") == final
        with sqlite3.connect(es.database) as db, pytest.raises(sqlite3.DatabaseError, match="immutable"):
            db.execute("UPDATE experiments SET state='FAILED' WHERE experiment_id='t'")
        with sqlite3.connect(es.database) as db, pytest.raises(sqlite3.DatabaseError):
            db.execute("INSERT INTO artifacts VALUES ('t','extra','text/plain',0,'0','x')")


def test_change_feed_revisions_monotonic_and_complete(clock):
    with running_service([sim(clock, duration=("uniform", 1, 20), seed=3)], pools={"default": 2}, clock=clock) as es:
        cursor, seen, last = "", {}, 0
        for i in range(6):
            put(es, f"e{i}", body(i))
            put(es, f"other{i}", body(i, campaign="c2"))
        cancel(es, "e5")
        while True:
            _, _, page = feed(es, after=cursor, limit=3)
            revisions = [int(item["revision"]) for item in page["items"]]
            assert revisions == sorted(revisions) and all(r > last for r in revisions)
            assert all(item["campaign_id"] == "c1" for item in page["items"])
            last = revisions[-1] if revisions else last
            seen.update({item["experiment_id"]: item for item in page["items"]})
            cursor = page["cursor"]
            if not page["more"] and not es.service.advance():
                if not page["items"]:
                    break
        assert set(seen) == {f"e{i}" for i in range(6)}
        assert all(seen[eid] == get(es, eid) for eid in seen)
        assert {item["state"] for item in seen.values()} == {"SUCCEEDED", "CANCELED"}
        assert feed(es, after="bogus")[0] == 400
        assert request(es.base_url, "GET", "/v1/experiments")[0] == 400


def test_long_poll_times_out_empty_and_wakes_on_change(clock):
    with running_service([sim(clock)], clock=clock) as es:
        put(es, "a", body(1))
        cursor = feed(es)[2]["cursor"]
        began = time.monotonic()
        _, _, page = feed(es, after=cursor, wait_seconds=0.3)
        assert page["items"] == [] and page["cursor"] == cursor and time.monotonic() - began >= 0.25
        threading.Timer(0.2, lambda: put(es, "b", body(2))).start()
        began = time.monotonic()
        _, _, page = feed(es, after=cursor, wait_seconds=10)
        assert [i["experiment_id"] for i in page["items"]] == ["b"] and time.monotonic() - began < 5


def test_restart_keeps_accepted_experiments_and_queue_order(clock, tmp_path):
    executor = sim(clock)  # an external system: it survives the ES restart
    database = tmp_path / "es.sqlite3"
    with running_service([executor], pools={"default": 1}, clock=clock, database=database) as es:
        for i in range(5):
            put(es, f"q{i}", body(i))
        es.service.run_until_idle()
        before = {f"q{i}": get(es, f"q{i}") for i in range(5)}
    assert before["q0"]["state"] == "RUNNING" and all(before[f"q{i}"]["state"] == "QUEUED" for i in range(1, 5))
    with running_service([executor], pools={"default": 1}, clock=clock, database=database) as es:
        assert {eid: get(es, eid) for eid in before} == before
        assert put(es, "q3", body(3))[0] == 200
        es.service.run_simulation()
        finals = [get(es, f"q{i}") for i in range(5)]
    assert all(f["state"] == "SUCCEEDED" for f in finals)
    assert [eid for eid, _ in executor.launches] == [f"q{i}" for i in range(5)]
    assert [f["started_at"] for f in finals] == sorted(f["started_at"] for f in finals)


def test_kind_metadata_auth_and_errors(clock):
    with running_service([sim(clock, units=2)], pools={"default": 3}, clock=clock, token="secret") as es:
        assert request(es.base_url, "GET", "/v1/kinds/sim.objective")[0] == 401
        status, _, info = request(es.base_url, "GET", "/v1/kinds/sim.objective", token="secret")
        assert status == 200 and info["pool"] == {"name": "default", "capacity": 3}
        assert info["capabilities"]["units"] == 2 and info["kind_version"] == "1"
        assert request(es.base_url, "GET", "/v1/kinds/nope", token="secret")[0] == 404
        assert request(es.base_url, "GET", "/v1/experiments/zzz", token="secret")[0] == 404
        assert request(es.base_url, "PUT", "/v1/experiments/zzz:cancel", {}, token="secret")[0] == 405
        assert request(es.base_url, "GET", "/v2/anything", token="secret")[0] == 404
