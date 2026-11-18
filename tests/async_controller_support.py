"""Shared harness for the controller tests: a real ES over HTTP, a toy codec and a drive loop.

The ES runs with ``autorun=False`` on a ``VirtualClock`` that the controller shares, so every run
is deterministic: the loop steps the controller until idle, lets the ES dispatch, and only then
moves virtual time to the next ES event or controller due time.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from experiment_service import SimulatedExecutor, VirtualClock
from experiment_service.testing import running_service
from ldm_tts.async_runtime import (CampaignConfig, Controller, ESClient, ExperimentRequest, FaultyTransport,
                                   FixedProposer, MappedCodec, RuntimeOptions, UrllibTransport, check_invariants)
from ldm_tts.contracts import Candidate

KIND = "toy.objective"
CAMPAIGN = "camp1"


class ToyCodec(MappedCodec):
    name = "toy"
    version = "1"

    def to_request(self, candidate):
        return ExperimentRequest(KIND, "1", {"x": candidate.payload}, {"max_runtime_seconds": 1000})

    def metrics(self, candidate, result):
        return {"objective": float(result["value"])}


MANIFEST_SEEN = []


def manifest_observer(event, view):
    MANIFEST_SEEN.append(event.seq)


class ManifestAction:
    def on_event(self, event, view):
        return [{"action": "request_pause", "reason": "manifest"}]


manifest_action = ManifestAction()


def toy_candidates(n, start=0):
    return [Candidate(f"cand-{i}", i, f"int:{i}") for i in range(start, start + n)]


class Harness:
    def __init__(self, tmp_path: Path, *, budget=6, candidates=10, concurrency=2, buffer_depth=0, pools=None,
                 sim=None, es_options=None, options=None, hooks=(), seed=0, objectives=None):
        self.tmp = Path(tmp_path)
        self.clock = VirtualClock()
        sim_options = {"duration": ("uniform", 5, 15), "seed": seed, "cancel_running": "cooperative",
                       "cancel_delay": 1.0}
        sim_options.update(sim or {})
        self.sim = SimulatedExecutor(self.clock, kind=KIND, **sim_options)
        self.pools = pools or {"default": concurrency}
        self.es_options = dict(es_options or {})
        self.config = CampaignConfig(CAMPAIGN, budget=budget, concurrency=concurrency, buffer_depth=buffer_depth)
        self.candidates = candidates if isinstance(candidates, list) else toy_candidates(candidates)
        self.options = options or RuntimeOptions(anti_entropy_interval=60.0, retry_base_seconds=1.0,
                                                 retry_max_seconds=8.0, unreachable_alert_seconds=30.0,
                                                 long_poll_seconds=0)
        self.hooks, self.objectives = list(hooks), objectives
        self.transport = FaultyTransport(UrllibTransport())
        self.db = self.tmp / "ldm" / "controller.sqlite3"
        self.es_db = self.tmp / "es" / "es.sqlite3"
        self.es = None
        self.ctrl = None
        self._es_stack = None
        self.restarts = 0
        self.on_step = None
        self.es_check = True  # cross-check against the ES (off when a test makes the ES misbehave)

    # ---- ES lifecycle ------------------------------------------------------------------------------

    def start_es(self):
        assert self.es is None
        self._es_stack = ExitStack()
        self.es = self._es_stack.enter_context(running_service([self.sim], pools=self.pools, clock=self.clock,
                                                               database=self.es_db, **self.es_options))
        if self.ctrl is not None:
            self.ctrl.client = self.client()
        return self.es

    def stop_es(self):
        self._es_stack.close()
        self.es, self._es_stack = None, None

    def client(self):
        url = self.es.base_url if self.es is not None else "http://127.0.0.1:9"
        return ESClient(url, timeout=5.0, transport=self.transport)

    # ---- controller lifecycle ----------------------------------------------------------------------

    def open(self, faults=None, **overrides):
        if self.ctrl is not None:
            self.ctrl.close()
        kwargs = dict(codec=ToyCodec(), client=self.client(), clock=self.clock,
                      proposer=FixedProposer(self.candidates), hooks=self.hooks, options=self.options,
                      faults=faults, objectives=self.objectives)
        config = overrides.pop("config", self.config)
        kwargs.update(overrides)
        self.ctrl = Controller(self.db, config, **kwargs)
        return self.ctrl

    def crash(self):
        """The controller process dies: drop it without any further work."""
        if self.ctrl is not None:
            self.ctrl.close()
            self.ctrl = None
        self.restarts += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.ctrl is not None:
            self.ctrl.close()
        if self.es is not None:
            self.stop_es()

    # ---- driving -----------------------------------------------------------------------------------

    def check(self, synced=False):
        items = self.es_items() if self.es is not None and self.es_check else None
        problems = check_invariants(self.ctrl.store, self.config, items, synced=synced)
        assert not problems, problems

    def settle(self, max_steps=10_000):
        """Step the controller until it reports no progress, checking invariants after each step."""
        progressed = False
        for _ in range(max_steps):
            if not self.ctrl.step():
                return progressed
            progressed = True
            self.check()
            if self.on_step is not None:
                self.on_step(self)
        raise AssertionError("controller did not settle")

    def drive(self, until=None, max_rounds=3_000, advance=True):
        """Run controller and ES to quiescence (or until ``until()``); time moves only when both idle."""
        for _ in range(max_rounds):
            if until is not None and until():
                return True
            progressed = self.settle()
            if self.es is not None and self.es.service.run_until_idle():
                continue
            if progressed:
                continue
            if self.ctrl.finished and (until is None):
                return True
            if not advance:
                return False
            times = [t for t in (self.es.service.next_event_time() if self.es is not None else None,
                                 self.ctrl.next_due()) if t is not None]
            if not times:
                return False
            target = min(times)
            assert target > self.clock.now(), f"controller is due at {target} but made no progress at {self.clock.now()}"
            self.clock.advance_to(target)
        raise AssertionError("drive did not finish")

    def es_items(self):
        page = self.es.service.changes(CAMPAIGN, "", limit=1000)
        assert not page["more"]
        return page["items"]

    def row(self, decision_id):
        return self.ctrl.store.row("SELECT * FROM experiments WHERE decision_id=?", (decision_id,))

    def mirror(self, decision_id):
        row = self.row(decision_id)
        return None if row is None else row["es_state"]

    def es_states(self):
        return {item["experiment_id"]: item["state"] for item in self.es_items()}
