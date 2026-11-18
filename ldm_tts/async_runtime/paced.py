"""The paced controller (design 5.1 loop, 5.4-5.6, 5.9, 6.5-6.7): reservoir, research lane, rules D/R, barrier.

``PacedController`` keeps everything controller-core does (store, outbox, ingestion, anti-entropy, hooks)
and replaces the selection stage:

    change feed -> anti-entropy -> lane (collect / time out) -> drain/abort -> rule D (select) -> rule R
    (research) -> cancels -> PUTs -> stop rules (only in a step without other progress) -> hooks

Candidates come from the resident reservoir (``ReservoirKeeper``), selection from a reservoir selector
(``ReferenceSelector`` = Kriging Believer by default, or a controller-core ``Selector`` labelled dedup-only).
Streaming decisions select one candidate each; a barrier batch is one stepwise selection of ``concurrency``
candidates stored as a plan and turned into intents one at a time (each its own decision).
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from ldm_tts.optimization.records import SelectionSeed

from .codec import ExperimentRequest, plugin_identity
from .controller import Controller, decision_seed
from .events import TASK_ACTIONS, Verdict
from .lane import InlineExecutor, ResearchLane, ThreadExecutor, empty_rounds
from .pacing import CampaignView, LaneStatus, PacingConfig, breaker_reason, pacing_for, research_guards, runtime_median
from .proposals import Proposal, ReservoirKeeper, results_booked
from .selection import DedupOnlySelector, SelectionContext, as_reservoir_selector
from .store import append_event, get_flag, set_flag
from .wire import TERMINAL_SQL, canonical, decision_label

ENDED = "('ready','rejected','abandoned')"


class ReservoirProposer:
    """Identity stand-in for the core's proposer slot: candidates come from the reservoir."""

    name = "reservoir"
    version = "1"

    def propose(self, view):
        return view.reservoir.candidates


class _PacingListener:
    """Same-transaction bookkeeping: barrier plan consumption (intent) and the failure breaker (settlement)."""

    def __init__(self, controller):
        self.controller = controller

    def on_intent(self, db, candidate, decision_id, snapshot, now):
        plan = snapshot.get("plan")
        if plan is not None:
            db.execute("""UPDATE plans SET consumed=consumed+1,
                closed_reason=CASE WHEN consumed+1 >= count THEN 'complete' ELSE closed_reason END WHERE plan_id=?""",
                       (plan["id"],))

    def on_settled(self, db, experiment_id, state, now):
        config = self.controller.pacing_config
        rows = db.execute("""SELECT e.es_state, e.es_started_at FROM events v JOIN experiments e
            ON e.experiment_id = v.subject_id WHERE v.type='experiment.finished' AND e.es_state IN ('SUCCEEDED','FAILED')
            ORDER BY v.seq DESC LIMIT ?""", (max(config.breaker_window, config.breaker_unstarted_failures),)).fetchall()
        reason = breaker_reason(rows, config)
        if reason is not None:
            self.controller.ledger.anomaly(db, "failure_breaker", "", {"reason": reason}, pause=True)


class PacingAdjudicator:
    """Task action hooks (5.9) under the guards of rule R: ``request_research`` (lane idle and not blocked, not
    draining, research and experiment budget left), ``notify_agents`` (length and per-round quota; written to the
    durable agent-message queue, delivered by history tools), ``request_pause`` (pause, or ``mode='drain'``)."""

    def __init__(self, controller):
        self.controller = controller

    def adjudicate(self, db, ledger, hook, event, request):
        if request.action not in TASK_ACTIONS:
            return Verdict(False, "unknown_action")
        if request.action not in hook.actions:
            return Verdict(False, "undeclared_action")
        now, params = ledger.clock.now(), request.params
        reason_text = f"hook.{hook.name}:{params.get('reason', '')}"
        if request.action == "request_pause":
            if params.get("mode") == "drain":
                self.controller._set_draining(db, reason_text, "drain")
                return Verdict(True, "draining")
            ledger.pause(db, reason_text)
            return Verdict(True, "paused")
        view = self.controller.view()
        config = self.controller.pacing_config
        if request.action == "request_research":
            reason = research_guards(view) or ("research_pending" if view.research_requested else None)
            if reason:
                return Verdict(False, reason)
            note = str(params.get("note", ""))[:config.max_message_chars]
            set_flag(db, "research_requested", {"request_id": request.request_id, "hook": hook.name, "note": note, "at": now})
            append_event(db, now, "research.requested", request.request_id, {"hook": hook.name})
            return Verdict(True, "queued")
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            return Verdict(False, "invalid_message")
        if len(text) > config.max_message_chars:
            return Verdict(False, "message_too_long")
        lane = view.lane
        target = lane.generation if lane.state == "running" else \
            db.execute("SELECT COALESCE(MAX(generation),0)+1 FROM generations").fetchone()[0]
        count = db.execute("SELECT COUNT(*) FROM agent_messages WHERE target_generation=?", (target,)).fetchone()[0]
        if count >= config.max_messages_per_round:
            return Verdict(False, "message_quota")
        db.execute("""INSERT INTO agent_messages (request_id, source, target_generation, text, payload_json, created_at)
            VALUES (?,?,?,?,?,?)""", (request.request_id, f"hook:{hook.name}", target, text,
                                      canonical({k: v for k, v in params.items() if k != "text"}), now))
        append_event(db, now, "agent.message", request.request_id, {"hook": hook.name, "target_generation": target,
                                                                    "kind": "task_hint"})
        return Verdict(True, "queued")


class PacedController(Controller):
    def __init__(self, path, config, *, codec, client, research=None, lane_executor=None, selector=None,
                 pacing=None, hooks=(), adjudicator=None, policy=None, history=None, **kwargs):
        if "pacing" not in config.extra:
            raise ValueError("PacedController needs CampaignConfig.extra['pacing'] (see pacing.campaign_extra)")
        if "proposer" in kwargs:
            raise ValueError("a paced controller takes its candidates from the reservoir, not from a proposer")
        self.policy = policy  # optional ``policy.PolicyPrior``: policy turns, frozen prior applications (6.6)
        if policy is not None and research is not None:
            research = policy.wrap(research)
        self.pacing_config = PacingConfig.from_json(config.extra["pacing"])
        if self.pacing_config.research_budget is None and any(
                getattr(h, "kind", "") == "action" and "request_research" in getattr(h, "actions", ()) for h in hooks):
            raise ValueError("an action hook may request research only with a finite pacing research_budget (5.9)")
        self.maintenance = self.pacing_config.maintenance()
        self.research_source = research
        self.reservoir_selector = as_reservoir_selector(selector or DedupOnlySelector())
        if policy is not None and not hasattr(self.reservoir_selector, "training_rows"):
            raise ValueError("a policy prior needs the reference selector (ReferenceSelector)")
        self.history_bridge = history  # history-tools: its publish hook runs before the pacing hook
        super().__init__(path, config, codec=codec, client=client, proposer=ReservoirProposer(),
                         selector=self.reservoir_selector, pacing=pacing or pacing_for(self.pacing_config), hooks=hooks,
                         adjudicator=adjudicator or PacingAdjudicator(self),
                         builtin_hooks=() if history is None else (history.hook(),), **kwargs)
        self.keeper = ReservoirKeeper(self.maintenance, self.ledger)
        self.ledger.listeners += [self.keeper, _PacingListener(self)]
        if history is not None:
            history.attach(self)
        self.lane = None
        if research is not None:
            executor = lane_executor or ThreadExecutor(self.waker.notify)
            self.lane = ResearchLane(store=self.store, ledger=self.ledger, keeper=self.keeper, source=research,
                                     executor=executor, config=self.pacing_config,
                                     directory=self.path.with_name(self.path.name + ".research"), clock=self.clock,
                                     extension=history)
        self._history: tuple[int, tuple] = (-1, ())
        if policy is not None:
            policy.bind(self)

    def frozen_config(self) -> dict:
        frozen = super().frozen_config()
        frozen["plugins"].update({
            "reservoir": self.maintenance.describe(),
            "research": None if self.research_source is None else plugin_identity(self.research_source, "research"),
            "selector_config": self.reservoir_selector.describe()})
        if self.policy is not None:
            frozen["plugins"]["policy"] = self.policy.describe()
        if self.history_bridge is not None:
            frozen["plugins"]["history"] = self.history_bridge.identity()
        return frozen

    # ---- the view ---------------------------------------------------------------------------------------

    def view(self) -> CampaignView:
        state = self.ledger.view()
        config = self.pacing_config
        with self.store.lock:
            db = self.store.db
            reservoir = self.keeper.view(db, state.dedup_keys)
            lane = self.lane.status(db) if self.lane is not None else LaneStatus()
            ended = db.execute(f"""SELECT final_watermark, decision_head FROM generations WHERE state IN {ENDED}
                ORDER BY generation DESC LIMIT 1""").fetchone()
            ready = db.execute("""SELECT generation FROM generations WHERE state='ready'
                ORDER BY generation DESC LIMIT 1""").fetchone()
            decisions = db.execute("SELECT COALESCE(MAX(decision_id),0) FROM decisions").fetchone()[0]
            runtimes = db.execute("""SELECT es_started_at, es_finished_at FROM experiments
                WHERE es_state IN ('SUCCEEDED','FAILED') AND es_started_at IS NOT NULL AND es_finished_at IS NOT NULL
                ORDER BY es_finished_at DESC LIMIT ?""", (config.runtime_window,)).fetchall()
            running = db.execute("SELECT COUNT(*) FROM experiments WHERE pending=1 AND es_state='RUNNING'").fetchone()[0]
            plan = db.execute("SELECT * FROM plans WHERE closed_reason IS NULL ORDER BY plan_id DESC LIMIT 1").fetchone()
            batch = db.execute("SELECT generation FROM plans ORDER BY plan_id DESC LIMIT 1").fetchone()
            return CampaignView(
                state=state, reservoir=reservoir, stale=self.keeper.stale(reservoir), lane=lane,
                delta=state.head_ingest_seq - (ended["final_watermark"] if ended is not None else 0),
                selected_since_ready=decisions - (ended["decision_head"] if ended is not None else 0),
                runtime_median=runtime_median(runtimes, config.runtime_window), observed_running=running,
                empty_rounds=empty_rounds(db),
                research_used=db.execute("SELECT COUNT(*) FROM generations").fetchone()[0],
                research_requested=get_flag(db, "research_requested"), draining=get_flag(db, "draining"),
                current_generation=ready["generation"] if ready is not None else 0,
                batch_generation=None if batch is None else batch["generation"],
                open_plan=None if plan is None else dict(plan), pacing=config, has_research=self.lane is not None)

    def history(self, head: int) -> tuple:
        if self._history[0] != head:
            self._history = (head, tuple(self.ledger.observations(0, head)))
        return self._history[1]

    # ---- the loop ---------------------------------------------------------------------------------------

    def step(self) -> bool:
        self._exhausted = False
        progressed = False
        # Collect-only mode never touches the research lane: recovering or polling it would run research (6).
        lane = self.lane if not self.collect_only else None
        if self._startup_pending:
            self._startup()
            self._initial_reservoir_flags()
            if lane is not None:
                lane.recover()
        progressed |= self.reconciler.pull_changes()
        progressed |= self.reconciler.anti_entropy(force=self._startup_pending)
        self._startup_pending = False
        if lane is not None:
            progressed |= lane.poll()
        if self.policy is not None:
            progressed |= self.policy.step()  # supplementary / lifetime re-applications
        progressed |= self._abort_step()
        progressed |= self._select()
        progressed |= self._research()
        progressed |= self.reconciler.send_cancels()
        progressed |= self.reconciler.send_puts()
        if not progressed:
            progressed |= self._check_stop()
        progressed |= self.dispatcher.dispatch()
        return progressed

    def _initial_reservoir_flags(self) -> None:
        """First start: ``reservoir.stale/exhausted`` start from the actual (empty) reservoir, so their initial edge
        is announced instead of being lost behind the ``False`` default (review item 7)."""
        if self.collect_only or self.ledger.flag("reservoir_exhausted") is not None:
            return
        with self.store.tx("proposals") as db:
            if get_flag(db, "reservoir_exhausted") is None:
                self.keeper.refresh(db, self.clock.now())

    def _abort_step(self) -> bool:
        draining = self.ledger.flag("draining")
        if draining is None or draining.get("mode") != "abort":
            return False
        # A 409-rejected intent names an id the ES holds for another experiment: cancelling it is refused (it would
        # cancel someone else's work); it already paused the campaign and waits for an operator write-off.
        rows = self.store.rows("""SELECT experiment_id FROM experiments WHERE pending=1 AND cancel_requested_at IS NULL
            AND NOT (submit_state='REJECTED' AND COALESCE(reject_status, 0)=409) ORDER BY decision_id""")
        return any([self.ledger.request_cancel(r["experiment_id"], f"abort:{draining['reason']}") for r in rows])

    def _select(self) -> bool:
        made = 0
        for _ in range(self.options.max_intents_per_step):
            if self.ledger.dispatch_blockers():
                break
            view = self.view()
            if view.open_plan is not None:
                if not self._consume_plan(view, view.open_plan):
                    break
                made += 1
                continue
            decision = self.pacing.decide(view)
            if decision.action == "stop":
                return self.ledger.stop(decision.reason or "pacing") or made > 0
            if decision.action != "dispatch" or not view.reservoir.candidates:
                break
            candidates, uncovered = view.reservoir.candidates, ()
            if self.policy is not None:  # candidates the active application does not cover wait for a supplement
                candidates, uncovered = self.policy.covered(candidates)
                if not candidates:
                    break
            barrier = self.pacing_config.mode == "barrier"
            count = min(self.config.concurrency, view.budget.free, view.eligible) if barrier else 1
            decision_id = self.ledger.next_decision_id()
            plan_id = self.store.value("SELECT COALESCE(MAX(plan_id),0)+1 FROM plans")
            label = f"plan{plan_id:06d}" if barrier else decision_label(decision_id)
            head = view.head_ingest_seq
            context = SelectionContext(view, tuple(candidates), view.reservoir.q0_by_key(), view.pending,
                                       SelectionSeed(self.config.seed, label), count, self.history(head), head,
                                       decision_seed(self.config.seed, decision_id))
            try:
                choice = self.reservoir_selector.choose(context)
            except Exception as exc:  # noqa: BLE001 - a selector defect pauses dispatch, it does not crash the loop
                with self.store.tx("selection") as db:
                    self.ledger.anomaly(db, "selector_failed", label, {"error": f"{type(exc).__name__}: {exc}"},
                                        pause=True)
                break
            if choice is None:
                self._exhausted = True
                break
            eligible = {c.canonical_key: c for c in candidates}
            if len(set(choice.keys)) != len(choice.keys) or any(k not in eligible for k in choice.keys):
                with self.store.tx("selection") as db:
                    self.ledger.anomaly(db, "selector_violation", label,
                                        {"selector": plugin_identity(self.reservoir_selector, "selector"),
                                         "keys": list(choice.keys)}, pause=True)
                break
            candidates = [eligible[k] for k in choice.keys]
            if barrier:
                if not self._write_plan(plan_id, view, candidates, choice):
                    break
                continue
            snapshot = {**self._snapshot(view, len(eligible)), "probability": choice.probabilities[0],
                        "record": dict(choice.record), "inputs": dict(choice.inputs),
                        "q0_selected": view.reservoir.q0_by_key()[choice.keys[0]]}
            if uncovered:
                snapshot["prior_uncovered"] = list(uncovered)
            if not self._intent(decision_id, candidates[0], snapshot):
                break
            made += 1
        return made > 0

    def _snapshot(self, view: CampaignView, eligible: int) -> dict:
        return {"proposer": "reservoir@1", "selector": plugin_identity(self.reservoir_selector, "selector"),
                "pacing": plugin_identity(self.pacing, "pacing"), "candidate_count": eligible,
                "budget": view.budget.to_dict(), "outstanding": view.outstanding, "unstarted": view.unstarted,
                "pending_keys": sorted(view.pending_keys), "head_ingest_seq": view.head_ingest_seq,
                "pacing_state": view.summary()}

    def _intent(self, decision_id: int, candidate, snapshot: dict) -> bool:
        try:
            request = self.codec.encode(candidate)
            if not isinstance(request, ExperimentRequest):
                raise TypeError(f"codec returned {type(request).__name__}, not ExperimentRequest")
            canonical(candidate.to_dict())
        except Exception as exc:  # noqa: BLE001
            with self.store.tx("selection") as db:
                self.ledger.anomaly(db, "encode_failed", candidate.canonical_key,
                                    {"error": f"{type(exc).__name__}: {exc}"}, pause=True)
            return False
        seed = decision_seed(self.config.seed, decision_id)
        return self.ledger.plan_intent(decision_id, candidate, request, seed, snapshot) is not None

    def _write_plan(self, plan_id: int, view: CampaignView, candidates, choice) -> bool:
        now = self.clock.now()
        with self.store.tx("plan") as db:
            if db.execute("SELECT COALESCE(MAX(plan_id),0)+1 FROM plans").fetchone()[0] != plan_id or \
                    db.execute("SELECT 1 FROM plans WHERE closed_reason IS NULL").fetchone():
                return False
            db.execute("""INSERT INTO plans (plan_id, generation, count, keys_json, probabilities_json, record_json,
                inputs_json, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                       (plan_id, view.current_generation, len(candidates), canonical([c.to_dict() for c in candidates]),
                        canonical(list(choice.probabilities)), canonical(dict(choice.record)),
                        canonical({**dict(choice.inputs), "q0_selected": [view.reservoir.q0_by_key()[k]
                                                                         for k in choice.keys]}), now))
            append_event(db, now, "selection.planned", str(plan_id),
                         {"count": len(candidates), "generation": view.current_generation})
        return True

    def _consume_plan(self, view: CampaignView, plan) -> bool:
        from ldm_tts.contracts import Candidate
        step = plan["consumed"]
        candidate = Candidate(**json.loads(plan["keys_json"])[step])
        # Draining (drain or abort) ends the batch: its remaining steps never become intents (review item 1).
        problem = "draining" if view.draining is not None else \
            "duplicate_key" if candidate.canonical_key in view.dedup_keys else "budget" if view.budget.free < 1 else None
        if problem:
            with self.store.tx("plan") as db:
                db.execute("UPDATE plans SET closed_reason=? WHERE plan_id=? AND closed_reason IS NULL",
                           (f"abandoned:{problem}", plan["plan_id"]))
            return True
        snapshot = {**self._snapshot(view, view.eligible), "plan": {"id": plan["plan_id"], "step": step},
                    "probability": json.loads(plan["probabilities_json"])[step]}
        return self._intent(self.ledger.next_decision_id(), candidate, snapshot)

    def _research(self) -> bool:
        if self.lane is None:
            return False
        view = self.view()
        request = view.research_requested
        if request is not None and research_guards(view) is None:
            return self.lane.start(f"hook:{request['request_id']}", request.get("note", "")) is not None
        if getattr(self.pacing, "research", None) is not None and self.pacing.research(view).action == "research":
            return self.lane.start("rule_r") is not None
        return False

    def _check_stop(self) -> bool:
        """Runs only in a step without other progress, so a decision about exhaustion never rests on state that
        the same step's cancels or PUT answers already changed (controller-core review item 1)."""
        if self.collect_only or self.ledger.campaign_row()["status"] != "active" or self.ledger.paused_reasons():
            return False
        view = self.view()
        if view.outstanding > 0 or view.lane.state == "running":
            return False  # a research round cannot be cancelled: wait for it (5.5 rule 4)
        if view.draining is not None:
            return self.ledger.stop(view.draining["reason"])
        if view.budget.free < 1:
            return self.ledger.stop("budget_exhausted")
        if self._exhausted:
            return self.ledger.stop("proposals_exhausted")
        if self.pacing.decide(view).reason not in ("reservoir_exhausted", "batch_done"):
            return False
        if self.lane is None:
            return self.ledger.stop("proposals_exhausted")
        research = self.pacing.research(view) if hasattr(self.pacing, "research") else None
        if research is not None and research.action == "wait" and research.reason == "research_budget":
            return self.ledger.stop("research_budget_exhausted")
        return False

    def next_due(self) -> float | None:
        times = [t for t in (super().next_due(), self.lane.next_due() if self.lane is not None else None) if t is not None]
        held, until = self.view().hold()
        if held and until is not None:
            times.append(until)
        return min(times) if times else None

    # ---- operator and agent-facing API --------------------------------------------------------------

    def _set_draining(self, db, reason: str, mode: str) -> bool:
        if get_flag(db, "draining") is not None:
            return False
        now = self.clock.now()
        set_flag(db, "draining", {"reason": reason, "mode": mode, "at": now})
        append_event(db, now, "campaign.draining", None, {"reason": reason, "mode": mode})
        return True

    def drain(self, reason: str = "operator_drain") -> bool:
        """No new dispatch or research; stop with ``reason`` once nothing is outstanding (5.6)."""
        with self.store.tx("drain") as db:
            changed = self._set_draining(db, reason, "drain")
        self.waker.notify()
        return changed

    def abort(self, reason: str = "aborted") -> bool:
        """Cancel every outstanding experiment, then drain (the end time depends on the ES)."""
        with self.store.tx("drain") as db:
            changed = self._set_draining(db, reason, "abort")
        self.waker.notify()
        return changed

    def add_proposals(self, proposals: Sequence[Proposal], *, generation: int | None = None,
                      watermark: int | None = None) -> dict:
        """Merge proposals outside a research round (seeding, history-tools re-proposals). Same lineage + key renews."""
        now = self.clock.now()
        with self.store.tx("proposals") as db:
            booked = results_booked(db)
            current = db.execute("SELECT COALESCE(MAX(generation),0) FROM occurrences").fetchone()[0]
            merged = self.keeper.merge(db, proposals, generation=current if generation is None else generation,
                                       base_watermark=booked if watermark is None else watermark, now=now)
            append_event(db, now, "reservoir.merged", None, merged)
        self.waker.notify()
        return merged

    def renew(self, lineage: str, keys: Iterable[str], watermark: int | None = None) -> dict:
        """The lineage re-proposes candidates still in the reservoir: weight back to 1 at the given watermark."""
        keys = set(keys)
        with self.store.lock:
            live = self.keeper.live(self.store.db)
        found = {o.key: o.candidate for _, o in live if o.key in keys}
        return self.add_proposals([Proposal(found[k], lineage, watermark) for k in sorted(found)])

    def withdraw(self, lineage: str, keys: Iterable[str]) -> int:
        """The lineage withdraws its own proposals (the agent-facing entry belongs to history-tools)."""
        with self.store.tx("proposals") as db:
            removed = self.keeper.withdraw(db, lineage, keys, self.clock.now())
        self.waker.notify()
        return removed

    def resolve_lane(self, action: str, operator: str) -> bool:
        if self.lane is None:
            return False
        changed = self.lane.resolve(action, operator)
        self.waker.notify()
        return changed

    def agent_messages(self, after_id: int = 0) -> list[dict]:
        return [{**dict(r), "payload": json.loads(r["payload_json"])}
                for r in self.store.rows("SELECT * FROM agent_messages WHERE id > ? ORDER BY id", (after_id,))]

    def generations(self) -> list[dict]:
        return self.lane.generations() if self.lane is not None else []

    def decisions(self) -> list[dict]:
        return [{"decision_id": r["decision_id"], "experiment_id": r["experiment_id"], "seed": r["seed"],
                 "snapshot": json.loads(r["snapshot_json"])}
                for r in self.store.rows("SELECT * FROM decisions ORDER BY decision_id")]

    def plans(self) -> list[dict]:
        return [dict(r) for r in self.store.rows("SELECT * FROM plans ORDER BY plan_id")]

    def audit(self, decision_id: int) -> list[str]:
        from .selectors import audit_decision
        return audit_decision(self.store, self.ledger, decision_id, self.reservoir_selector,
                              None if self.policy is None else self.policy.application)


__all__ = ["InlineExecutor", "PacedController", "PacingAdjudicator", "ReservoirProposer", "TERMINAL_SQL"]
