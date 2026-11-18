"""Asynchronous LDM controller components (see ``docs/impl-notes/controller-core.md``).

Talks to the experiment service only over HTTP; never imports ``experiment_service``. The pacing layer is
documented in ``docs/impl-notes/controller-pacing.md``; ``selectors`` (numpy) is imported on demand.
"""
from .clock import RealClock, VirtualClock
from .codec import ExperimentCodec, ExperimentRequest, MappedCodec, expected_status
from .config import CampaignConfig, CampaignConfigMismatch, RuntimeOptions
from .controller import Controller
from .es_client import ESBackpressure, ESClient, ESError, ESRejected, ESUnavailable, TransportError, UrllibTransport
from .events import ActionHook, CoreAdjudicator, EffectHook, Event, HookDispatcher, Verdict, WakeHook, task_hooks
from .faults import CRASH_POINTS, CrashPlan, FaultyTransport, SimulatedCrash
from .invariants import check_invariants
from .ledger import Budget, PendingItem, StateView
from .lock import RunLock, RunLockHeld
from .selection import (DedupOnlySelector, FixedProposer, GreedyPacing, Pacing, PacingDecision, ProposalSource,
                        Selection, Selector)
from .store import Store
from .lane import (DeterministicProposer, ExpanderSource, InlineExecutor, ResearchOutcome, ScriptedSource,
                   ThreadExecutor)
from .paced import PacedController, PacingAdjudicator
from .pacing import BarrierPacing, CampaignView, PacingConfig, StreamingPacing, campaign_extra
from .proposals import Proposal

__all__ = ["ActionHook", "Budget", "CRASH_POINTS", "CampaignConfig", "CampaignConfigMismatch", "Controller",
           "CoreAdjudicator", "CrashPlan", "DedupOnlySelector", "ESBackpressure", "ESClient", "ESError", "ESRejected",
           "ESUnavailable", "EffectHook", "Event", "ExperimentCodec", "ExperimentRequest", "FaultyTransport",
           "FixedProposer", "GreedyPacing", "HookDispatcher", "MappedCodec", "Pacing", "PacingDecision", "PendingItem",
           "ProposalSource", "RealClock", "RunLock", "RunLockHeld", "RuntimeOptions", "Selection", "Selector",
           "SimulatedCrash", "StateView", "Store", "TransportError", "UrllibTransport", "Verdict", "VirtualClock",
           "WakeHook", "check_invariants", "expected_status", "task_hooks",
           "BarrierPacing", "CampaignView", "DeterministicProposer", "ExpanderSource", "InlineExecutor", "PacedController",
           "PacingAdjudicator", "PacingConfig", "Proposal", "ResearchOutcome", "ScriptedSource", "StreamingPacing",
           "ThreadExecutor", "campaign_extra"]
