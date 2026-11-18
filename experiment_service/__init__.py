"""Experiment Service (ES): a task-neutral, standalone experiment queue behind an HTTP contract.

Standard library only; never imports ``ldm_tts``. See ``docs/impl-notes/es-core.md``.
"""
from .clock import RealClock, VirtualClock
from .executors import (Artifact, AttemptContext, AttemptStatus, BlockingExecutor, DispatchUncertain,
                        ExecutorCapabilities, ExperimentExecutor, ExperimentFailure)
from .service import ExperimentService, Problem, canonical, digest
from .simulation import SimulatedExecutor
from .store import ServiceAlreadyRunning

__all__ = ["Artifact", "AttemptContext", "AttemptStatus", "BlockingExecutor", "DispatchUncertain",
           "ExecutorCapabilities", "ExperimentExecutor", "ExperimentFailure", "ExperimentService", "Problem",
           "RealClock", "ServiceAlreadyRunning", "SimulatedExecutor", "VirtualClock", "canonical", "digest"]
