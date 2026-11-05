"""Controller-owned public interface gate using the existing isolated executor."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.contracts.evaluation import EvaluationPaused
from ..core.execution import digest, freeze_model
from ..core.geometry import read_model
from ..core.runner import input_digest
from .data import save_problem
from .fixtures import fixtures, FIXTURE_VERSION
from .geometry import canonicalize
from .suite import file_hash


class InterfaceChecks:
    fixture_count = 6

    def __init__(self, root, execution, slots=None):
        self.root, self.execution = Path(root).resolve(), execution
        self.slots = slots
        self._input_lock = threading.Lock()
        self._inputs = {}

    def __call__(self, payload):
        try:
            return self._check(payload)
        except OSError as exc:
            raise EvaluationPaused("interface infrastructure I/O failed; no scientific rejection charged") from exc
        except ValueError:
            if self.slots is not None:
                self.slots.reserve(payload["sha256"], interface_rejected=True)
            raise

    def _check(self, payload):
        source = self.root / payload["source_path"]
        if source.is_symlink() or not source.resolve().is_relative_to(self.root):
            raise ValueError("invalid candidate path")
        if file_hash(source) != payload["sha256"]:
            raise ValueError("candidate changed before interface checks")
        problems = tuple(fixtures())
        # Prepare once before workers start. Multiple admissions share fixtures;
        # never truncate an NPZ file while another container is reading it.
        with self._input_lock:
            for problem in problems:
                fixture_digest = digest(dict(version=FIXTURE_VERSION, problem=input_digest(problem)))
                if fixture_digest not in self._inputs:
                    inputs = self.root / "interface-inputs" / fixture_digest
                    save_problem(problem, inputs)
                    atomic_json_write(inputs / "budget.json", dict(seed=47, wall_time_seconds=5, cpu_count=1, memory_mb=2048))
                    self._inputs[fixture_digest] = inputs
        # Queue limits govern actual containers. Consume results in fixture order
        # so parallel completion cannot change the reported rejection.
        with ThreadPoolExecutor(max_workers=self.fixture_count) as pool:
            futures = [pool.submit(self._check_fixture, payload, source, problem) for problem in problems]
            for future in futures:
                future.result()
        return True

    def _check_fixture(self, payload, source, problem):
        fixture_digest = digest(dict(version=FIXTURE_VERSION, problem=input_digest(problem)))
        inputs = self._inputs[fixture_digest]
        identity = dict(source=payload["sha256"], fixture=fixture_digest)
        job, receipt = self.execution.run(identity, phase="interface", inputs=inputs,
            source=source.parent, timeout=5, science_module="v2")
        if receipt["exit_code"] or receipt["reason"] or receipt["oom"]:
            raise ValueError(f"public interface {problem['case_id']} failed: {receipt['reason'] or receipt['exit_code']}")
        frozen = self.root / "interface-models" / digest(identity)
        try:
            read_model(job / "output/model.npy", problem["shape"])
        except (FileNotFoundError, PermissionError) as exc:
            raise ValueError("candidate did not produce a readable model") from exc
        freeze_model(job / "output/model.npy", frozen, problem["shape"])
        canonicalize(problem, read_model(frozen / "model.npy", problem["shape"]))
