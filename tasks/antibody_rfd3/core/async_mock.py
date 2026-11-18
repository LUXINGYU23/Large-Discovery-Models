"""PoC task entry for asynchronous offline fixtures."""

import json

from ldm_tts.async_release import run_fixture
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.engine.run_store import atomic_json_write, unique_run_dir


TASK = "antibody_rfd3"
METRIC = "mock_energy"


def validate(args):
    if not args.mock or args.oracle != "mock" or args.method != "fixed" or args.initialization_run is not None:
        raise ValueError("async scheduling requires the offline fixed mock fixture without an initialization run")


def describe(args):
    validate(args)
    return LDMTaskSpec(
        TASK,
        CandidateDomainSpec("fixture_id", "integer", 1, "Opaque fixture index"),
        (ObjectiveSpec(METRIC, "minimize", "Deterministic software fixture score"),),
        (ResponseSpaceSpec("fixtures", "json"),),
        AcquisitionSpec("fixed", (METRIC,), "minimize", "Fixed fixture order"),
        ReservoirSpec(
            "offline_fixtures",
            (ReservoirExpansionSpec("load_fixtures", "emit_candidate", "fixtures", True),),
            "Integer fixture ID",
            "fixture_id",
        ),
        SurrogateSpaceSpec("none", "no surrogate for fixture scheduling", "none"),
        ProposalSearchSpec("fixed", parameters={"scheduler": "async", "concurrency": args.concurrency}),
    )


def run(args):
    task_spec = describe(args)
    directory = unique_run_dir(args.out_dir.resolve())
    directory.mkdir(parents=True)
    atomic_json_write(directory / "ldm_task_spec.json", task_spec.to_dict())
    atomic_json_write(
        directory / "config.json",
        {
            "task": TASK,
            "scheduler": "async",
            "method": "fixed",
            "mock": True,
            "oracle": "mock",
            "iterations": args.iterations,
            "concurrency": args.concurrency,
            "seed": args.seed,
        },
    )
    status = run_fixture(
        "run_async_fixture.py",
        directory,
        [
            "--run-dir", str(directory),
            "--task-id", TASK,
            "--metric", METRIC,
            "--budget", str(args.iterations),
            "--concurrency", str(args.concurrency),
            "--seed", str(args.seed),
        ],
    )
    print(json.dumps({"run_dir": str(directory), "scheduler": "async", "exit_code": status}))
    return status
