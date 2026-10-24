"""Default task entry points must select the shared research architecture."""
from importlib import import_module
import pytest
from ldm_tts.harness.research_strategy import load_research_strategy

@pytest.mark.parametrize("task", ["nucleobench", "iron_mind", "synthonbench", "atomworld", "reasyn", "alphabench"])
def test_default_pi_research(task):
    if task == "alphabench":
        from tasks.alphabench.core.protocol import T3Protocol
        protocol = T3Protocol()
        method, strategy = protocol.method, load_research_strategy(protocol.research_strategy_config)
    else:
        # Real argument parsing without launching a provider or evaluator.
        args = import_module(f"tasks.{task}.core.workflow_args").parse_args(["--data-dir", "/unused-test-data"] if task == "atomworld" else [])
        method, strategy = args.search_method, args._research_strategy
    assert method == "ldm_harness_compiled"
    assert strategy.fork_policy.identity()["name"] == "main_history_1plus3"


@pytest.mark.parametrize("task", ["iron_mind", "synthonbench"])
def test_default_harness_mode_passes_full_validation(task, tmp_path):
    module = import_module(f"tasks.{task}.core.workflow_args")
    argv = ["--data-dir", str(tmp_path)]
    if task == "synthonbench":
        argv += ["--source-dir", str(tmp_path)]
    args = module.parse_args(argv)
    assert args.proposal_mode == "none"
    module.validate_args(args)
