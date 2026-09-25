"""Program admission, canonical identity, identity-free features, and coverage."""

import pytest

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from tasks.researchgym.core.candidate import FEATURE_NAMES, ProgramDomain, ProgramEncoder
from tasks.researchgym.core.cases import CASE_IDS, load_case
from tasks.researchgym.core.grading import CoverageError, score_summary


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_released_seed_programs_satisfy_their_slot(case_id):
    case = load_case(case_id)
    assert case.check_program(case.seed_program()) == []


def codes(case, source):
    return {error["code"] for error in case.check_program(source)}


def test_static_rules_reject_unsafe_or_malformed_programs():
    case = load_case("improving_replay_buffers")
    seed = case.seed_program()
    assert "missing_entry" in codes(case, "def other():\n    pass\n")
    assert "missing_method" in codes(case, "class ReplayBuffer:\n    def add(self):\n        pass\n")
    assert "forbidden_import" in codes(case, "import subprocess\n" + seed)
    assert "forbidden_import" in codes(case, "from rl_runner import SAC\n" + seed)
    assert "forbidden_call" in codes(case, seed + "\nx = eval('1')\n")
    assert "degenerate_repetition" in codes(case, seed + "\n" + "y = 1\n" * 30)
    assert "syntax_error" in codes(case, "class ReplayBuffer(:\n")
    tse = load_case("time_series_explanation")
    assert "entry_arguments" in codes(tse, "def attribute(classifier, loader):\n    return None\n")
    cmr = load_case("cross_modal_retrieval")
    assert "forbidden_import" in codes(cmr, "from models.tta_baselines.tent import Tent\n" + cmr.seed_program())


def test_canonical_identity_ignores_comments_formatting_and_docstrings():
    case = load_case("improving_replay_buffers")
    domain = ProgramDomain(case)
    seed = case.seed_program()
    variant = "# comment\n" + seed.replace("class ReplayBuffer:", 'class ReplayBuffer:\n    """doc"""', 1) + "\n\n"
    first, second = (domain.admit(RawProposal({"program": p}, "test")) for p in (seed, variant))
    assert isinstance(first, Candidate) and first.canonical_key == second.canonical_key
    assert isinstance(domain.admit(RawProposal({"program": seed, "extra": 1}, "test")), CandidateRejection)


def test_features_describe_program_content_not_identity():
    case = load_case("cross_modal_retrieval")
    encoder = ProgramEncoder(case)
    seed = case.seed_program()
    base = encoder.encode(Candidate("a", {"program": seed}, "a"))
    renamed = encoder.encode(Candidate("b", {"program": seed.replace("outputs", "results")}, "b"))
    assert len(base.values) == len(FEATURE_NAMES) == 12
    assert base.values == renamed.values  # same-length rename: no content hash or identity dimension
    assert all(0.0 <= value <= 1.0 for value in base.values)


def tse_summary(datasets, count=5):
    return {"real": {d: {b: {"mean": 0.9, "se": 0.0, "count": count} for b in ("Average", "Zeros")} for d in datasets}}


def test_tse_requires_all_ten_cells_with_five_folds():
    case = load_case("time_series_explanation")
    full = ["PAM", "Boiler", "Epilepsy", "Wafer", "Freezer"]
    metrics = score_summary(case, tse_summary(full))
    assert metrics["tse_real_mean_cpd"] == pytest.approx(0.9) and len([k for k in metrics if k.startswith("cell.")]) == 10
    # The review's reproduction: Epilepsy, Wafer and Freezer only (6 cells) must not score.
    with pytest.raises(CoverageError, match="missing PAM.Average"):
        score_summary(case, tse_summary(["Epilepsy", "Wafer", "Freezer"]))
    with pytest.raises(CoverageError, match="4 folds"):
        score_summary(case, tse_summary(full, count=4))


def test_cl_cmr_and_rl_coverage_identities():
    cl = load_case("continual_learning")
    row = {"dataset": "CIFAR100", "num_tasks": 10, "num_runs": 3, "seeds": [1992, 1993, 1994],
           "metrics": {"acc_mean": 80.5, "aaa_mean": 86.0, "acc_se": 0.1, "aaa_se": 0.1}}
    assert score_summary(cl, {"tables": [{"table": "cifar100", "datasets": [row]}]})["cl_cifar100_acc"] == 80.5
    with pytest.raises(CoverageError, match="runs"):
        score_summary(cl, {"tables": [{"table": "cifar100", "datasets": [{**row, "num_runs": 2, "seeds": [1992, 1993]}]}]})
    cmr = load_case("cross_modal_retrieval")
    columns = [{"id": c, "value": 60.0, "sources": [{"value": 60.0}]} for c in cmr.raw["coverage"]["columns"]]
    summary = {"rows": [{"row": "qs_image_blip_base", "average": 60.0, "columns": columns}]}
    assert score_summary(cmr, summary)["cmr_coco_c_i2t_r1"] == 60.0
    columns[3] = {"id": columns[3]["id"], "value": None}
    with pytest.raises(CoverageError, match="missing speckle_noise"):
        score_summary(cmr, {"rows": [{"row": "qs_image_blip_base", "average": 60.0, "columns": columns}]})
    rl = load_case("improving_replay_buffers")
    env = {"average_return": {"mean": 250.0, "std": 40.0, "count": 2},
           "samples": [{"seed": 0, "average_return": 290.0}, {"seed": 1, "average_return": 210.0}]}
    assert score_summary(rl, {"environments": {"Cheetah-Run": env}})["rl_cheetah_run_return"] == 250.0
    with pytest.raises(CoverageError):
        score_summary(rl, {"environments": {"Cheetah-Run": {**env, "samples": env["samples"][:1]}}})
