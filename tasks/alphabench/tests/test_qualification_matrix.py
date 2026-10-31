"""The functional inventory must retain every required T3 combination."""

from copy import deepcopy
import json

from tasks.alphabench.qualify_matrix import DATA_EVIDENCE, OUTPUT, ROOT, TASK_EVIDENCE, build_matrix


def test_full_matrix_retains_blocked_markets_and_all_methods():
    data = json.loads((ROOT / DATA_EVIDENCE).read_text(encoding="utf-8"))
    task = json.loads((ROOT / TASK_EVIDENCE).read_text(encoding="utf-8"))
    matrix = build_matrix(data, task)
    cells = matrix["cells"]
    assert matrix["required"] == 72 and matrix["counts"] == {"passed": 0, "failed": 0, "blocked": 72}
    assert len({cell["id"] for cell in cells}) == 72 and matrix["complete_t3"] is False
    assert {cell["backend"] for cell in cells} == {"qlib", "assay"}
    assert all(cell["contract_profile"] is None and cell["run_id"] is None for cell in cells)
    assert all(cell["capabilities"]["data"] == "unverified" for cell in cells)
    assert matrix["comparison_data_policy"] == "partial_comparison"
    assert len(matrix["market_data_limitations"]) == 9
    assert all("unqualified_market_data" not in cell["blockers"] for cell in cells)
    assert all(cell["data_evidence"]["file"] == DATA_EVIDENCE for cell in cells)
    assert json.loads((ROOT / OUTPUT).read_text(encoding="utf-8")) == matrix

    qualified = deepcopy(data)
    for section in ("cn", "us"):
        for audit in qualified[section].values():
            audit["qualification"] = "qualified"
            audit["issues"] = []
    without_data_blocker = build_matrix(qualified, task)
    assert without_data_blocker["counts"] == {"passed": 0, "failed": 0, "blocked": 72}
    assert all("unqualified_market_data" not in cell["blockers"] and
               "frozen_backend_manifest_missing" in cell["blockers"] and
               cell["capabilities"]["data"] == "unverified" for cell in without_data_blocker["cells"])
    assert without_data_blocker["complete_t3"] is False
    strict = deepcopy(data)
    strict.pop("comparison_data_policy")
    assert all("unqualified_market_data" in cell["blockers"]
               for cell in build_matrix(strict, task)["cells"])
