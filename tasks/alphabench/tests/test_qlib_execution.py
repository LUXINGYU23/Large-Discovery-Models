"""Run in the pinned Qlib environment; these are synthetic market fixtures."""

import os
from pathlib import Path

import numpy as np
import pytest

qlib = pytest.importorskip("qlib")
import pandas as pd
from qlib.data.storage.file_storage import FileFeatureStorage

from tasks.alphabench.core.oracle_worker import load_source, qlib_evaluate
from tasks.alphabench.core.protocol import T3Protocol


def test_actual_qlib_nan_filter_and_paper_finite_time_boundaries(tmp_path):
    source = Path(os.environ.get("ALPHABENCH_SOURCE_ROOT", "/mnt/data1/Large-Discovery-Models/data/alphabench/AlphaBench"))
    if not (source / "ffo/utils/utils.py").exists():
        pytest.skip("pinned AlphaBench source checkout is required")
    days = pd.bdate_range("2016-01-01", periods=100)
    for directory in ("calendars", "instruments", "features/s00"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "calendars/day.txt").write_text("\n".join(days.strftime("%Y-%m-%d")) + "\n")
    (tmp_path / "instruments/csi300.txt").write_text(f"S00\t{days[0]:%Y-%m-%d}\t{days[-1]:%Y-%m-%d}\n")
    qlib.init(provider_uri=str(tmp_path), region="cn", expression_cache=None, dataset_cache=None, kernels=1)
    cases = [("close", np.nan, 1, True, True), ("open", np.nan, 2, False, False),
             ("high", np.inf, 1, True, True), ("low", np.inf, 2, True, False),
             ("volume", np.nan, 100, False, False)]
    arrays = {}
    for field, value, count, _, _ in cases:
        array = np.full(100, 20., dtype=np.float32)
        array[:count] = value
        FileFeatureStorage("S00", field, "day", provider_uri={"day": str(tmp_path)}).write(array, index=0)
        arrays[field] = array
    protocol = T3Protocol(filter_profile="qlib_code_filter_v1")
    request = {"protocol": protocol.to_dict(), "operation": "check", "start": str(days[0].date()),
               "end": str((days[-1] + pd.Timedelta(days=1)).date()), "fast": True}
    config = {"upstream_root": str(source), "data_root": str(tmp_path)}
    source_utils = load_source("t3_filter_reference", source / "ffo/utils/utils.py")
    for field, _, _, source_pass, paper_pass in cases:
        raw = qlib_evaluate(request | {"expression": "$" + field}, config)
        expected = source_utils._check_single_column("$" + field, pd.Series(arrays[field]))
        assert all(raw[key] == value for key, value in expected.items())
        assert raw["success"] is source_pass
        assert raw["nan_ratio"] == np.isnan(arrays[field]).mean()
        assert raw["non_finite_ratio"] == (~np.isfinite(arrays[field])).mean()
        assert protocol.check_passed(raw | {"elapsed_seconds": 30}) is source_pass
        assert protocol.check_passed(raw | {"elapsed_seconds": 30}, paper=True) is paper_pass
        assert protocol.check_passed(raw | {"elapsed_seconds": 30.001}, paper=True) is False
        assert raw["metrics"] == {} and raw["daily"] == [] and raw["scores"] == [] and raw["portfolio"] is None
