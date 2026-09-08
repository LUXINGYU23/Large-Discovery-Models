from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ldm_tts.optimization.records import BOPrediction
from ldm_tts.registration.dependencies import (
    DependencyCheck,
    arg_value,
    bool_arg,
    check_cuda_visibility,
    check_llm_settings,
    checks_to_json,
    cli_args_to_map,
    first_env,
    format_checks,
    is_local_url,
    load_yaml_object,
    mask_secret,
    parse_device_ids,
    query_visible_gpu_ids,
    fail,
    ok,
)
from tasks.antibody.core.dependencies import check_antibody
from tasks.nanogpt.core.dependencies import check_nanogpt
from tasks.small_molecule.core.dependencies import resolve_reasyn_python
from ldm_tts.engine.runtime import LDMSearchRoundResult, run_budgeted_search
from tasks.nanogpt.core.expansion_schema import (
    OperationParameter,
    OperationSchema,
    choice_values_equal,
    initial_operation_feature_names,
    load_operation_schema,
    normalize_operation_numeric,
    normalize_operation_parameter,
    operation_feature_version,
    operation_parameter_from_payload,
    operation_parameter_to_json,
    operation_schema_signature,
    operation_schema_to_json,
    replace_operation_schema,
    validate_operation_payload,
    validate_operation_value,
)
from ldm_tts.transport.parsing import (
    extract_json_object_text,
    load_json_object,
    reject_keys,
    require_allowed_keys,
    require_list,
    require_nonnegative_int,
    require_number,
    require_str,
    strip_json_fence,
)
from ldm_tts.cli.runner import (
    apply_override,
    args_to_cli,
    build_plan,
    expand_env_vars,
    expand_experiments,
    expand_string,
    expand_value,
    extra_args_to_cli,
    list_configs,
    load_config,
    main,
    make_context,
    parse_args,
    parse_override_value,
    patched_env,
    plan_for_json,
    preflight_plan,
    pushd,
    resolve_child_config_path,
    resolve_config_path,
    resolve_repo_path,
    resolve_repo_relative_reference,
    run_plan,
    validate_config_keys,
)
from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    ProposalSearchSpec,
    SurrogateSpaceSpec,
)
from ldm_tts.engine.run_store import CandidateTraceRecord, LDMRoundTrace
from ldm_tts.engine.run_store import AtomicJsonLog, JsonlTrajectoryRecorder, load_jsonl, utc_timestamp


def _schema() -> OperationSchema:
    return OperationSchema(
        version="v1",
        description="test schema",
        parameters={
            "COUNT": OperationParameter("COUNT", "int", 1, 5),
            "RATE": OperationParameter("RATE", "float", 0.01, 1.0, scale="log"),
            "MODE": OperationParameter("MODE", "choice", choices=("fast", 2, True, 1.5)),
        },
    )


class TestRunnerHelpers:
    def test_parse_args_defaults_and_options(self) -> None:
        args = parse_args([
            "config.json",
            "--dry-run",
            "--keep-going",
            "--skip-preflight",
            "--set",
            "args.x=2",
        ])
        assert args.config == "config.json"
        assert args.dry_run and args.keep_going and args.skip_preflight
        assert args.set == ["args.x=2"]

    def test_load_config_supports_json_and_yaml_and_rejects_non_objects(self, tmp_path: Path) -> None:
        json_path = tmp_path / "config.json"
        json_path.write_text('{"task": "nanogpt"}', encoding="utf-8")
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("task: antibody\n", encoding="utf-8")
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("[]", encoding="utf-8")

        assert load_config(json_path) == {"task": "nanogpt"}
        assert load_config(yaml_path) == {"task": "antibody"}
        with pytest.raises(SystemExit, match="must be an object"):
            load_config(bad_path)

    def test_resolve_config_paths(self, tmp_path: Path) -> None:
        child = tmp_path / "child.json"
        child.write_text("{}", encoding="utf-8")
        assert resolve_config_path(str(child)) == child.resolve()
        assert resolve_child_config_path("child.json", tmp_path) == child.resolve()
        with pytest.raises(SystemExit, match="Config not found"):
            resolve_config_path(str(tmp_path / "missing.json"))
        with pytest.raises(SystemExit, match="Suite child config not found"):
            resolve_child_config_path("missing.json", tmp_path)

    def test_expand_experiments_supports_every_entry_shape(self, tmp_path: Path) -> None:
        suite = tmp_path / "suite.json"
        child = tmp_path / "child.json"
        child.write_text('{"task":"nanogpt","args":{"budget":1}}', encoding="utf-8")
        config = {
            "experiments": [
                "child.json",
                {"config": "child.json", "set": ["args.budget=3"]},
                {"task": "antibody"},
            ]
        }

        expanded = expand_experiments(config, suite)

        assert [item[0]["task"] for item in expanded] == ["nanogpt", "nanogpt", "antibody"]
        assert expanded[1][0]["args"]["budget"] == 3
        assert expand_experiments({"task": "nanogpt"}, suite) == [({"task": "nanogpt"}, suite)]

    @pytest.mark.parametrize("entries", [[], "child.json", [7]])
    def test_expand_experiments_rejects_invalid_suites(self, tmp_path: Path, entries: object) -> None:
        with pytest.raises(SystemExit, match="Suite configs|Unsupported suite"):
            expand_experiments({"experiments": entries}, tmp_path / "suite.json")

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({}, "Missing required field"),
            ({"task": "missing"}, "Unknown task"),
            ({"task": "nanogpt", "runner": ["invalid"]}, "runner must be"),
            ({"task": "nanogpt", "env": ["invalid"]}, "env must be"),
            ({"task": "nanogpt", "unexpected": 1}, "Unknown top-level"),
        ],
    )
    def test_build_plan_rejects_invalid_configs(self, config: dict[str, object], message: str) -> None:
        with pytest.raises(SystemExit, match=message):
            build_plan(config, Path("config/test.json"))

    def test_build_plan_expands_runner_env_lists_and_extra_args(self) -> None:
        config = {
            "name": "run-1",
            "task": "nanogpt",
            "algorithm": "beam",
            "mode": "mock",
            "runner": {"cwd": "{task_dir}", "module": "custom.{task}"},
            "env": {"FIRST": "one", "SECOND": "$FIRST/two"},
            "args": {
                "enabled": True,
                "disabled": False,
                "allow_early_stop": False,
                "items": ["a", "config/file.json"],
                "payload": {"b": 2, "a": 1},
                "nothing": None,
            },
            "extra_args": ["--literal", "{name}", 7],
        }

        plan = build_plan(config, Path("config/test.json"))

        assert plan["module"] == "custom.nanogpt"
        assert plan["env_overrides"]["FIRST"] == "one"
        assert plan["env_overrides"]["SECOND"] == "one/two"
        assert plan["env_overrides"]["LDM_EXPERIMENT_CONTRACT_PATH"].endswith(
            "tasks/nanogpt/experiment.json"
        )
        assert "--enabled" in plan["argv"]
        assert "--disabled" not in plan["argv"]
        assert "--no-allow-early-stop" in plan["argv"]
        assert plan["argv"].count("--items") == 2
        assert '{"a":1,"b":2}' in plan["argv"]
        assert plan["argv"][-3:] == ["--literal", "run-1", "7"]
        assert plan_for_json(plan)["command_display"] == plan["command_display"]

    def test_context_and_expansion_helpers(self) -> None:
        context = make_context({"name": "named"}, Path("config/x.yaml"), "nanogpt")
        assert context["config_name"] == "x"
        assert resolve_repo_path("nanogpt", context).name == "nanogpt"
        assert expand_env_vars("$A/${B}/$C", {"A": "x", "B": "y"}) == "x/y/$C"
        assert expand_string("{name}/{unknown}/$A", context, {"A": "z"}) == "named/{unknown}/z"
        assert expand_value([2, 1], context) == "[2,1]"
        assert expand_value({"b": 2, "a": 1}, context) == '{"a":1,"b":2}'
        assert resolve_repo_relative_reference("plain.txt") == "plain.txt"
        assert resolve_repo_relative_reference("config/x.yaml").endswith("config/x.yaml")
        assert resolve_repo_relative_reference("/absolute/x") == "/absolute/x"

    def test_plan_output_redacts_secrets(self) -> None:
        config = {
            "name": "secret-check",
            "task": "nanogpt",
            "mode": "mock",
            "env": {"LLM_API_KEY": "env-secret", "PUBLIC_SETTING": "visible"},
            "args": {
                "api-key": "cli-secret",
                "api-key-file": "/private/cli-key-file",
                "model": "visible-model",
            },
            "extra_args": ["--access-token=extra-secret"],
        }

        plan = build_plan(config, Path("config/secret-check.yaml"))
        public_plan = plan_for_json(plan)

        assert "cli-secret" in plan["argv"]
        assert "/private/cli-key-file" in plan["argv"]
        assert "cli-secret" not in plan["command_display"]
        assert "/private/cli-key-file" not in plan["command_display"]
        assert "extra-secret" not in plan["command_display"]
        assert public_plan["env_overrides"]["LLM_API_KEY"] == "***"
        assert public_plan["env_overrides"]["PUBLIC_SETTING"] == "visible"
        assert public_plan["env_overrides"]["LDM_EXPERIMENT_CONTRACT_PATH"].endswith(
            "tasks/nanogpt/experiment.json"
        )
        assert "cli-secret" not in json.dumps(public_plan)
        assert "/private/cli-key-file" not in json.dumps(public_plan)
        assert "extra-secret" not in json.dumps(public_plan)

    def test_cli_conversion_validation(self) -> None:
        context = {"name": "x"}
        assert args_to_cli(None, context) == []
        with pytest.raises(SystemExit, match="args must be"):
            args_to_cli([], context)
        assert extra_args_to_cli(None, context) == []
        with pytest.raises(SystemExit, match="extra_args must be"):
            extra_args_to_cli({}, context)

    def test_overrides_create_nested_objects_and_parse_values(self) -> None:
        config: dict[str, object] = {}
        apply_override(config, "args.budget=4")
        apply_override(config, "mode=mock")
        assert config == {"args": {"budget": 4}, "mode": "mock"}
        assert parse_override_value("true") is True
        assert parse_override_value("word") == "word"
        with pytest.raises(SystemExit, match="must look like"):
            apply_override(config, "bad")
        with pytest.raises(SystemExit, match="path is empty"):
            apply_override(config, "=1")
        with pytest.raises(SystemExit, match="is not an object"):
            apply_override({"args": 1}, "args.x=2")

    def test_context_managers_restore_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        original_cwd = Path.cwd()
        monkeypatch.setenv("KEPT_TEST_ENV", "old")
        monkeypatch.delenv("NEW_TEST_ENV", raising=False)
        with pushd(tmp_path), patched_env({"KEPT_TEST_ENV": "new", "NEW_TEST_ENV": "value"}):
            assert Path.cwd() == tmp_path
            assert os.environ["KEPT_TEST_ENV"] == "new"
            assert os.environ["NEW_TEST_ENV"] == "value"
        assert Path.cwd() == original_cwd
        assert os.environ["KEPT_TEST_ENV"] == "old"
        assert "NEW_TEST_ENV" not in os.environ

    def test_run_plan_calls_module_main_and_restores_process_state(self, tmp_path: Path) -> None:
        fake_main = Mock(return_value=7)
        fake_module = SimpleNamespace(main=fake_main)
        plan = {"cwd": str(tmp_path), "env_overrides": {"PLAN_ENV": "yes"}, "module": "fake", "argv": ["--x"]}
        with patch("ldm_tts.cli.runner.importlib.import_module", return_value=fake_module):
            assert run_plan(plan) == 7
        fake_main.assert_called_once_with(["--x"])
        assert "PLAN_ENV" not in os.environ

    def test_run_plan_accepts_none_and_rejects_missing_main(self, tmp_path: Path) -> None:
        plan = {"cwd": str(tmp_path), "env_overrides": {}, "module": "fake", "argv": []}
        with patch("ldm_tts.cli.runner.importlib.import_module", return_value=SimpleNamespace(main=lambda _argv: None)):
            assert run_plan(plan) == 0
        with patch("ldm_tts.cli.runner.importlib.import_module", return_value=SimpleNamespace()):
            with pytest.raises(SystemExit, match="has no main"):
                run_plan(plan)

    def test_preflight_skips_mock_plans(self) -> None:
        with patch("ldm_tts.cli.runner.check_plan") as check_plan_mock:
            assert preflight_plan({"task": "nanogpt", "mode": "mock"}) == []
        check_plan_mock.assert_not_called()

    def test_preflight_prints_successful_checks(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [ok("nanogpt", "training data", "Ready.")]
        with patch("ldm_tts.cli.runner.check_plan", return_value=checks):
            assert preflight_plan({"name": "real", "task": "nanogpt", "mode": "real"}) == checks
        assert "[OK] nanogpt: training data: Ready." in capsys.readouterr().out

    def test_preflight_blocks_failed_checks(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [fail("small_molecule", "G12D activity model", "Missing.")]
        with (
            patch("ldm_tts.cli.runner.check_plan", return_value=checks),
            pytest.raises(SystemExit, match="Dependency preflight failed"),
        ):
            preflight_plan({"name": "real", "task": "small_molecule", "mode": "real"})
        assert "[FAIL] small_molecule: G12D activity model: Missing." in capsys.readouterr().out

    def test_list_configs_skips_invalid_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "good.json").write_text('{"task":"nanogpt","mode":"mock"}', encoding="utf-8")
        (config_dir / "bad.json").write_text("[]", encoding="utf-8")
        (config_dir / "ignore.txt").write_text("x", encoding="utf-8")
        with patch("ldm_tts.cli.runner.REPO_ROOT", tmp_path):
            list_configs()
        rows = json.loads(capsys.readouterr().out)
        assert rows == [{"algorithm": "", "description": "", "mode": "mock", "path": "config/good.json", "task": "nanogpt"}]

    def test_main_list_missing_config_dry_run_and_execution_paths(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("ldm_tts.cli.runner.list_configs") as listed:
            assert main(["--list"]) == 0
            listed.assert_called_once()
        with pytest.raises(SystemExit, match="Provide a config"):
            main([])

        plan = {"name": "p", "command_display": "cmd", "task": "nanogpt", "algorithm": "", "mode": "", "config_path": "x", "cwd": ".", "module": "m", "argv": [], "env_overrides": {}}
        with (
            patch("ldm_tts.cli.runner.resolve_config_path", return_value=Path("x")),
            patch("ldm_tts.cli.runner.load_config", return_value={"task": "nanogpt"}),
            patch("ldm_tts.cli.runner.expand_experiments", return_value=[({"task": "nanogpt"}, Path("x"))]),
            patch("ldm_tts.cli.runner.build_plan", return_value=plan),
            patch("ldm_tts.cli.runner.preflight_plan") as preflight,
        ):
            assert main(["x", "--dry-run"]) == 0
        preflight.assert_not_called()
        assert json.loads(capsys.readouterr().out)[0]["name"] == "p"

        with (
            patch("ldm_tts.cli.runner.resolve_config_path", return_value=Path("x")),
            patch("ldm_tts.cli.runner.load_config", return_value={}),
            patch("ldm_tts.cli.runner.expand_experiments", return_value=[({}, Path("x")), ({}, Path("x"))]),
            patch("ldm_tts.cli.runner.build_plan", side_effect=[dict(plan, name="a"), dict(plan, name="b")]),
            patch("ldm_tts.cli.runner.run_plan", side_effect=[2, 3]),
        ):
            assert main(["x", "--keep-going"]) == 3
        output = capsys.readouterr()
        assert '"failed"' in output.err

        with (
            patch("ldm_tts.cli.runner.resolve_config_path", return_value=Path("x")),
            patch("ldm_tts.cli.runner.load_config", return_value={}),
            patch("ldm_tts.cli.runner.expand_experiments", return_value=[({}, Path("x"))]),
            patch("ldm_tts.cli.runner.build_plan", return_value=dict(plan, mode="real")),
            patch("ldm_tts.cli.runner.preflight_plan") as preflight,
            patch("ldm_tts.cli.runner.run_plan", return_value=0),
        ):
            assert main(["x", "--skip-preflight"]) == 0
        preflight.assert_not_called()


class TestResponseValidation:
    def test_fence_and_prose_extraction(self) -> None:
        assert strip_json_fence(" plain ") == "plain"
        assert strip_json_fence("```\nvalue\n```") == "value"
        assert extract_json_object_text('before {"x": 1} after') == '{"x": 1}'
        assert extract_json_object_text('```json\n{"x": 1}\n```') == '{"x": 1}'
        with pytest.raises(ValueError, match="no JSON object"):
            extract_json_object_text("none")

    def test_json_loader_reports_invalid_json_and_non_object(self) -> None:
        with pytest.raises(ValueError, match="invalid JSON"):
            load_json_object('{"x":}')
        with pytest.raises(ValueError, match="JSON object"):
            load_json_object("[1]")

    def test_nested_rejection_and_allowed_keys(self) -> None:
        reject_keys({"items": [{"ok": 1}]}, {"score"})
        with pytest.raises(ValueError, match="score"):
            reject_keys({"items": [{"score": 1}]}, {"score"})
        require_allowed_keys({"a": 1}, {"a"})
        with pytest.raises(ValueError, match="unknown top-level"):
            require_allowed_keys({"b": 1}, {"a"})

    def test_typed_require_helpers(self) -> None:
        assert require_list({"items": [{"x": 1}]}, "items") == [{"x": 1}]
        assert require_str({"name": " x "}, "name") == "x"
        assert require_number({"n": 2}, "n") == 2.0
        assert require_nonnegative_int({"n": 2.0}, "n") == 2
        with pytest.raises(ValueError, match="must be a list"):
            require_list({"items": {}}, "items")
        with pytest.raises(ValueError, match="entries must be objects"):
            require_list({"items": [1]}, "items")
        with pytest.raises(ValueError, match="non-empty string"):
            require_str({"name": " "}, "name")
        with pytest.raises(ValueError, match="must be a number"):
            require_number({"n": True}, "n")
        with pytest.raises(ValueError, match="non-negative integer"):
            require_nonnegative_int({"n": -1}, "n")
        with pytest.raises(ValueError, match="non-negative integer"):
            require_nonnegative_int({"n": 1.5}, "n")


class TestOperationSpaceCoverage:
    def test_schema_serialization_signature_and_replacement(self) -> None:
        schema = _schema()
        payload = operation_schema_to_json(schema)
        assert payload["source_path"] is None
        assert payload["parameters"]["MODE"]["choices"] == ["fast", 2, True, 1.5]
        assert operation_feature_version(schema) == "operation_schema:v1"
        assert len(operation_schema_signature(schema)) == 12
        replaced = replace_operation_schema(
            schema,
            {"count": schema.parameters["COUNT"]},
            version_suffix="active",
            description_prefix="Subset.",
        )
        assert list(replaced.parameters) == ["COUNT"]
        assert replaced.version.startswith("v1:active:")

    def test_load_schema_validation(self, tmp_path: Path) -> None:
        valid = tmp_path / "valid.json"
        valid.write_text(json.dumps({"version": "1", "parameter_order": ["rate", "mode"], "parameters": {"mode": {"type": "choice", "choices": ["a"]}, "rate": {"type": "float", "min": 0, "max": 1}}}), encoding="utf-8")
        loaded = load_operation_schema(valid, tmp_path)
        assert list(loaded.parameters) == ["RATE", "MODE"]

        bad_cases = [
            ([], "JSON object"),
            ({"parameters": {"x": {"type": "int", "min": 0, "max": 1}}}, "version"),
            ({"version": "1", "parameters": {}}, "define parameters"),
            ({"version": "1", "parameters": {"x": {}}, "parameter_order": ["y"]}, "unknown parameter"),
            ({"version": "1", "parameters": {"x": []}}, "spec must be"),
        ]
        for index, (payload, message) in enumerate(bad_cases):
            path = tmp_path / f"bad-{index}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with pytest.raises(ValueError, match=message):
                load_operation_schema(path, tmp_path)

    @pytest.mark.parametrize(
        "parameter",
        [
            OperationParameter("x", "bad"),
            OperationParameter("x", "choice"),
            OperationParameter("x", "float", None, 2),
            OperationParameter("x", "float", 2, 1),
            OperationParameter("x", "float", 1, 2, scale="bad"),
            OperationParameter("x", "float", 0, 2, scale="log"),
        ],
    )
    def test_parameter_normalization_rejects_invalid_specs(self, parameter: OperationParameter) -> None:
        with pytest.raises(ValueError):
            normalize_operation_parameter(parameter)

    def test_parameter_payload_and_json_roundtrip_shapes(self) -> None:
        choice = operation_parameter_from_payload({"parameter": "mode", "kind": "choice", "choices": ["a"]})
        numeric = operation_parameter_from_payload({"name": "rate", "type": "numeric", "min": 1, "max": 2})
        assert operation_parameter_to_json(choice) == {"name": "MODE", "type": "choice", "choices": ["a"]}
        assert operation_parameter_to_json(numeric)["type"] == "float"
        for payload, message in [
            ([], "must be an object"),
            ({}, "non-empty name"),
            ({"name": "x", "type": "choice", "choices": []}, "non-empty choices"),
            ({"name": "x", "type": "text"}, "unsupported type"),
        ]:
            with pytest.raises(ValueError, match=message):
                operation_parameter_from_payload(payload)

    def test_initial_feature_selection(self) -> None:
        schema = _schema()
        assert initial_operation_feature_names(schema, "all") == ["COUNT", "RATE", "MODE"]
        assert initial_operation_feature_names(schema, "0") == ["COUNT"]
        assert initial_operation_feature_names(schema, "99") == ["COUNT", "RATE", "MODE"]
        assert initial_operation_feature_names(schema, "rate, rate, mode") == ["RATE", "MODE"]
        with pytest.raises(ValueError, match="Unknown expansion-schema parameter"):
            initial_operation_feature_names(schema, "missing")
        with pytest.raises(ValueError, match="did not select"):
            initial_operation_feature_names(schema, ",")

    def test_numeric_normalization_and_choice_comparison(self) -> None:
        assert normalize_operation_numeric(2, OperationParameter("x", "float", 2, 2)) == 0
        assert normalize_operation_numeric(10, OperationParameter("x", "float", 1, 100, scale="log")) == pytest.approx(0.5)
        assert normalize_operation_numeric(2, OperationParameter("x", "float", 1, 3)) == 0.5
        assert choice_values_equal(2, "2")
        assert choice_values_equal(True, True)
        assert not choice_values_equal(1, True)
        assert choice_values_equal("2", 2)
        assert choice_values_equal("1.5", 1.5)
        assert choice_values_equal(None, None)

    def test_value_validation_covers_choice_integer_float_and_errors(self) -> None:
        schema = _schema()
        assert validate_operation_value("2", schema.parameters["MODE"], index=1) == 2
        assert validate_operation_value(3.0, schema.parameters["COUNT"], index=1) == 3
        assert validate_operation_value("0.5", schema.parameters["RATE"], index=1) == 0.5
        for value, parameter, message in [
            ("missing", schema.parameters["MODE"], "not in choices"),
            (True, schema.parameters["COUNT"], "must not be boolean"),
            ("bad", schema.parameters["RATE"], "must be numeric"),
            (10, schema.parameters["COUNT"], "outside"),
            (2.5, schema.parameters["COUNT"], "integer"),
        ]:
            with pytest.raises(ValueError, match=message):
                validate_operation_value(value, parameter, index=1)

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ([], "payload must be"),
            ({}, "non-empty list"),
            ({"operations": [{"name": "COUNT", "op": "set_numeric", "value": 1}, {"name": "RATE", "op": "set_numeric", "value": 1}, {"name": "MODE", "op": "set_choice", "value": "fast"}]}, "too many"),
            ({"operations": [1]}, "must be an object"),
            ({"operations": [{"name": "missing", "op": "set_numeric", "value": 1}]}, "unknown parameter"),
            ({"operations": [{"name": "COUNT", "op": "set_numeric", "value": 1}, {"name": "COUNT", "op": "set_numeric", "value": 2}]}, "repeats parameter"),
            ({"operations": [{"name": "MODE", "op": "set_numeric", "value": "fast"}]}, "must use op"),
        ],
    )
    def test_payload_validation_errors(self, payload: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            validate_operation_payload(payload, _schema(), max_operations=2)


class TestLoopTrajectoryAndSpecs:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"budget": -1}, "budget"),
            ({"budget": 1, "start_round": -1}, "start_round"),
            ({"budget": 1, "max_empty_reservoir_rounds": 0}, "max_empty"),
        ],
    )
    def test_loop_rejects_invalid_policy(self, kwargs: dict[str, int], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            run_budgeted_search([], build_round=lambda *_: LDMSearchRoundResult(), **kwargs)

    def test_loop_honors_explicit_stop_reason(self) -> None:
        result = run_budgeted_search(
            [],
            budget=2,
            build_round=lambda *_: LDMSearchRoundResult(stop_reason="done"),
        )
        assert result.early_stop_reason == "done"
        assert result.rounds_run == 1

    def test_disabled_recorder_and_jsonl_loading(self, tmp_path: Path) -> None:
        recorder = JsonlTrajectoryRecorder(None, existing_rounds=[{"x": 1}])
        recorder.append_round({"x": 2})
        assert recorder.rounds == [{"x": 1}, {"x": 2}]
        assert recorder.write_json("x.json", {}) is None
        assert load_jsonl(tmp_path / "missing.jsonl") == []
        path = tmp_path / "rows.jsonl"
        path.write_text('{"x":1}\n\n{"x":2}\n', encoding="utf-8")
        assert load_jsonl(path) == [{"x": 1}, {"x": 2}]
        assert utc_timestamp().count(":") == 2

    def test_recorder_reset_and_atomic_log_update(self, tmp_path: Path) -> None:
        rounds = tmp_path / "rounds.jsonl"
        rounds.write_text("old\n", encoding="utf-8")
        recorder = JsonlTrajectoryRecorder(tmp_path, reset_rounds_file=True)
        assert rounds.read_text(encoding="utf-8") == ""
        log = AtomicJsonLog(tmp_path / "log.json", {"items": []})
        updated = log.update(lambda payload: payload["items"].append(1))
        assert updated == {"items": [1]}
        assert log.read() == updated

    def test_atomic_log_cleans_temp_file_on_replace_failure(self, tmp_path: Path) -> None:
        log = AtomicJsonLog(tmp_path / "log.json", {"ok": True})
        with patch("ldm_tts.engine.run_store.os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                log.write({"ok": False})
        assert list(tmp_path.glob("*.tmp")) == []

    def test_specs_and_trace_records_serialize(self) -> None:
        task = LDMTaskSpec(
            task="demo",
            candidate_domain=CandidateDomainSpec("items", "vector", dimension=2),
            objectives=(ObjectiveSpec("loss", "minimize"),),
            response_spaces=(ResponseSpaceSpec("items", "structured", {"type": "object"}),),
            acquisition=AcquisitionSpec("ei", ("loss",), "maximize", "argmax"),
            reservoir=ReservoirSpec(
                name="items",
                expansions=(
                    ReservoirExpansionSpec(
                        name="emit_items",
                        action_kind="emit_candidate",
                        response_space="items",
                        produces_candidates=True,
                    ),
                ),
                candidate_validator="validate item",
                deduplication_key="item id",
            ),
            surrogate=SurrogateSpaceSpec(
                kind="vector",
                representation="two numeric values",
                dimension_policy="fixed",
                dimension=2,
            ),
            proposal_search=ProposalSearchSpec(
                name="beam_search",
                breadth=3,
                depth=2,
                beam_width=2,
            ),
        )
        assert task.to_dict()["candidate_domain"]["dimension"] == 2
        assert task.to_dict()["reservoir"]["expansions"][0]["action_kind"] == "emit_candidate"
        assert task.to_dict()["proposal_search"]["name"] == "beam_search"
        assert BOPrediction("x").to_dict()["candidate_id"] == "x"
        candidate = CandidateTraceRecord("x", {"a": 1})
        assert candidate.to_dict()["candidate_id"] == "x"
        trace = LDMRoundTrace(0, "demo", 0, 1, "items", "ei", candidates=(candidate,))
        assert trace.to_dict()["candidates"] == (candidate.to_dict(),)

    def test_task_spec_rejects_unbound_reservoir_response_space(self) -> None:
        with pytest.raises(ValueError, match="unknown response space"):
            LDMTaskSpec(
                task="demo",
                candidate_domain=CandidateDomainSpec("items", "vector", dimension=2),
                objectives=(ObjectiveSpec("loss", "minimize"),),
                response_spaces=(
                    ResponseSpaceSpec("items", "structured", {"type": "object"}),
                ),
                acquisition=AcquisitionSpec("ei", ("loss",), "maximize", "argmax"),
                reservoir=ReservoirSpec(
                    name="items",
                    expansions=(
                        ReservoirExpansionSpec(
                            name="broken",
                            action_kind="emit_candidate",
                            response_space="missing",
                            produces_candidates=True,
                        ),
                    ),
                    candidate_validator="validate item",
                    deduplication_key="item id",
                ),
                surrogate=SurrogateSpaceSpec(
                    kind="vector",
                    representation="two numeric values",
                    dimension_policy="fixed",
                    dimension=2,
                ),
            )


class TestDependencyCoverage:
    def test_cli_map_accumulates_repeated_values_and_helpers(self) -> None:
        parsed = cli_args_to_map(["positional", "--tag", "a", "--tag", "b", "--flag", "--no-cache"])
        assert parsed == {"tag": ["a", "b"], "flag": True, "cache": False}
        assert arg_value(parsed, "tag") == "b"
        assert arg_value({"x": []}, "x") == ""
        assert bool_arg({"x": "YES"}, "x")
        assert not bool_arg({"x": False}, "x")

    def test_llm_setting_statuses_and_secret_helpers(self) -> None:
        checks = check_llm_settings(
            "task",
            {},
            {},
            url_arg="url",
            model_arg="model",
            api_arg="key",
            url_env=("URL",),
            model_env=("MODEL",),
            api_env=("KEY",),
            required=True,
        )
        assert [check.status for check in checks] == ["fail", "warn", "fail"]
        local = check_llm_settings(
            "task", {"url": "http://localhost:1"}, {},
            url_arg="url", model_arg="model", api_arg="key",
            url_env=("URL",), model_env=("MODEL",), api_env=("KEY",), required=True,
        )
        assert local[-1].status == "warn"
        assert first_env({"A": "", "B": "x"}, ("A", "B")) == "x"
        assert is_local_url("http://127.0.0.1") and is_local_url("http://0.0.0.0")
        assert not is_local_url("https://example.com")
        assert mask_secret("EMPTY") == "EMPTY"
        assert mask_secret("short") == "***"
        assert mask_secret("123456789") == "***"

    def test_device_parsing_and_cuda_outcomes(self) -> None:
        assert parse_device_ids("0, bad, 2,,") == [0, 2]
        assert check_cuda_visibility("t", "gpu", requested_device="cpu", env={}).status == "skip"
        with patch("ldm_tts.registration.dependencies.query_visible_gpu_ids", return_value=None):
            assert check_cuda_visibility("t", "gpu", requested_device="cuda", env={}).status == "warn"
        with patch("ldm_tts.registration.dependencies.query_visible_gpu_ids", return_value=set()):
            assert check_cuda_visibility("t", "gpu", requested_device="cuda", env={}).status == "fail"
        with patch("ldm_tts.registration.dependencies.query_visible_gpu_ids", return_value={0, 1}):
            assert check_cuda_visibility("t", "gpu", requested_device="cuda", env={}, requested_devices=[2]).status == "fail"
            assert check_cuda_visibility("t", "gpu", requested_device="cuda", env={"CUDA_VISIBLE_DEVICES": "0"}, requested_devices=[0]).status == "ok"
        assert check_cuda_visibility(
            "t",
            "gpu",
            requested_device="cuda",
            env={"CUDA_VISIBLE_DEVICES": ""},
        ).status == "fail"

    def test_gpu_query_handles_success_failure_and_os_errors(self) -> None:
        success = subprocess.CompletedProcess([], 0, stdout="0\ninvalid\n2\n", stderr="")
        with patch("ldm_tts.registration.dependencies.subprocess.run", return_value=success):
            assert query_visible_gpu_ids() == {0, 2}
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="bad")
        with patch("ldm_tts.registration.dependencies.subprocess.run", return_value=failed):
            assert query_visible_gpu_ids() == set()
        with patch("ldm_tts.registration.dependencies.subprocess.run", side_effect=FileNotFoundError):
            assert query_visible_gpu_ids() is None

    def test_llm_checks_treat_unexpanded_environment_references_as_missing(self) -> None:
        checks = check_llm_settings(
            "task",
            {
                "llm-url": "${LLM_BASE_URL}",
                "llm-model-name": "$LLM_MODEL_NAME",
                "api-key": "${LLM_API_KEY}",
            },
            {},
            url_arg="llm-url",
            model_arg="llm-model-name",
            api_arg="api-key",
            url_env=("LLM_BASE_URL",),
            model_env=("LLM_MODEL_NAME",),
            api_env=("LLM_API_KEY",),
            required=True,
        )

        by_name = {check.name: check for check in checks}
        assert by_name["LLM URL"].status == "fail"
        assert by_name["LLM model"].status == "warn"
        assert by_name["LLM API key"].status == "fail"

    def test_reasyn_python_resolution(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        bin_dir = repo / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        versioned = bin_dir / "python3.11"
        versioned.write_text("", encoding="utf-8")
        assert resolve_reasyn_python("", repo) == versioned
        explicit = tmp_path / "python"
        assert resolve_reasyn_python(str(explicit), repo) == explicit

    def test_antibody_checks_cover_mock_and_real_inputs(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text("device: cuda\nbbox:\n  path: missing-absolut\n", encoding="utf-8")
        antigens = tmp_path / "antigens.txt"
        antigens.write_text("a\n", encoding="utf-8")
        mock_checks = check_antibody(
            {"mock": True, "antigens-file": str(antigens), "config": str(config)},
            {}, tmp_path, mode="mock",
        )
        assert not any(check.status == "fail" for check in mock_checks)
        assert next(check for check in mock_checks if check.name == "Absolut").status == "skip"

        real_checks = check_antibody(
            {
                "antigen": "A",
                "config": str(config),
                "device": "cpu",
                "llm-url": "https://api",
                "api-key": "secret",
            },
            {}, tmp_path, mode="real",
        )
        device_check = next(check for check in real_checks if check.name == "AntBO device")
        assert device_check.status == "skip"
        assert "cpu" in device_check.message
        assert next(check for check in real_checks if check.name == "Absolut").status == "fail"
        absolut_executable = tmp_path / "src" / "bin" / "Absolut"
        absolut_executable.parent.mkdir(parents=True)
        absolut_executable.write_text("#!/bin/sh\n", encoding="utf-8")
        absolut_executable.chmod(0o755)
        env_path_checks = check_antibody(
            {
                "antigen": "A",
                "config": str(config),
                "device": "cpu",
                "llm-url": "https://api",
                "api-key": "secret",
            },
            {"ABSOLUT_PATH": str(tmp_path)},
            tmp_path,
            mode="real",
        )
        assert next(check for check in env_path_checks if check.name == "Absolut").status == "ok"
        missing = check_antibody({"mock": True, "config": "missing.yaml"}, {}, tmp_path, mode="mock")
        assert any(check.name == "antigen input" and check.status == "fail" for check in missing)
        assert any(check.name == "AntBO config" and check.status == "fail" for check in missing)

    def test_nanogpt_mock_and_real_dependency_paths(self, tmp_path: Path) -> None:
        train_file = tmp_path / "train.py"
        schema = tmp_path / "schema.json"
        train_file.write_text("", encoding="utf-8")
        schema.write_text("{}", encoding="utf-8")
        mock_checks = check_nanogpt(
            {
                "generator": "mock",
                "gp-device": "cpu",
                "train-file": str(train_file),
                "operation-schema": str(schema),
            },
            {},
            tmp_path,
            mode="mock",
        )
        assert not any(check.status == "fail" for check in mock_checks)
        real_checks = check_nanogpt(
            {"generator": "api", "llm-url": "https://api", "api-key": "key", "gp-device": "cpu", "train-file": "missing.py", "data-dir": "missing-data"},
            {}, tmp_path, mode="real",
        )
        assert any(check.status == "fail" for check in real_checks)

    def test_yaml_and_check_formatting(self, tmp_path: Path) -> None:
        mapping = tmp_path / "mapping.yaml"
        mapping.write_text("a: 1\n", encoding="utf-8")
        sequence = tmp_path / "sequence.yaml"
        sequence.write_text("- 1\n", encoding="utf-8")
        assert load_yaml_object(mapping) == {"a": 1}
        assert load_yaml_object(sequence) == {}
        checks = [
            DependencyCheck("t", "ok", "ok", "fine", "detail"),
            DependencyCheck("t", "custom", "x", "message"),
        ]
        assert "[OK] t: ok: fine (detail)" in format_checks(checks)
        assert "[X]" in format_checks(checks)
        assert json.loads(checks_to_json(checks))[0]["detail"] == "detail"
