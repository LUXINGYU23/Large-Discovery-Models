from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tasks.nanogpt.resources.train import mock_train
from tasks.nanogpt.core import workflow as model_based_procedure
from tasks.nanogpt.core.expansion_schema import load_operation_schema
from tasks.nanogpt.core.search_core import (
    ProgressBar,
    SearchConfig,
    SearchEngine,
    SearchState,
    add_token_usage,
    apply_blank_line_tolerant_replace,
    apply_search_replace_blocks,
    apply_unified_diff,
    apply_unified_diff_to_text,
    collect_token_logprobs,
    extract_description,
    extract_generation_logprobs,
    extract_replacement_train_file,
    extract_search_replace_blocks,
    extract_tool_call_edit_blocks,
    extract_tool_call_summary,
    extract_unified_diff,
    find_blank_line_tolerant_spans,
    format_duration,
    format_prior_edits,
    format_state_logprob_summary,
    indent_text,
    jsonable,
    lines_equal_ignoring_trailing_space,
    looks_like_unified_diff,
    make_mock_candidate_text,
    make_mock_patch,
    make_search_replace_response,
    make_unified_diff,
    matched_line_end_len,
    normalize_token_usage,
    parse_metrics,
    strip_code_fence_edges,
    strip_line_ending,
    summarize_edit_blocks,
    summarize_generation_logprobs,
    truncate_text,
    tune_mock_train_text,
)
from tasks.nanogpt.core.single_search import default_run_name, make_unique_run_dir, parse_args, safe_path_tag


def _state(tmp_path: Path, state_id: str, depth: int, score: float | None = None) -> SearchState:
    workdir = tmp_path / state_id
    workdir.mkdir(parents=True, exist_ok=True)
    train_path = workdir / "train.py"
    train_path.write_text("print('x')\n", encoding="utf-8")
    return SearchState(state_id, None, depth, workdir, train_path, score=score)


class TestSearchEngineIntegration:
    def test_engine_reward_orients_scores_and_rejects_failed_states(self, tmp_path: Path) -> None:
        config = SearchConfig(
            project_root=tmp_path,
            seed_train_path=Path(mock_train.__file__).resolve(),
            out_dir=tmp_path / "run",
            generator="mock",
            show_progress=False,
        )
        engine = SearchEngine(config)
        better = _state(tmp_path, "better", 1, 1.0)
        worse = _state(tmp_path, "worse", 1, 3.0)
        failed = _state(tmp_path, "failed", 1, engine.config.failure_score)
        better.status = worse.status = "evaluated"
        failed.status = "crash"
        engine.states = [better, worse, failed]

        assert engine.reward(better) == 1.0
        assert engine.reward(worse) == 0.0
        assert engine.reward(failed) == 0.0
        engine.config.minimize = False
        assert engine.reward(better) == 0.0
        assert engine.reward(worse) == 1.0

    def test_engine_can_be_constructed_after_an_event_loop_closes(self, tmp_path: Path) -> None:
        asyncio.run(asyncio.sleep(0))
        config = SearchConfig(
            project_root=tmp_path,
            seed_train_path=Path(mock_train.__file__).resolve(),
            out_dir=tmp_path / "run",
            generator="mock",
            show_progress=False,
        )

        engine = SearchEngine(config)
        root = engine.create_seed_state()
        assert asyncio.run(engine.expand_state(root, 1, search_note="loop lifecycle"))

    def test_mock_generation_evaluation_and_summary(self, tmp_path: Path) -> None:
        seed = Path(mock_train.__file__).resolve()
        out_dir = tmp_path / "run"
        config = SearchConfig(
            project_root=tmp_path,
            seed_train_path=seed,
            out_dir=out_dir,
            eval_command=f"{sys.executable} {{train_path}}",
            generator="mock",
            show_progress=False,
            timeout_seconds=10,
        )
        engine = SearchEngine(config)
        root = engine.create_seed_state()
        children = asyncio.run(engine.expand_state(root, 2, search_note="integration"))
        asyncio.run(engine.evaluate_many(children))

        assert len(children) == 2
        assert all(state.status == "evaluated" for state in children)
        assert all(state.score is not None and state.score < 2 for state in children)
        assert engine.best_state() in children
        assert len(engine.ranked_states(children)) == 2
        summary = engine.write_summary(method="best_of_n", args={"path": tmp_path}, best=engine.best_state())
        assert json.loads(summary.read_text(encoding="utf-8"))["evaluation_count"] == 2
        assert (out_dir / "best_train.py").exists()
        assert len((out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) >= 5

    def test_evaluation_exposes_task_scripts_and_local_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_root = tmp_path / "nanogpt"
        scripts_dir = project_root / "scripts"
        cache_dir = project_root / "cache" / "autoresearch"
        scripts_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        seed = project_root / "train.py"
        seed.write_text(
            "\n".join(
                [
                    "import os",
                    f"assert {str(scripts_dir)!r} in os.environ.get('PYTHONPATH', '').split(os.pathsep)",
                    f"assert os.environ.get('AUTORESEARCH_CACHE_DIR') == {str(cache_dir)!r}",
                    "print('val_bpb: 0.5')",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("AUTORESEARCH_CACHE_DIR", raising=False)
        engine = SearchEngine(
            SearchConfig(
                project_root=project_root,
                seed_train_path=seed,
                out_dir=project_root / "run",
                eval_command=f"{sys.executable} {{train_path}}",
                generator="mock",
                show_progress=False,
                timeout_seconds=10,
            )
        )

        root = engine.create_seed_state()
        engine.evaluate_state(root)

        assert root.status == "evaluated"
        assert root.score == pytest.approx(0.5)

    def test_skipped_and_generation_error_evaluations(self, tmp_path: Path) -> None:
        config = SearchConfig(
            project_root=tmp_path,
            seed_train_path=Path(mock_train.__file__).resolve(),
            out_dir=tmp_path / "run",
            generator="mock",
            run_evaluation=False,
            show_progress=False,
        )
        engine = SearchEngine(config)
        root = engine.create_seed_state()
        child = asyncio.run(engine.expand_state(root, 1))[0]
        engine.evaluate_state(child)
        assert child.status == "evaluation_skipped"
        failed = _state(tmp_path, "failed", 1)
        failed.status = "generation_error"
        engine.evaluate_state(failed)
        assert failed.status == "generation_error"
        engine.defer_evaluation(failed, reason="ignored")
        assert failed.status == "generation_error"

    def test_engine_depth_ranking_inline_and_command_helpers(self, tmp_path: Path) -> None:
        config = SearchConfig(
            project_root=tmp_path,
            seed_train_path=Path(mock_train.__file__).resolve(),
            out_dir=tmp_path / "run",
            eval_command="python {train_path} --diag {diagnostics_path}",
            eval_shell=False,
            eval_each_num_steps=2,
            response_log_chars=4,
            top_logprobs=-3,
            show_progress=False,
        )
        engine = SearchEngine(config)
        assert engine.config.top_logprobs == 0
        assert engine.evaluation_depths(3) == [2, 3]
        assert engine.should_evaluate_depth(0)
        assert not engine.should_evaluate_depth(1)
        inline, truncated = engine._inline_llm_response("abcdef")
        assert inline == "abcd" and truncated
        state = _state(tmp_path, "cmd", 1)
        command = engine._format_eval_command(state, tmp_path / "diag.json")
        assert isinstance(command, list) and str(state.train_path) in command
        assert engine.best_state([]) is None


class TestSearchCoreUtilities:
    def test_metrics_parsing_prefers_diagnostics_and_reads_inline(self, tmp_path: Path) -> None:
        diagnostics = tmp_path / "diagnostics.json"
        diagnostics.write_text('{"val_bpb": 0.9, "nested": {"ok": true}}', encoding="utf-8")
        output = 'val_bpb: 1.2\nother: 3\ndiagnostics_json_inline: {"inline": 4}\n'
        metrics = parse_metrics(output, diagnostics)
        assert metrics["val_bpb"] == 0.9
        assert metrics["other"] == 3.0
        assert metrics["inline"] == 4
        diagnostics.write_text("invalid", encoding="utf-8")
        assert parse_metrics("score: 2", diagnostics)["score"] == 2

    def test_json_token_and_logprob_helpers(self, tmp_path: Path) -> None:
        assert jsonable({"path": tmp_path, "items": (tmp_path,)}) == {"path": str(tmp_path), "items": [str(tmp_path)]}
        assert normalize_token_usage(3) == (3, {"prompt_tokens": 0, "completion_tokens": 3, "total_tokens": 3})
        usage = {"prompt_tokens": 2, "completion_tokens": 3, "logprobs": ["x"]}
        assert normalize_token_usage(usage)[0] == 3
        assert extract_generation_logprobs(usage) == ["x"]
        assert extract_generation_logprobs(2) is None
        assert collect_token_logprobs({"content": [{"logprob": -1}, {"x": 2}]}) == [-1.0]
        assert collect_token_logprobs({"token_logprobs": [-1, None, -2]}) == [-1.0, -2.0]
        assert collect_token_logprobs([{"logprob": -1}, -2, "bad"]) == [-1.0, -2.0]
        assert collect_token_logprobs(None) == []
        summary = summarize_generation_logprobs([-1.0, -2.0])
        assert summary["token_count"] == 2
        assert summary["perplexity"] > 1
        assert summarize_generation_logprobs([]) == {}
        assert add_token_usage({"prompt_tokens": 1}, {"completion_tokens": 2})["total_tokens"] == 0

    def test_state_logprob_and_text_formatters(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "s", 1)
        assert format_state_logprob_summary(state) == ""
        state.edits = [{"logprob_summary": {"mean_logprob": -1.2, "perplexity": 3.4, "token_count": 5}}]
        assert format_state_logprob_summary(state) == "lp=-1.2 ppl=3.4 lptok=5"
        assert indent_text("", "  ") == "  (empty)"
        assert indent_text("a\nb", ">") == ">a\n>b"
        assert truncate_text("abcdef", 3).startswith("abc")
        assert truncate_text("abcdef", -1) == "abcdef"
        assert format_prior_edits([]) == "- None yet."
        assert "Edit 1" in format_prior_edits([{"edit_index": 1, "description": "change", "patch": "+x"}])
        assert summarize_edit_blocks([{"search": "abcdef", "replace": "uvwxyz"}], limit=3)[0]["search"].startswith("abc")

    def test_search_replace_parsing_application_and_errors(self) -> None:
        response = "Summary: change\n\ntrain.py\n<<<<<<< SEARCH\na = 1\n=======\na = 2\n>>>>>>> REPLACE\n"
        blocks = extract_search_replace_blocks(response)
        assert blocks == [{"search": "a = 1", "replace": "a = 2"}]
        assert apply_search_replace_blocks("a = 1\n", blocks) == "a = 2\n"
        assert apply_search_replace_blocks("a = 2\n", blocks) == "a = 2\n"
        with pytest.raises(ValueError, match="empty SEARCH"):
            apply_search_replace_blocks("x", [{"search": "", "replace": "y"}])
        with pytest.raises(ValueError, match="matched 2 times"):
            apply_search_replace_blocks("x x", [{"search": "x", "replace": "y"}])
        with pytest.raises(ValueError, match="not found"):
            apply_search_replace_blocks("x", [{"search": "z", "replace": "y"}])

    def test_blank_line_tolerant_matching(self) -> None:
        text = "a  \n\n\n b\n"
        search = "a\n\n b"
        spans = find_blank_line_tolerant_spans(text, search)
        assert len(spans) == 1
        assert apply_blank_line_tolerant_replace(text, search, "c") == "c\n"
        assert apply_blank_line_tolerant_replace("x", "z", "y") is None
        assert lines_equal_ignoring_trailing_space("a  \r\n", "a\n")
        assert strip_line_ending("a\r\n") == "a"
        assert strip_line_ending("a\r") == "a"
        assert matched_line_end_len("a\n", "a") == 1
        assert matched_line_end_len("a\n", "a\n") == 2

    def test_tool_call_parsing(self) -> None:
        payload = [{"name": "edit_train_py", "arguments": {"summary": " update ", "edits": [{"search": "a", "replace": "b"}]}}]
        response = f"<tool_calls>{json.dumps(payload)}</tool_calls>"
        assert extract_tool_call_edit_blocks(response) == [{"search": "a", "replace": "b"}]
        assert extract_tool_call_summary(response) == "update"
        string_args = [{"name": "edit_train_py", "arguments": json.dumps({"edits": [{"search": "x", "replace": "y"}]})}]
        assert extract_tool_call_edit_blocks(f"<tool_calls>{json.dumps(string_args)}</tool_calls>")[0]["search"] == "x"
        assert extract_tool_call_edit_blocks("<tool_calls>bad</tool_calls>") == []
        assert extract_tool_call_summary("none") == ""

    def test_diff_replacement_and_description_helpers(self, tmp_path: Path) -> None:
        old = "a = 1\n"
        new = "a = 2\n"
        diff = make_unified_diff(old, new, fromfile="a/train.py", tofile="b/train.py")
        assert looks_like_unified_diff(diff)
        assert extract_unified_diff(f"```diff\n{diff}```") == diff
        assert apply_unified_diff_to_text(old, diff) == new
        path = tmp_path / "train.py"
        path.write_text(old, encoding="utf-8")
        assert apply_unified_diff(path, diff) == new
        assert extract_description("Description: concise") == "concise"
        assert extract_description(diff) == "a = 2"
        assert extract_unified_diff("none") is None
        with pytest.raises(ValueError, match="Could not apply"):
            apply_unified_diff_to_text(old, "--- train.py\n@@ bad\n")

    def test_complete_file_and_code_fence_helpers(self) -> None:
        mock_file = "val_bpb = 1\ndiagnostics_json_inline = 1\nAUTORESEARCH_DIAGNOSTICS_JSON = 1\n"
        assert extract_replacement_train_file(f"<train.py>{mock_file}</train.py>") == mock_file
        assert extract_replacement_train_file(f"```python\n{mock_file}```") == mock_file
        assert extract_replacement_train_file("not code") is None
        assert strip_code_fence_edges("```python\nx = 1\n```") == "x = 1"

    def test_mock_edit_generation_roundtrip(self) -> None:
        source = Path(mock_train.__file__).read_text(encoding="utf-8")
        tuned, description = tune_mock_train_text(source, "state_1")
        assert tuned != source and description.startswith("move")
        response = make_search_replace_response(source, tuned, description)
        assert apply_search_replace_blocks(source, extract_search_replace_blocks(response)) == tuned
        assert make_mock_patch(source, "state_2")
        generic, generic_description = make_mock_candidate_text("#!/usr/bin/env python\nprint(1)\n", "s1")
        assert generic.startswith("#!/usr/bin/env python\n# TTS mock")
        assert "comment-only" in generic_description

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(1.2, "1s"), (65, "1m05s"), (3665, "1h01m")],
    )
    def test_duration_formatting(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    def test_progress_bar_can_update_and_finish(self, capsys: pytest.CaptureFixture[str]) -> None:
        bar = ProgressBar(total=2, label="test", width=4)
        bar.update(1, best_score=0.5, status="working")
        bar.finish(2, best_score=0.4)
        assert "test" in capsys.readouterr().err


class TestMockTrainAndSingleSearch:
    def test_mock_metric_helpers_and_validation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert mock_train.penalty(2, 1, 1, 2) == 2
        assert mock_train.log_penalty(1, 1, 2) == 0
        with patch.object(mock_train, "file_jitter", return_value=0.0):
            metrics = mock_train.compute_mock_metrics()
        assert metrics["val_bpb"] > 0
        assert metrics["mock_knobs"]["DEPTH"] == mock_train.DEPTH

        with patch.object(mock_train, "DEPTH", 0):
            with pytest.raises(ValueError, match="positive"):
                mock_train.compute_mock_metrics()
        with patch.object(mock_train, "DEPTH", 20), patch.object(mock_train, "WIDTH", 600):
            with pytest.raises(RuntimeError, match="OOM"):
                mock_train.compute_mock_metrics()

        output = tmp_path / "diag.json"
        monkeypatch.setenv("AUTORESEARCH_DIAGNOSTICS_JSON", str(output))
        mock_train.write_diagnostics({"ok": True})
        assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
        monkeypatch.setenv("AUTORESEARCH_DIAGNOSTICS_JSON", "")
        mock_train.write_diagnostics({"ignored": True})

    def test_mock_main_success_and_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        diagnostics = tmp_path / "diag.json"
        monkeypatch.setenv("AUTORESEARCH_DIAGNOSTICS_JSON", str(diagnostics))
        monkeypatch.setenv("TTS_MOCK_SLEEP", "0")
        assert mock_train.main() == 0
        assert "val_bpb:" in capsys.readouterr().out
        assert diagnostics.exists()
        monkeypatch.setenv("TTS_MOCK_FAIL", "1")
        assert mock_train.main() == 1
        assert "FAIL" in capsys.readouterr().out

    def test_single_search_name_and_argument_helpers(self, tmp_path: Path) -> None:
        assert safe_path_tag(" a/b c ") == "a_b_c"
        assert safe_path_tag("***", default="fallback") == "fallback"
        assert make_unique_run_dir(tmp_path, "run") == tmp_path / "run"
        (tmp_path / "run").mkdir()
        assert make_unique_run_dir(tmp_path, "run") == tmp_path / "run_02"
        args = parse_args(["--method", "best_of_n", "--breadth", "3", "--no-progress"])
        assert args.method == "best_of_n" and args.breadth == 3 and args.no_progress
        with patch("tasks.nanogpt.core.single_search.time.strftime", return_value="STAMP"):
            name = default_run_name(args, Path("train.py"))
        assert name.endswith("STAMP")


class TestOperationMockSearch:
    def test_zero_iteration_contract_has_no_progress_work(self) -> None:
        args = SimpleNamespace(
            iterations=0,
            warmup=0,
            skip_eval=True,
            evaluate_root=False,
            max_real_evaluations=0,
        )

        assert model_based_procedure.estimate_progress_total(args, generated_per_iteration=2) == 0

    def test_generation_uses_a_semaphore_bound_to_the_active_loop(self, tmp_path: Path) -> None:
        task_root = Path(mock_train.__file__).resolve().parents[2]
        schema = load_operation_schema(
            task_root / "resources" / "schemas" / "mock_operations.json",
            task_root,
        )
        args = model_based_procedure.parse_args(
            [
                "--generator",
                "operation_mock",
                "--operation-schema",
                str(schema.path),
                "--project-root",
                str(task_root),
                "--no-progress",
            ]
        )
        config = SearchConfig(
            project_root=task_root,
            seed_train_path=Path(mock_train.__file__).resolve(),
            out_dir=tmp_path / "run",
            generator="operation_mock",
            show_progress=False,
        )
        engine = model_based_procedure.OperationSearchEngine(config, schema, args)
        root = engine.create_seed_state()

        child = asyncio.run(engine.expand_state(root, 1))[0]

        assert child.status == "generated"
        assert child.error is None
        assert child.edits
        assert engine._generation_sem is not None
