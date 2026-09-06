"""Config-driven runner for all LDM-TTS task procedures."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ldm_tts.registration.experiment import (
    ACTIVE_CONTRACT_PATH_ENV,
    ACTIVE_CONTRACT_PROFILE_ENV,
    ExperimentContractError,
    load_experiment_contract,
    validate_profile_args,
)
from ldm_tts.registration.dependencies import (
    DependencyCheck,
    check_plan,
    format_checks,
    has_failures,
)
from ldm_tts.registration.registry import (
    REPOSITORY_RELATIVE_PREFIXES,
    TASK_DEFINITIONS,
    TaskRegistrationError,
    get_task_definition,
)
from ldm_tts.repository import resolve_repository_root


REPO_ROOT = resolve_repository_root(source_file=Path(__file__))
ENV_VAR_PATTERN = re.compile(r"\$(?P<brace>\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\})|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)")
SENSITIVE_NAME_SUFFIXES = (
    "api-key",
    "api-key-file",
    "password",
    "secret",
    "access-token",
    "auth-token",
)

NEGATABLE_BOOLEAN_KEYS = {
    "allow-early-stop",
    "fallback-random",
    "include-antigen-context",
    "endpoint-preflight",
    "qwen35-sampling-defaults",
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_configs()
        return 0
    if args.config is None:
        raise SystemExit("Provide a config path, or use --list.")

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    for raw_override in args.set:
        apply_override(config, raw_override)

    experiments = expand_experiments(config, config_path)
    plans = [build_plan(exp, path) for exp, path in experiments]

    if args.dry_run:
        print(json.dumps([plan_for_json(plan) for plan in plans], indent=2, sort_keys=True))
        return 0

    failures: list[tuple[str, int]] = []
    for plan in plans:
        if not args.skip_preflight:
            preflight_plan(plan)
        print(f"[ldm-tts] {plan['name']}: {plan['command_display']}")
        return_code = run_plan(plan)
        if return_code != 0:
            failures.append((plan["name"], return_code))
            if not args.keep_going:
                return return_code

    if failures:
        print(json.dumps({"failed": failures}, indent=2), file=sys.stderr)
        return failures[-1][1]
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a config from config/ and run the matching LDM-TTS task procedure."
    )
    parser.add_argument("config", nargs="?", help="YAML/JSON experiment config, or suite config.")
    parser.add_argument("--dry-run", action="store_true", help="Print procedure calls without running them.")
    parser.add_argument("--list", action="store_true", help="List available configs under config/.")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the automatic dependency preflight for advanced diagnostics only.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override a config value. Example: --set args.budget=16. "
            "VALUE is parsed as JSON when possible."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="When running a suite, continue after a failed experiment.",
    )
    return parser.parse_args(argv)


def resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("YAML configs require PyYAML. Install pyyaml or use JSON.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be an object: {path}")
    return data


def expand_experiments(config: dict[str, Any], config_path: Path) -> list[tuple[dict[str, Any], Path]]:
    if "experiments" not in config:
        return [(config, config_path)]

    entries = config.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("Suite configs must contain a non-empty experiments list.")

    expanded: list[tuple[dict[str, Any], Path]] = []
    for entry in entries:
        if isinstance(entry, str):
            child_path = resolve_child_config_path(entry, config_path.parent)
            expanded.append((load_config(child_path), child_path))
            continue
        if isinstance(entry, dict) and "config" in entry:
            child_path = resolve_child_config_path(str(entry["config"]), config_path.parent)
            child = load_config(child_path)
            for override in entry.get("set", []) or []:
                apply_override(child, str(override))
            expanded.append((child, child_path))
            continue
        if isinstance(entry, dict):
            expanded.append((entry, config_path))
            continue
        raise SystemExit(f"Unsupported suite experiment entry: {entry!r}")
    return expanded


def resolve_child_config_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise SystemExit(f"Suite child config not found: {path}")
    return path.resolve()


def build_plan(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    validate_config_keys(config, config_path)

    task = str(config.get("task", "")).strip()
    if not task:
        raise SystemExit(f"Missing required field `task` in {config_path}")
    try:
        definition = get_task_definition(task)
    except (KeyError, TaskRegistrationError) as exc:
        raise SystemExit(str(exc).strip("'")) from exc
    runner = config.get("runner") or {}
    if not isinstance(runner, dict):
        raise SystemExit("runner must be an object when provided.")

    context = make_context(config, config_path, task)
    cwd = resolve_repo_path(
        str(runner.get("cwd", definition.relative_root.as_posix())), context
    )
    module_name = expand_string(
        str(runner.get("module", definition.module)), context
    )

    env_overrides = config.get("env") or {}
    if not isinstance(env_overrides, dict):
        raise SystemExit("env must be an object when provided.")
    env: dict[str, str] = {}
    for key, value in env_overrides.items():
        env[str(key)] = resolve_repo_relative_reference(
            expand_value(value, context, {**os.environ, **env})
        )

    variables = {**os.environ, **env}
    argv = args_to_cli(config.get("args", {}), context, variables)
    argv.extend(extra_args_to_cli(config.get("extra_args", []), context, variables))

    contract = None
    contract_profile = str(config.get("contract_profile", "")).strip()
    if definition.experiment_contract_path is not None:
        contract_path = REPO_ROOT / definition.experiment_contract_path
        try:
            contract = load_experiment_contract(contract_path)
            if contract_profile:
                validate_profile_args(contract, contract_profile, config.get("args", {}))
        except ExperimentContractError as exc:
            raise SystemExit(str(exc)) from exc
        env[ACTIVE_CONTRACT_PATH_ENV] = str(contract_path.resolve())
        if contract_profile:
            env[ACTIVE_CONTRACT_PROFILE_ENV] = contract_profile
    elif contract_profile:
        raise SystemExit(
            f"Config selects contract_profile {contract_profile!r}, but task {task!r} "
            "has no experiment.json contract."
        )

    command = [sys.executable, "-m", module_name, *argv]
    display_command = [sys.executable, "-m", module_name, *redact_cli_values(argv)]
    return {
        "name": str(config.get("name") or config_path.stem),
        "task": task,
        "algorithm": str(config.get("algorithm", "")),
        "mode": str(config.get("mode", "")),
        "config_path": str(config_path),
        "cwd": str(cwd),
        "module": module_name,
        "argv": argv,
        "command": command,
        "command_display": shlex.join(display_command),
        "env_overrides": env,
        "contract_path": str(contract.path) if contract is not None else "",
        "contract_sha256": contract.digest if contract is not None else "",
        "contract_profile": contract_profile,
    }


def validate_config_keys(config: dict[str, Any], config_path: Path) -> None:
    allowed = {
        "name",
        "task",
        "algorithm",
        "mode",
        "description",
        "contract_profile",
        "env",
        "args",
        "extra_args",
        "runner",
        "experiments",
    }
    unknown = sorted(str(key) for key in config if str(key) not in allowed)
    if unknown:
        formatted = ", ".join(unknown)
        raise SystemExit(
            f"Unknown top-level config field(s) in {config_path}: {formatted}. "
            "Task CLI options belong under `args:`; environment variables belong under `env:`."
        )


def make_context(config: dict[str, Any], config_path: Path, task: str) -> dict[str, str]:
    task_dir = REPO_ROOT / get_task_definition(task).relative_root
    return {
        "repo_root": str(REPO_ROOT),
        "config_dir": str(config_path.parent),
        "config_name": config_path.stem,
        "task": task,
        "task_dir": str(task_dir),
        "name": str(config.get("name") or config_path.stem),
        "algorithm": str(config.get("algorithm", "")),
        "mode": str(config.get("mode", "")),
    }


def resolve_repo_path(raw: str, context: dict[str, str]) -> Path:
    path = Path(expand_string(raw, context))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def run_plan(plan: dict[str, Any]) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    with pushd(Path(plan["cwd"])), patched_env(plan["env_overrides"]):
        module = importlib.import_module(plan["module"])
        main_func = getattr(module, "main", None)
        if main_func is None:
            raise SystemExit(f"Task module {plan['module']} has no main(argv) function.")
        result = main_func(list(plan["argv"]))
    return 0 if result is None else int(result)


def preflight_plan(plan: dict[str, Any]) -> list[DependencyCheck]:
    """Block non-mock execution when a declared dependency check fails."""
    if str(plan.get("mode", "")).strip().lower() in {"", "mock"}:
        return []
    checks = check_plan(plan)
    report = format_checks(checks)
    if report:
        print(report)
    if has_failures(checks):
        raise SystemExit(
            f"Dependency preflight failed for {plan.get('name', plan.get('task', 'task'))}. "
            "Resolve every FAIL before running, or use --skip-preflight only for "
            "controlled diagnostics."
        )
    return checks


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def patched_env(overrides: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def args_to_cli(args: Any, context: dict[str, str], variables: dict[str, str] | None = None) -> list[str]:
    if args is None:
        return []
    if not isinstance(args, dict):
        raise SystemExit("args must be an object mapping CLI option names to values.")
    cli: list[str] = []
    for key, value in args.items():
        append_cli_arg(cli, str(key), value, context, variables)
    return cli


def append_cli_arg(
    cli: list[str],
    key: str,
    value: Any,
    context: dict[str, str],
    variables: dict[str, str] | None = None,
) -> None:
    if value is None:
        return
    flag = key if key.startswith("-") else "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            cli.append(flag)
        elif flag.startswith("--") and flag[2:] in NEGATABLE_BOOLEAN_KEYS:
            cli.append("--no-" + flag[2:])
        return
    if isinstance(value, list):
        for item in value:
            cli.append(flag)
            cli.append(resolve_repo_relative_reference(expand_value(item, context, variables)))
        return
    cli.append(flag)
    cli.append(resolve_repo_relative_reference(expand_value(value, context, variables)))


def extra_args_to_cli(
    extra_args: Any,
    context: dict[str, str],
    variables: dict[str, str] | None = None,
) -> list[str]:
    if extra_args is None:
        return []
    if not isinstance(extra_args, list):
        raise SystemExit("extra_args must be a list when provided.")
    return [expand_value(item, context, variables) for item in extra_args]


def is_sensitive_name(value: str) -> bool:
    normalized = value.lower().replace("_", "-").lstrip("-")
    return any(
        normalized == suffix or normalized.endswith("-" + suffix)
        for suffix in SENSITIVE_NAME_SUFFIXES
    )


def redact_cli_values(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for value in argv:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        if value.startswith("--") and "=" in value:
            flag, _, _ = value.partition("=")
            redacted.append(f"{flag}=***" if is_sensitive_name(flag) else value)
            continue
        redacted.append(value)
        if value.startswith("--") and is_sensitive_name(value):
            hide_next = True
    return redacted


def redact_environment(env: dict[str, str]) -> dict[str, str]:
    return {
        key: "***" if is_sensitive_name(key) else value
        for key, value in env.items()
    }


def expand_value(value: Any, context: dict[str, str], variables: dict[str, str] | None = None) -> str:
    if isinstance(value, str):
        return expand_string(value, context, variables)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def expand_string(value: str, context: dict[str, str], variables: dict[str, str] | None = None) -> str:
    class PreserveUnknown(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    expanded = value.format_map(PreserveUnknown(context))
    return expand_env_vars(expanded, variables or os.environ)


def expand_env_vars(value: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group("braced") or match.group("plain")
        if key is None or key not in variables:
            return match.group(0)
        return str(variables[key])

    return ENV_VAR_PATTERN.sub(replace, value)


def resolve_repo_relative_reference(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return value
    normalized = value.replace("\\", "/")
    if not any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in REPOSITORY_RELATIVE_PREFIXES
    ):
        return value
    return str((REPO_ROOT / path).resolve())


def apply_override(config: dict[str, Any], raw: str) -> None:
    if "=" not in raw:
        raise SystemExit(f"Override must look like PATH=VALUE, got {raw!r}")
    path_text, value_text = raw.split("=", 1)
    keys = [part for part in path_text.split(".") if part]
    if not keys:
        raise SystemExit(f"Override path is empty in {raw!r}")
    value = parse_override_value(value_text)
    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        current = cursor.get(key)
        if current is None:
            current = {}
            cursor[key] = current
        if not isinstance(current, dict):
            raise SystemExit(f"Cannot set {raw!r}; {key!r} is not an object.")
        cursor = current
    cursor[keys[-1]] = value


def parse_override_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def list_configs() -> None:
    base = REPO_ROOT / "config"
    rows: list[dict[str, str]] = []
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        try:
            data = load_config(path)
        except SystemExit:
            continue
        rows.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "task": str(data.get("task", "suite" if "experiments" in data else "")),
            "algorithm": str(data.get("algorithm", "")),
            "mode": str(data.get("mode", "")),
            "description": str(data.get("description", "")),
        })
    print(json.dumps(rows, indent=2, sort_keys=True))


def plan_for_json(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": plan["name"],
        "task": plan["task"],
        "algorithm": plan["algorithm"],
        "mode": plan["mode"],
        "config_path": plan["config_path"],
        "cwd": plan["cwd"],
        "module": plan["module"],
        "argv": redact_cli_values(plan["argv"]),
        "command_display": plan["command_display"],
        "env_overrides": redact_environment(plan["env_overrides"]),
        "experiment_contract": {
            "path": plan.get("contract_path", ""),
            "sha256": plan.get("contract_sha256", ""),
            "profile": plan.get("contract_profile", ""),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
