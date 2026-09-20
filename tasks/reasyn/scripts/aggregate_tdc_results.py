#!/usr/bin/env python3
"""Aggregate only a complete real released-oracle three-seed result grid."""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.reasyn.core.metrics import ORACLES, top_auc, top_mean

MAX_ORACLE_CALLS = 10000
SEEDS = (0, 1, 2)
VARIABLE_IDENTITY_FIELDS = frozenset(("oracle", "seed"))
ORACLE_VARIANT_IDENTITY_FIELDS = frozenset(("search_width",))
REQUIRED_CONFIGURATION_IDENTITY_FIELDS = frozenset(
    (
        "benchmark",
        "mock",
        "proposal_mode",
        "search_method",
        "max_oracle_calls",
        "reservoir_size",
        "evaluations_per_round",
        "proposal_batch_size",
        "bo_pool_size",
        "max_replenishment_batches",
        "num_cycles",
        "search_width",
        "exhaustiveness",
        "num_editflow_samples",
        "max_results",
        "llm_model",
        "llm_max_tokens",
        "gp_history_limit",
        "acquisition_beta",
        "acquisition_alpha",
        "acquisition_eta",
        "acquisition_z_clip",
        "initialization_mode",
        "original_target",
        "source_archive_digest",
        "asset_digests",
    )
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def close_metric(saved, measured, context):
    require(
        isinstance(saved, (int, float))
        and not isinstance(saved, bool)
        and math.isfinite(saved)
        and math.isclose(float(saved), measured, rel_tol=1e-9, abs_tol=1e-10),
        f"{context}: recomputed metric differs from saved result",
    )


def configuration_identity(identity, context):
    require(isinstance(identity, dict), f"{context}: scientific identity must be an object")
    shared = {
        key: identity[key]
        for key in sorted(identity)
        if key not in VARIABLE_IDENTITY_FIELDS
    }
    missing = REQUIRED_CONFIGURATION_IDENTITY_FIELDS - set(shared)
    require(not missing, f"{context}: missing configuration identity fields {sorted(missing)}")
    require(
        shared.get("benchmark") == "tdc"
        and shared.get("mock") is False
        and shared.get("max_oracle_calls") == MAX_ORACLE_CALLS,
        f"{context}: invalid TDC configuration identity",
    )
    require(
        isinstance(shared.get("asset_digests"), dict) and shared["asset_digests"],
        f"{context}: missing frozen asset identity",
    )
    return shared


def normalize_expected_identity(raw):
    require(raw is not None, "TDC aggregation requires a predeclared configuration identity")
    require(isinstance(raw, dict), "expected configuration identity must be a JSON object")
    if "configuration_identity" in raw or "oracle_overrides" in raw:
        require(
            set(raw) <= {"configuration_identity", "oracle_overrides"},
            "expected identity wrapper may contain only configuration_identity and oracle_overrides",
        )
        shared = raw.get("configuration_identity")
        oracle_overrides = raw.get("oracle_overrides", {})
    else:
        shared = raw
        oracle_overrides = {}
    require(isinstance(shared, dict), "configuration_identity must be a JSON object")
    require(
        not (VARIABLE_IDENTITY_FIELDS & set(shared)),
        "configuration_identity must omit oracle and seed",
    )
    require(isinstance(oracle_overrides, dict), "oracle_overrides must be a JSON object")
    unknown_oracles = set(oracle_overrides) - set(ORACLES)
    require(not unknown_oracles, f"unknown oracle override(s): {sorted(unknown_oracles)}")
    overrides = {}
    for oracle, value in oracle_overrides.items():
        require(isinstance(value, dict), f"{oracle}: oracle override must be an object")
        extra = set(value) - ORACLE_VARIANT_IDENTITY_FIELDS
        require(not extra, f"{oracle}: non-oracle-specific identity override(s) {sorted(extra)}")
        overrides[oracle] = dict(value)
    missing_base = (
        REQUIRED_CONFIGURATION_IDENTITY_FIELDS
        - set(shared)
        - ORACLE_VARIANT_IDENTITY_FIELDS
    )
    require(
        not missing_base,
        f"configuration_identity is missing shared fields {sorted(missing_base)}",
    )
    by_oracle = {}
    for oracle in ORACLES:
        expected = {**shared, **overrides.get(oracle, {})}
        missing = REQUIRED_CONFIGURATION_IDENTITY_FIELDS - set(expected)
        require(
            not missing,
            f"{oracle}: missing oracle-specific identity field(s) {sorted(missing)}",
        )
        by_oracle[oracle] = configuration_identity(expected, f"{oracle} expected identity")
    return {
        "configuration_identity": dict(shared),
        "oracle_overrides": overrides,
        "by_oracle": by_oracle,
    }


def _expected_for_oracle(expected_identity, oracle):
    if "by_oracle" not in expected_identity:
        expected_identity = normalize_expected_identity(expected_identity)
    require(oracle in expected_identity["by_oracle"], f"{oracle}: missing expected identity")
    return expected_identity["by_oracle"][oracle]


def _check_identity_config(identity, config, folder):
    for key, value in identity.items():
        if key in {"asset_digests", "source_archive_digest", "original_target"}:
            continue
        if key in config:
            require(
                config[key] == value,
                f"{folder}: config differs from scientific identity field {key}",
            )


def verify_result(path, *, expected_identity=None):
    path = Path(path)
    folder = path.parent
    data = read(path)
    config = read(folder / "config.json")
    identity = read(folder / "scientific_identity.json")
    status = read(folder / "status.json")
    ledger = read(folder / "budget.json")
    cache = read(folder / "oracle_cache.json")

    require(status.get("status") == "completed", f"{folder}: unfinished campaign")
    require(
        data.get("mock") is False
        and config.get("mock") is False
        and identity.get("mock") is False
        and cache.get("mock") is False,
        f"{folder}: mock or missing real-mode receipt",
    )
    require(
        data.get("benchmark") == config.get("benchmark") == identity.get("benchmark") == "tdc",
        f"{folder}: not a TDC result",
    )
    oracle = identity.get("oracle")
    seed = identity.get("seed")
    require(oracle in ORACLES, f"{folder}: unknown oracle in scientific identity")
    require(type(seed) is int and seed in SEEDS, f"{folder}: invalid seed in scientific identity")
    require(
        data.get("oracle") == config.get("oracle") == cache.get("oracle") == oracle,
        f"{folder}: oracle identity mismatch",
    )
    require(config.get("seed") == seed, f"{folder}: seed identity mismatch")
    require(
        data.get("official_budget") is True
        and data.get("oracle_budget_exhausted") is True
        and config.get("max_oracle_calls") == identity.get("max_oracle_calls") == MAX_ORACLE_CALLS,
        f"{folder}: not an official full-budget TDC run",
    )
    require(
        isinstance(identity.get("asset_digests"), dict) and identity["asset_digests"],
        f"{folder}: missing frozen asset identity",
    )
    _check_identity_config(identity, config, folder)
    run_identity = configuration_identity(identity, str(folder))
    if expected_identity is not None:
        expected = _expected_for_oracle(expected_identity, oracle)
        require(
            run_identity.get("asset_digests") == expected.get("asset_digests"),
            f"{folder}: frozen asset identity differs from predeclared configuration",
        )
        require(
            run_identity == expected,
            f"{folder}: scientific configuration identity mismatch",
        )

    counters = ledger.get("counters", {})
    limits = ledger.get("limits", {})
    require(data.get("budget") == ledger, f"{folder}: result budget snapshot differs from sibling ledger")
    for key in ("successful_evaluations", "oracle_calls", "expensive_evaluation_attempts"):
        require(
            counters.get(key) == MAX_ORACLE_CALLS and limits.get(key) == MAX_ORACLE_CALLS,
            f"{folder}: incomplete or changed {key} budget ledger",
        )

    entries = cache.get("entries")
    require(
        isinstance(entries, list) and len(entries) == MAX_ORACLE_CALLS,
        f"{folder}: oracle cache must contain exactly {MAX_ORACLE_CALLS} entries",
    )
    seen, scores = set(), []
    for index, entry in enumerate(entries, start=1):
        smiles = entry.get("smiles")
        score = entry.get("score")
        require(
            entry.get("status") == "completed"
            and entry.get("call_index") == index
            and isinstance(smiles, str)
            and smiles
            and smiles not in seen
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score),
            f"{folder}: invalid oracle cache entry at call {index}",
        )
        seen.add(smiles)
        scores.append(float(score))

    require(
        data.get("successful_evaluations") == MAX_ORACLE_CALLS
        and data.get("oracle_calls") == MAX_ORACLE_CALLS
        and data.get("completed_oracle_calls") == MAX_ORACLE_CALLS,
        f"{folder}: result summary does not match completed oracle receipts",
    )
    metrics = data.get("metrics", {})
    auc = top_auc(scores, max_calls=MAX_ORACLE_CALLS)
    close_metric(metrics.get("top10"), top_mean(scores), f"{folder}: top10")
    close_metric(metrics.get("auc_top10_observed"), auc, f"{folder}: auc_top10_observed")
    close_metric(metrics.get("auc_top10"), auc, f"{folder}: auc_top10")
    return (oracle, seed), auc, run_identity


def aggregate(paths, expected_identity=None):
    expected_identity = normalize_expected_identity(expected_identity)
    found = {}
    for path in paths:
        key, auc, _identity = verify_result(path, expected_identity=expected_identity)
        if key in found:
            raise ValueError(f"duplicate oracle/seed {key}")
        found[key] = auc
    expected = {(o, s) for o in ORACLES for s in SEEDS}
    if set(found) != expected:
        raise ValueError(
            f"incomplete or unexpected result grid: missing={sorted(expected - set(found))}, extra={sorted(set(found) - expected)}"
        )
    values = {o: [found[o, s] for s in SEEDS] for o in ORACLES}
    return {
        "task": "reasyn",
        "benchmark": "released_tdc_13",
        "seeds": list(SEEDS),
        "configuration_identity_sha256": json_digest(
            {
                "configuration_identity": expected_identity["configuration_identity"],
                "oracle_overrides": expected_identity["oracle_overrides"],
            }
        ),
        "configuration_identity": expected_identity["configuration_identity"],
        "oracle_overrides": expected_identity["oracle_overrides"],
        "asset_digests_sha256": json_digest(
            expected_identity["configuration_identity"]["asset_digests"]
        ),
        "oracles": {
            o: {"mean": statistics.mean(v), "std": statistics.pstdev(v)}
            for o, v in values.items()
        },
        "mean_auc_top10": statistics.mean(found.values()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument(
        "--identity",
        type=Path,
        required=True,
        help="Predeclared shared scientific_identity JSON with oracle/seed omitted",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = aggregate(args.results, expected_identity=read(args.identity))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
