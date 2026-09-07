"""Bounded official projector process and deterministic infrastructure fixture."""

from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from ldm_tts.engine.run_store import atomic_json_write


class Projector:
    def __init__(self, args, run_dir):
        self.args = args
        self.run_dir = Path(run_dir)
        self.before_project = None

    def project(self, targets, *, sampling_seed, identity):
        folder = self.run_dir / "projections" / identity
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "result.json"
        request = {
            "asset_digests": getattr(self.args, "asset_digests", {}),
            "targets": targets,
            "sampling_seed": sampling_seed,
            "model_paths": [str(p.resolve()) for p in self.args.model_paths],
            "fpindex": str(self.args.fpindex.resolve()),
            "rxn_matrix": str(self.args.rxn_matrix.resolve()),
            "additional_fpindex": str(self.args.additional_fpindex.resolve())
            if self.args.additional_fpindex
            else None,
            "device": self.args.device,
            "settings": {
                "search_width": self.args.search_width,
                "exhaustiveness": self.args.exhaustiveness,
                "num_cycles": self.args.num_cycles,
                "num_editflow_samples": self.args.num_editflow_samples,
                "max_results": self.args.max_results,
                "exact_break": True,
                "time_limit": self.args.projection_time_limit,
            },
        }
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        request_path = folder / "request.json"
        if output.exists() and request_path.exists():
            prior = json.loads(request_path.read_text())
            if prior != request:
                raise ValueError("projection resume request mismatch")
            rows = json.loads(output.read_text())["rows"]
            self._validate(rows, targets)
            return rows, str(output.relative_to(self.run_dir))
        if self.before_project:
            self.before_project(len(targets))
        atomic_json_write(request_path, request)
        if self.args.mock:
            rows = []
            for target in targets:
                rows.append(
                    {
                        "target": target,
                        "smiles": target,
                        "synthesis": target,
                        "num_steps": 0,
                        "pathway_verified": True,
                        "mock_fixture": True,
                    }
                )
            atomic_json_write(
                output,
                {
                    "rows": rows,
                    "target_count": len(targets),
                    "mock": True,
                    "request_sha256": digest,
                },
            )
        else:
            command = [
                self.args.evaluator_python,
                str(Path(__file__).with_name("projection_worker.py")),
                "--request",
                str(request_path.resolve()),
                "--output",
                str(output.resolve()),
            ]
            with (folder / "worker.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=self.args.upstream_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.args.projection_timeout,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"ReaSyn projector exited {completed.returncode}; see {folder / 'worker.log'}"
                )
            if not output.exists():
                raise RuntimeError("ReaSyn projector omitted result.json")
        data = json.loads(output.read_text())
        rows = data["rows"]
        self._validate(rows, targets)
        return rows, str(output.relative_to(self.run_dir))

    def _validate(self, rows, targets):
        from .chemistry import canonicalize

        for row in rows:
            if (
                row.get("pathway_verified") is not True
                or row["target"] not in targets
                or not row.get("synthesis")
            ):
                raise ValueError("projector output failed trusted-pathway boundary")
            canonicalize(row["smiles"], mock=self.args.mock)
