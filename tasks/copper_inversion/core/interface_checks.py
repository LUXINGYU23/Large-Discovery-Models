"""Read-only public fixtures, checked by the controller outside candidate processes."""

import json
from pathlib import Path
from ldm_tts.engine.run_store import atomic_json_write
from .data import load_problem, save_problem
from .execution import digest, freeze_model
from .geometry import read_model, signed_distance


class InterfaceChecks:
    def __init__(self, root, execution, case):
        self.root = Path(root)
        self.execution = execution
        self.case = case

    def __call__(self, payload):
        # Two fixtures exercise absence/presence of contact labels.
        for contacts in (False, True):
            p = load_problem(self.case, fixture=True)
            if contacts:
                index = tuple(n // 3 for n in p["shape"])
                p["observations"][0]["values"][index] = 0.5
                p["observations"][0]["mask"][index] = True
            inputs = self.root / "interface-inputs" / str(contacts)
            save_problem(p, inputs)
            atomic_json_write(
                inputs / "budget.json",
                {
                    "seed": 47,
                    "wall_time_seconds": 30,
                    "cpu_count": 1,
                    "memory_mb": 2048,
                },
            )
            identity = {
                "source": payload["sha256"],
                "fixture": digest(
                    {"case": self.case, "contacts": contacts, "version": 1}
                ),
            }
            root, r = self.execution.run(
                identity,
                phase="interface",
                inputs=inputs,
                timeout=30,
                source=(self.root / payload["source_path"]).parent,
            )
            if r["exit_code"] or r["oom"] or r["reason"]:
                log = root / "log.txt"
                diagnostic = "\n".join(line for line in log.read_text(errors="replace").splitlines()
                    if not line.startswith("COPPER_RESOURCE_V1 "))[-1200:] if log.exists() else ""
                raise ValueError(
                    f"interface fixture contacts={contacts} failed: {r['reason'] or r['exit_code']}. "
                    f"Untrusted candidate output: {diagnostic}"
                )
            field = read_model(root / "output/model.npy", p["shape"])
            signed_distance(field, p["shape"], p["spacing"])
        return True
