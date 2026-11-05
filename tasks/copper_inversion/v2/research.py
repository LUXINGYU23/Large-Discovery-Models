"""Allowlisted public export; reuse the established Pi/research-tree runtime."""
import json
import shutil
from pathlib import Path
from ldm_tts.engine.run_store import atomic_json_write
from ..core.data import RESOURCES
from .data import read_problem, save_problem
from .public_catalog import load_catalog


def prepare_resources(root):
    resources = Path(root) / "v2-harness-resources"
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RESOURCES / "harness/image", resources / "image", dirs_exist_ok=True)
    profile = resources / "profiles/researcher"
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RESOURCES / "v2/AGENTS.md", profile / "AGENTS.md")
    shutil.copy2(RESOURCES / "harness/profiles/researcher/PROBE.md", profile / "PROBE.md")
    tools = resources / "tools"; tools.mkdir(exist_ok=True)
    text = (RESOURCES / "harness/tools/copper.mjs").read_text()
    text = text.replace("Read TASK.md, history.json and objective.py. Load arrays from problem.npz and metadata from problem.json.",
        "Read TASK.md, history.json, cases.json and tasks/copper_inversion/v2/objective.py. Load public cases using the supplied data.read_problem.")
    (tools / "copper.mjs").write_text(text)
    return resources


class PublicTrainingExport:
    def __init__(self, catalog_path):
        self.path = Path(catalog_path).resolve()
        self.catalog = load_catalog(self.path)

    def __call__(self, root, _problem):
        if load_catalog(self.path) != self.catalog:
            raise ValueError("training data changed before export")
        public = Path(root) / "public"; public.mkdir(parents=True, exist_ok=True)
        cases = []
        for index, row in enumerate(self.catalog["cases"]):
            problem = read_problem(self.path.parent / row["directory"])
            folder = public / "cases" / f"{index:04d}"
            save_problem(problem, folder)
            cases.append({"case_id": row["case_id"], "directory": str(folder.relative_to(public))})
        atomic_json_write(public / "cases.json", cases)
        package = public / "tasks/copper_inversion/v2"
        package.mkdir(parents=True, exist_ok=True)
        for name in ("__init__.py", "contract.py", "data.py", "fixtures.py", "geometry.py", "objective.py", "baselines.py", "geological_policy.py"):
            shutil.copy2(Path(__file__).parent / name, package / name)
        for name in ("selfcheck.py", "TASK.md"):
            shutil.copy2(RESOURCES / "v2" / name, public / name)
        if not (public / "history.json").exists():
            atomic_json_write(public / "history.json", [])
