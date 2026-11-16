"""Identity of shared Python code used by the controller and evaluator."""

import hashlib
from pathlib import Path

from ..core.execution import digest


def shared_identity():
    repo = Path(__file__).resolve().parents[3]
    package = repo / "ldm_tts"
    return digest({str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(package.rglob("*.py")) if "__pycache__" not in path.parts})
