"""Code identity, immutable snapshots and shared candidate admission."""

import ast
import hashlib
from pathlib import Path
from ldm_tts.contracts import Candidate, CandidateRejection

MAX_SOURCE_BYTES = 1024 * 1024


def validate_source(source):
    if not source or len(source) > MAX_SOURCE_BYTES:
        raise ValueError("empty or oversized solver")
    text = source.decode("utf-8")
    tree = ast.parse(text, "solver.py")
    if not any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve"
        for n in tree.body
    ):
        raise ValueError("solver must define solve(problem, budget, seed)")
    return hashlib.sha256(source).hexdigest()


def seal(root, source):
    sha = validate_source(source)
    p = Path(root) / "candidates" / sha / "solver.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_bytes() != source:
        raise ValueError("candidate snapshot changed")
    if not p.exists():
        p.write_bytes(source)
    p.chmod(0o444)
    return {"sha256": sha, "source_path": str(p.relative_to(root))}


class Domain:
    def __init__(self, root):
        self.root = Path(root)

    def admit(self, proposal):
        try:
            p = proposal.payload
            path = self.root / p["source_path"]
            if path.is_symlink():
                raise ValueError("symlink source path")
            path = path.resolve()
            if not path.is_relative_to(self.root.resolve()) or path.is_symlink():
                raise ValueError("invalid source path")
            if validate_source(path.read_bytes()) != p["sha256"]:
                raise ValueError("candidate digest mismatch")
            return Candidate(
                p["sha256"], p, p["sha256"], proposal.source, metadata=proposal.metadata
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            SyntaxError,
            UnicodeError,
        ) as exc:
            return CandidateRejection("invalid_source", str(exc), proposal.source)
