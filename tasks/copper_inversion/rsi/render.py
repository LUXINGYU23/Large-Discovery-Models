"""Deterministic measurements-only context and presentation source."""

import ast
from pathlib import Path

from .ledger import validate


class _RemoveDocstrings(ast.NodeTransformer):
    def _body(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
            if not node.body:
                node.body = [ast.copy_location(ast.Pass(), node)]
        return node

    visit_Module = _body
    visit_ClassDef = _body
    visit_FunctionDef = _body
    visit_AsyncFunctionDef = _body


def stripped_source(source):
    """Remove comments and docstrings for display; never change executed bytes."""
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    tree = _RemoveDocstrings().visit(ast.parse(text))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def render(records):
    rows = sorted((validate(dict(r)) for r in records), key=lambda r: r["seq"])
    if len({r["seq"] for r in rows}) != len(rows):
        raise ValueError("duplicate record sequence")
    ids = [r["program_id"][:12] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("source prefix collision")
    last = max((r["round"] for r in rows), default=0)
    lines = [f"# history v1  records={len(rows)}  rounds=0..{last}",
             "# id            ep rd br  status    risk      legal  s/solve  sel"]
    for row in rows:
        branch = "-" if row["branch"] == -1 else str(row["branch"])
        line = (f"{row['program_id'][:12]:12}  {row['epoch']:2d} {row['round']:2d} "
                f"{branch:>2}  {row['status']:<8}  {row['risk']:.6f}  "
                f"{row['legal_rate']:.3f}  {row['seconds_per_solve']:.2f}  {row['selection']}")
        if row["message"]:
            line += "  " + row["message"]
        lines.append(line)
    lines.append("# per-case risk, 10 lowest-risk valid programs")
    for row in sorted((r for r in rows if r["status"] == "valid"),
                      key=lambda r: (r["risk"], r["program_id"]))[:10]:
        values = " ".join(f"{key}={value:.4f}" for key, value in sorted(row["risk_by_case"].items()))
        lines.append(f"{row['program_id'][:12]}  {values}")
    return "\n".join(lines) + "\n"


def visible_history(records, *, branch, epoch_start_round):
    """Pooled records at the boundary plus only this branch's later records."""
    return [r for r in records if r["round"] < epoch_start_round or r["branch"] == branch]


def write_sources(directory, records, candidate_root):
    """Publish only stripped sources in the supplied branch's view."""
    directory, candidate_root = Path(directory), Path(candidate_root)
    directory.mkdir(parents=True, exist_ok=True)
    expected = {r["program_id"][:12] + ".py" for r in records}
    if len(expected) != len(records):
        raise ValueError("source prefix collision in view")
    for old in directory.glob("*.py"):
        if old.name not in expected:
            old.unlink()
    for row in records:
        sha = row["program_id"]
        source = candidate_root / "candidates" / sha / "solver.py"
        if source.is_symlink():
            raise ValueError("candidate source cannot be a symlink")
        from ..core.candidate import validate_source
        body = source.read_bytes()
        if validate_source(body) != sha:
            raise ValueError("candidate source changed")
        (directory / (sha[:12] + ".py")).write_text(stripped_source(body), encoding="utf-8")
