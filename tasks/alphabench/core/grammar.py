"""Safe, ordered syntax trees for the complete pinned T3 expression guides."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tokenize

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources"
REGISTRY = json.loads((RESOURCE_ROOT / "operators.json").read_text(encoding="utf-8"))
GRAMMAR_VERSION = REGISTRY["version"] + ":" + hashlib.sha256((RESOURCE_ROOT / "operators.json").read_bytes()).hexdigest()
FIELDS = ("open", "high", "low", "close", "volume", "vwap")
BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
CMPOPS = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}


class ExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class Expression:
    original: str
    dialect: str
    canonical: str
    tree: ast.Expression
    depth: int
    nodes: int

    @property
    def candidate_id(self):
        return hashlib.sha256(f"{GRAMMAR_VERSION}:{self.dialect}:{self.canonical}".encode()).hexdigest()


def parse_expression(expression: str, *, backend: str = "qlib", max_depth: int | None = None,
                     qlib_execution: bool = False) -> Expression:
    if backend not in {"qlib", "assay"}:
        raise ExpressionError("unknown backend")
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 32768:
        raise ExpressionError("expression must be nonempty text of at most 32768 characters")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(expression).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise ExpressionError("malformed expression") from exc
    if any(token.type == tokenize.COMMENT for token in tokens):
        raise ExpressionError("comments are not expressions")
    dialect = "qlib" if any(token.string == "$" or token.string in REGISTRY["qlib"] for token in tokens) else "assay"
    if dialect == "assay" and backend == "qlib":
        raise ExpressionError("Assay syntax requires the Assay backend")
    # Replace dollar tokens only outside literals; reserved names cannot be supplied by a caller.
    if any(token.type == tokenize.NAME and token.string.startswith("__field_") for token in tokens):
        raise ExpressionError("reserved identifier")
    transformed = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.string == "$":
            index += 1
            if index >= len(tokens) or tokens[index].type != tokenize.NAME:
                raise ExpressionError("dollar must precede a field name")
            transformed.append((tokenize.NAME, "__field_" + tokens[index].string))
        else:
            transformed.append((token.type, token.string))
        index += 1
    try:
        tree = ast.parse(tokenize.untokenize(transformed).strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ExpressionError("malformed expression") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 2048:
        raise ExpressionError("expression exceeds 2048 syntax nodes")

    def visit(node, role="value"):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
            value = node.operand.value
            if type(value) not in (int, float):
                raise ExpressionError("numeric unary operand required")
            return visit(ast.Constant(value if isinstance(node.op, ast.UAdd) else -value), role)
        if role != "value":
            if not isinstance(node, ast.Constant):
                raise ExpressionError(f"{role} requires a literal")
            value = node.value
            if role == "group":
                if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
                    raise ExpressionError("invalid group literal")
                return json.dumps(value), 0
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ExpressionError("finite numeric literal required")
            if role in {"window", "lag"} and (type(value) is not int or value < (0 if role == "lag" else 1)):
                raise ExpressionError(f"{role} must be a causal {'nonnegative' if role == 'lag' else 'positive'} integer")
            if role == "quantile" and not 0 <= value <= 1:
                raise ExpressionError("quantile must be in [0, 1]")
            return repr(value), 0
        if isinstance(node, ast.Constant):
            return visit(node, "number")
        if isinstance(node, ast.Name):
            field = node.id.removeprefix("__field_")
            allowed = FIELDS[:5] if dialect == "qlib" else FIELDS
            if field in allowed and node.id == ("__field_" + field if dialect == "qlib" else field):
                return ("$" if dialect == "qlib" else "") + field, 0
            macro = re.fullmatch(r"adv([1-9][0-9]*)", node.id)
            if macro and dialect == "assay":
                return f"ts_mean(volume,{int(macro[1])})", 1
            raise ExpressionError(f"unknown or mixed-dialect field: {node.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            signature = REGISTRY[dialect].get(name)
            if signature is None:
                raise ExpressionError(f"unknown or mixed-dialect operator: {name}")
            args = list(node.args)
            for keyword in node.keywords:
                if name != "safe_div" or keyword.arg != "fill" or len(args) != 2:
                    raise ExpressionError(f"unsupported or duplicated keyword for {name}")
                args.append(keyword.value)
            required = sum(not item.endswith("?") for item in signature)
            if not required <= len(args) <= len(signature):
                raise ExpressionError(f"{name} expects {required}..{len(signature)} arguments")
            values = [visit(arg, param.removesuffix("?").split(":")[-1]) for arg, param in zip(args, signature)]
            return f"{name}({','.join(value for value, _ in values)})", 1 + max((depth for _, depth in values), default=0)
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            left, ld = visit(node.left)
            right, rd = visit(node.right)
            return f"({left}{BINOPS[type(node.op)]}{right})", 1 + max(ld, rd)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMPOPS:
            left, ld = visit(node.left)
            right, rd = visit(node.comparators[0])
            return f"({left}{CMPOPS[type(node.ops[0])]}{right})", 1 + max(ld, rd)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value, depth = visit(node.operand)
            if qlib_execution and dialect == "qlib":
                return (f"Mul(-1,{value})", depth + 1) if isinstance(node.op, ast.USub) else (value, depth)
            return f"({'-' if isinstance(node.op, ast.USub) else '+'}{value})", depth + 1
        raise ExpressionError(f"unsupported syntax: {type(node).__name__}")

    try:
        canonical, depth = visit(tree.body)
    except (RecursionError, OverflowError) as exc:
        raise ExpressionError("expression nesting is too deep") from exc
    if max_depth is not None and depth > max_depth:
        raise ExpressionError(f"operator depth {depth} exceeds {max_depth}")
    canonical_tree = ast.parse(re.sub(r"\$([A-Za-z]+)", r"__field_\1", canonical), mode="eval")
    return Expression(expression, dialect, canonical, canonical_tree, depth, len(list(ast.walk(canonical_tree))))
