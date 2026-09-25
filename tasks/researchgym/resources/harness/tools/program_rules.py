"""Static candidate-program rules shared by task admission and the guest checker.

Standard library only: the host task validator imports this exact file, and the
Harness tool runs it unchanged in the sidecar. Passing these checks means the
program has the slot's shape; it says nothing about whether it trains or scores.
"""

from __future__ import annotations

import ast
import hashlib
from collections import Counter


def _error(code, message, hint):
    return {"code": code, "message": message, "hint": hint}


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def canonical_key(source):
    """Identity that ignores comments, formatting, and docstrings."""
    tree = _strip_docstrings(ast.parse(source))
    return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode("utf-8")).hexdigest()


def check_program(source, *, entry, rules, forbidden_prefixes=()):
    """Return a list of indexed-free error dicts; empty means statically admissible."""
    if not isinstance(source, str):
        return [_error("program_not_string", "program must be a string of Python source",
                       "Store the full module source as one JSON string.")]
    program = source.strip()
    if not program:
        return [_error("empty_program", "program is empty", "Write the complete module source.")]
    errors = []
    if len(program) > rules["max_program_chars"]:
        errors.append(_error("program_too_long", f"program has {len(program)} characters; the limit is "
                             f"{rules['max_program_chars']}", "Remove unused code or helpers."))
    lines = [line.strip() for line in program.splitlines() if line.strip()]
    if len(lines) > rules["max_program_lines"]:
        errors.append(_error("program_too_many_lines", f"program has {len(lines)} non-empty lines; the limit "
                             f"is {rules['max_program_lines']}", "Simplify the method."))
    if lines:
        line, repeats = Counter(lines).most_common(1)[0]
        if repeats > rules["max_repeated_line"]:
            errors.append(_error("degenerate_repetition", f"one line repeats {repeats} times: {line[:60]!r}",
                                 "Replace repeated statements with a loop or remove them."))
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        errors.append(_error("syntax_error", f"line {exc.lineno}: {exc.msg}", "Run python -m py_compile on the file."))
        return errors
    symbol = entry["symbol"]
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and node.name == symbol]
    if not matches:
        errors.append(_error("missing_entry", f"no top-level {symbol} is defined",
                             f"Define {symbol} at module level exactly as the case interface states."))
    else:
        node = matches[-1]
        if entry["kind"] == "function":
            if not isinstance(node, ast.FunctionDef):
                errors.append(_error("entry_kind", f"{symbol} must be a synchronous function",
                                     f"Use def {symbol}(...)."))
            else:
                names = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                if names[:len(entry["arguments"])] != list(entry["arguments"]) or \
                        len(node.args.args) + len(node.args.posonlyargs) - len(node.args.defaults) > len(entry["arguments"]):
                    errors.append(_error("entry_arguments", f"{symbol} arguments are {names}; expected "
                                         f"{entry['arguments']}", "Keep the exact positional signature."))
        else:
            if not isinstance(node, ast.ClassDef):
                errors.append(_error("entry_kind", f"{symbol} must be a class", f"Use class {symbol}(...)."))
            else:
                defined = {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
                missing = [name for name in entry["methods"] if name not in defined]
                if missing:
                    errors.append(_error("missing_method", f"{symbol} does not define {', '.join(missing)}",
                                         "Implement every method named by the case interface."))
    forbidden_roots = set(rules["forbidden_import_roots"])
    forbidden_calls = set(rules["forbidden_calls"])
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                errors.append(_error("relative_import", "relative imports are not allowed",
                                     "Import through absolute task module paths."))
            if node.module:
                modules = [node.module]
        for module in modules:
            root = module.split(".")[0]
            prefix = next((p for p in forbidden_prefixes if module == p or module.startswith(p.rstrip(".") + ".")
                           or (p.endswith("_") and module.startswith(p))), None)
            if root in forbidden_roots or prefix:
                errors.append(_error("forbidden_import", f"import of {module} is not allowed",
                                     "Implement the method directly with the allowed libraries."))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            errors.append(_error("forbidden_call", f"{node.func.id}() is not allowed",
                                 "Write the logic as ordinary Python code."))
    return errors
