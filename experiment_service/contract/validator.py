"""A minimal JSON Schema subset validator (standard library only).

Supported keywords: type (name or list), required, properties, enum, pattern (re.search), items,
additionalProperties (bool or schema), minLength, minimum and local ``$ref`` ("#/..."). Annotation keywords
($schema, $id, $comment, $defs, title, description, x-*) are ignored; any other keyword raises ``SchemaError``,
so a schema can never silently use a constraint this validator does not enforce.
"""
from __future__ import annotations

import re

ANNOTATIONS = frozenset({"$schema", "$id", "$comment", "$defs", "title", "description"})
KEYWORDS = frozenset({"type", "required", "properties", "enum", "pattern", "items", "additionalProperties",
                      "minLength", "minimum", "$ref"})
_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


class SchemaError(ValueError):
    """The schema itself uses something this validator does not support."""


def _resolve(ref, root):
    if not ref.startswith("#/"):
        raise SchemaError(f"only local $ref is supported: {ref!r}")
    node = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _same(a, b):
    return type(a) is type(b) and a == b  # no True == 1 confusion


def check(instance, schema, root=None, path="$") -> list[str]:
    """Return the list of violations of ``instance`` against ``schema`` (empty when valid)."""
    root = schema if root is None else root
    if schema is True or schema == {}:
        return []
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: a schema must be an object, got {schema!r}")
    unknown = {k for k in schema if k not in KEYWORDS and k not in ANNOTATIONS and not k.startswith("x-")}
    if unknown:
        raise SchemaError(f"unsupported keyword(s) {sorted(unknown)} at {path}")
    if "$ref" in schema:
        return check(instance, _resolve(schema["$ref"], root), root, path)
    errors = []
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        for name in names:
            if name not in _TYPES:
                raise SchemaError(f"unsupported type {name!r} at {path}")
        if not any(_TYPES[name](instance) for name in names):
            return [f"{path}: expected {'|'.join(names)}, got {type(instance).__name__} {str(instance)[:60]!r}"]
    if "enum" in schema and not any(_same(instance, option) for option in schema["enum"]):
        errors.append(f"{path}: {str(instance)[:60]!r} is not one of {schema['enum']}")
    if isinstance(instance, str):
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: {instance[:80]!r} does not match {schema['pattern']!r}")
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
    if _TYPES["number"](instance) and instance < schema.get("minimum", instance):
        errors.append(f"{path}: {instance} is below {schema['minimum']}")
    if isinstance(instance, dict):
        errors += [f"{path}: missing required property {name!r}" for name in schema.get("required", ())
                   if name not in instance]
        properties, extra = schema.get("properties", {}), schema.get("additionalProperties", True)
        for name, value in instance.items():
            if name in properties:
                errors += check(value, properties[name], root, f"{path}.{name}")
            elif extra is False:
                errors.append(f"{path}: unexpected property {name!r}")
            elif extra is not True:
                errors += check(value, extra, root, f"{path}.{name}")
    if isinstance(instance, list) and "items" in schema:
        for index, value in enumerate(instance):
            errors += check(value, schema["items"], root, f"{path}[{index}]")
    return errors
