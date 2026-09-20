"""Public geometry-only interface; neither targets nor evaluators are accepted."""

from .executor import OperationError, execute_operations
from .schema import OPERATIONS_SCHEMA, OPERATION_INSTRUCTIONS

__all__ = ["OperationError", "execute_operations", "OPERATIONS_SCHEMA", "OPERATION_INSTRUCTIONS"]
