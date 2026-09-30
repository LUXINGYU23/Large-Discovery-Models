"""Guide-specific extensions over Qlib's existing numerical operators.

Mask follows the pinned T3 guide's Mask(condition, value), rather than Qlib's
instrument-redirection operator. This semantic difference is a protocol delta.
"""

import numpy as np
from qlib.data.ops import ElemOperator, If, NpElemOperator


class Sqrt(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "sqrt")


class Tanh(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "tanh")


class Clip(ElemOperator):
    def __init__(self, feature, lower, upper):
        super().__init__(feature)
        self.lower, self.upper = lower, upper

    def __str__(self):
        return f"Clip({self.feature},{self.lower},{self.upper})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(instrument, start_index, end_index, *args).clip(self.lower, self.upper)


class Mask(If):
    def __init__(self, condition, feature):
        super().__init__(condition, feature, np.nan)


CUSTOM_OPS = [Sqrt, Tanh, Clip, Mask]
