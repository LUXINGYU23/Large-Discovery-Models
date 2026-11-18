#!/bin/sh
set -eu
python3 - <<'PY'
import io
import numpy as np
assert len("SNGDWGSPGTV") == 11
buffer = io.BytesIO()
np.savez(buffer, history_utilities=np.asarray([1.0, 2.0]))
buffer.seek(0)
with np.load(buffer, allow_pickle=False) as saved:
    assert saved["history_utilities"].tolist() == [1.0, 2.0]
PY
printf 'cdrh3_guest_ok\n' > /workspace/research/proof.txt
