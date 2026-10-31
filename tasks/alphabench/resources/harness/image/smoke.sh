#!/bin/sh
set -eu
python3 - <<'PY'
import json
from pathlib import Path

path = Path('/workspace/candidates.json')
path.write_text(json.dumps({'candidates': [{'name': 'smoke', 'expression': 'Mean($close,5)'}]}))
assert json.loads(path.read_text())['candidates'][0]['expression'] == 'Mean($close,5)'
Path('/workspace/research/proof.txt').write_text('t3_guest_ok\n')
PY
