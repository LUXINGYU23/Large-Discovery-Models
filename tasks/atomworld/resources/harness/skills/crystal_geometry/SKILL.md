---
name: crystal_geometry
description: Public CIF manipulation, Cartesian geometry and syntax auditing.
---
Use get_public_task to obtain the exact original CIF and public action. Copy that CIF to /workspace/input.cif with write, preserving atom order. Use get_geometry_contract for the delivered bounded tool semantics. The guest installs atomworld_tools, ASE, NumPy and pymatgen; use Python in bash for scratch analysis.

```python
from pathlib import Path
from atomworld_tools import execute_operations
source = Path('/workspace/input.cif').read_text()
# Replace this illustrative operation with the actual requested public action.
plan = [{'op': 'move', 'index': 0, 'd_pos': [1, 0, 0]}]
output = execute_operations(source, plan)
Path('/workspace/answer.cif').write_text(output)
```

Inspect species, cell and Cartesian coordinates with ASE. For direct edits use ASE operations or NumPy; do not write arbitrary code into an operation plan. Read output with pymatgen CifParser.from_str(output).parse_structures(primitive=False, check_occu=False) and check the requested invariants. Parsing is a public syntax check, never an official correctness measurement. The bounded tools cover the ten paper actions; unsupported extension operations may be implemented with scratch ASE code using the public instruction. No task source, hidden targets, oracle errors or other questions are accessible. Write the complete raw CIF to /workspace/answer.cif and call submit_answer with artifact_path="answer.cif" and a concise English rationale. The host verifies the immutable file digest. Use get_public_history for the compact draft index; request detail="detailed" and draft_ids for earlier CIFs and rationales. On rejection, repair the reported syntax or schema defect and submit again before the bounded repair allowance expires.
