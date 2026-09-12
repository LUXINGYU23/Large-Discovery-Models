# Construct a Working Panel Incrementally

Load `get_task_context`'s `guest_file.path` with JSON; it contains the complete
original sequence and exact editable positions. For a measured parent, copy
its `mutations` from the detailed history file. Start with that background
and apply only the replacements your hypothesis needs. This avoids manually
padding or transcribing entire sequence windows.

Adapt this example in a short workspace script. The context path is a tool
result; `parent_patch` and `placements` describe your chosen design. Placements
use absolute zero-based coordinates and concrete A/C/G/T strings.

```python
import json
from pathlib import Path
from Bio.Seq import MutableSeq

context = json.loads(Path(context_path).read_text())["paired_start"]
reference = context["start_sequence"]
editable = set(context["editable_positions"])

def build_patch(parent_patch, placements):
    sequence = MutableSeq(reference)
    for edit in parent_patch:
        position, base = edit["position"], edit["base"]
        if position not in editable or base not in ("A", "C", "G", "T"):
            raise ValueError("Invalid measured-parent edit")
        sequence[position] = base
    written = {}
    for position, bases in placements:
        if not isinstance(position, int) or not bases or set(bases) - set("ACGT"):
            raise ValueError("A placement needs an integer position and concrete DNA")
        for index, base in enumerate(bases, position):
            if index not in editable:
                raise ValueError(f"Position {index} is outside the editable mask")
            if index in written and written[index] != base:
                raise ValueError(f"Conflicting module writes at position {index}")
            sequence[index] = base
            written[index] = base
    assert len(sequence) == len(reference)
    patch = [{"position": i, "base": str(sequence[i])}
             for i in sorted(editable) if sequence[i] != reference[i]]
    if not patch:
        raise ValueError("The design reproduces the already-measured original start")
    return patch
```

This builds the final difference from the original start, including inherited
parent edits outside the newly changed window. A no-op write is omitted from
the patch. Compatible overlapping modules are allowed; conflicting writes
need an explicit redesign. For a full-window rewrite, compute its length in
code and check the assembled module before placement. Do not silently truncate
an overlong design or claim a shorter insert replaced an entire window.

Run the smallest useful construct first. Add one valid candidate object at a
time to a saved draft, with its actual change summary and testable rationale.
Use exact reconstructed identities for history exclusion and the turn's
uniqueness rule. The task tools and `submit_candidates` remain authoritative;
this construction example does not replace those checks or choose candidates.

Keep per-design biological diagnostics separate from construction. If a
consensus match fails, inspect the string, strand and source; revise the
affected claim or design. If fixed modules already contain a supposedly
forbidden motif, changing a spacer cannot make the whole sequence motif-free.
Such exclusions are hypotheses, not benchmark rules. Check their feasibility
before a random search, and use an explicit time bound for any local search.
Keep unrelated validated entries while reconsidering the failed design.

Reach the requested panel size early with scientifically justified designs;
continue improving entries if useful time remains. Save concise diagnostics
and unresolved questions in notes. Submit only the exact candidate schema,
after checking the complete final array, count, uniqueness and eligibility.
