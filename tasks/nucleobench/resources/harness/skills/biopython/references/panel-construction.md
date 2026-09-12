# Compile Chosen Sequence Designs

Write the complete chosen panel as data with the native `write` tool. Pass its
path to `compile_candidate_panel({"designs_path":"designs.json"})`. The tool
performs fixed-length replacements, inherits the measured parent, and writes
the existing mutation-patch submission format to `/workspace/candidates.json`.
It does not choose designs or impose biological motif filters.

The file has one field, `designs`. Each design contains:

- `placements`: an array of `{ "start": absolute_zero_based_position,
  "bases": "ACGT" }` objects. Choose positions from the exact editable mask;
  each DNA string replaces a span of the same length. Other bases are retained.
- `parent_candidate_id`: optional exact measured ID. Omit for the paired start.
- `change_summary` and `rationale`: concise English notes describing the actual
  change and its testable hypothesis or control purpose.
- `comparison_candidate_ids`: optional array of exact measured IDs.

Use a one-base string for a point edit and a longer string for a chosen module.
There are no implicit spacers, padding, insertions or deletions. Compatible
overlapping placements are allowed; conflicting writes are rejected. Reverting
a parent base to the original start is allowed and removes it from the patch.
All inherited parent edits outside the replacements remain in the candidate.

Keep all intended designs in the input on every call. The compiler rebuilds
the complete draft and checks each design independently. `rejected` names the
input `design_index` and concrete reason; `output_design_indices` maps retained
candidates back to that input. Fix those designs and recompile. A failed design
does not prevent the other valid designs from being written.

Check `candidate_count`, `unique_candidate_count`, and `duplicate_design_groups`
against the turn contract. The compiler preserves repetitions so permitted
occurrence weights remain intact; replace repeats when this turn requires
uniqueness. Only measured history excludes cross-round reuse.

Reach a complete draft before optional extended analysis. Motif scores and
unintended matches can motivate changing a placement or qualifying its rationale;
they do not require a motif-free background or a perfect shuffled control.
Finally call `submit_candidates({"artifact_path":"candidates.json"})`.
Compilation alone neither submits nor evaluates a candidate, and incomplete
panels still fail the authoritative submission contract.
