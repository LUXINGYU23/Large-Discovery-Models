# LDM Data Pipeline Contract

## Contents

- [Artifacts](#artifacts)
- [Public Interface](#public-interface)
- [Task Hook Status](#task-hook-status)
- [Collection Boundary](#collection-boundary)
- [Environment](#environment)
- [Reasoning Policy](#reasoning-policy)
- [Known Limitations](#known-limitations)

## Artifacts

| File | Role |
| --- | --- |
| `ldm_ir.jsonl` | Immutable `ldm-2.0` source, one accepted teacher action per row. |
| `ldm_sft.jsonl` | Convenience Alpaca rendering of collected IR. |
| `ldm_ir_augmented.jsonl` | Derived IR with expert text in `action.reasoning`. |
| `ldm_sft_augmented.jsonl` | Alpaca rendering used for reasoning-augmented SFT. |
| `augmentation.checkpoint.jsonl` | Resumable expert-response cache; never training input. |
| `dataset_info.json` | LlamaFactory registration for the selected SFT filename. |

An IR row has `schema_version`, `task`, `search_state`, `request`, and `action`.
It may also have `raw_context` and `collection`. The renderer maps
`task + search_state + request + allowed raw_context` to `instruction` and maps
`action` to JSON in `output`. It never renders `collection`.

## Public Interface

Import task hooks from `ldm_tts.data`:

```python
from ldm_tts.data import (
    DataCollectionSink,
    make_complete_design_ir,
    make_parameter_edit_ir,
    render_record,
    validate_ir_record,
)
```

Use `complete_design` for full molecules or sequences. Use `parameter_edits` for
actions that edit a parent state or activate design-space dimensions. Valid
action types are `propose`, `expand_design_space`, and `add_new_parameter`; each
action must appear in `request.allowed_actions`.

## Task Hook Status

| Task | Runtime status | Mapping |
| --- | --- | --- |
| Small molecule | Implemented for accepted direct-LLM attempts in `tasks/small_molecule/core/ldm_tilted_case2/trace.py`. | `smallmol`, `complete_design`, `propose`. |
| nanoGPT | Implemented in `tasks/nanogpt/core/workflow.py` for validated operation edits and expansion-schema actions. | `nanogpt`, `parameter_edits`, `propose`, `expand_design_space`, or `add_new_parameter`. |
| Antibody | Implemented in `tasks/antibody/core/ldm_light/ldm_acq.py` for validated direct LLM sequence actions; fallback and policy-DSL decisions are rejected. | `protein`, `complete_design`, `propose`; sequence rows use `reasoning_available:false`. |
| SynthonBench Harness | Native sessions and redacted provider transport are retained under the run's Harness artifact root. No `ldm-2.0` converter is implemented. | Raw multi-turn research trace only; do not register as SFT data. |

The small-molecule adapter is intentionally narrow. Seed-plan and ReaSyn
analogue prompts are not collected by the current direct-LLM adapter. Each hook
is method-specific, so confirm that the selected execution path reaches it
before spending model or evaluator budget.

## Collection Boundary

Freeze the record after parsing and validation identifies the teacher's accepted
action. Preserve the teacher's full action even if it contains multiple
candidates. The write may happen later to attach BO or evaluator outcomes, but
the model-visible state and action must remain the pre-selection versions.

```text
build teacher-visible state
call teacher
parse and reject invalid attempts
construct accepted action
freeze model-visible IR
run selection/evaluation
append frozen IR, retaining outcomes only as collection metadata
```

Do not turn a selected post-BO candidate into a teacher target unless a separate,
explicit distillation transform defines and labels that policy.

Harness traces require a separate conversion contract because they contain
multiple model turns, tool calls, rejected provisional submissions, and one
terminal accepted minibatch. Raw trace retention alone does not identify the
training input/output boundary or prove that task-private and post-evaluation
state is absent.

## Environment

`DataCollectionSink.from_env` reads:

| Variable | Meaning |
| --- | --- |
| `LDM_DATA_COLLECTION_ENABLED` | Truthy switch. An explicit collection directory also enables the sink. |
| `LDM_DATA_COLLECTION_DIR` | Shared output directory; otherwise the task hook may choose a run-local default. |
| `LDM_DATA_COLLECTION_RENDER` | `prose` or `json`; use `prose` for ordinary SFT. |
| `LDM_DATA_COLLECTION_STRIP_PARENT_ARTIFACT` | Omits large parent artifacts; use only when context limits require it. |

Protected expert credentials use `LLM_BASE_URL`, `LLM_MODEL_NAME`, and
`LLM_API_KEY`. Keep the API key process-only. A protected JSON file with `url`,
`model`, and `key` may be loaded with `jq`, but must never be copied, committed,
or printed.

## Reasoning Policy

The expert receives the rendered model-visible context and a copy of the
accepted action whose `reasoning` is set to `null`. The result is stored as one
string in `action.reasoning`. Augmentation must not change the payload, action
type, candidate rationales, or summary.

Skip existing reasoning and records marked `reasoning_available:false` by
default. The production `ldm-2.0` path keeps reasoning inside the JSON action;
`<think>` wrapping is only a compatibility fallback for unstructured legacy
Alpaca outputs.

## Known Limitations

- Runtime collection is action-level, not candidate-level. One row may contain
  a full proposal list.
- The collector does not perform empirical-acquisition-tilt sampling or top-k
  candidate filtering. Selection results are audit metadata.
- Small-molecule and protein IR normally have `surrogate_feedback:null`; do not
  claim their rationales were conditioned on hidden acquisition values.
- Protein sequence-only traces have no supported rationale source and are
  skipped by normal augmentation.
- `expand_design_space` is rare in existing nanoGPT data and requires deliberate
  collection or oversampling if it should be learned.
- Split by whole run, seed, or antigen to avoid trajectory leakage.
- Native Harness sessions are not accepted-action IR and have no released
  `ldm-2.0` conversion path.
