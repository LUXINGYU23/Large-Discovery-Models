# iron_mind

Closed-loop optimization over source-pinned reaction-condition tables.

## Current Status

The task is registered with the shared runner, but its source contract and
campaign workflow are still under implementation. Both mock and real campaigns
fail explicitly until the task-specific core components are available. At this
stage, validate registration with `scripts/validate_tasks.py`; do not treat the
generated mock config as a completed campaign.

The implementation will replace this draft boundary with a single shared-engine
workflow. Keep `qualification` set to `draft` until a real seed and tiny
LDM-selected evaluation pass.
