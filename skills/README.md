# Repository Agent Skills

These Skills guide repository development and execution. Research Agents load
their task-owned Skills from `tasks/<task_id>/resources/harness/skills/`, not
from this directory.

| Skill | Purpose |
| --- | --- |
| [`collect-ldm-data`](collect-ldm-data/SKILL.md) | Collect, augment, render, and validate `ldm-2.0` fine-tuning data. |
| [`register-ldm-task`](register-ldm-task/SKILL.md) | Scaffold, implement, register, and verify a direct, research-Harness, or Harness-Compiled LDM task adapter. |
| [`run-ldm-task`](run-ldm-task/SKILL.md) | Validate and progressively run an existing direct, research-Harness, or Harness-Compiled LDM task. |

Each skill is self-contained and includes `agents/openai.yaml` metadata. Invoke
the relevant skill by name in a skill-aware agent, for example
`$collect-ldm-data`, `$run-ldm-task`, or `$register-ldm-task`.
