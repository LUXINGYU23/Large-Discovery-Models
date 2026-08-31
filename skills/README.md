# Repository Agent Skills

Repository-local agent workflows live in one folder so additional skills can be
added without mixing them into task implementations.

| Skill | Purpose |
| --- | --- |
| [`collect-ldm-data`](collect-ldm-data/SKILL.md) | Collect, augment, render, and validate `ldm-2.0` fine-tuning data. |
| [`register-ldm-task`](register-ldm-task/SKILL.md) | Scaffold, implement, register, and verify a direct or research-Harness task adapter. |
| [`run-ldm-task`](run-ldm-task/SKILL.md) | Validate and progressively run an existing direct or research-Harness task. |

Each skill is self-contained and includes `agents/openai.yaml` metadata. Invoke
the relevant skill by name in a skill-aware agent, for example
`$collect-ldm-data`, `$run-ldm-task`, or `$register-ldm-task`.
