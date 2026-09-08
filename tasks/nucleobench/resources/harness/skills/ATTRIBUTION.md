# Scientific Research Skills

The `biopython`, `experimental-design`, `scientific-critical-thinking`, and
`statsmodels` directories are task-local adaptations of
[Scientific Agent Skills by K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/tree/1e5eeffbdad3749125afe7ab48a39694e27f181c/skills),
pinned to commit `1e5eeffbdad3749125afe7ab48a39694e27f181c`.
Their source paths are `skills/<skill-name>/SKILL.md` and the associated
reference material in that revision. Each adapted directory retains the
upstream MIT copyright notice and license, including in session snapshots.
Library packages keep their own licenses; these notices cover the skill text.

The adaptations focus on fixed-length DNA reconstruction, two-strand motif
analysis, interpretable candidate panels, evidence evaluation, and small-model
diagnostics. They use the task's installed packages and existing web/Context7
tools. General clinical, manuscript, hosted-service, and unrelated domain
workflows are not included. DOE examples use pyDOE3 directly instead of
vendoring an additional script interface.

Biopython supports exact sequence reconstruction and both-strand motif checks;
experimental design and critical thinking support comparisons and interpretation.
Statsmodels is conditional on identifiable measured contrasts, not a bootstrap
requirement. A matched comparison can be preferable to a factorial panel, and
descriptive analysis can be preferable to fitting a model. No Skill requires a
particular package call or a separate report every round.

All candidate-generation profiles, including `direct_research` and repeated
`comprehensive_research` sessions, expose the same four Skills through Pi's
on-demand loader. `policy_architect` retains its separate task-owned
`compile-ldm-policy` Skill. The scientific research Skills neither replace
that policy contract nor change the evaluator, GP, or LDM selection rules.

Skill names and descriptions are available at session creation; full content
is read on demand from the immutable guest-visible resource snapshot. Skill
digests are recorded by the existing Harness profile/manifest mechanism.
