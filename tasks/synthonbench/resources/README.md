# SynthonBench Resources

`upstream_contract.json` records the immutable official source and dataset
revisions used by this task. It is not a copy of released score tables or
synthon spaces. Run `python -m tasks.synthonbench.scripts.prepare_official_data` to place those large,
source-pinned artifacts outside the repository.

`qualification_evidence.json` and `verification_record.json` summarize the
source-pinned Official Example, Surrogate Oracle, and Glide
Ligand-Efficiency verification boundary. Campaign outputs remain outside the
repository.

`harness/` contains the comprehensive researcher and policy profiles,
task-local on-demand Skills, the official SynthonSpace and measured-history
tools, and a dependency-pinned chemistry research guest. Skill sources and
licenses are in [harness/skills/ATTRIBUTION.md](harness/skills/ATTRIBUTION.md).
External MCP tools are user-configured at runtime.
