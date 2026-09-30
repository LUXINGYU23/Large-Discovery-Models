# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Root package metadata and a reproducible development lockfile.
- Public CI lanes for the shared runner and the three built-in task packages.
- Project license, contribution, security, conduct, and citation guidance.
- Draft experiment contracts for all three built-in tasks.
- Integrity verification and trust documentation for the external G12D joblib
  model artifact.
- Automatic manifest-declared dependency preflight before non-mock runs.
- Slime as an RL submodule (`rl/slime`) plus a task-neutral RL environment
  (`rl/ldm_rl`) that maps one LDM campaign to one RL episode: policy candidate
  proposals are actions, evaluation feedback is the observation, and objective
  improvement is the reward. Includes mock-mode factories for the
  `ai4bio_mutation_effect_prediction` and `causal_discovery_discrete` tasks,
  a Slime custom generate/reward bridge, and engine-parity tests.

### Changed

- NucleoBench GP selection computes the candidate posterior as one batch,
  preserving the UCB formula, compiled prior mean, and selection ordering.
- Clarified that v0.1 is a release candidate and that built-in campaign
  contracts remain drafts pending qualification evidence.
- Removed the G12D joblib binary from Git while retaining its checksum and
  provenance metadata.
- Consolidated project-specific technical guides under `docs/` while retaining
  conventional public project and governance files at the repository root.
- Removed the protein inverse-folding task until its public contract and
  dependencies are ready.
- Corrected stale module paths and local-only artifact metadata.
- Small-molecule real runs: keep macrocycles rigid during Meeko ligand
  preparation so the legacy AutoDock Vina 1.1.2 binary no longer rejects the
  ring-breaking "glue" atom types (G0/CG0/CG1) that Meeko injects by default;
  those rejections previously surfaced as "non-finite objective score after
  retries" and wasted most of the evaluation budget.
- Small-molecule real runs: scale the engine `proposal_attempts` limit by the
  number of LLM chunks per reservoir round so the stratified direct-LLM
  reservoir plus its refill loops no longer exhaust the proposal budget before
  the target number of successful evaluations completes.

### Removed

- Historical `ldm_tts` package-root aliases and their migration tests. Import
  shared interfaces from their owning packages, such as `ldm_tts.contracts`,
  `ldm_tts.engine`, and `ldm_tts.transport`; import task symbols from `tasks.<task>`.

[Unreleased]: https://github.com/yzailab/Large-Discovery-Models/commits/ldm_engine
