# LDM Discovery Concepts

This context defines the task-neutral language used to describe Large Discovery
Model search across scientific domains.

## Language

**Candidate**:
A task-valid solution that can be selected for evaluation.
_Avoid_: Sample, point, feature

**Candidate Domain**:
The complete set of solutions permitted by a task's scientific validity rules.
_Avoid_: Candidate space, feature space, design space

**Reservoir**:
The finite, run-local collection of valid candidates available for selection in
one discovery step.
_Avoid_: Candidate pool, feature pool

**Reservoir Expansion**:
The LDM-guided process that adds candidates to a reservoir, either directly or
through edits, generators, or policies.
_Avoid_: Feature expansion, candidate generation

**Proposal Backend**:
The execution mechanism used by a reservoir expander to obtain raw proposals,
such as direct model requests, a persistent research Harness, a deterministic
generator, or a dataset search.
_Avoid_: Optimizer, evaluator

**Research Harness**:
A persistent, tool-using Agent backend behind reservoir expansion. It owns
research sessions and raw traces while the Campaign remains authoritative for
optimization history, candidate admission, acquisition, and evaluation.
_Avoid_: Agent optimizer, second campaign

**Expansion Schema**:
The structured parameters and actions currently available for reservoir
expansion. The schema may be fixed or may evolve during discovery.
_Avoid_: Feature set, active features, design space

**Surrogate Representation**:
The task-owned numerical or kernel representation of candidates consumed by a
surrogate model and acquisition rule.
_Avoid_: Candidate dimension, feature space

**Observation**:
A candidate paired with its authoritative evaluation outcome, including
measured objectives or a classified failure.
_Avoid_: Training point, result row

**Candidate Admission**:
Task-owned canonicalization and scientific validation that converts one raw
proposal into a candidate or an explicit rejection.
_Avoid_: Cleanup, post-processing

**Evaluation**:
The externally measured outcome for one admitted candidate, including status,
metrics, artifacts, resource usage, and failure information.
_Avoid_: Acquisition score, surrogate prediction

**Campaign**:
One durable LDM discovery run with a task contract, budgets, observations,
events, checkpoints, and terminal status.
_Avoid_: Script invocation, temporary run

**Acquisition**:
The rule that scores or samples reservoir candidates using observations and,
when configured, surrogate predictions.
_Avoid_: Evaluator, objective
