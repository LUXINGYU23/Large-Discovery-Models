"""NucleoBench evidence adapter for the shared research-tree runtime."""

from __future__ import annotations

import math
from dataclasses import replace

from ldm_tts.harness.research_runtime import ResearchCoordinator
from ldm_tts.harness.research_store import ResearchMeasurement
from tasks.nucleobench.core.candidate import prepare_candidate_payload
from tasks.nucleobench.core.harness import NucleoBenchHarnessExpander, _forbidden_query_terms, _history_delta, _profile
from tasks.nucleobench.core.research import (
    serialize_measured_observations, summarize_measured_observations, write_measured_history,
)


def main_research_profile():
    return (_profile("main_research"),)


def _descriptor(payloads, context):
    positions = {p: i for i, p in enumerate(context.editable_positions)}
    reference = ["ACGT".index(context.start_sequence[p]) for p in positions]
    counts = [0.0] * (4 * len(positions))
    for payload in payloads:
        bases = reference.copy()
        for mutation in payload["mutations"]:
            bases[positions[mutation["position"]]] = "ACGT".index(mutation["base"])
        for i, base in enumerate(bases):
            counts[4 * i + base] += 1 / (len(payloads) * math.sqrt(len(positions)))
    return tuple(counts)


class ForkedResearchExpander:
    def __init__(self, *, main_client, root, branch_client, expander_options, strategy):
        self.options, self.branch_client = expander_options, branch_client
        self.domain = expander_options["domain"]
        self.coordinator = ResearchCoordinator(root, strategy, expander_options["profiles"], main_client,
                                               account=expander_options.get("account"))

    def expand(self, request):
        coordinator, context = self.coordinator, self.domain.context
        write_measured_history(coordinator.root, request.observations, context)
        write_measured_history(coordinator.root / "main", request.observations, context)
        measured = [o for o in request.observations if o.evaluation.status == "succeeded"
                    and math.isfinite(float(o.evaluation.metrics["utility"]))]
        leader = max(measured, key=lambda o: (o.evaluation.metrics["utility"], o.canonical_key), default=None)
        prepared = coordinator.prepare(
            request.round_idx, serialize_measured_observations(request.observations, context),
            tuple(ResearchMeasurement(o.canonical_key, float(o.evaluation.metrics["utility"]), o.round_idx)
                  for o in measured),
            descriptor=_descriptor([leader.candidate.payload if leader else {"mutations": []}], context),
            forbidden_query_terms=_forbidden_query_terms(request),
            message={"new_measurements": summarize_measured_observations(
                serialize_measured_observations(_history_delta(request, 1), context))},
        )
        with self.branch_client(prepared["root"], prepared["sources"]) as client:
            expanded = NucleoBenchHarnessExpander(
                client, artifact_root=prepared["root"],
                inherited_history_counts={p: s.history_cursor for p, s in prepared["sources"].items()},
                **self.options,
            ).expand(request)
        assignments = {a["profile_id"]: a for a in prepared["plan"]["assignments"]}
        expanded = replace(expanded, proposals=tuple(replace(p, metadata={
            **p.metadata, "research_fork": assignments[p.metadata["harness_lineage"]["profile_id"]],
        }) for p in expanded.proposals))
        batches = {profile: [prepare_candidate_payload(p.payload, context) for p in expanded.proposals
                            if p.metadata["harness_lineage"]["profile_id"] == profile]
                   for profile in assignments}
        coordinator.commit(
            prepared, {s["profile_id"]: s for s in expanded.metadata["sessions"]},
            {profile: tuple(p.canonical_key for p in batch) for profile, batch in batches.items()},
            descriptors={profile: _descriptor([p.payload for p in batch], context) for profile, batch in batches.items()},
            proposals=[{"payload": p.payload, "annotations": p.metadata["research_annotations"],
                        "lineage": p.metadata["harness_lineage"]} for p in expanded.proposals],
        )
        return replace(expanded, metadata={
            **expanded.metadata, "sampling_mode": "main_research_round_forks",
            "main_turn_id": prepared["main_turn_id"], "fork_source": prepared["source"].to_dict(),
            "fork_policy": coordinator.tree.fork_policy.identity(), "fork_plan": prepared["plan"]["assignments"],
        })
