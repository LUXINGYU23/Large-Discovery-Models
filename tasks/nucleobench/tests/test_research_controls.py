import json
from tasks.nucleobench.core.harness import NucleoBenchHarnessExpander, harness_profiles
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.tests.test_harness import FakeHarnessClient, _observation
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness.research_instructions import ROLES


def test_independent_roles_refresh_evidence_without_claiming_fork(tmp_path):
    expander = NucleoBenchHarnessExpander(None, NucleoBenchCandidateDomain(MOCK_CONTEXT),
        profiles=harness_profiles(("comprehensive_research",)*4), candidates_per_profile=1,
        campaign_id="control", first_active_round=1, attach_empirical_q0=True,
        allow_repeated_occurrences=True, artifact_root=tmp_path, research_roles=True)
    observations = (_observation({"mutations": []}, round_idx=0),)
    first = expander._turns(ExpansionRequest(1, 4, observations=observations))
    observations += (_observation({"mutations": [{"position": 0, "base": "C"}]}, round_idx=1),)
    second = expander._turns(ExpansionRequest(2, 4, observations=observations))
    for index, (a,b) in enumerate(zip(first,second)):
        x = json.loads(a.message.split("\n\n",1)[1])["research_branch"]
        y = json.loads(b.message.split("\n\n",1)[1])["research_branch"]
        assert y["role"] == ROLES[index][0]
        assert y["parent_node"] is None and y["node_selection_rule"] == "independent_persistent"
        assert "persistent session" in y["requirements"][0]
        assert x["shared_evidence"]["history_count"] == 1
        assert y["shared_evidence"]["history_count"] == 2
        assert x["shared_evidence"]["history_digest"] != y["shared_evidence"]["history_digest"]
        assert b.history_from_seq == 1
