"""Reuse source admission, then enforce the v2 interface gate before search."""
from ldm_tts.contracts import CandidateRejection
from ..core.candidate import Domain


class CheckedDomain(Domain):
    def __init__(self, root, checks):
        super().__init__(root)
        self.checks = checks

    def admit(self, proposal):
        candidate = super().admit(proposal)
        if isinstance(candidate, CandidateRejection):
            return candidate
        try:
            self.checks(candidate.payload)
        except (ValueError, OSError) as exc:
            return CandidateRejection("invalid_interface", str(exc), proposal.source)
        return candidate
