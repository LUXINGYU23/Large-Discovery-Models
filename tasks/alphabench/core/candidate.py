from ldm_tts.contracts import Candidate, CandidateRejection
from .grammar import ExpressionError, parse_expression


class FactorDomain:
    def __init__(self, backend="qlib", max_depth=None):
        self.backend, self.max_depth = backend, max_depth

    def admit(self, proposal):
        payload = proposal.payload
        if not isinstance(payload, dict) or not isinstance(payload.get("expression"), str):
            return CandidateRejection("invalid_payload", "expression text is required", proposal.source)
        try:
            parsed = parse_expression(payload["expression"], backend=self.backend, max_depth=self.max_depth)
        except (ExpressionError, SyntaxError, ValueError) as exc:
            return CandidateRejection("invalid_expression", str(exc), proposal.source)
        return Candidate(parsed.candidate_id, {**payload, "expression": parsed.canonical, "dialect": parsed.dialect},
                         parsed.candidate_id, proposal.source, metadata=dict(proposal.metadata))
