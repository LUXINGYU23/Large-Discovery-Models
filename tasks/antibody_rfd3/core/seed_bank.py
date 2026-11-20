"""Immutable shared candidate initialization for paired campaigns."""
import json

from ldm_tts.contracts import RawProposal, CandidateRejection
from tasks.antibody.core.engine_adapters import AntibodyCandidateDomain


def load_seed_bank(args):
    if args.seed_bank is None:
        return ()
    if args.initialization_run or args.mock:
        raise ValueError("A seed bank cannot be combined with replayed designs or mock generation")
    bank = json.loads(args.seed_bank.read_text(encoding="utf-8"))
    if bank["antigen"] != args.antigen or bank["oracle"] != args.oracle:
        raise ValueError("Seed bank antigen and oracle must match the campaign")
    sequences = bank["sequences"]
    if len(sequences) < 2 or len(sequences) >= args.iterations or len(set(sequences)) != len(sequences):
        raise ValueError("Seed bank must contain unique candidates and leave room for online evaluations")
    candidates = tuple(AntibodyCandidateDomain(11).admit(RawProposal({"sequence": seq}, "shared_initialization"))
                       for seq in sequences)
    if any(isinstance(candidate, CandidateRejection) for candidate in candidates):
        raise ValueError("Seed bank contains inadmissible candidates")
    return candidates
