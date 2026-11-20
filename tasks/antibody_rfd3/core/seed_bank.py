"""Read and verify shared sequence initialization and its reusable backbone inputs."""
import json

from ldm_tts.contracts import CandidateRejection, RawProposal
from ldm_tts.harness import file_sha256
from tasks.antibody.core.engine_adapters import AntibodyCandidateDomain


def load_seed_bank(args):
    if args.seed_bank is None:
        return (), {}
    if args.initialization_run or args.mock:
        raise ValueError("A seed bank cannot be combined with replayed designs or mock generation")
    bank = json.loads(args.seed_bank.read_text(encoding="utf-8"))
    if bank["antigen"] != args.antigen or bank["oracle"] != args.oracle:
        raise ValueError("Seed bank antigen and oracle must match the campaign")
    seeds = bank["candidates"]
    sequences = [seed["sequence"] for seed in seeds]
    if len(sequences) < 2 or len(sequences) >= args.iterations or len(set(sequences)) != len(sequences):
        raise ValueError("Seed bank must contain unique candidates and leave room for online evaluations")
    root = args.seed_bank.resolve().parent
    backbones = {}
    for key, record in bank["backbones"].items():
        manifest = (root / record["manifest"]).resolve()
        if not manifest.is_relative_to(root) or file_sha256(manifest) != record["manifest_sha256"]:
            raise ValueError(f"Seed backbone manifest differs: {key}")
        data = json.loads(manifest.read_text())
        if data["status"] != "success" or len(data["candidates"]) != 1:
            raise ValueError(f"Seed backbone manifest is not a successful single backbone: {key}")
        pdb = manifest.parent / "backbone.pdb"
        cif = manifest.parent / "structure.cif.gz"
        if (data["candidates"][0]["backbone_path"] != str(pdb) or
                data["candidates"][0]["structure_path"] != str(cif) or
                file_sha256(pdb) != record["pdb_sha256"] or file_sha256(cif) != record["cif_sha256"]):
            raise ValueError(f"Seed backbone files differ: {key}")
        backbones[key] = manifest
    if {row["backbone_id"] for row in seeds} != set(backbones):
        raise ValueError("Seed candidates and backbone registrations differ")
    domain = AntibodyCandidateDomain(11)
    candidates = tuple(domain.admit(RawProposal(row, "shared_initialization")) for row in seeds)
    if any(isinstance(candidate, CandidateRejection) for candidate in candidates):
        raise ValueError("Seed bank contains inadmissible candidates")
    return candidates, backbones
