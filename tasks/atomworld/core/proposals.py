"""Target-blind CIF reasoning and self-review over the shared ProposalClient seam."""

from __future__ import annotations
import hashlib
import json
import warnings
from pathlib import Path

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, make_complete_design_ir
from ldm_tts.engine.expansion import ExpansionResult
from ldm_tts.transport import ProposalRequest, ProposalResponse
from tasks.atomworld.core.data import write_json

MAX_OUTPUT_CHARS = 1_000_000


def extract_cif(text: str) -> str | None:
    # Same last-tag behavior as the pinned upstream evaluator.
    start = text.rfind("<cif>")
    end = text.rfind("</cif>", start)
    if start == -1 or end == -1:
        return None
    return text[start + len("<cif>") : end].strip()


def public_validation(text: str, *, mock=False) -> dict:
    cif = extract_cif(text)
    if cif is None:
        return {
            "parseable": False,
            "message": "Return a complete <cif>...</cif> block.",
        }
    if mock:
        return {
            "parseable": cif.startswith("data_"),
            "check": "synthetic_mock_syntax_only",
        }
    try:
        from pymatgen.io.cif import CifParser

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = CifParser.from_str(cif).parse_structures(
                primitive=False, check_occu=False
            )
        if not structures:
            return {"parseable": False, "message": "CIF parser returned no structure."}
        structure = structures[0]
        return {
            "parseable": True,
            "atom_count": len(structure),
            "composition": structure.composition.as_dict(),
            "message": "Parsing checks syntax only; independently verify the requested geometry.",
        }
    except Exception:
        return {
            "parseable": False,
            "message": "CIF parser could not read this output. Check cell and atom-loop syntax.",
        }


def official_prompt(sample: dict) -> str:
    # Verbatim upstream src/prompts/cif_action_prompt.py formatting, pinned in source manifest.
    parts = [
        "You are a CIF operation assistant.",
        "You will be given an input CIF content and an action prompt.",
        "Your task is to apply the action described in the action prompt to the initial CIF content.",
        "The coordinates in the action are in Cartesian format, and the indices of atoms are started from 0.\n",
        'Return the modified CIF content in cif format within "<cif>" and "</cif>" tags.\n',
        "Please ensure the output is a valid CIF file, with correct formula, and atom positions.",
        f"Input CIF content:\n{sample['input_cif']}\n",
        f"Action prompt: {sample['action_prompt']} ",
    ]
    return " ".join(parts)


def build_messages(
    sample: dict, previous: dict | None = None, *, operation_instructions=None
) -> tuple[dict, ...]:
    prompt = official_prompt(sample)
    if operation_instructions is not None:
        prompt = (
            "Apply the following action to the input crystal using the bounded geometry tool. "
            "Return only a JSON array of operation objects; the tool will write the CIF.\n"
            + operation_instructions
            + "\nInput CIF:\n"
            + sample["input_cif"]
            + "\nRequested action: "
            + sample["action_prompt"]
        )
    messages = [{"role": "user", "content": prompt}]
    if previous is not None:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": previous.get(
                        "model_output", previous["generated_output"]
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Review your previous answer against the original input and requested action. "
                        "Work through the Cartesian/fractional coordinate conversion, zero-based indices, "
                        "periodic boundaries and unchanged atoms carefully. Use the cell matrix for non-orthogonal cells. "
                        + (
                            "Correct any mistakes and return your final JSON operation array. "
                            if operation_instructions is not None
                            else "Correct any mistakes and return your final complete CIF in <cif> tags. "
                        )
                        + "Tool execution error: "
                        + str(previous.get("tool_error"))
                        + ". "
                        + "Public syntax feedback (does not assess task correctness): "
                        + json.dumps(previous["public_validation"], sort_keys=True)
                    ),
                },
            ]
        )
    return tuple(messages)


def canonical_key(sample_id: str, text: str) -> str:
    # Preserve malformed outputs in the official scoring denominator; only whitespace is canonicalized.
    cif = extract_cif(text)
    normalized = cif if cif is not None else text.strip()
    return hashlib.sha256((sample_id + "\0" + normalized).encode("utf-8")).hexdigest()


class AtomWorldDomain:
    def __init__(self, samples: list[dict], *, sink=None, mock=False):
        self.public = {row["sample_id"]: dict(row) for row in samples}
        self.sink = sink or DataCollectionSink.disabled()
        self.mock = mock

    def admit(self, proposal):
        payload = proposal.payload
        fields = {"sample_id", "action_name", "generated_output"}
        if not isinstance(payload, dict) or set(payload) != fields:
            return CandidateRejection(
                "invalid_payload",
                "Exactly sample_id, action_name, generated_output required",
                proposal.source,
            )
        sample = self.public.get(payload["sample_id"])
        if sample is None or sample["action_name"] != payload["action_name"]:
            return CandidateRejection(
                "unknown_sample",
                "Sample and action must match the public dataset",
                proposal.source,
            )
        text = payload["generated_output"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_OUTPUT_CHARS
        ):
            return CandidateRejection(
                "invalid_output_size",
                "Nonempty output of at most 1,000,000 characters required",
                proposal.source,
            )
        key = canonical_key(payload["sample_id"], text)
        candidate = Candidate(f"aw-{key[:24]}", dict(payload), key, proposal.source)
        # Malformed CIF is a scoreable benchmark answer (official format/parsing error),
        # but never becomes training data. This preserves the one-shot denominator.
        if (
            proposal.metadata.get("collectable")
            and public_validation(text, mock=self.mock)["parseable"]
        ):
            ir = make_complete_design_ir(
                task_id="atomworld",
                domain="crystal structure manipulation",
                task_description="Apply the requested operation to the input CIF.",
                objectives=[{"name": "correct", "direction": "maximize"}],
                design_space_description="Complete CIF structure inside <cif> tags.",
                observations=[],
                candidates=[{"generated_output": text}],
                request_description=json.dumps(
                    proposal.metadata.get("messages", build_messages(sample)),
                    ensure_ascii=False,
                ),
                num_candidates=1,
                allows_new_parameters=False,
                reasoning_available=False,
            )
            self.sink.append(
                ir,
                provenance={
                    "sample_id": sample["sample_id"],
                    "candidate_id": candidate.candidate_id,
                    "source": proposal.source,
                    "round_idx": proposal.metadata.get("round_idx"),
                },
            )
        return candidate


class BlindRefinementExpander:
    """Round schedule and drafts depend solely on public tasks, never judge observations."""

    def __init__(
        self,
        samples: list[dict],
        client,
        *,
        attempts_per_sample: int,
        run_dir: Path,
        mock=False,
        tools_root: Path | None = None,
        operations: bool = False,
    ):
        self.samples = [dict(row) for row in samples]
        self.client = client
        self.attempts_per_sample = attempts_per_sample
        self.run_dir = run_dir
        self.mock = mock
        self.runtime = None
        self.operation_tools = None
        if tools_root is not None or operations:
            from tasks.atomworld.core import geometry

            self.operation_tools = geometry

    def expand(self, request):
        sample_index, attempt = divmod(request.round_idx, self.attempts_per_sample)
        sample = self.samples[sample_index]
        path = self.run_dir / "attempts" / f"{request.round_idx:06d}.json"
        previous = None
        if attempt:
            previous_path = (
                self.run_dir / "attempts" / f"{request.round_idx - 1:06d}.json"
            )
            previous = json.loads(previous_path.read_text())
        messages = build_messages(
            sample,
            previous,
            operation_instructions=self.operation_tools.OPERATION_INSTRUCTIONS
            if self.operation_tools
            else None,
        )
        if path.exists():
            record = json.loads(path.read_text())
            if record["sample_id"] != sample["sample_id"] or record["messages"] != list(
                messages
            ):
                raise ValueError(
                    "Resume draft does not match immutable prompt schedule"
                )
            response = ProposalResponse(**record["response"])
        else:
            if self.runtime is not None:
                self.runtime.consume(
                    "mock_model_requests" if self.mock else "llm_requests"
                )
            response = self.client.propose(
                ProposalRequest(
                    messages,
                    metadata={
                        "sample_id": sample["sample_id"],
                        "round_idx": request.round_idx,
                    },
                )
            )
            generated_output = response.text
            tool_error = None
            if self.operation_tools:
                try:
                    text = response.text.strip()
                    if text.startswith("```") and text.endswith("```"):
                        text = "\n".join(text.splitlines()[1:-1])
                    operations = json.loads(text)
                    if isinstance(operations, dict) and set(operations) == {
                        "operations"
                    }:
                        operations = operations["operations"]
                    if self.runtime is not None:
                        self.runtime.consume("geometry_tool_calls")
                    generated_output = (
                        "<cif>"
                        + self.operation_tools.execute_operations(
                            sample["input_cif"], operations
                        )
                        + "</cif>"
                    )
                except ValueError as exc:
                    tool_error = str(exc)[:2000]
                    generated_output = "ToolExecutionError: " + tool_error
            record = {
                "sample_id": sample["sample_id"],
                "action_name": sample["action_name"],
                "attempt": attempt,
                "round_idx": request.round_idx,
                "messages": list(messages),
                "generated_output": generated_output,
                "model_output": response.text,
                "tool_error": tool_error,
                "response": response.to_dict(),
                "canonical_key": canonical_key(sample["sample_id"], generated_output),
                "public_validation": public_validation(
                    generated_output, mock=self.mock
                ),
            }
            write_json(path, record)
        payload = {
            "sample_id": sample["sample_id"],
            "action_name": sample["action_name"],
            "generated_output": record["generated_output"],
        }
        # request.parent, request.observations, acquisition_feedback are intentionally unread.
        return ExpansionResult(
            proposals=(
                RawProposal(
                    payload,
                    "mock_transport"
                    if self.mock
                    else (
                        "openai_operations"
                        if self.operation_tools
                        else "openai_reasoning"
                    ),
                    metadata={
                        "collectable": self.operation_tools is None,
                        "messages": list(messages),
                        "round_idx": request.round_idx,
                    },
                ),
            ),
            attempts=(response,),
            selection_mode="reservoir_order",
            metadata={
                "sample_id": sample["sample_id"],
                "attempt": attempt,
                "feedback": "public_syntax_only",
            },
        )
