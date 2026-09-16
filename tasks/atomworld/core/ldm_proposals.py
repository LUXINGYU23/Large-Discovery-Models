"""Durable direct-model reservoir batches for measured-feedback CIF optimization."""

import json
from dataclasses import asdict

from ldm_tts.engine.expansion import attach_proposal_attempt_receipt
from ldm_tts.harness import canonical_sha256
from ldm_tts.transport import ProposalRequest, ProposalResponse
from .data import write_json
from .proposals import official_prompt
from .selection import measured_history, pool_result


class AtomWorldLDMExpander:
    def __init__(self, samples, args, run_dir, client):
        self.samples, self.args, self.run_dir, self.client = samples, args, run_dir, client
        self.runtime = None

    def expand(self, request):
        index = request.round_idx // self.args.attempts_per_sample
        sample = self.samples[index]
        history = measured_history(request.observations, sample["sample_id"])
        messages = ({"role": "user", "content": official_prompt(sample) +
                    "\nThis is measured-feedback optimization. Past submitted answers and scalar correctness:\n" +
                    json.dumps(history[-self.args.gp_history_limit:], sort_keys=True) +
                    "\nPropose one complete CIF. Unmeasured candidates have no correctness label; never seek targets or judge internals."},)
        digest = canonical_sha256(messages)
        drafts, responses = [], []
        for batch in range(self.args.proposal_samples):
            relative = f"proposal_batches/{request.round_idx:06d}/{batch:04d}.json"
            path = self.run_dir / relative
            if path.exists():
                receipt = json.loads(path.read_text())
                if receipt["input_digest"] != digest:
                    raise ValueError("Direct LDM batch differs from measured-history receipt")
                response = ProposalResponse(**receipt["response"])
            else:
                if self.runtime is not None:
                    self.runtime.consume("mock_model_requests" if self.args.mock else "llm_requests")
                response = self.client.propose(ProposalRequest(
                    messages=messages, metadata={"sample_id": sample["sample_id"],
                    "round_idx": request.round_idx, "batch_idx": batch, "history": history},
                ))
                write_json(path, {"input_digest": digest, "response": asdict(response)})
            responses.append(attach_proposal_attempt_receipt(response, relative))
            drafts.append({"generated_output": response.text, "rationale": "Direct CIF proposal"})
        record = {"sample_id": sample["sample_id"], "action_name": sample["action_name"],
                  "round_idx": request.round_idx, "input_digest": digest, "drafts": drafts}
        write_json(self.run_dir / "proposal_pools" / f"{request.round_idx:06d}.json", record)
        return pool_result(record, attempts=responses)
