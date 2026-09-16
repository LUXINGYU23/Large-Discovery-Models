"""Measured-feedback LDM selection over a finite CIF proposal reservoir."""

from collections import Counter
from dataclasses import replace
import json

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, RawProposal, SurrogateSpaceSpec
from ldm_tts.engine.expansion import ExpansionResult, attach_proposal_attempt_receipt
from ldm_tts.harness import canonical_sha256
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOSelectionResult, SurrogateVector
from ldm_tts.transport import ProposalResponse
from .data import write_json
from .optimization_policy import FEATURE_NAMES, _geometry_features
from .proposals import canonical_key, public_validation

Q0_KEY = "atomworld_empirical_q0"
FEATURE_VERSION = "atomworld_public_geometry_signed_log_v1"


def summarize_rounds(rounds):
    compiled = [row["compiled_policy"] for row in rounds if "compiled_policy" in row]
    return {
        "rounds": rounds, "compiled_rounds": len(compiled),
        "mean_prior_prediction_change": sum(row["prediction_mean_abs_change"] for row in compiled) / max(1, len(compiled)),
        "mean_selection_entropy": sum(row["probability_entropy"] for row in rounds) / max(1, len(rounds)),
        "mean_effective_sample_size": sum(row["probability_effective_sample_size"] for row in rounds) / max(1, len(rounds)),
    }


def measured_history(observations, sample_id):
    """Allow only past scalar correctness, not private CIFs or judge diagnostics."""
    return [
        {"candidate_id": item.candidate_id, "round_idx": item.round_idx,
         "correct": float(item.metrics["correct"]),
         "generated_output": item.candidate.payload["generated_output"]}
        for item in observations
        if item.evaluation.succeeded and item.candidate.payload["sample_id"] == sample_id
    ]


def pool_result(record, *, attempts=()):
    drafts = record["drafts"]
    counts = Counter(canonical_key(record["sample_id"], d["generated_output"]) for d in drafts)
    if not attempts and record.get("sessions"):
        attempts = tuple(attach_proposal_attempt_receipt(
            ProposalResponse(text=draft["generated_output"]),
            f"atomworld:{record['sample_id']}:{session['turn_id']}",
        ) for draft, session in zip(drafts, record["sessions"], strict=True))
    return ExpansionResult(
        proposals=tuple(RawProposal(
            {"sample_id": record["sample_id"], "action_name": record["action_name"],
             "generated_output": draft["generated_output"]},
            "atomworld_ldm_reservoir",
            metadata={"submission_round": record["round_idx"], "rationale": draft["rationale"],
                      Q0_KEY: {"occurrence_count": counts[canonical_key(record["sample_id"], draft["generated_output"])],
                               "valid_occurrence_count": len(drafts)}},
        ) for draft in drafts),
        attempts=tuple(attempts), selection_mode="acquisition",
        metadata={"feedback": "past_measured_correctness", "valid_proposal_occurrences": len(drafts)},
    )


class GeometryEncoder:
    version = FEATURE_VERSION

    def __init__(self, samples, *, mock=False):
        self.samples = {s["sample_id"]: s for s in samples}
        self.mock = mock

    def describe(self):
        return SurrogateSpaceSpec(
            "vector", "Signed log1p public input-to-draft geometry features",
            "fixed", dimension=len(FEATURE_NAMES), version=self.version,
        )

    def encode(self, candidate):
        sample = self.samples[candidate.payload["sample_id"]]
        text = candidate.payload["generated_output"]
        check = public_validation(text, mock=self.mock)
        raw = np.asarray([float(check["parseable"]), check.get("atom_count", 0),
                          len(check.get("composition", {})), len(text),
                          *_geometry_features(sample, text, mock=self.mock)], dtype=float)
        values = np.sign(raw) * np.log1p(np.abs(raw))
        return SurrogateVector(tuple(values), self.version, candidate.candidate_id)


def make_gp(history, args, prior=None):
    selected = history[-args.gp_history_limit:]
    if prior is not None:
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (len(history),):
            raise ValueError("Compiled prior must align with the full measured history")
    return RBFGPSurrogate(
        selected, lengthscale=args.gp_lengthscale, noise=args.gp_noise,
        feature_version=FEATURE_VERSION,
        residual_prior_mean=np.zeros(len(selected)) if prior is None else prior[-args.gp_history_limit:],
    )


def gp_projection(gp, history_size, vectors):
    projection = gp.posterior_projection(vectors)
    weights = np.zeros((len(vectors), history_size))
    weights[:, -len(gp.observations):] = projection["weights"]
    return {**projection, "weights": weights}


def base_masses(candidates):
    counts, totals = [], set()
    for candidate in candidates:
        record = candidate.metadata[Q0_KEY]
        count, total = record["occurrence_count"], record["valid_occurrence_count"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (count, total)) or not 1 <= count <= total:
            raise ValueError("Invalid CIF proposal occurrence receipt")
        counts.append(count)
        totals.add(total)
    if len(totals) != 1 or sum(counts) > next(iter(totals)):
        raise ValueError("Inconsistent CIF proposal frequencies")
    return np.asarray(counts, dtype=float) / sum(counts)


def selection_distribution(q0, acquisition, *, alpha, eta, z_clip):
    if any(not np.isfinite(v) or v < 0 for v in (alpha, eta)):
        raise ValueError("LDM weights must be finite and nonnegative")
    values = np.asarray(acquisition, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    if values.shape != q0.shape or not np.isfinite(values).all() or np.any(q0 <= 0):
        raise ValueError("Finite UCB values must align with positive q0")
    center = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - center))
    if scale <= 1e-12:
        scale = np.std(values)
    z = np.zeros_like(values) if scale <= 1e-12 else np.clip((values - center) / (scale + 1e-12), -z_clip, z_clip)
    logits = alpha * np.log(q0 + 1e-12) + eta * z
    probability = np.exp(logits - logits.max())
    return probability / probability.sum(), logits, z


def sample_indices(probability, count, *, seed):
    rng = np.random.default_rng(seed)
    return np.argsort(np.log(probability) + rng.gumbel(size=len(probability)))[::-1][:count]


class AtomWorldLDMSelector:
    def __init__(self, args, samples, run_dir, *, expander=None):
        self.args, self.samples, self.run_dir = args, samples, run_dir
        self.expander = expander
        self.history = ()

    def describe(self):
        return AcquisitionSpec(
            "ldm_rbf_gp_ucb", ("correct",), "sample",
            "q0^alpha * exp(eta * robust_z(UCB)); Gumbel top-k without replacement",
            parameters={"alpha": self.args.acquisition_alpha, "eta": self.args.acquisition_eta,
                        "beta": self.args.ucb_beta, "bo_pool_size": self.args.bo_pool_size,
                        "kernel": "rbf", "feature_version": FEATURE_VERSION},
        )

    def fit(self, history):
        self.history = tuple(history)

    def select(self, candidates, representations, *, count=1, round_idx=0):
        if count != 1:
            raise ValueError("AtomWorld schedules one measured submission per round")
        args = self.args
        index, attempt = divmod(round_idx, args.attempts_per_sample)
        sample = self.samples[index]
        if any(c.payload["sample_id"] != sample["sample_id"] for c in candidates):
            raise ValueError("LDM reservoir crosses the scheduled sample boundary")
        history = [h for h in self.history
                   if index * args.attempts_per_sample <= h.metadata["round_idx"] < round_idx]
        reservoir = tuple(sorted(candidates, key=lambda c: c.candidate_id))
        seed = int(canonical_sha256({"seed": args.campaign_index, "round": round_idx,
                                    "candidates": [c.candidate_id for c in reservoir]})[:16], 16)
        q0 = base_masses(reservoir)
        indices = sample_indices(q0, min(args.bo_pool_size, len(reservoir)), seed=seed)
        pool = tuple(sorted((reservoir[i] for i in indices), key=lambda c: c.candidate_id))
        q0 = base_masses(pool)
        for h in history:
            if h.feature.version != FEATURE_VERSION:
                raise ValueError("Measured history feature version mismatch")
        if any(representations[c.candidate_id].version != FEATURE_VERSION for c in pool):
            raise ValueError("CIF query feature version mismatch")
        gp = make_gp(history, args)
        baseline = tuple(gp.predict_record(c.candidate_id, representations[c.candidate_id].values,
                                           beta=args.ucb_beta) for c in pool)
        active = baseline
        alpha, eta = args.acquisition_alpha, args.acquisition_eta
        policy, controller = None, None
        if history and args.search_method == "ldm_harness_compiled":
            from .compiled_policy import compiled_controller, build_policy_input

            controller = compiled_controller(self.expander, index, round_idx, pool)
            policy = controller.resolve(build_policy_input(
                args, sample, round_idx, history, pool, representations, baseline, q0, gp,
            ))
            gp = make_gp(history, args, policy.history_prior_mean)
            active = tuple(gp.predict_record(
                c.candidate_id, representations[c.candidate_id].values, beta=args.ucb_beta,
                query_prior_mean=float(policy.query_prior_mean[i]),
            ) for i, c in enumerate(pool))
            alpha, eta = policy.alpha, policy.eta
        probability, logits, z = selection_distribution(
            q0, [p.acquisition_score for p in active], alpha=alpha,
            eta=eta if history else 0.0, z_clip=args.z_clip,
        )
        records = [{
            "candidate_id": c.candidate_id, "q0": float(q0[i]),
            "occurrence_count": c.metadata[Q0_KEY]["occurrence_count"],
            "baseline_mean": baseline[i].scalar_mean, "baseline_std": baseline[i].scalar_std,
            "active_mean": active[i].scalar_mean, "active_std": active[i].scalar_std,
            "baseline_acquisition": baseline[i].acquisition_score,
            "active_acquisition": active[i].acquisition_score,
            "selection_probability": float(probability[i]), "first_draw_probability": float(probability[i]),
            "normalized_acquisition": float(z[i]), "logit": float(logits[i]),
        } for i, c in enumerate(pool)]
        selected = pool[int(sample_indices(probability, 1, seed=seed ^ 0xA701)[0])]
        metadata = {
            "method": args.search_method, "feedback_protocol": "measured_correctness_optimization",
            "training_observations": len(gp.observations), "measured_history_size": len(history),
            "surrogate": gp.summary(),
            "alpha": alpha, "eta": eta, "warm_start": not history,
            "proposal_reservoir_size": len(reservoir), "bo_pool_size": len(pool),
            "selection_seed": seed ^ 0xA701, "distribution": records,
            "probability_entropy": float(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))),
            "probability_effective_sample_size": float(1 / np.sum(probability**2)),
            "tilted_kl_from_q0": float(np.sum(probability * np.log(np.maximum(probability, 1e-300) / q0))),
        }
        if policy is not None:
            controller.record_predictions(round_idx, records)
            metadata["compiled_policy"] = {
                **dict(policy.metadata), "epoch_id": policy.epoch_id, "artifact_sha256": policy.artifact_digest,
                "source": policy.source, "degraded": policy.degraded, "stage": policy.stage,
                "alpha": alpha, "eta": eta,
                "prediction_mean_abs_change": float(np.mean([abs(a.scalar_mean-b.scalar_mean)
                                                              for a, b in zip(active, baseline)])),
            }
        record = {**selected.payload, "round_idx": round_idx, "attempt": attempt,
                  "canonical_key": selected.canonical_key, "candidate_id": selected.candidate_id,
                  "rationale": selected.metadata.get("rationale", ""),
                  "public_validation": public_validation(selected.payload["generated_output"], mock=args.mock),
                  "selection": metadata}
        path = self.run_dir / "attempts" / f"{round_idx:06d}.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("Resumed LDM selection differs from its durable attempt receipt")
        write_json(path, record)
        return BOSelectionResult(
            (selected.candidate_id,),
            tuple(replace(p, metadata={**p.metadata, **records[i]}) for i, p in enumerate(active)),
            metadata=metadata,
        )
