"""Read-only forward validation of GP kernels on completed HER2 controller ledgers."""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sqlite3
from time import perf_counter

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import spearmanr

FOLDS = ((16, 32), (32, 64), (64, 128), (128, 192), (256, 320), (512, 576))
NOISE = 1e-4 + 1e-8
GAP_DECAY = 0.5
TK_PENALTY = 0.5
KERNELS = ("K0", "K1-equal", "K1-ARD", "K2", "K3", "K4")


def read_scores(directory):
    uri = (directory.resolve() / "controller.sqlite3").as_uri() + "?mode=ro"
    config = json.loads((directory / "config.json").read_text())
    if config["antigen"] != "1N8Z_C" or config["oracle"] != "absolut":
        raise ValueError("Expected an existing HER2 Absolut ledger")
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = db.execute("SELECT ingest_seq,canonical_key,status,observation_json "
                          "FROM observations ORDER BY ingest_seq").fetchall()
    scores = np.asarray([json.loads(row[3])["evaluation"]["metrics"]["absolut_energy"] for row in rows])
    sequences = [row[1] for row in rows]
    if (len(rows) < FOLDS[-1][1] or len(set(sequences)) != len(rows)
            or any(row[0] != i or row[2] != "succeeded" for i, row in enumerate(rows, 1))
            or not np.isfinite(scores).all()
            or any(len(s) != 11 or set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in sequences)):
        raise ValueError("Expected a complete, unique, successful CDRH3 ledger")
    source_digest = hashlib.sha256(json.dumps(list(zip(sequences, scores.tolist()))).encode()).hexdigest()
    return sequences, scores, source_digest


def string_features(sequences, *, gaps):
    if gaps:
        indices = list(combinations(range(11), 3))
    else:
        indices = [(i, i + 1, i + 2) for i in range(9)]
    alphabet = {aa: index for index, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    columns, weights = [], []
    for seq in sequences:
        for i, j, k in indices:
            columns.append(400 * alphabet[seq[i]] + 20 * alphabet[seq[j]] + alphabet[seq[k]])
            weights.append(GAP_DECAY ** (k - i - 2))
    offsets = np.arange(len(sequences) + 1) * len(indices)
    raw = csr_matrix((weights, columns, offsets), shape=(len(sequences), 8000))
    norms = np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).ravel())
    return raw.multiply(1 / norms[:, None]).tocsr()


def gp_fit(gram, y):
    factor = cholesky(gram + NOISE * np.eye(len(y)), lower=True)
    alpha = cho_solve((factor, True), y)
    return factor, alpha


def train_tk(mismatches, y):
    def loss(logits):
        weights = softmax(logits)
        kernel = np.exp(-(mismatches @ weights))
        factor, alpha = gp_fit(kernel, y)
        inverse = cho_solve((factor, True), np.eye(len(y)))
        derivative = -0.5 * (inverse - np.outer(alpha, alpha)) * kernel
        grad_w = np.einsum("ij,ijk->k", derivative, mismatches)
        grad_logits = weights * (grad_w - weights @ grad_w) + TK_PENALTY * logits
        loss_value = 0.5 * y @ alpha + np.log(np.diag(factor)).sum() + 0.5 * TK_PENALTY * (logits @ logits)
        return loss_value, grad_logits

    result = minimize(loss, np.zeros(11), jac=True, method="L-BFGS-B",
                      options={"maxiter": 100, "ftol": 1e-9})
    if not result.success:
        raise RuntimeError(f"TK optimization failed: {result.message}")
    return softmax(result.x), result.nit


def train_embedding_rbf(x_train, x_test, y):
    sq_distance = cdist(x_train, x_train, metric="sqeuclidean")
    typical = np.sqrt(np.median(sq_distance[np.triu_indices(len(y), k=1)]))
    if not np.isfinite(typical) or typical <= 0:
        raise ValueError("Degenerate training embedding distances")

    def loss(log_relative_scale):
        lengthscale = typical * np.exp(log_relative_scale)
        gram = np.exp(-0.5 * sq_distance / lengthscale ** 2)
        factor, alpha = gp_fit(gram, y)
        return 0.5 * y @ alpha + np.log(np.diag(factor)).sum()

    search = minimize_scalar(loss, method="bounded", bounds=(-4, 4), options={"xatol": 1e-4})
    if not search.success:
        raise RuntimeError(f"Embedding RBF optimization failed: {search.message}")
    lengthscale = float(typical * np.exp(search.x))
    return (np.exp(-0.5 * sq_distance / lengthscale ** 2),
            lambda: np.exp(-0.5 * cdist(x_test, x_train, metric="sqeuclidean") / lengthscale ** 2),
            {"lengthscale": lengthscale, "optimizer_evaluations": search.nfev})


def covariances(kernel, train, test, y, embeddings):
    parameters = {}
    if kernel in ("K2", "K3"):
        x_train, x_test = train, test
        return (x_train @ x_train.T).toarray(), lambda: (x_test @ x_train.T).toarray(), parameters
    if kernel == "K4":
        return train_embedding_rbf(embeddings[0], embeddings[1], y)
    mismatch = train[:, None, :] != train[None, :, :]
    if kernel == "K1-ARD":
        weights, iterations = train_tk(mismatch.astype(float), y)
        parameters = {"position_weights": weights.tolist(), "optimizer_iterations": iterations}
    else:
        weights = np.full(11, 1 / (9 if kernel == "K0" else 11))
    return np.exp(-(mismatch @ weights)), lambda: np.exp(-((test[:, None, :] != train[None, :, :]) @ weights)), parameters


def extract_embeddings(sequences, model_path, device, output):
    import torch
    import transformers
    from transformers import BertModel, BertTokenizer

    before = perf_counter()
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
    model = BertModel.from_pretrained(model_path, local_files_only=True, weights_only=True)
    model = model.eval().to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    loaded = perf_counter()
    rows, batch_seconds = [], []
    with torch.inference_mode():
        for start in range(0, len(sequences), 64):
            batch_started = perf_counter()
            encoded = tokenizer([" ".join(seq) for seq in sequences[start:start + 64]],
                                add_special_tokens=False, return_tensors="pt")
            if encoded.input_ids.shape[1] != 11 or (encoded.input_ids == tokenizer.unk_token_id).any():
                raise ValueError("Expected one known token per residue")
            states = model(**encoded.to(device)).last_hidden_state
            rows.append(states.mean(dim=1).cpu().numpy())
            batch_seconds.append(perf_counter() - batch_started)
    embeddings = np.concatenate(rows).astype(np.float64)
    finished = perf_counter()
    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid ProtBERT embeddings")
    np.savez_compressed(output, embeddings=embeddings,
                        sequence_sha256=np.array([hashlib.sha256(seq.encode()).hexdigest() for seq in sequences]))
    return dict(zip(sequences, embeddings)), {"model_directory": str(model_path), "device": device,
        "torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32",
        "pooling": "mean of 11 final-layer residue tokens, no special tokens", "sequences": len(sequences),
        "model_load_seconds": loaded - before, "embedding_seconds": finished - loaded,
        "batch_size": 64, "batch_seconds": batch_seconds}


def evaluate(sequences, scores, embed, kinds, predict_rows):
    sequences_array = np.array([list(s) for s in sequences], dtype="U1")
    results = []
    for kernel in kinds:
        before = perf_counter()
        features = string_features(sequences, gaps=kernel == "K3") if kernel in ("K2", "K3") else sequences_array
        embedding_matrix = np.stack([embed[s] for s in sequences]) if kernel == "K4" else None
        feature_time = perf_counter() - before
        for train_end, test_end in FOLDS:
            y, test_y = scores[:train_end], scores[train_end:test_end]
            center, scale = y.mean(), max(y.std(), 0.01)
            start = perf_counter()
            emb = None if embedding_matrix is None else (embedding_matrix[:train_end], embedding_matrix[train_end:test_end])
            gram, query, parameters = covariances(kernel, features[:train_end], features[train_end:test_end],
                                                   (y - center) / scale, emb)
            # The online RBF's Cholesky path uses the same noise and jitter.
            factor, alpha = gp_fit(gram, (y - center) / scale)
            fitted = perf_counter()
            cross = query()
            mean = center + scale * (cross @ alpha)
            solved = solve_triangular(factor, cross.T, lower=True)
            variance = (np.maximum(1 - (solved ** 2).sum(axis=0), 0) + NOISE) * scale ** 2
            predicted = perf_counter()
            error = test_y - mean
            spearman = None
            if np.ptp(test_y) > 0 and np.ptp(mean) > 0:
                spearman = float(spearmanr(test_y, mean).statistic)
            results.append({"kernel": kernel, "train": train_end, "test": len(test_y),
                "spearman": spearman, "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt((error ** 2).mean())),
                "nll": float((0.5 * (np.log(2 * np.pi * variance) + error ** 2 / variance)).mean()),
                "coverage_90": float((np.abs(error) <= 1.64485362695 * np.sqrt(variance)).mean()),
                "feature_seconds_per_ledger": feature_time, "fit_seconds": fitted - start,
                "predict_seconds": predicted - fitted, "parameters": parameters})
            predict_rows.extend((kernel, train_end, train_end + i, actual, mu, np.sqrt(var))
                                for i, (actual, mu, var) in enumerate(zip(test_y, mean, variance), 1))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    inputs = {name: read_scores(path) for name, path in (("stream", args.stream), ("barrier", args.barrier))}
    args.output.mkdir(parents=True, exist_ok=False)
    embed, embedding_info = None, None
    if args.model is not None:
        sequences = sorted({seq for data in inputs.values() for seq in data[0]})
        embed, embedding_info = extract_embeddings(sequences, args.model, args.device,
                                                   args.output / "embeddings.npz")
    kinds = KERNELS if embed is not None else KERNELS[:-1]
    report = {"folds": FOLDS, "noise": NOISE, "embedding": embedding_info,
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "python": platform.python_version(), "numpy": np.__version__, "runs": {}}
    with (args.output / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "kernel", "train_count", "ingest_seq", "observed_energy", "mean", "predictive_std"])
        for name, (seq, scores, digest) in inputs.items():
            rows = []
            metrics = evaluate(seq, scores, embed, kinds, rows)
            writer.writerows((name, *row) for row in rows)
            report["runs"][name] = {"ledger_sha256": digest, "observations": len(scores), "metrics": metrics}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
