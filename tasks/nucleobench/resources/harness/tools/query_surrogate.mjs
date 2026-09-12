import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { context, exportData, validatePatch, workspaceFile } from "./sequence_context.mjs";

const snapshotRoot = process.env.LDM_NUCLEOBENCH_SURROGATE;
if (!snapshotRoot) throw new Error("LDM_NUCLEOBENCH_SURROGATE is required");
const runner = join(dirname(fileURLToPath(import.meta.url)), "query_posterior.py");
const baseCodes = { A: 0, C: 1, G: 2, T: 3 };
const positions = context.paired_start.editable_positions;
const start = context.paired_start.start_sequence;

function predict(input, signal) {
    return new Promise((resolve, reject) => {
        const child = spawn("python3", [runner], { stdio: ["pipe", "pipe", "pipe"], signal });
        const output = [], errors = [];
        child.stdout.on("data", chunk => output.push(chunk));
        child.stderr.on("data", chunk => errors.push(chunk));
        child.on("error", reject);
        child.on("close", code => {
            if (code !== 0) return reject(new Error(Buffer.concat(errors).toString() || `Posterior worker exited with ${code}`));
            try { resolve(JSON.parse(Buffer.concat(output).toString())); }
            catch (error) { reject(error); }
        });
        child.stdin.on("error", reject);
        child.stdin.end(JSON.stringify(input));
    });
}

export default function surrogateTools(pi) {
    // Each loaded extension belongs to one session; cache hits never reveal peers' drafts.
    let cachedSnapshot = null;
    const cache = new Map();
    pi.registerTool({
        name: "query_surrogate",
        label: "Query frozen sequence surrogate",
        description: "Predict candidate utility, latent uncertainty and raw UCB from this round's frozen baseline GP. Supply candidates or artifact_path, and the snapshot_id from the turn message. This does not evaluate sequences or submit proposals. All results are exported to guest_file; the inline preview preserves input order.",
        promptSnippet: "query_surrogate: compare research hypotheses against the current baseline GP before submitting candidates",
        promptGuidelines: [
            "Use batched queries and inspect guest_file with local code. Predictions are model estimates, not measurements. A neutral_prior cannot rank candidates from data.",
            "Compare alternative designs and controls, including scientific hypotheses that disagree with the surrogate. The later compiled-policy GP and final selection may differ from this baseline.",
        ],
        parameters: {
            type: "object",
            properties: {
                snapshot_id: { type: "string", minLength: 1 },
                artifact_path: { type: "string", description: "JSON file with a candidates array; annotated candidates.json files are accepted." },
                candidates: { type: "array", minItems: 1, items: {
                    type: "object", properties: { mutations: { type: "array", items: {
                        type: "object", properties: { position: { type: "integer" }, base: { type: "string" } },
                        required: ["position", "base"], additionalProperties: false,
                    } } }, required: ["mutations"],
                } },
            },
            required: ["snapshot_id"], additionalProperties: false,
        },
        async execute(_id, params, signal, _update, ctx) {
            const began = performance.now();
            const pointer = JSON.parse(readFileSync(join(snapshotRoot, "current.json"), "utf8"));
            if (params.snapshot_id !== pointer.snapshot_id) throw new Error(
                "stale_snapshot: use the snapshot_id in the current round message; no candidates were queried");
            if ((params.candidates !== undefined) === (params.artifact_path !== undefined)) {
                throw new Error("Provide exactly one of candidates or artifact_path");
            }
            const candidates = params.candidates ?? JSON.parse(readFileSync(workspaceFile(ctx.cwd, params.artifact_path), "utf8")).candidates;
            if (!Array.isArray(candidates) || candidates.length === 0) throw new Error("candidates must be a non-empty array");
            const snapshotFile = join(snapshotRoot, pointer.file);
            const snapshot = JSON.parse(readFileSync(snapshotFile, "utf8"));
            if (cachedSnapshot !== pointer.snapshot_id) { cache.clear(); cachedSnapshot = pointer.snapshot_id; }
            const missing = new Map();
            let cacheHits = 0;
            const rows = candidates.map((candidate, index) => {
                try {
                    const patch = validatePatch(candidate?.mutations, true);
                    const key = patch.sequence_sha256;
                    if (cache.has(key)) cacheHits += 1;
                    else if (!missing.has(key)) {
                        const mutations = new Map(patch.mutations.map(item => [item.position, item.base]));
                        missing.set(key, positions.map(position => baseCodes[mutations.get(position) ?? start[position]]));
                    }
                    return { index, valid: true, sequence_sha256: key };
                } catch (error) {
                    return { index, valid: false, error: { code: "invalid_mutations", message: error.message,
                        repair_hint: "Use legal, unique editable positions and A/C/G/T bases different from the original start." } };
                }
            });
            if (missing.size) {
                const scores = await predict({ snapshot_file: snapshotFile, snapshot_id: pointer.snapshot_id,
                    codes: [...missing.values()] }, signal);
                if (scores.length !== missing.size) throw new Error("Posterior returned an inconsistent candidate count");
                [...missing.keys()].forEach((key, index) => cache.set(key, scores[index]));
            }
            for (const row of rows) if (row.valid) {
                const score = cache.get(row.sequence_sha256);
                row.objectives = { utility: { direction: "maximize", mean: score.mean, std: score.std, std_kind: "latent" } };
                row.acquisition = { name: "ucb", direction: "maximize", beta: snapshot.posterior.beta, score: score.acquisition_score };
            }
            const details = {
                snapshot_id: pointer.snapshot_id, round_index: pointer.round_index, model_role: "baseline",
                history_size: snapshot.posterior.history_size, working_set_size: snapshot.posterior.working_set_size,
                fit_status: snapshot.posterior.fit_status, candidate_count: rows.length,
                invalid_count: rows.filter(row => !row.valid).length, cache_hits: cacheHits,
                predicted_unique_candidates: missing.size, elapsed_seconds: (performance.now() - began) / 1000,
                predictions: rows,
            };
            const result = { ...details, guest_file: exportData(ctx, details), predictions: rows.slice(0, 8) };
            return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
        },
    });
}
