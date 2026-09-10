import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const catalogPath = process.env.LDM_IRON_MIND_CATALOG;
if (!catalogPath) throw new Error("LDM_IRON_MIND_CATALOG is required");

const historyPath = process.env.LDM_IRON_MIND_HISTORY;
if (!historyPath) throw new Error("LDM_IRON_MIND_HISTORY is required");

function measuredHistory() {
    return JSON.parse(readFileSync(historyPath, "utf8")).observations;
}

const catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
if (catalog.schema_version !== 1 || !Array.isArray(catalog.factors) || !Array.isArray(catalog.candidates)) {
	throw new Error("invalid Iron Mind reaction-space catalog");
}

const factorNames = catalog.factors.map((factor) => factor.name);

function jsonResult(value) {
	return { content: [{ type: "text", text: JSON.stringify(value) }], details: value };
}

function exportData(ctx, value) {
	const body = JSON.stringify(value);
	const sha256 = createHash("sha256").update(body).digest("hex");
	const directory = join(ctx.cwd, ".ldm-resources", "research");
	mkdirSync(directory, { recursive: true });
	const path = join(directory, `${sha256}.json`);
	if (!existsSync(path)) writeFileSync(path, body);
	return { path: `/workspace/.ldm-resources/research/${sha256}.json`, sha256 };
}

function normalizedCandidate(value) {
	if (!value || value.dataset_id !== catalog.dataset_id || typeof value.conditions !== "object" || value.conditions === null) {
		throw new Error(`candidate must use dataset_id ${catalog.dataset_id} and a conditions object`);
	}
	const names = Object.keys(value.conditions);
	if (names.length !== factorNames.length || names.some((name) => !factorNames.includes(name))) {
		throw new Error(`conditions must contain exactly: ${factorNames.join(", ")}`);
	}
	return {
		dataset_id: catalog.dataset_id,
		conditions: Object.fromEntries(factorNames.map((name) => [name, value.conditions[name]])),
	};
}

function key(value) {
	return JSON.stringify(normalizedCandidate(value));
}

const candidatesByKey = new Map(catalog.candidates.map((candidate) => [key(candidate), candidate]));

const searchParameters = {
	type: "object",
	properties: {
		conditions: { type: "object", description: "Optional exact partial factor filter." },
		query: { type: "string", description: "Optional case-insensitive text filter over condition values." },
		offset: { type: "integer", minimum: 0 },
		limit: { type: "integer", minimum: 1, maximum: 100 },
	},
	additionalProperties: false,
};

const validateParameters = {
	type: "object",
	properties: {
		dataset_id: { type: "string" },
		conditions: { type: "object" },
	},
	required: ["dataset_id", "conditions"],
	additionalProperties: false,
};

export default function reactionSpaceTools(pi) {

    pi.registerTool({
        name: "get_measured_history",
        promptGuidelines: ["Read exact records from guest_file.path in sandbox scripts. The file includes all matching detailed rows, independent of pagination or response_format. Omit candidate_ids and round_index to export the complete authoritative evaluated set; previous proposal files are not exclusions."],
        label: "Read measured research history",
        description: "Query evaluated candidates by ID or round. Concise results contain IDs, round, status and reaction_score; request detailed for exact candidates and original research annotations. Follow next_offset. Unmeasured proposals are not exposed.",
        parameters: {
            type: "object",
            properties: {
                candidate_ids: { type: "array", items: { type: "string", minLength: 1 } },
                round_index: { type: "integer", minimum: 0 },
                sort_by: { type: "string", enum: ["recent", "utility_desc", "utility_asc"] },
                response_format: { type: "string", enum: ["concise", "detailed"] },
                offset: { type: "integer", minimum: 0 },
                limit: { type: "integer", minimum: 1, maximum: 128 },
            },
            additionalProperties: false,
        },
        async execute(_id, params, _signal, _update, ctx) {
            const observations = measuredHistory();
            const ids = new Set(params.candidate_ids ?? []);
            const matched = observations.filter((row) =>
                (ids.size === 0 || ids.has(row.candidate_id))
                && (params.round_index === undefined || row.round_index === params.round_index),
            );
            const sortBy = params.sort_by ?? "recent";
            matched.sort((a, b) => {
                if (sortBy === "recent") return b.round_index - a.round_index;
                if (a.reaction_score === null) return b.reaction_score === null ? 0 : 1;
                if (b.reaction_score === null) return -1;
                return sortBy === "utility_desc" ? b.reaction_score - a.reaction_score : a.reaction_score - b.reaction_score;
            });
            const offset = params.offset ?? 0;
            const page = [];
            let bytes = 0;
            for (const row of matched.slice(offset, offset + (params.limit ?? 16))) {
                const value = params.response_format === "detailed" ? row : {
                    candidate_id: row.candidate_id, round_index: row.round_index,
                    evaluation_status: row.evaluation_status, reaction_score: row.reaction_score,
                };
                const size = Buffer.byteLength(JSON.stringify(value), "utf8");
                if (page.length && bytes + size > 32000) break;
                page.push(value);
                bytes += size;
            }
            const known = new Set(observations.map((row) => row.candidate_id));
            return jsonResult({
                guest_file: exportData(ctx, {
                    complete_evaluated_history: ids.size === 0 && params.round_index === undefined,
                    observations: matched,
                }),
                total: matched.length, offset,
                next_offset: offset + page.length < matched.length ? offset + page.length : null,
                observations: page,
                unmeasured_or_unknown_ids: [...ids].filter((id) => !known.has(id)),
            });
        },
    });

	pi.registerTool({
		name: "describe_reaction_space",
		label: "Describe reaction space",
		description: "Return source-pinned factors and legal options. guest_file contains the complete legal condition catalog for scripts, without hidden outcomes.",
		promptSnippet: "describe_reaction_space: inspect the exact factors and legal options before choosing conditions",
		parameters: { type: "object", properties: {}, additionalProperties: false },
		async execute(_id, _params, _signal, _update, ctx) {
			return jsonResult({
				guest_file: exportData(ctx, catalog),
				dataset_id: catalog.dataset_id,
				schema_sha256: catalog.schema_sha256,
				condition_count: catalog.condition_count,
				factors: catalog.factors,
			});
		},
	});

	pi.registerTool({
		name: "search_reaction_conditions",
		label: "Search reaction conditions",
		description: "Retrieve legal complete reaction-condition candidates using exact partial factor filters or text search.",
		promptSnippet: "search_reaction_conditions: inspect exact legal complete condition combinations",
		parameters: searchParameters,
		async execute(_id, params) {
			const filters = params.conditions ?? {};
			for (const name of Object.keys(filters)) {
				if (!factorNames.includes(name)) throw new Error(`unknown factor: ${name}`);
			}
			const query = params.query?.trim().toLowerCase();
			const matches = catalog.candidates.filter((candidate) =>
				Object.entries(filters).every(([name, value]) => candidate.conditions[name] === value)
				&& (!query || JSON.stringify(candidate.conditions).toLowerCase().includes(query))
			);
			const offset = params.offset ?? 0;
			const limit = params.limit ?? 30;
			return jsonResult({
				total_matches: matches.length,
				offset,
				next_offset: offset + limit < matches.length ? offset + limit : null,
				candidates: matches.slice(offset, offset + limit),
			});
		},
	});

	pi.registerTool({
		name: "validate_reaction_candidate",
		label: "Validate reaction candidate",
		description: "Check that a complete dataset_id and conditions object is legal in the source-pinned reaction table.",
		promptSnippet: "validate_reaction_candidate: verify exact candidate legality before final submission",
		parameters: validateParameters,
		async execute(_id, params) {
			const normalized = normalizedCandidate(params);
			if (!candidatesByKey.has(JSON.stringify(normalized))) {
				throw new Error("candidate is not present in the source-pinned reaction table");
			}
			const measured = measuredHistory().find((row) => key(row) === key(normalized));
            return jsonResult({ valid: true, candidate: normalized,
                already_evaluated: measured !== undefined, candidate_id: measured?.candidate_id ?? null });
		},
	});
}
