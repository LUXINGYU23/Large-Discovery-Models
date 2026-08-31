import { readFileSync } from "node:fs";

const catalogPath = process.env.LDM_IRON_MIND_CATALOG;
if (!catalogPath) throw new Error("LDM_IRON_MIND_CATALOG is required");

const catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
if (catalog.schema_version !== 1 || !Array.isArray(catalog.factors) || !Array.isArray(catalog.candidates)) {
	throw new Error("invalid Iron Mind reaction-space catalog");
}

const factorNames = catalog.factors.map((factor) => factor.name);

function jsonResult(value) {
	return { content: [{ type: "text", text: JSON.stringify(value) }], details: value };
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
		name: "describe_reaction_space",
		label: "Describe reaction space",
		description: "Return the source-pinned dataset identity, factors, legal options, and number of complete condition combinations.",
		promptSnippet: "describe_reaction_space: inspect the exact factors and legal options before choosing conditions",
		parameters: { type: "object", properties: {}, additionalProperties: false },
		async execute() {
			return jsonResult({
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
			return jsonResult({ valid: true, candidate: normalized });
		},
	});
}
