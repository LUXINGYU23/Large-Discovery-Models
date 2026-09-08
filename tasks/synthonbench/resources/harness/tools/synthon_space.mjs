import { readFileSync } from "node:fs";

const catalogPath = process.env.LDM_SYNTHON_SPACE_CATALOG;
if (!catalogPath) throw new Error("LDM_SYNTHON_SPACE_CATALOG is required");

const historyPath = process.env.LDM_SYNTHONBENCH_HISTORY;
if (!historyPath) throw new Error("LDM_SYNTHONBENCH_HISTORY is required");

function measuredHistory() {
    return JSON.parse(readFileSync(historyPath, "utf8")).observations;
}

const catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
if (catalog.schema_version !== 1 || !Array.isArray(catalog.reactions)) {
	throw new Error("invalid SynthonSpace tool catalog");
}

const reactions = new Map(catalog.reactions.map((reaction) => [reaction.reaction_id, reaction]));

function jsonResult(value) {
	return { content: [{ type: "text", text: JSON.stringify(value) }], details: value };
}

function reaction(value) {
	const result = reactions.get(value);
	if (!result) throw new Error(`unknown official reaction_id: ${value}`);
	return result;
}

const listParameters = {
	type: "object",
	properties: {
		query: { type: "string", description: "Optional case-insensitive text filter over reaction metadata." },
		reaction_ids: { type: "array", items: { type: "string" }, maxItems: 42 },
	},
	additionalProperties: false,
};

const searchParameters = {
	type: "object",
	properties: {
		reaction_id: { type: "string" },
		position: { type: "integer", minimum: 1 },
		query: { type: "string", description: "Optional case-insensitive substring of SMILES or synthon ID." },
		offset: { type: "integer", minimum: 0 },
		limit: { type: "integer", minimum: 1, maximum: 100 },
	},
	required: ["reaction_id"],
	additionalProperties: false,
};

const validateParameters = {
	type: "object",
	properties: {
		reaction_id: { type: "string" },
		synthon_ids: { type: "array", items: { type: "integer" }, minItems: 1, maxItems: 4 },
	},
	required: ["reaction_id", "synthon_ids"],
	additionalProperties: false,
};

export default function synthonSpaceTools(pi) {

    pi.registerTool({
        name: "get_measured_history",
        label: "Read measured research history",
        description: "Query evaluated candidates by ID or round. Concise results contain IDs, round, status and synthon_utility; request detailed for exact candidates and original research annotations. Follow next_offset. Unmeasured proposals are not exposed.",
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
        async execute(_id, params) {
            const observations = measuredHistory();
            const ids = new Set(params.candidate_ids ?? []);
            const matched = observations.filter((row) =>
                (ids.size === 0 || ids.has(row.candidate_id))
                && (params.round_index === undefined || row.round_index === params.round_index),
            );
            const sortBy = params.sort_by ?? "recent";
            matched.sort((a, b) => {
                if (sortBy === "recent") return b.round_index - a.round_index;
                if (a.synthon_utility === null) return b.synthon_utility === null ? 0 : 1;
                if (b.synthon_utility === null) return -1;
                return sortBy === "utility_desc" ? b.synthon_utility - a.synthon_utility : a.synthon_utility - b.synthon_utility;
            });
            const offset = params.offset ?? 0;
            const page = [];
            let bytes = 0;
            for (const row of matched.slice(offset, offset + (params.limit ?? 16))) {
                const value = params.response_format === "detailed" ? row : {
                    candidate_id: row.candidate_id, round_index: row.round_index,
                    evaluation_status: row.evaluation_status, synthon_utility: row.synthon_utility,
                };
                const size = Buffer.byteLength(JSON.stringify(value), "utf8");
                if (page.length && bytes + size > 32000) break;
                page.push(value);
                bytes += size;
            }
            const known = new Set(observations.map((row) => row.candidate_id));
            return jsonResult({
                total: matched.length, offset,
                next_offset: offset + page.length < matched.length ? offset + page.length : null,
                observations: page,
                unmeasured_or_unknown_ids: [...ids].filter((id) => !known.has(id)),
            });
        },
    });

	pi.registerTool({
		name: "list_synthon_reactions",
		label: "List SynthonSpace reactions",
		description: "List official reaction types, slot counts, synthon counts, product-space sizes, and public reaction metadata.",
		promptSnippet: "list_synthon_reactions: inspect available official reaction types before choosing a search direction",
		parameters: listParameters,
		async execute(_id, params) {
			const selected = params.reaction_ids?.length
				? params.reaction_ids.map(reaction)
				: [...reactions.values()];
			const query = params.query?.trim().toLowerCase();
			const filtered = query
				? selected.filter((item) => JSON.stringify(item.metadata ?? {}).toLowerCase().includes(query)
					|| item.reaction_id.toLowerCase().includes(query))
				: selected;
			return jsonResult({ reactions: filtered.map((item) => ({
				reaction_id: item.reaction_id,
				positions: item.positions.map((slot) => ({ position: slot.position, synthon_count: slot.synthons.length })),
				product_count: item.positions.reduce((total, slot) => total * slot.synthons.length, 1),
				metadata: item.metadata ?? {},
			})) });
		},
	});

	pi.registerTool({
		name: "search_synthon_space",
		label: "Search official SynthonSpace",
		description: "Retrieve valid official synthons and public SMILES for one reaction, optionally restricted to a slot or text query.",
		promptSnippet: "search_synthon_space: inspect exact legal synthon IDs and structures for a chosen reaction",
		parameters: searchParameters,
		async execute(_id, params) {
			const item = reaction(params.reaction_id);
			const offset = params.offset ?? 0;
			const limit = params.limit ?? 30;
			const query = params.query?.trim().toLowerCase();
			const slots = params.position === undefined
				? item.positions
				: item.positions.filter((slot) => slot.position === params.position);
			if (slots.length === 0) throw new Error("position is not valid for this reaction");
			return jsonResult({
				reaction_id: item.reaction_id,
				slots: slots.map((slot) => {
					const matches = query
						? slot.synthons.filter((synthon) => synthon.smiles.toLowerCase().includes(query)
							|| String(synthon.synthon_id).includes(query))
						: slot.synthons;
					return {
						position: slot.position,
						total_matches: matches.length,
						offset,
						next_offset: offset + limit < matches.length ? offset + limit : null,
						synthons: matches.slice(offset, offset + limit),
					};
				}),
			});
		},
	});

	pi.registerTool({
		name: "validate_synthon_candidate",
		label: "Validate SynthonSpace candidate",
		description: "Check that a reaction_id plus ordered synthon_ids tuple is legal in the official SynthonSpace snapshot.",
		promptSnippet: "validate_synthon_candidate: verify exact candidate legality before final submission",
		parameters: validateParameters,
		async execute(_id, params) {
			const item = reaction(params.reaction_id);
			if (params.synthon_ids.length !== item.positions.length) {
				throw new Error(`reaction ${item.reaction_id} requires ${item.positions.length} ordered synthon IDs`);
			}
			const synthons = params.synthon_ids.map((synthonId, index) => {
				const slot = item.positions[index];
				const synthon = slot.synthons.find((candidate) => candidate.synthon_id === synthonId);
				if (!synthon) throw new Error(`synthon ${synthonId} is invalid for position ${slot.position}`);
				return { position: slot.position, ...synthon };
			});
			const measured = measuredHistory().find((row) => row.reaction_id === item.reaction_id
                && row.synthon_ids.length === params.synthon_ids.length
                && row.synthon_ids.every((id, index) => id === params.synthon_ids[index]));
            return jsonResult({ valid: true, reaction_id: item.reaction_id, synthons,
                already_evaluated: measured !== undefined, candidate_id: measured?.candidate_id ?? null });
		},
	});
}
