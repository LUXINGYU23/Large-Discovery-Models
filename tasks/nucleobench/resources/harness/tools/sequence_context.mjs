import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const contextPath = process.env.LDM_NUCLEOBENCH_CONTEXT;
if (!contextPath) throw new Error("LDM_NUCLEOBENCH_CONTEXT is required");
const historyPath = process.env.LDM_NUCLEOBENCH_HISTORY;
if (!historyPath) throw new Error("LDM_NUCLEOBENCH_HISTORY is required");

const context = JSON.parse(readFileSync(contextPath, "utf8"));
if (context.schema_version !== 1 || !context.case || !context.paired_start) {
	throw new Error("invalid sequence-design context");
}
const pairedStart = context.paired_start;
const startSequence = pairedStart.start_sequence;
const editablePositions = pairedStart.editable_positions;
if (typeof startSequence !== "string" || !Array.isArray(editablePositions)) {
	throw new Error("invalid paired start");
}
const editable = new Set(editablePositions);

function jsonResult(value) {
	return { content: [{ type: "text", text: JSON.stringify(value) }], details: value };
}

function exactObject(value, fields, label) {
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error(label + " must be an object");
	}
	const keys = Object.keys(value);
	const unexpected = keys.filter((key) => !fields.includes(key));
	const missing = fields.filter((key) => !(key in value));
	if (unexpected.length || missing.length) {
		throw new Error(label + " fields are invalid; missing=" + missing.join(",") + ", unexpected=" + unexpected.join(","));
	}
}

function mutableRegions() {
	const regions = [];
	for (const position of editablePositions) {
		const previous = regions.at(-1);
		if (previous && previous.end_exclusive === position) {
			previous.end_exclusive += 1;
			previous.position_count += 1;
		} else {
			regions.push({ start: position, end_exclusive: position + 1, position_count: 1 });
		}
	}
	return regions;
}

function sequenceFromPatch(mutations) {
	const sequence = [...startSequence];
	for (const mutation of mutations) sequence[mutation.position] = mutation.base;
	return sequence.join("");
}

function validatePatch(mutations) {
	if (!Array.isArray(mutations) || mutations.length === 0) {
		throw new Error("mutations must be a non-empty array");
	}
	const seen = new Set();
	const normalized = mutations.map((mutation, index) => {
		exactObject(mutation, ["position", "base"], "mutations[" + index + "]");
		const { position, base } = mutation;
		if (!Number.isInteger(position) || position < 0 || position >= startSequence.length) {
			throw new Error("mutations[" + index + "].position is outside the sequence");
		}
		if (seen.has(position)) throw new Error("position " + position + " occurs more than once");
		if (!editable.has(position)) throw new Error("position " + position + " is not editable");
		if (!/^[ACGT]$/.test(base)) throw new Error("mutations[" + index + "].base must be A, C, G, or T");
		if (startSequence[position] === base) throw new Error("position " + position + " already has base " + base);
		seen.add(position);
		return { position, base };
	}).sort((left, right) => left.position - right.position);
	return {
		valid: true,
		mutations: normalized,
		hamming_distance: normalized.length,
		sequence_sha256: createHash("sha256").update(sequenceFromPatch(normalized), "ascii").digest("hex"),
	};
}

export default function sequenceContextTools(pi) {
    pi.registerTool({
        name: "get_measured_history",
        label: "Read measured sequence history",
        description: "Query measured history by ID or round. Default concise results contain IDs, utility and mutation count; request detailed for exact patches and original design notes. Sort by utility or recency, and follow next_offset for more. Unmeasured proposals are not exposed.",
        promptSnippet: "get_measured_history: revisit measured results and the hypotheses that motivated them",
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
            const { observations } = JSON.parse(readFileSync(historyPath, "utf8"));
            const ids = new Set(params.candidate_ids ?? []);
            const matched = observations.filter((row) =>
                (ids.size === 0 || ids.has(row.candidate_id))
                && (params.round_index === undefined || row.round_index === params.round_index),
            );
            const offset = params.offset ?? 0;
            const sortBy = params.sort_by ?? "recent";
            matched.sort(sortBy === "recent"
                ? (a, b) => b.round_index - a.round_index
                : sortBy === "utility_desc" ? (a, b) => b.utility - a.utility : (a, b) => a.utility - b.utility);
            const limit = params.limit ?? 16;
            const page = [];
            let bytes = 0;
            for (const row of matched.slice(offset, offset + limit)) {
                const value = params.response_format === "detailed" ? row : {
                    candidate_id: row.candidate_id, round_index: row.round_index,
                    utility: row.utility, hamming_distance: row.mutations.length,
                };
                const size = Buffer.byteLength(JSON.stringify(value), "utf8");
                if (page.length && bytes + size > 32000) break;
                page.push(value);
                bytes += size;
            }
            return jsonResult({
                total: matched.length,
                offset,
                next_offset: offset + page.length < matched.length ? offset + page.length : null,
                observations: page,
                unmeasured_or_unknown_ids: [...ids].filter((id) =>
                    !observations.some((row) => row.candidate_id === id),
                ),
            });
        },
    });
    pi.registerTool({
		name: "get_task_context",
		label: "Get sequence-design task context",
		description: "Return paired-start metadata and the public biological target.",
		promptSnippet: "get_task_context: inspect the configured sequence-design context",
		parameters: { type: "object", properties: {}, additionalProperties: false },
		async execute() {
			return jsonResult({
				case: context.case,
				paired_start: {
					start_set_digest: pairedStart.start_set_digest,
					start_index: pairedStart.start_index,
					sequence_length: startSequence.length,
					editable_position_count: editablePositions.length,
				},
			});
		},
	});

	pi.registerTool({
		name: "get_sequence_window",
		label: "Get authoritative sequence window",
		description: "Read a zero-based half-open sequence window. Omit candidate_id for the paired start; otherwise use an exact ID from get_measured_history. Returns bases, their SHA-256, and editable positions. Unmeasured candidates are not exposed.",
		promptSnippet: "get_sequence_window: retrieve exact start or measured-parent bases and verify their checksum before editing",
		parameters: {
			type: "object",
			properties: {
				candidate_id: { type: "string", minLength: 1, description: "Exact measured candidate ID from get_measured_history; omit for the paired start." },
				start: { type: "integer", minimum: 0 },
				end_exclusive: { type: "integer", minimum: 1 },
			},
			required: ["start", "end_exclusive"],
			additionalProperties: false,
		},
		async execute(_id, params) {
			if (!Number.isInteger(params.start) || !Number.isInteger(params.end_exclusive)
				|| params.start < 0 || params.start >= params.end_exclusive
				|| params.end_exclusive > startSequence.length) {
				throw new Error("sequence window must be a non-empty range inside the paired start");
			}
			let sequence = startSequence;
			if (params.candidate_id !== undefined) {
				const { observations } = JSON.parse(readFileSync(historyPath, "utf8"));
				const measured = observations.find((row) => row.candidate_id === params.candidate_id);
				if (!measured) throw new Error("candidate_id is unmeasured or unknown: " + params.candidate_id
					+ ". Use an ID from get_measured_history, or omit candidate_id for the paired start.");
				sequence = sequenceFromPatch(measured.mutations);
			}
			const bases = sequence.slice(params.start, params.end_exclusive);
			return jsonResult({
				start: params.start,
				end_exclusive: params.end_exclusive,
				bases,
				bases_sha256: createHash("sha256").update(bases, "ascii").digest("hex"),
				editable_positions: editablePositions.filter(
					(position) => params.start <= position && position < params.end_exclusive,
				),
			});
		},
	});

	pi.registerTool({
		name: "list_mutable_regions",
		label: "List editable regions",
		description: "List contiguous zero-based half-open regions in the official editable mask.",
		promptSnippet: "list_mutable_regions: identify where legal mutations may be placed",
		parameters: { type: "object", properties: {}, additionalProperties: false },
		async execute() {
			return jsonResult({ regions: mutableRegions() });
		},
	});

	pi.registerTool({
		name: "validate_mutations",
		label: "Validate mutation patch",
		description: "Validate one start-relative mutation patch and return its normalized identity metadata.",
		promptSnippet: "validate_mutations: verify every exact patch before final submission",
		parameters: {
			type: "object",
			properties: {
				mutations: {
					type: "array",
					minItems: 1,
					items: {
						type: "object",
						properties: {
							position: { type: "integer", minimum: 0 },
							base: { type: "string", enum: ["A", "C", "G", "T"] },
						},
						required: ["position", "base"],
						additionalProperties: false,
					},
				},
			},
			required: ["mutations"],
			additionalProperties: false,
		},
		async execute(_id, params) {
			const validated = validatePatch(params.mutations);
			const { observations } = JSON.parse(readFileSync(historyPath, "utf8"));
			// Both patches are canonical and relative to the same paired start.
			const previous = observations.find((row) =>
				row.mutations.length === validated.mutations.length && row.mutations.every((mutation, index) =>
					mutation.position === validated.mutations[index].position
					&& mutation.base === validated.mutations[index].base));
			return jsonResult({
				...validated, already_evaluated: previous !== undefined,
				evaluated_candidate_id: previous?.candidate_id ?? null,
			});
		},
	});
}
