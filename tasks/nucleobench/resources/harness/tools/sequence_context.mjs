import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const contextPath = process.env.LDM_NUCLEOBENCH_CONTEXT;
if (!contextPath) throw new Error("LDM_NUCLEOBENCH_CONTEXT is required");

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
	const sequence = [...startSequence];
	for (const mutation of normalized) sequence[mutation.position] = mutation.base;
	return {
		valid: true,
		mutations: normalized,
		hamming_distance: normalized.length,
		sequence_sha256: createHash("sha256").update(sequence.join(""), "ascii").digest("hex"),
	};
}

export default function sequenceContextTools(pi) {
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
		label: "Get paired-start sequence window",
		description: "Read one zero-based half-open window of the paired start and its editable positions.",
		promptSnippet: "get_sequence_window: inspect exact start bases around positions under consideration",
		parameters: {
			type: "object",
			properties: {
				start: { type: "integer", minimum: 0 },
				end_exclusive: { type: "integer", minimum: 1 },
			},
			required: ["start", "end_exclusive"],
			additionalProperties: false,
		},
		async execute(_id, params) {
			if (params.start >= params.end_exclusive || params.end_exclusive > startSequence.length) {
				throw new Error("sequence window must be a non-empty range inside the paired start");
			}
			return jsonResult({
				start: params.start,
				end_exclusive: params.end_exclusive,
				start_bases: startSequence.slice(params.start, params.end_exclusive),
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
			return jsonResult(validatePatch(params.mutations));
		},
	});
}
