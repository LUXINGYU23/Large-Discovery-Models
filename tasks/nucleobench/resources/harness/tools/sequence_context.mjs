import { createHash, randomUUID } from "node:crypto";
import { existsSync, lstatSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative, sep } from "node:path";

const contextPath = process.env.LDM_NUCLEOBENCH_CONTEXT;
if (!contextPath) throw new Error("LDM_NUCLEOBENCH_CONTEXT is required");
const historyPath = process.env.LDM_NUCLEOBENCH_HISTORY;
if (!historyPath) throw new Error("LDM_NUCLEOBENCH_HISTORY is required");

export const context = JSON.parse(readFileSync(contextPath, "utf8"));
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

export function exportData(ctx, value) {
	const body = JSON.stringify(value);
	const sha256 = createHash("sha256").update(body).digest("hex");
	const directory = join(ctx.cwd, ".ldm-resources", "research");
	mkdirSync(directory, { recursive: true });
	const path = join(directory, `${sha256}.json`);
	if (!existsSync(path)) writeFileSync(path, body);
	return { path: `/workspace/.ldm-resources/research/${sha256}.json`, sha256 };
}

export function workspaceFile(cwd, path) {
    if (typeof path !== "string" || !path) throw new Error("path must name a workspace JSON file");
    const local = path.startsWith("/workspace/") ? path.slice("/workspace/".length) : path;
    if (isAbsolute(local)) throw new Error("path must be relative to /workspace");
    const resolved = realpathSync(join(cwd, local));
    const offset = relative(realpathSync(cwd), resolved);
    if (offset === ".." || offset.startsWith(".." + sep) || isAbsolute(offset)) {
        throw new Error("path must stay inside this session's workspace");
    }
    return resolved;
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

export function validatePatch(mutations, allowEmpty = false) {
    if (!Array.isArray(mutations) || (!allowEmpty && mutations.length === 0)) {
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

function compileDesign(design, measured) {
    if (!design || typeof design !== "object" || Array.isArray(design)) throw new Error("design must be an object");
    const { parent_candidate_id, placements, change_summary, rationale, comparison_candidate_ids, ...extra } = design;
    if (Object.keys(extra).length) throw new Error("unknown design fields: " + Object.keys(extra).join(", "));
    if (typeof change_summary !== "string" || !change_summary.trim()
        || typeof rationale !== "string" || !rationale.trim()) {
        throw new Error("change_summary and rationale must be non-empty English notes");
    }
    if (!Array.isArray(placements) || !placements.length) throw new Error("placements must be a non-empty array");
    const parent = parent_candidate_id === undefined ? null : measured.get(parent_candidate_id);
    if (parent_candidate_id !== undefined && !parent) throw new Error("unknown measured parent_candidate_id: " + parent_candidate_id);
    const edits = new Map((parent?.mutations ?? []).map(({ position, base }) => [position, base]));
    const writes = new Map();
    for (const [index, placement] of placements.entries()) {
        exactObject(placement, ["start", "bases"], `placements[${index}]`);
        if (!Number.isInteger(placement.start) || typeof placement.bases !== "string"
            || !/^[ACGT]+$/.test(placement.bases)) {
            throw new Error(`placements[${index}] needs an integer start and a non-empty A/C/G/T string`);
        }
        for (const [offset, base] of [...placement.bases].entries()) {
            const position = placement.start + offset;
            if (!editable.has(position)) throw new Error(`placements[${index}] writes non-editable position ${position}`);
            if (writes.has(position) && writes.get(position) !== base) throw new Error(`conflicting placements at position ${position}`);
            writes.set(position, base);
            if (startSequence[position] === base) edits.delete(position);
            else edits.set(position, base);
        }
    }
    const mutations = validatePatch([...edits].map(([position, base]) => ({ position, base }))).mutations;
    const candidate = { mutations, change_summary: change_summary.trim(), rationale: rationale.trim() };
    if (comparison_candidate_ids !== undefined) {
        if (!Array.isArray(comparison_candidate_ids) || comparison_candidate_ids.some(id => !measured.has(id))) {
            throw new Error("comparison_candidate_ids must contain only exact measured IDs");
        }
        candidate.comparison_candidate_ids = comparison_candidate_ids;
    }
    return candidate;
}

export default function sequenceContextTools(pi) {
    pi.registerTool({
        name: "compile_candidate_panel",
        label: "Compile sequence designs into candidate patches",
        description: "Read the complete designs JSON file and rebuild candidates.json without writing a sequence-construction script. File shape: {designs:[{parent_candidate_id?: measured ID, placements:[{start: absolute integer, bases: concrete DNA}], change_summary: string, rationale: string, comparison_candidate_ids?: measured IDs}]}. Omit parent_candidate_id for the original start. Placements replace equal-length spans; other parent bases are retained. Each design is checked independently. This does not evaluate, rank, fill missing slots, or submit candidates.",
        promptSnippet: "compile_candidate_panel: reliably construct the full candidate file from your chosen edits and notes",
        promptGuidelines: [
            "Write compact design data with the write tool, then compile; do not rewrite a whole-panel constructor or search for motif-free filler. Use exact parent backgrounds unless a specific design requires replacement.",
            "Keep every intended design in the input file. Fix rejected design indices and recompile; successful designs remain in the output. Duplicates are reported and preserved; follow the turn's uniqueness contract. Submit only after the requested count and legality checks pass.",
            "Motif/composition diagnostics belong in research notes, not compilation gates. A failed proxy claim requires correcting the affected rationale or redesigning that entry; it must not block unrelated candidates.",
        ],
        parameters: {
            type: "object", properties: { designs_path: { type: "string", minLength: 1 } },
            required: ["designs_path"], additionalProperties: false,
        },
        async execute(_id, params, _signal, _update, ctx) {
            const inputPath = workspaceFile(ctx.cwd, params.designs_path);
            const outputPath = join(ctx.cwd, "candidates.json");
            if (inputPath === outputPath) throw new Error("keep the designs input separate from candidates.json output");
            const input = JSON.parse(readFileSync(inputPath, "utf8"));
            exactObject(input, ["designs"], "design file");
            if (!Array.isArray(input.designs) || !input.designs.length) throw new Error("designs must be a non-empty array");
            const { observations } = JSON.parse(readFileSync(historyPath, "utf8"));
            const measured = new Map(observations.map(row => [row.candidate_id, row]));
            const identity = mutations => JSON.stringify(mutations.map(({ position, base }) => [position, base]).sort((a, b) => a[0] - b[0]));
            const evaluated = new Set(observations.map(row => identity(row.mutations)));
            const candidates = [], rejected = [], designIndices = [], groups = new Map();
            for (const [index, design] of input.designs.entries()) {
                try {
                    const candidate = compileDesign(design, measured);
                    const key = identity(candidate.mutations);
                    if (evaluated.has(key)) throw new Error("historical_duplicate: the complete rebuilt sequence was already measured");
                    candidates.push(candidate);
                    designIndices.push(index);
                    if (!groups.has(key)) groups.set(key, []);
                    groups.get(key).push(index);
                } catch (error) {
                    rejected.push({ design_index: index, reason: error.message });
                }
            }
            const stat = lstatSync(outputPath, { throwIfNoEntry: false });
            if (stat && !stat.isFile()) throw new Error("candidate output must be a regular workspace file");
            const temporary = outputPath + "." + randomUUID() + ".tmp";
            try {
                writeFileSync(temporary, JSON.stringify({ candidates }), { flag: "wx" });
                renameSync(temporary, outputPath);
            } finally {
                rmSync(temporary, { force: true });
            }
            return jsonResult({
                artifact_path: "candidates.json", candidate_count: candidates.length,
                unique_candidate_count: groups.size, output_design_indices: designIndices,
                duplicate_design_groups: [...groups.values()].filter(indices => indices.length > 1), rejected,
                next_step: "Repair rejected entries in the complete design file and recompile. Check the turn's count and uniqueness contract, then call submit_candidates. Compilation alone is not submission.",
            });
        },
    });
    pi.registerTool({
        name: "get_measured_history",
        promptGuidelines: ["Read exact records from guest_file.path in sandbox scripts. The file includes all matching detailed rows, independent of pagination or response_format. These rows contain mutations; compute their mutation count as len(row['mutations']), rather than expecting the concise preview's hamming_distance field. Omit candidate_ids and round_index to export the complete authoritative evaluated set; previous proposal files are not exclusions."],
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
        async execute(_id, params, _signal, _update, ctx) {
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
                guest_file: exportData(ctx, {
                    complete_evaluated_history: ids.size === 0 && params.round_index === undefined,
                    observations: matched,
                }),
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
		description: "Return paired-start metadata and the public biological target. guest_file contains the exact start_sequence and editable_positions for scripts.",
		promptSnippet: "get_task_context: inspect the configured sequence-design context",
		parameters: { type: "object", properties: {}, additionalProperties: false },
		async execute(_id, _params, _signal, _update, ctx) {
			return jsonResult({
				guest_file: exportData(ctx, context),
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
		async execute(_id, params, _signal, _update, ctx) {
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
			const window = {
				candidate_id: params.candidate_id ?? null,
				start: params.start,
				end_exclusive: params.end_exclusive,
				bases,
				bases_sha256: createHash("sha256").update(bases, "ascii").digest("hex"),
				editable_positions: editablePositions.filter(
					(position) => params.start <= position && position < params.end_exclusive,
				),
			};
			return jsonResult({ ...window, guest_file: exportData(ctx, window) });
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
