import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.env.LDM_ATOMWORLD_PUBLIC_ROOT;
if (!root) throw new Error("LDM_ATOMWORLD_PUBLIC_ROOT is required");
const result = (value) => ({ content: [{ type: "text", text: JSON.stringify(value) }], details: value });
const empty = { type: "object", properties: {}, additionalProperties: false };

export default function publicContextTools(pi) {
    pi.registerTool({
        name: "get_public_task", label: "Read public crystal question",
        description: "Get the exact public input CIF, action and sample identifier. No private labels exist here.",
        parameters: empty,
        async execute() {
            const raw = JSON.parse(readFileSync(join(root, "public_task.json"), "utf8"));
            const { sample_id, action_name, input_cif, action_prompt } = raw;
            return result({ sample_id, action_name, input_cif, action_prompt });
        },
    });
    pi.registerTool({
        name: "get_public_history", label: "Read public draft history",
        description: "Query drafts by ID for this question. Blind runs expose no labels; feedback-optimization runs include only explicitly measured past scalar correctness, never targets or judge internals.",
        parameters: { type: "object", additionalProperties: false, properties: {
            offset: { type: "integer", minimum: 0 },
            limit: { type: "integer", minimum: 1, maximum: 16 },
            draft_ids: { type: "array", maxItems: 16, items: { type: "string" } },
            detail: { type: "string", enum: ["concise", "detailed"] },
        } },
        async execute(_id, params) {
            const offset = params.offset ?? 0, limit = params.limit ?? 4;
            if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(limit) || limit < 1 || limit > 16) throw new Error("Invalid pagination");
            const { drafts } = JSON.parse(readFileSync(join(root, "public_history.json"), "utf8"));
            const detail = params.detail ?? "concise";
            if (!["concise", "detailed"].includes(detail) || (params.draft_ids !== undefined && (!Array.isArray(params.draft_ids) || params.draft_ids.length > 16 || params.draft_ids.some(id => typeof id !== "string")))) throw new Error("Invalid draft query");
            const rows = drafts.filter(row => !params.draft_ids || params.draft_ids.includes(row.draft_id));
            const page = rows.slice(offset, offset + limit).map(row => detail === "detailed" ? row : {
                draft_id: row.draft_id, round_idx: row.round_idx, public_validation: row.public_validation,
                output_sha256: row.output_sha256, rationale_preview: (row.rationale ?? "").slice(0, 160),
                ...(row.correct !== undefined ? { correct: row.correct, candidate_id: row.candidate_id } : {}),
            });
            return result({ drafts: page, total: rows.length, next_offset: offset + limit < rows.length ? offset + limit : null });
        },
    });
    pi.registerTool({
        name: "get_geometry_contract", label: "Read bounded geometry semantics",
        description: "Read the shipped geometry operation schema and instructions; execute atomworld_tools with Python inside the research guest using bash.",
        parameters: empty,
        async execute() {
            return result({ public_geometry_module: "atomworld_tools",
                instruction_source: readFileSync("/resources/image/atomworld_tools/schema.py", "utf8"),
                execution: "from atomworld_tools import execute_operations; output = execute_operations(public_input_cif, operations)",
                feedback: "Geometry and syntax only; no target comparison is available." });
        },
    });
}
