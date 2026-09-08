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
        description: "Page through previously submitted drafts and syntax checks for this question only. No judge measurements or correctness labels are available.",
        parameters: { type: "object", additionalProperties: false, properties: {
            offset: { type: "integer", minimum: 0 },
            limit: { type: "integer", minimum: 1, maximum: 16 },
        } },
        async execute(_id, params) {
            const offset = params.offset ?? 0, limit = params.limit ?? 4;
            if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(limit) || limit < 1 || limit > 16) throw new Error("Invalid pagination");
            const { drafts } = JSON.parse(readFileSync(join(root, "public_history.json"), "utf8"));
            const page = drafts.slice(offset, offset + limit).map(({ round_idx, generated_output, public_validation }) => ({ round_idx, generated_output, public_validation }));
            return result({ drafts: page, total: drafts.length, next_offset: offset + limit < drafts.length ? offset + limit : null });
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
