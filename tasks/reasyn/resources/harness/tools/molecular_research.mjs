import { readFileSync } from "node:fs";

const contextPath = process.env.LDM_REASYN_CONTEXT;
const historyPath = process.env.LDM_REASYN_HISTORY;
if (!contextPath || !historyPath) throw new Error("ReaSyn public context/history mounts are required");
const result = value => ({ content: [{ type: "text", text: JSON.stringify(value) }], details: value });
const history = () => JSON.parse(readFileSync(historyPath, "utf8")).observations;

export default function molecularTools(pi) {
  pi.registerTool({
    name: "describe_reasyn_task", label: "Describe ReaSyn research task",
    description: "Read the public reconstruction target or TDC objective and legal feedback protocol.",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    async execute() { return result(JSON.parse(readFileSync(contextPath, "utf8"))); }
  });
  pi.registerTool({
    name: "get_measured_history", label: "Query measured molecular history",
    description: "Page through authoritative measurements. Query an exact canonical SMILES or candidate ID; unmeasured proposals are absent.",
    parameters: { type: "object", properties: {
      offset: { type: "integer", minimum: 0 }, limit: { type: "integer", minimum: 1, maximum: 128 },
      smiles: { type: "string" }, candidate_id: { type: "string" }
    }, additionalProperties: false },
    async execute(_id, params) {
      const offset = params.offset ?? 0, limit = params.limit ?? 32;
      if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(limit) || limit < 1 || limit > 128)
        throw new Error("History offset must be nonnegative and limit an integer between 1 and 128");
      const rows = history().filter(row => (!params.smiles || row.smiles === params.smiles || row.target_smiles === params.smiles)
        && (!params.candidate_id || row.candidate_id === params.candidate_id));
      return result({ observations: rows.slice(offset, offset + limit), total: rows.length,
        next_offset: offset + limit < rows.length ? offset + limit : null });
    }
  });
  pi.registerTool({
    name: "check_measured_product", label: "Check historical canonical products",
    description: "Check exact canonical product SMILES membership. Canonicalize with RDKit in the sandbox first; this tool does not run a projector or an oracle.",
    parameters: { type: "object", properties: { smiles: { type: "array", minItems: 1, maxItems: 128,
      items: { type: "string", minLength: 1 } } }, required: ["smiles"], additionalProperties: false },
    async execute(_id, params) {
      if (!Array.isArray(params.smiles) || params.smiles.length < 1 || params.smiles.length > 128
          || params.smiles.some(smiles => typeof smiles !== "string" || !smiles.trim()))
        throw new Error("Provide between 1 and 128 canonical SMILES strings");
      const measured = new Set(history().map(row => row.smiles).filter(Boolean));
      const reconstruction = JSON.parse(readFileSync(contextPath, "utf8")).benchmark === "reconstruction";
      return result({
        identity_space: reconstruction ? "canonical_query_and_sampling_seed" : "canonical_product",
        products: params.smiles.map(smiles => ({ smiles,
          already_measured: reconstruction ? false : measured.has(smiles),
          ...(reconstruction ? { query_seen: measured.has(smiles) } : {}) })),
        ...(reconstruction ? { explanation: "A seen reconstruction query remains eligible with a new independently controlled projection seed." } : {})
      });
    }
  });
}
