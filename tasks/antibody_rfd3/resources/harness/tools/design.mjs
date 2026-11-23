import { request } from "node:http";
import { basename, dirname } from "node:path";

function generate(action, params, callId, profileId) {
  const body = JSON.stringify({action, ...params, call_id: callId, profile_id: profileId});
  return new Promise((resolve, reject) => {
    const req = request({socketPath: "/worker/design.sock", path: "/", method: "POST",
      headers: {"Content-Length": Buffer.byteLength(body)}}, res => {
      let text = "";
      res.setEncoding("utf8").on("data", part => { text += part; });
      res.on("end", () => res.statusCode === 200
        ? resolve({content: [{type: "text", text}], details: JSON.parse(text)})
        : reject(new Error(text)));
    });
    req.on("error", reject);
    req.end(body);
  });
}

export default function register(pi) {
  for (const name of ["design_h3", "sequence_h3"]) {
    const properties = {count: {type: "integer", minimum: 1},
      temperature: {type: "number", minimum: 0.01, maximum: 1}};
    if (name === "sequence_h3") properties.backbone_id = {type: "string"};
    pi.registerTool({name, label: name, description: name === "design_h3"
      ? "Generate a new RFD3 H3 backbone, then sample sequences on it; read the returned candidate_catalog when present."
      : "Sample ProteinMPNN sequences on a previously generated backbone; read the returned candidate_catalog when present.",
    parameters: {type: "object", properties, required: Object.keys(properties), additionalProperties: false},
    execute: (id, params, _signal, _onUpdate, ctx) =>
      generate(name, params, id, basename(dirname(ctx.cwd)))});
  }
  pi.registerTool({name: "mutate_h3", label: "mutate_h3",
    description: "Generate unique local sequence variants around an explicit H3 seed, including a measured historical sequence. Positions are 1-based within the 11-residue H3. Unlisted positions stay fixed. Proposals undergo the usual admission and real scoring.",
    parameters: {type: "object", additionalProperties: false,
      properties: {
        seed_sequence: {type: "string", pattern: "^[ACDEFGHIKLMNPQRSTVWY]{11}$"},
        count: {type: "integer", minimum: 1},
        mutable_positions: {type: "array", minItems: 1, uniqueItems: true,
          items: {type: "integer", minimum: 1, maximum: 11}},
        min_mutations: {type: "integer", minimum: 1, maximum: 11},
        max_mutations: {type: "integer", minimum: 1, maximum: 11},
        allowed_amino_acids: {type: "object", additionalProperties: {type: "string", pattern: "^[ACDEFGHIKLMNPQRSTVWY]+$"}},
        rationale: {type: "string", minLength: 1}},
      required: ["seed_sequence", "count", "rationale"]},
    execute: (id, params, _signal, _onUpdate, ctx) =>
      generate("mutate_h3", params, id, basename(dirname(ctx.cwd)))});
  pi.registerTool({name: "register_h3", label: "register_h3",
    description: "Issue candidate IDs for explicit H3 sequences you design. Supply the design rationale and optionally the parent seed. This spends proposal budget and does not create a backbone or claim a measured score.",
    parameters: {type: "object", additionalProperties: false,
      properties: {
        sequences: {type: "array", minItems: 1, uniqueItems: true,
          items: {type: "string", pattern: "^[ACDEFGHIKLMNPQRSTVWY]{11}$"}},
        seed_sequence: {type: "string", pattern: "^[ACDEFGHIKLMNPQRSTVWY]{11}$"},
        rationale: {type: "string", minLength: 1}},
      required: ["sequences", "rationale"]},
    execute: (id, params, _signal, _onUpdate, ctx) =>
      generate("register_h3", params, id, basename(dirname(ctx.cwd)))});
  pi.registerTool({name: "candidate_catalog", label: "candidate_catalog",
    description: "Refresh this role's current-generation candidate catalog using the latest measured and pending history without spending generation budget.",
    parameters: {type: "object", properties: {}, additionalProperties: false},
    execute: (id, params, _signal, _onUpdate, ctx) =>
      generate("candidate_catalog", params, id, basename(dirname(ctx.cwd)))});
}
