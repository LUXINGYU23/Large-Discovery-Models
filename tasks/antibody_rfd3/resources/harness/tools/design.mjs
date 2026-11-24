import { request } from "node:http";
import { existsSync } from "node:fs";
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
  if (existsSync("/worker/protein-tools.enabled")) {
    pi.registerTool({name: "sequence_h3_soluble", label: "sequence_h3_soluble",
      description: "Sample soluble ProteinMPNN sequences on an existing RFD3 backbone. Shares the sequence budget; does not establish expression or solubility.",
      parameters: {type: "object", additionalProperties: false,
        properties: {backbone_id: {type: "string"}, count: {type: "integer", minimum: 1}, temperature: {type: "number", minimum: 0.01, maximum: 1}},
        required: ["backbone_id", "count", "temperature"]},
      execute: (id, params, _signal, _onUpdate, ctx) => generate("sequence_h3_soluble", params, id, basename(dirname(ctx.cwd)))});
    pi.registerTool({name: "score_interface", label: "score_interface",
      description: "Compute auxiliary ipSAE for a host-prepared genuine structure/PAE artifact ID. Does not predict structures, establish affinity, or change Absolut scores.",
      parameters: {type: "object", additionalProperties: false, properties: {artifact_id: {type: "string"}}, required: ["artifact_id"]},
      execute: (id, params, _signal, _onUpdate, ctx) => generate("score_interface", params, id, basename(dirname(ctx.cwd)))});
  }
  if (existsSync("/worker/boltzgen.enabled")) pi.registerTool({name: "design_h3_boltzgen", label: "design_h3_boltzgen",
    description: "Generate one fixed-framework H3 backbone and count native inverse-folded sequences. Consumes one shared backbone and count sequence slots. Returned backbones cannot enter ProteinMPNN resequencing.",
    parameters: {type: "object", additionalProperties: false, properties: {count: {type: "integer", minimum: 1}}, required: ["count"]},
    execute: (id, params, _signal, _onUpdate, ctx) => generate("design_h3_boltzgen", params, id, basename(dirname(ctx.cwd)))});

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
