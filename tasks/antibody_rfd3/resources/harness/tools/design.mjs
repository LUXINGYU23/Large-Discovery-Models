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
  pi.registerTool({name: "candidate_catalog", label: "candidate_catalog",
    description: "Refresh this role's current-generation candidate catalog using the latest measured and pending history without spending generation budget.",
    parameters: {type: "object", properties: {}, additionalProperties: false},
    execute: (id, params, _signal, _onUpdate, ctx) =>
      generate("candidate_catalog", params, id, basename(dirname(ctx.cwd)))});
}
