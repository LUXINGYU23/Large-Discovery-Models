import { request } from "node:http";

function generate(action, params) {
  const body = JSON.stringify({action, ...params});
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
  for (const name of ["design_h3", "sequence_h3", "sequence_h3_soluble"]) {
    const properties = {count: {type: "integer", minimum: 1},
      temperature: {type: "number", minimum: 0.01, maximum: 1}};
    if (name !== "design_h3") properties.backbone_id = {type: "string"};
    pi.registerTool({name, label: name, description: name === "design_h3"
      ? "Generate a new RFD3 H3 backbone, then sample sequences on it."
      : name === "sequence_h3_soluble" ? "Sample soluble-weight ProteinMPNN on an existing backbone with the same fixed-framework constraints." : "Sample ProteinMPNN sequences on a previously generated backbone.",
    parameters: {type: "object", properties, required: Object.keys(properties), additionalProperties: false},
    execute: (_id, params) => generate(name, params)});
  }
  pi.registerTool({name: "score_interface", label: "score_interface",
    description: "Compute ipSAE for a host-prepared structure/PAE artifact ID. Does not predict structures or change Absolut scores.",
    parameters: {type: "object", properties: {artifact_id: {type: "string"}}, required: ["artifact_id"], additionalProperties: false},
    execute: (_id, params) => generate("score_interface", params)});

}
