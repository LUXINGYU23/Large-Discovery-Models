import { request } from "node:http";

function generate(action, params, callId) {
  const body = JSON.stringify({action, ...params, call_id: callId});
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
      ? "Generate a new RFD3 H3 backbone, then sample sequences on it."
      : "Sample ProteinMPNN sequences on a previously generated backbone.",
    parameters: {type: "object", properties, required: Object.keys(properties), additionalProperties: false},
    execute: (id, params) => generate(name, params, id)});
  }
}
