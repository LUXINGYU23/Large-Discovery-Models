import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const contextPath = process.env.LDM_RG_CONTEXT;
const historyPath = process.env.LDM_RG_HISTORY;
if (!contextPath || !historyPath) throw new Error("ResearchGym context and measured-history mounts are required");
const checker = join(dirname(fileURLToPath(import.meta.url)), "check_candidates.py");
const result = value => ({ content: [{ type: "text", text: JSON.stringify(value) }], details: value });
const context = () => JSON.parse(readFileSync(contextPath, "utf8"));
const history = () => JSON.parse(readFileSync(historyPath, "utf8")).observations;

function exportData(ctx, value) {
    const body = JSON.stringify(value);
    const sha256 = createHash("sha256").update(body).digest("hex");
    const directory = join(ctx.cwd, ".ldm-resources", "research");
    mkdirSync(directory, { recursive: true });
    const path = join(directory, `${sha256}.json`);
    if (!existsSync(path)) writeFileSync(path, body);
    return { path: `/workspace/.ldm-resources/research/${sha256}.json`, sha256 };
}

function workspaceFile(cwd, path) {
    if (typeof path !== "string" || !path) throw new Error("artifact_path must name a workspace JSON file");
    const local = path.startsWith("/workspace/") ? path.slice("/workspace/".length) : path;
    if (isAbsolute(local)) throw new Error("artifact_path must be relative to /workspace");
    const resolved = realpathSync(join(cwd, local));
    const offset = relative(realpathSync(cwd), resolved);
    if (offset === ".." || offset.startsWith(".." + sep) || isAbsolute(offset)) {
        throw new Error("artifact_path must stay inside this session's workspace");
    }
    return resolved;
}

function runChecker(input, signal) {
    return new Promise((resolve, reject) => {
        const child = spawn("python3", [checker], { stdio: ["pipe", "pipe", "pipe"], signal });
        const output = [], errors = [];
        child.stdout.on("data", chunk => output.push(chunk));
        child.stderr.on("data", chunk => errors.push(chunk));
        child.on("error", reject);
        child.on("close", code => {
            if (code !== 0) return reject(new Error(Buffer.concat(errors).toString() || `checker exited with ${code}`));
            try { resolve(JSON.parse(Buffer.concat(output).toString())); } catch (error) { reject(error); }
        });
        child.stdin.on("error", reject);
        child.stdin.end(JSON.stringify(input));
    });
}

export default function researchgymTools(pi) {
    pi.registerTool({
        name: "describe_researchgym_case",
        label: "Describe the ResearchGym case",
        description: "Return the public case interface, scoring protocol, static rules and submission facts. guest_file contains the same contract plus the pinned public reference source files for sandbox reading.",
        parameters: { type: "object", properties: {}, additionalProperties: false },
        async execute(_id, _params, _signal, _update, ctx) {
            const value = context();
            const { reference_sources, ...contract } = value;
            return result({ ...contract,
                reference_files: Object.fromEntries(Object.entries(reference_sources || {}).map(([k, v]) => [k, v.length])),
                guest_file: exportData(ctx, value) });
        },
    });
    pi.registerTool({
        name: "get_measured_history",
        label: "Query measured candidates",
        description: "Query authoritative evaluated candidates (including failures) by ID, round or status, sorted by evaluation order or objective. concise rows omit program source; detailed rows include source, all metrics, errors and original research notes. guest_file always contains every matching detailed row.",
        parameters: { type: "object", properties: {
            candidate_ids: { type: "array", maxItems: 64, items: { type: "string" } },
            round: { type: "integer", minimum: 0 },
            status: { type: "string", enum: ["succeeded", "failed", "timed_out", "invalid"] },
            sort_by: { type: "string", enum: ["seq", "objective"] },
            order: { type: "string", enum: ["asc", "desc"] },
            offset: { type: "integer", minimum: 0 },
            limit: { type: "integer", minimum: 1, maximum: 64 },
            detail: { type: "string", enum: ["concise", "detailed"] },
        }, additionalProperties: false },
        async execute(_id, params, _signal, _update, ctx) {
            const offset = params.offset ?? 0, limit = params.limit ?? 16;
            const sort = params.sort_by ?? "seq", order = params.order ?? "asc", detail = params.detail ?? "concise";
            if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(limit) || limit < 1 || limit > 64
                || !["seq", "objective"].includes(sort) || !["asc", "desc"].includes(order)
                || !["concise", "detailed"].includes(detail)) throw new Error("Invalid history query");
            const rows = history().filter(row => (!params.candidate_ids || params.candidate_ids.includes(row.candidate_id))
                && (params.round === undefined || row.round_idx === params.round)
                && (!params.status || row.status === params.status));
            const rank = row => sort === "seq" ? row.seq : (row.objective ?? -Infinity);
            rows.sort((a, b) => (order === "asc" ? 1 : -1) * (rank(a) - rank(b) || a.seq - b.seq));
            const page = rows.slice(offset, offset + limit).map(row => detail === "detailed" ? row : {
                seq: row.seq, candidate_id: row.candidate_id, round_idx: row.round_idx, status: row.status,
                objective: row.objective, error: row.error ? row.error.slice(0, 200) : "",
                annotation_count: row.research_annotations.length, program_chars: row.program.length,
            });
            return result({ observations: page, total: rows.length,
                next_offset: offset + limit < rows.length ? offset + limit : null,
                guest_file: exportData(ctx, { filter: params, observations: rows }) });
        },
    });
    pi.registerTool({
        name: "check_candidate_programs",
        label: "Check a candidates.json draft",
        description: "Statically check a workspace candidates.json with the task's admission rules: fields, entry point, imports, measured repeats and duplicates within the file. Passing does not mean the program trains or scores well.",
        parameters: { type: "object", properties: {
            artifact_path: { type: "string", minLength: 1 },
        }, required: ["artifact_path"], additionalProperties: false },
        async execute(_id, params, signal, _update, ctx) {
            const data = JSON.parse(readFileSync(workspaceFile(ctx.cwd, params.artifact_path), "utf8"));
            if (!data || !Array.isArray(data.candidates)) throw new Error("the file must contain a candidates array");
            const value = context();
            const measured = history().map(row => row.canonical_key);
            return result(await runChecker({ catalog: { global_rules: value.static_rules }, case: value.checker_case,
                candidates: data.candidates, measured_keys: measured }, signal));
        },
    });
}
