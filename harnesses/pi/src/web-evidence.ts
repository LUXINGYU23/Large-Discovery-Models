import { constants } from "node:fs";
import { mkdir, open } from "node:fs/promises";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { sha256 } from "./trace.js";
import { EXTERNAL_WEB_CONTENT_NOTICE } from "./vendor/deepseek-web/contract.js";

export const WEB_EVIDENCE_GUEST = "/workspace/.web-evidence";
export const WEB_EVIDENCE_THRESHOLD = 8192;
export const WEB_EVIDENCE_MARKER = "ldm_web_evidence_v1";
export function isWebResult(message: AgentMessage): boolean {
	return message.role === "toolResult" && ["web_search", "web_fetch"].includes(message.toolName);
}

/** Preserve every byte of an accepted textual result; only its model-context projection changes. */
export class WebEvidenceArchive {
	constructor(readonly root: string) {}

	async project(messages: AgentMessage[], underPressure: boolean): Promise<{ messages: AgentMessage[]; archived: number }> {
		let archived = 0;
		const projected: AgentMessage[] = [];
		for (const message of messages) {
			if (message.role !== "toolResult" || !isWebResult(message) || message.isError
				|| !message.content.every((block) => block.type === "text")) {
				projected.push(message); continue;
			}
			const text = message.content.map((block) => block.type === "text" ? block.text : "").join("\n\n");
			if (text.startsWith(WEB_EVIDENCE_MARKER) || Array.from(text).length <= WEB_EVIDENCE_THRESHOLD) {
				projected.push(message); continue;
			}
			const digest = sha256(text);
			const path = join(this.root, `${digest}.md`);
			const exists = await this.verify(path, digest);
			if (!underPressure && !exists) { projected.push(message); continue; }
			if (!exists) {
				await mkdir(this.root, { recursive: true });
				try {
					const file = await open(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
					try { await file.writeFile(text, "utf8"); await file.sync(); }
					finally { await file.close(); }
				} catch (error) {
					if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
				}
				await this.verify(path, digest);
			}
			const lines = text.split("\n");
			const headings = lines.flatMap((line, index) => /^\s{0,3}#{1,6}\s+\S/u.test(line)
				? [`${index + 1}: ${Array.from(line).slice(0, 120).join("")}`] : []);
			const receipt = [WEB_EVIDENCE_MARKER, EXTERNAL_WEB_CONTENT_NOTICE,
				`Full ${message.toolName} result: ${WEB_EVIDENCE_GUEST}/${digest}.md`,
				`SHA-256: ${digest}; ${Buffer.byteLength(text)} bytes; ${lines.length} lines.`,
				"Body archived because of context pressure; this preview is not a summary or complete evidence.",
				"Use read with path, offset (1-based line), and limit to inspect any section before relying on it. Preserve this path in summaries.",
				headings.length ? `Section line numbers (${Math.min(32, headings.length)} of ${headings.length}):\n${headings.slice(0, 32).join("\n")}`
					: "No Markdown headings found; read the file in line ranges or search it with bash.",
				`Opening preview:\n${Array.from(text).slice(0, 768).join("")}`,
			].join("\n\n");
			projected.push({ ...message, content: [{ type: "text", text: receipt }] });
			archived++;
		}
		return { messages: projected, archived };
	}

	private async verify(path: string, digest: string): Promise<boolean> {
		try {
			const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
			try {
				if (!(await file.stat()).isFile() || sha256(await file.readFile()) !== digest) throw new Error("Web evidence integrity check failed");
			} finally { await file.close(); }
			return true;
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
			throw error;
		}
	}
}
