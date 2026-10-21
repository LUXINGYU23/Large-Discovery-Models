import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { mkdir, open, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { Type, type TUnsafe } from "typebox";
import type { ExtensionFactory, ToolDefinition } from "@earendil-works/pi-coding-agent";
import type {
	InitializeFrame,
	SubmissionValidator,
	HarnessArtifactRuleConfig,
	SubmissionError,
	SubmittedArtifact,
} from "./protocol.js";
import { atomicJson, canonicalJson, sha256 } from "./trace.js";

export interface TerminalSubmission {
	submissionStatus: "accepted" | "rejected";
	submissionId: string;
	submissionJson: string;
	submissionDigest: string;
	submission: Record<string, unknown>;
	submittedArtifacts: SubmittedArtifact[];
	validationErrors: SubmissionError[];
}

export class SubmissionController {
	private readonly parameters: TUnsafe<Record<string, unknown>>;
	private providerRequests = 0;
	private attemptIndex = 0;
	private profileId = "";
	private turnId = "";
	private turnRoot = "";
	private value: TerminalSubmission | undefined;
	private persist: ((submission: TerminalSubmission) => Promise<void>) | undefined;
	private validate: SubmissionValidator | undefined;

	constructor(
		private readonly config: InitializeFrame,
		private readonly workspace: string,
	) {
		this.parameters = Type.Unsafe<Record<string, unknown>>(
			config.submissionContract.payloadSchema,
		);
	}

	async begin(
		profileId: string,
		turnId: string,
		turnRoot: string,
		persist: (submission: TerminalSubmission) => Promise<void>,
		validate: SubmissionValidator,
	): Promise<void> {
		this.profileId = profileId;
		this.turnId = turnId;
		this.turnRoot = turnRoot;
		this.providerRequests = 0;
		this.attemptIndex = 0;
		try {
			for (const name of await readdir(join(turnRoot, "attempts"))) {
				if (/^\d+$/.test(name)) this.attemptIndex = Math.max(this.attemptIndex, Number(name));
			}
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
		}
		this.value = undefined;
		this.persist = persist;
		this.validate = validate;
	}

	get submission(): TerminalSubmission | undefined {
		return this.value;
	}

	createExtension(): ExtensionFactory {
		return (pi) => {
			pi.on("before_provider_request", (event) => {
				const payload = event.payload;
				if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
					throw new Error("provider payload must be a JSON object");
				}
				this.providerRequests += 1;
				if (this.value) return payload;
				if (this.providerRequests > 1 || this.config.forceFirstToolCall === false) return payload;
				return {
					...payload,
					tool_choice: "required",
				};
			});
		};
	}

	tool(): ToolDefinition<TUnsafe<Record<string, unknown>>> {
		const toolName = this.config.submissionContract.toolName;
		return {
			name: toolName,
			label: "Submit result",
			description: "Submit the complete structured result. Validation errors must be repaired before resubmission.",
			promptSnippet: `${toolName}: submit the complete structured result`,
			parameters: this.parameters,
			executionMode: "sequential",
			execute: async (_toolCallId, params) => {
				if (this.value) throw new Error(`${toolName} may be called only once per turn`);
				this.attemptIndex += 1;
				if (!this.validate) throw new Error("submission validator is not initialized");
				const submission = { ...params };
				const attemptRoot = join(
					this.turnRoot,
					"attempts",
					this.attemptIndex.toString().padStart(4, "0"),
				);
				let artifacts: SubmittedArtifact[];
				try {
					artifacts = await snapshotSubmissionArtifacts(
						this.config.submissionContract.artifactRules,
						this.workspace,
						this.config.artifactRoot,
						this.turnRoot,
						this.attemptIndex,
						submission,
					);
				} catch (error) {
					await rm(attemptRoot, { recursive: true, force: true });
					throw error;
				}
				const submissionJson = canonicalJson({ artifacts, submission });
				const submissionDigest = sha256(submissionJson);
				const decision = await this.validate({
					profileId: this.profileId,
					turnId: this.turnId,
					attemptIndex: this.attemptIndex,
					submission,
					artifacts,
					submissionJson,
					submissionDigest,
				});
				if ((decision.decision === "accept") !== (decision.errors.length === 0)) {
					throw new Error("submission validator returned an inconsistent decision");
				}
				await atomicJson(join(attemptRoot, "validation.json"), {
					attemptIndex: this.attemptIndex,
					decision: decision.decision,
					errors: decision.errors,
					submission,
					submissionJson,
					submissionDigest,
					artifacts,
				});
				if (decision.decision === "retry") {
					throw new Error(
						"Submission rejected by the task validator. Repair the reported paths and resubmit. "
						+ `Validation report: ${JSON.stringify(decision)}`,
					);
				}
				const terminal: TerminalSubmission = {
					submissionStatus: decision.decision === "accept" ? "accepted" : "rejected",
					submissionId: `${this.turnId}-submission-${this.attemptIndex}`,
					submissionDigest,
					submission,
					submittedArtifacts: artifacts,
					validationErrors: decision.errors,
					submissionJson,
				};
				if (!this.persist) throw new Error("submission turn is not initialized");
				await this.persist(terminal);
				this.value = terminal;
				return {
					content: [{
						type: "text",
						text: decision.decision === "accept"
							? "Submission accepted. End this turn now."
							: "Submission rejected and the turn is closed.",
					}],
					details: {
						submissionId: terminal.submissionId,
						status: terminal.submissionStatus,
					},
				};
			},
		};
	}

}

export async function snapshotSubmissionArtifacts(
	rules: HarnessArtifactRuleConfig[],
	workspace: string,
	artifactRoot: string,
	turnRoot: string,
	attemptIndex: number,
	submission: Record<string, unknown>,
): Promise<SubmittedArtifact[]> {
	const artifacts: SubmittedArtifact[] = [];
	for (const [index, rule] of rules.entries()) {
		const rawPath = pointerValue(submission, rule.pathPointer);
		if (rawPath === undefined) continue;
		if (typeof rawPath !== "string" || !rawPath) {
			throw artifactError(rule.pathPointer, "invalid_artifact_path", "Artifact path must be a non-empty string.");
		}
		if (
			rawPath.includes("\\")
			|| rawPath.startsWith("/")
			|| rawPath.split("/").some((part) => part === "" || part === "." || part === "..")
		) {
			throw artifactError(rule.pathPointer, "unsafe_artifact_path", "Artifact path must stay inside the session workspace.");
		}
		const suffix = rule.allowedSuffixes.find((value) => rawPath.endsWith(value));
		if (!suffix) {
			throw artifactError(rule.pathPointer, "invalid_artifact_suffix", `Artifact suffix must be one of: ${rule.allowedSuffixes.join(", ")}.`);
		}
		const sourcePath = resolve(workspace, rawPath);
		if (!isInside(workspace, sourcePath)) {
			throw artifactError(rule.pathPointer, "unsafe_artifact_path", "Artifact path escapes the session workspace.");
		}
		let resolvedSource: string;
		try {
			resolvedSource = await realpath(sourcePath);
		} catch {
			throw artifactError(rule.pathPointer, "artifact_not_found", "Artifact file does not exist.");
		}
		if (!isInside(workspace, resolvedSource)) {
			throw artifactError(rule.pathPointer, "artifact_symlink_escape", "Artifact resolves outside the session workspace.");
		}
		const handle = await open(resolvedSource, "r");
		let body: Buffer;
		try {
			const stat = await handle.stat();
			if (!stat.isFile()) {
				throw artifactError(rule.pathPointer, "artifact_not_file", "Artifact path must reference a regular file.");
			}
			if (stat.size > rule.maxBytes) {
				throw artifactError(rule.pathPointer, "artifact_too_large", `Artifact exceeds the ${rule.maxBytes}-byte limit.`);
			}
			body = await handle.readFile();
		} finally {
			await handle.close();
		}
		if (body.length > rule.maxBytes) {
			throw artifactError(rule.pathPointer, "artifact_too_large", `Artifact exceeds the ${rule.maxBytes}-byte limit.`);
		}
		const snapshotPath = resolve(
			turnRoot,
			"attempts",
			attemptIndex.toString().padStart(4, "0"),
			"artifacts",
			`artifact-${index.toString().padStart(2, "0")}${suffix}`,
		);
		if (!isInside(turnRoot, snapshotPath)) throw new Error("artifact snapshot path escapes turn root");
		await mkdir(dirname(snapshotPath), { recursive: true });
		await writeFile(snapshotPath, body, { flag: "wx", mode: 0o400 });
		artifacts.push({
			pathPointer: rule.pathPointer,
			relativePath: rawPath,
			snapshotPath: relative(artifactRoot, snapshotPath).replaceAll("\\", "/"),
			sha256: sha256(body),
			sizeBytes: body.length,
		});
	}
	return artifacts;
}

export async function verifySubmissionRecord(
	value: Pick<TerminalSubmission, "submissionStatus" | "submissionJson" | "submissionDigest" | "submission" | "submittedArtifacts" | "validationErrors">,
	artifactRoot: string,
): Promise<void> {
	const expectedJson = canonicalJson({
		artifacts: value.submittedArtifacts,
		submission: value.submission,
	});
	if (value.submissionJson !== expectedJson || sha256(value.submissionJson) !== value.submissionDigest) {
		throw new Error("terminal submission digest mismatch");
	}
	if ((value.submissionStatus === "accepted") !== (value.validationErrors.length === 0)) {
		throw new Error("terminal submission status is inconsistent with validation errors");
	}
	for (const artifact of value.submittedArtifacts) {
		const snapshotPath = resolve(artifactRoot, artifact.snapshotPath);
		if (!isInside(artifactRoot, snapshotPath)) throw new Error("artifact snapshot path escapes artifact root");
		const body = await readFile(snapshotPath);
		if (body.length !== artifact.sizeBytes || sha256(body) !== artifact.sha256) {
			throw new Error(`artifact snapshot digest mismatch: ${artifact.snapshotPath}`);
		}
	}
}

function pointerValue(root: Record<string, unknown>, pointer: string): unknown {
	let value: unknown = root;
	for (const rawToken of pointer.slice(1).split("/")) {
		if (/~(?:[^01]|$)/.test(rawToken)) throw new Error(`invalid JSON Pointer in submission contract: ${pointer}`);
		const token = rawToken.replaceAll("~1", "/").replaceAll("~0", "~");
		if (!value || typeof value !== "object") return undefined;
		value = (value as Record<string, unknown>)[token];
	}
	return value;
}

function artifactError(path: string, code: string, message: string): Error {
	return new Error(`Submission artifact rejected: ${JSON.stringify({
		path,
		code,
		message,
		hint: "Repair the artifact reference and resubmit.",
	})}`);
}

function isInside(root: string, value: string): boolean {
	const path = relative(resolve(root), resolve(value));
	return path === "" || (path !== ".." && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}
