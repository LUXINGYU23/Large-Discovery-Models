import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { mkdir, open, readFile, realpath, writeFile } from "node:fs/promises";
import type {
	HarnessArtifactRuleConfig,
	SubmissionError,
	SubmittedArtifact,
} from "./protocol.js";
import { canonicalSha256, sha256 } from "./trace.js";

export interface TerminalSubmission {
	submissionStatus: "accepted" | "rejected";
	submissionId: string;
	submissionDigest: string;
	submission: Record<string, unknown>;
	submittedArtifacts: SubmittedArtifact[];
	validationErrors: SubmissionError[];
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
	value: Pick<TerminalSubmission, "submissionStatus" | "submissionDigest" | "submission" | "submittedArtifacts" | "validationErrors">,
	artifactRoot: string,
): Promise<void> {
	if (canonicalSha256({ artifacts: value.submittedArtifacts, submission: value.submission }) !== value.submissionDigest) {
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
