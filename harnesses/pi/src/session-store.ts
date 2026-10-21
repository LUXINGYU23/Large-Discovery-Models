import { dirname, join, relative } from "node:path";
import { copyFile, mkdir, readFile, readdir, rename, rm } from "node:fs/promises";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import type { ForkSource } from "./protocol.js";
import { atomicJson, canonicalSha256, sha256 } from "./trace.js";

export async function optionalJson<T>(path: string): Promise<T | undefined> {
	try {
		return JSON.parse(await readFile(path, "utf8")) as T;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
		throw error;
	}
}

export async function directorySha256(root: string): Promise<string> {
	const files: Array<{ path: string; sha256: string }> = [];
	async function visit(directory: string): Promise<void> {
		const entries = await readdir(directory, { withFileTypes: true });
		entries.sort((left, right) => Buffer.compare(
			Buffer.from(left.name),
			Buffer.from(right.name),
		));
		for (const entry of entries) {
			const path = join(directory, entry.name);
			if (entry.isDirectory()) await visit(path);
			else if (entry.isFile()) files.push({
				path: relative(root, path).replaceAll("\\", "/"),
				sha256: sha256(await readFile(path)),
			});
			else throw new Error(`unsupported skill resource entry: ${path}`);
		}
	}
	await visit(root);
	return canonicalSha256(files);
}

export async function pinInitialization(root: string, identity: unknown): Promise<string> {
	const digest = canonicalSha256(identity);
	const manifestPath = join(root, "manifest.json");
	const previous = await optionalJson<{ initializationSha256?: string }>(manifestPath);
	if (previous) {
		if (previous.initializationSha256 !== digest) {
			throw new Error("Harness initialization identity changed; restore the original configuration and runtime or use a new artifact root");
		}
	} else {
		try {
			if ((await readdir(join(root, "sessions"))).length) {
				throw new Error("Harness sessions exist without an initialization manifest");
			}
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
		}
		await atomicJson(manifestPath, { initializationSha256: digest });
	}
	return digest;
}

export async function copyDirectory(source: string, target: string): Promise<void> {
	await mkdir(target, { recursive: true });
	const entries = await readdir(source, { withFileTypes: true });
	for (const entry of entries) {
		const sourcePath = join(source, entry.name);
		const targetPath = join(target, entry.name);
		if (entry.isDirectory()) await copyDirectory(sourcePath, targetPath);
		else if (entry.isFile()) await copyFile(sourcePath, targetPath);
		else throw new Error(`unsupported skill resource entry: ${sourcePath}`);
	}
}

/** Commit both inherited files and the native Pi fork before any branch runs. */
export async function initializeFork(snapshot: string, profileRoot: string, source: ForkSource): Promise<void> {
	const sessionPath = join(snapshot, "session.jsonl");
	if (sha256(await readFile(sessionPath)) !== source.sessionSha256
		|| await directorySha256(join(snapshot, "workspace")) !== source.workspaceSha256) {
		throw new Error("fork source digest mismatch");
	}
	const previous = await optionalJson(join(profileRoot, "fork.json"));
	if (previous) {
		if (canonicalSha256(previous) !== canonicalSha256(source)) throw new Error("fork identity changed");
		return;
	}
	try {
		await readdir(profileRoot);
		throw new Error("branch exists without an atomic fork record");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
	const staging = `${profileRoot}.fork-staging`;
	await rm(staging, { recursive: true, force: true });
	await copyDirectory(join(snapshot, "workspace"), join(staging, "workspace"));
	SessionManager.forkFrom(sessionPath, join(profileRoot, "workspace"), join(staging, "pi-session"));
	await atomicJson(join(staging, "fork.json"), source);
	await mkdir(dirname(profileRoot), { recursive: true });
	await rename(staging, profileRoot);
}
