import { lstat, readFile, realpath } from "node:fs/promises";
import { dirname, join, posix, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { GuestRuntimeConfig } from "./protocol.js";
import { canonicalSha256, sha256 } from "./trace.js";

const APP_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPOSITORY_ROOT = resolve(APP_ROOT, "..", "..");
const TASK_ID_PATTERN = /^[a-z][a-z0-9_]*$/;
const IMAGE_ID_PATTERN = /^[a-z][a-z0-9-]*$/;
const ROOTFS_SIZE_PATTERN = /^[1-9][0-9]*[KMGT]$/;

export interface TaskGuestRecipe {
	taskId: string;
	imageDirectory: string;
	imageId: string;
	smokeScript: string;
	baseImageDigests: string[];
	guestRuntime: GuestRuntimeConfig;
}

export interface TaskGuestCommand {
	taskId: string;
	cacheDir: string;
}

function record(value: unknown, name: string): Record<string, unknown> {
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error(`${name} must be an object`);
	}
	return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[], name: string): void {
	const actual = Object.keys(value).sort();
	const required = [...expected].sort();
	if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
		throw new Error(`${name} has unexpected or missing fields`);
	}
}

function string(value: unknown, name: string): string {
	if (typeof value !== "string" || value.length === 0) throw new Error(`${name} must be a non-empty string`);
	return value;
}

function recipePath(value: string): string {
	if (!value || value.includes("\\")) {
		throw new Error("guest recipe paths must be non-empty POSIX relative paths");
	}
	const normalized = posix.normalize(value);
	if (normalized === "." || normalized.startsWith("../") || normalized === ".." || posix.isAbsolute(normalized)) {
		throw new Error("guest recipe paths must stay inside the image directory");
	}
	return normalized;
}

function imageDirectory(taskId: string): string {
	if (!TASK_ID_PATTERN.test(taskId)) throw new Error("--task must be a lowercase task identifier");
	return join(REPOSITORY_ROOT, "tasks", taskId, "resources", "harness", "image");
}

function commandUsage(): string {
	return "Usage: --task <task_id> --cache-dir <cache_root>";
}

export function parseTaskGuestCommand(argv: string[]): TaskGuestCommand {
	let taskId: string | undefined;
	let cacheDir: string | undefined;
	for (let index = 0; index < argv.length; index += 1) {
		const option = argv[index];
		const value = argv[index + 1];
		if (option === "--task" && value) {
			taskId = value;
			index += 1;
			continue;
		}
		if (option === "--cache-dir" && value) {
			cacheDir = value;
			index += 1;
			continue;
		}
		throw new Error(commandUsage());
	}
	if (!taskId || !cacheDir) throw new Error(commandUsage());
	if (!TASK_ID_PATTERN.test(taskId)) throw new Error("--task must be a lowercase task identifier");
	return { taskId, cacheDir: resolve(cacheDir) };
}

export async function loadTaskGuestRecipe(taskId: string): Promise<TaskGuestRecipe> {
	const root = imageDirectory(taskId);
	const resolvedRoot = await realpath(root);
	let descriptor: unknown;
	try {
		descriptor = JSON.parse(await readFile(join(root, "guest-image.json"), "utf8"));
	} catch (error) {
		throw new Error(`Unable to load task guest recipe for ${taskId}: ${(error as Error).message}`);
	}
	const data = record(descriptor, "guest-image.json");
	exactKeys(data, [
		"schemaVersion", "imageId", "rootfsSize", "installPolicy", "recipeFiles", "smokeScript",
	], "guest-image.json");
	if (data.schemaVersion !== 1) throw new Error("unsupported guest-image.json schemaVersion");
	const imageId = string(data.imageId, "guest-image.json.imageId");
	if (!IMAGE_ID_PATTERN.test(imageId)) throw new Error("guest-image.json.imageId must be lowercase hyphenated");
	const rootfsSize = string(data.rootfsSize, "guest-image.json.rootfsSize");
	if (!ROOTFS_SIZE_PATTERN.test(rootfsSize)) throw new Error("guest-image.json.rootfsSize must be a positive size");
	if (data.installPolicy !== "session_overlay") {
		throw new Error("guest-image.json.installPolicy is unsupported");
	}
	if (!Array.isArray(data.recipeFiles) || data.recipeFiles.length === 0) {
		throw new Error("guest-image.json.recipeFiles must be a non-empty array");
	}
	const recipeFiles = [...new Set(data.recipeFiles.map((item) => recipePath(string(item, "guest-image.json.recipeFiles[]"))))].sort();
	if (recipeFiles.length !== data.recipeFiles.length || !recipeFiles.includes("Dockerfile")) {
		throw new Error("guest-image.json.recipeFiles must be unique and include Dockerfile");
	}
	const smokeScript = recipePath(string(data.smokeScript, "guest-image.json.smokeScript"));
	if (!recipeFiles.includes(smokeScript)) {
		throw new Error("guest-image.json.smokeScript must be listed in recipeFiles");
	}
	const recipeEntries = await Promise.all(recipeFiles.map(async (path) => {
		const source = resolve(root, ...path.split("/"));
		if (relative(root, source).startsWith("..")) throw new Error(`guest recipe file escapes image directory: ${path}`);
		const resolvedSource = await realpath(source);
		if (relative(resolvedRoot, resolvedSource).startsWith("..")) {
			throw new Error(`guest recipe file escapes image directory: ${path}`);
		}
		const info = await lstat(source);
		if (!info.isFile()) throw new Error(`guest recipe file is not a regular file: ${path}`);
		return { path, sha256: sha256(await readFile(resolvedSource)) };
	}));
	const recipeSha256 = canonicalSha256({
		imageId,
		installPolicy: "session_overlay",
		recipeFiles: recipeEntries,
		rootfsSize,
		schemaVersion: 1,
		smokeScript,
	});
	const dockerfile = await readFile(join(root, "Dockerfile"), "utf8");
	const baseImageDigests = parseBaseImageDigests(dockerfile);
	const imageRef = `ldm/${imageId}:${recipeSha256.slice(0, 12)}`;
	const guestRuntime: GuestRuntimeConfig = {
		imageRef,
		recipeSha256,
		rootfsSize,
		installPolicy: "session_overlay",
	};
	return {
		taskId,
		imageDirectory: root,
		imageId,
		smokeScript,
		baseImageDigests,
		guestRuntime,
	};
}

function parseBaseImageDigests(dockerfile: string): string[] {
	const matches = [...dockerfile.matchAll(/^\s*FROM\s+(?:--[^\s]+\s+)*([^\s]+)/gim)];
	if (matches.length === 0) throw new Error("task guest Dockerfile must declare a FROM image");
	const digests = matches.map((match) => {
		const image = match[1];
		const digest = image?.match(/@sha256:([a-f0-9]{64})$/)?.[1];
		if (!digest) throw new Error("every task guest Dockerfile FROM image must be digest-pinned");
		return `sha256:${digest}`;
	});
	return [...new Set(digests)].sort();
}

export function rootfsSizeMb(value: string): number {
	const match = value.match(/^([1-9][0-9]*)([KMGT])$/);
	if (!match) throw new Error(`invalid rootfs size: ${value}`);
	const count = Number(match[1]);
	const multiplier = { K: 1 / 1024, M: 1, G: 1024, T: 1024 * 1024 }[match[2] as "K" | "M" | "G" | "T"];
	const result = Math.ceil(count * multiplier);
	if (!Number.isSafeInteger(result) || result < 1) throw new Error(`unsupported rootfs size: ${value}`);
	return result;
}
