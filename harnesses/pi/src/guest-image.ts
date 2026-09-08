import { readFile } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import type { Architecture, ResolvedImage } from "@earendil-works/gondolin";
import type { GuestRuntimeConfig } from "./protocol.js";
import { sha256 } from "./trace.js";

const METADATA_FIELDS = [
	"schemaVersion",
	"taskId",
	"imageId",
	"imageRef",
	"recipeSha256",
	"rootfsSize",
	"installPolicy",
	"gondolinBuildId",
	"guestManifestSha256",
	"guestArchitecture",
	"baseImageDigests",
	"ociSource",
] as const;

interface GuestBuildMetadata {
	schemaVersion: 1;
	taskId: string;
	imageId: string;
	imageRef: string;
	recipeSha256: string;
	rootfsSize: string;
	installPolicy: "session_overlay";
	gondolinBuildId: string;
	guestManifestSha256: string;
	guestArchitecture: Architecture;
	baseImageDigests: string[];
	ociSource: Record<string, unknown>;
}

export interface ResolvedGuestRuntime {
	assetDir: string;
	imageRef: string;
	recipeSha256: string;
	rootfsSize: string;
	installPolicy: "session_overlay";
	buildId: string;
	manifestSha256: string;
	architecture: Architecture;
}

function runtimeFailure(taskId: string, imageRef: string, detail: string): never {
	throw new Error(
		`Task guest is not ready for ${taskId} (${imageRef}): ${detail}. `
		+ `Build it with: npm --prefix harnesses/pi run build:task-guest -- --task ${taskId} --cache-dir <harness-cache-dir>`,
	);
}

function record(value: unknown): Record<string, unknown> | undefined {
	return value && typeof value === "object" && !Array.isArray(value)
		? value as Record<string, unknown>
		: undefined;
}

function exactMetadata(value: Record<string, unknown>): boolean {
	const actual = Object.keys(value).sort();
	const expected = [...METADATA_FIELDS].sort();
	return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function isArchitecture(value: unknown): value is Architecture {
	return value === "x86_64" || value === "aarch64";
}

export function hostGuestArchitecture(): Architecture {
	if (process.arch === "x64") return "x86_64";
	if (process.arch === "arm64") return "aarch64";
	throw new Error(`unsupported host architecture for task guest: ${process.arch}`);
}

function cacheRoot(): string {
	const value = process.env.LDM_HARNESS_CACHE_ROOT;
	if (!value || !isAbsolute(value)) {
		throw new Error("LDM_HARNESS_CACHE_ROOT must be an absolute path before starting the Harness");
	}
	const root = resolve(value);
	const expectedStore = join(root, "images");
	const expectedSessions = join(root, "sessions");
	if (resolve(process.env.GONDOLIN_IMAGE_STORE ?? "") !== expectedStore) {
		throw new Error("GONDOLIN_IMAGE_STORE must equal LDM_HARNESS_CACHE_ROOT/images");
	}
	if (resolve(process.env.GONDOLIN_SESSIONS_DIR ?? "") !== expectedSessions) {
		throw new Error("GONDOLIN_SESSIONS_DIR must equal LDM_HARNESS_CACHE_ROOT/sessions");
	}
	return root;
}

function metadataPath(root: string, taskId: string, recipeSha256: string): string {
	return join(root, "task-builds", taskId, recipeSha256, "build.json");
}

async function readMetadata(path: string): Promise<GuestBuildMetadata | undefined> {
	let raw: unknown;
	try {
		raw = JSON.parse(await readFile(path, "utf8"));
	} catch {
		return undefined;
	}
	const data = record(raw);
	if (!data || !exactMetadata(data)) return undefined;
	if (
		data.schemaVersion !== 1
		|| typeof data.taskId !== "string"
		|| typeof data.imageId !== "string"
		|| typeof data.imageRef !== "string"
		|| typeof data.recipeSha256 !== "string"
		|| typeof data.rootfsSize !== "string"
		|| data.installPolicy !== "session_overlay"
		|| typeof data.gondolinBuildId !== "string"
		|| typeof data.guestManifestSha256 !== "string"
		|| !isArchitecture(data.guestArchitecture)
		|| !Array.isArray(data.baseImageDigests)
		|| data.baseImageDigests.some((item) => typeof item !== "string" || !/^sha256:[a-f0-9]{64}$/.test(item))
		|| !record(data.ociSource)
	) return undefined;
	return data as unknown as GuestBuildMetadata;
}

export function configureGuestCache(cacheDir: string): string {
	const root = resolve(cacheDir);
	process.env.LDM_HARNESS_CACHE_ROOT = root;
	process.env.GONDOLIN_IMAGE_STORE = join(root, "images");
	process.env.GONDOLIN_SESSIONS_DIR = join(root, "sessions");
	process.env.TMPDIR = join(root, "build-tmp");
	process.env.HOME = join(root, "build-tmp", "home");
	process.env.XDG_CACHE_HOME = join(root, "build-tmp", "home", ".cache");
	return root;
}

export async function resolveGuestRuntime(
	taskId: string,
	guestRuntime: GuestRuntimeConfig,
): Promise<ResolvedGuestRuntime> {
	const root = cacheRoot();
	const metadata = await readMetadata(metadataPath(root, taskId, guestRuntime.recipeSha256));
	if (!metadata) runtimeFailure(taskId, guestRuntime.imageRef, "build metadata is missing or malformed");
	const imageId = guestRuntime.imageRef.match(/^ldm\/([a-z][a-z0-9-]*):[a-f0-9]{12}$/)?.[1];
	if (
		!imageId
		|| metadata.taskId !== taskId
		|| metadata.imageId !== imageId
		|| metadata.imageRef !== guestRuntime.imageRef
		|| metadata.recipeSha256 !== guestRuntime.recipeSha256
		|| metadata.rootfsSize !== guestRuntime.rootfsSize
		|| metadata.installPolicy !== guestRuntime.installPolicy
	) runtimeFailure(taskId, guestRuntime.imageRef, "build metadata does not match this task recipe");

	const architecture = hostGuestArchitecture();
	if (metadata.guestArchitecture !== architecture) {
		runtimeFailure(taskId, guestRuntime.imageRef, `guest architecture is ${metadata.guestArchitecture}, host requires ${architecture}`);
	}
	const { loadAssetManifest, resolveImageSelector, verifyAssets } = await import("@earendil-works/gondolin");
	let image: ResolvedImage;
	try {
		image = resolveImageSelector(guestRuntime.imageRef, architecture);
	} catch (error) {
		runtimeFailure(taskId, guestRuntime.imageRef, `local image ref cannot be resolved (${(error as Error).message})`);
	}
	if (!image.buildId || image.buildId !== metadata.gondolinBuildId || image.arch !== architecture) {
		runtimeFailure(taskId, guestRuntime.imageRef, "local image ref has unexpected build identity");
	}
	if (!verifyAssets(image.assetDir)) {
		runtimeFailure(taskId, guestRuntime.imageRef, "guest asset checksum verification failed");
	}
	const manifest = loadAssetManifest(image.assetDir);
	const manifestPath = join(image.assetDir, "manifest.json");
	if (!manifest || manifest.buildId !== metadata.gondolinBuildId || manifest.config.arch !== architecture) {
		runtimeFailure(taskId, guestRuntime.imageRef, "guest manifest does not match build metadata");
	}
	const manifestSha256 = sha256(await readFile(manifestPath));
	if (manifestSha256 !== metadata.guestManifestSha256) {
		runtimeFailure(taskId, guestRuntime.imageRef, "guest manifest checksum does not match build metadata");
	}
	return {
		assetDir: image.assetDir,
		imageRef: guestRuntime.imageRef,
		recipeSha256: guestRuntime.recipeSha256,
		rootfsSize: guestRuntime.rootfsSize,
		installPolicy: "session_overlay",
		buildId: metadata.gondolinBuildId,
		manifestSha256,
		architecture,
	};
}
