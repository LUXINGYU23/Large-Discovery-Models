import { spawn, spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { atomicJson, sha256 } from "./trace.js";
import {
	configureGuestCache,
	hostGuestArchitecture,
	resolveGuestRuntime,
} from "./guest-image.js";
import {
	loadTaskGuestRecipe,
	parseTaskGuestCommand,
	rootfsSizeMb,
} from "./task-guest-recipe.js";

function run(command: string, args: string[]): Promise<void> {
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, { stdio: "inherit" });
		child.once("error", reject);
		child.once("exit", (code) => {
			if (code === 0) resolve();
			else reject(new Error(`${command} exited with code ${code ?? "unknown"}`));
		});
	});
}

function hasCommand(command: string): boolean {
	return spawnSync(command, [], { stdio: "ignore" }).error === undefined;
}

function requireBuildTools(): void {
	const missing = ["docker", "cpio", "lz4"].filter((command) => !hasCommand(command));
	if (!hasCommand("mke2fs") && !hasCommand("mkfs.ext4")) missing.push("mke2fs or mkfs.ext4");
	if (missing.length > 0) {
		throw new Error(
			`Task guest build requires host tools: ${missing.join(", ")}. ` +
			"Install Docker, e2fsprogs, cpio, and lz4 before building a task guest.",
		);
	}
}

async function main(): Promise<void> {
	const command = parseTaskGuestCommand(process.argv.slice(2));
	const cacheRoot = configureGuestCache(command.cacheDir);
	await Promise.all([
		mkdir(join(cacheRoot, "images"), { recursive: true }),
		mkdir(join(cacheRoot, "task-builds"), { recursive: true }),
		mkdir(join(cacheRoot, "runtime-overlays"), { recursive: true }),
		mkdir(join(cacheRoot, "build-tmp"), { recursive: true }),
	]);
	const recipe = await loadTaskGuestRecipe(command.taskId);
	const existing = await resolveGuestRuntime(recipe.taskId, recipe.guestRuntime).catch(() => undefined);
	if (existing) {
		process.stdout.write(`${JSON.stringify({ status: "already_built", imageRef: existing.imageRef, buildId: existing.buildId })}\n`);
		return;
	}
	requireBuildTools();
	const { buildAssets, importImageFromDirectory, setImageRef } = await import("@earendil-works/gondolin");

	const buildRoot = join(cacheRoot, "task-builds", recipe.taskId, recipe.guestRuntime.recipeSha256);
	const localImage = `ldm-task-guest-${recipe.imageId}:${recipe.guestRuntime.recipeSha256.slice(0, 12)}`;
	const buildArgs = ["PIP_INDEX_URL", "PIP_TRUSTED_HOST"].flatMap((name) => {
		const value = process.env[name];
		return value ? ["--build-arg", `${name}=${value}`] : [];
	});
	await mkdir(buildRoot, { recursive: true });
	const temporaryAssets = await mkdtemp(join(cacheRoot, "build-tmp", `${recipe.taskId}-assets-`));
	let dockerImageBuilt = false;
	try {
		await run("docker", [
			"build",
			"--pull=false",
			...buildArgs,
			"--tag", localImage,
			"--file", join(recipe.imageDirectory, "Dockerfile"),
			recipe.imageDirectory,
		]);
		dockerImageBuilt = true;
		const architecture = hostGuestArchitecture();
		const result = await buildAssets({
			arch: architecture,
			distro: "alpine",
			oci: {
				image: localImage,
				runtime: "docker",
				platform: architecture === "aarch64" ? "linux/arm64" : "linux/amd64",
				pullPolicy: "never",
			},
			rootfs: { sizeMb: rootfsSizeMb(recipe.guestRuntime.rootfsSize) },
			runtimeDefaults: { rootfsMode: "cow" },
		}, {
			outputDir: temporaryAssets,
			workDir: join(cacheRoot, "build-tmp", `${recipe.taskId}-${recipe.guestRuntime.recipeSha256.slice(0, 12)}`),
		});
		if (!result.manifest.buildId) throw new Error("Gondolin build did not produce a build ID");
		const imported = importImageFromDirectory(temporaryAssets);
		if (imported.buildId !== result.manifest.buildId || imported.arch !== result.manifest.config.arch) {
			throw new Error("imported Gondolin image does not match the build result");
		}
		setImageRef(recipe.guestRuntime.imageRef, imported.buildId, imported.arch);
		await atomicJson(join(buildRoot, "build.json"), {
			schemaVersion: 1,
			taskId: recipe.taskId,
			imageId: recipe.imageId,
			imageRef: recipe.guestRuntime.imageRef,
			recipeSha256: recipe.guestRuntime.recipeSha256,
			rootfsSize: recipe.guestRuntime.rootfsSize,
			installPolicy: recipe.guestRuntime.installPolicy,
			gondolinBuildId: imported.buildId,
			guestManifestSha256: sha256(await readFile(join(temporaryAssets, "manifest.json"))),
			guestArchitecture: imported.arch,
			baseImageDigests: recipe.baseImageDigests,
			ociSource: result.manifest.ociSource ?? {},
		});
		const resolved = await resolveGuestRuntime(recipe.taskId, recipe.guestRuntime);
		process.stdout.write(`${JSON.stringify({ status: "built", imageRef: resolved.imageRef, buildId: resolved.buildId })}\n`);
	} finally {
		await rm(temporaryAssets, { recursive: true, force: true });
		if (dockerImageBuilt) await run("docker", ["image", "rm", "--force", localImage]).catch(() => undefined);
	}
}

await main();
