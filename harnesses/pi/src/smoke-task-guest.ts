import { spawnSync } from "node:child_process";
import { copyFile, mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import type { VM } from "@earendil-works/gondolin";
import { configureGuestCache, hostGuestArchitecture, resolveGuestRuntime } from "./guest-image.js";
import { loadTaskGuestRecipe, parseTaskGuestCommand } from "./task-guest-recipe.js";

function requireQemu(): void {
	const command = hostGuestArchitecture() === "x86_64"
		? "qemu-system-x86_64"
		: "qemu-system-aarch64";
	if (spawnSync(command, ["--version"], { stdio: "ignore" }).error) {
		throw new Error(`Task guest smoke requires ${command} and Linux KVM.`);
	}
}

async function main(): Promise<void> {
	const command = parseTaskGuestCommand(process.argv.slice(2));
	const cacheRoot = configureGuestCache(command.cacheDir);
	requireQemu();
	process.env.TMPDIR = join(cacheRoot, "runtime-overlays");
	await Promise.all([
		mkdir(join(cacheRoot, "build-tmp"), { recursive: true }),
		mkdir(process.env.TMPDIR, { recursive: true }),
	]);
	const recipe = await loadTaskGuestRecipe(command.taskId);
	const guest = await resolveGuestRuntime(recipe.taskId, recipe.guestRuntime);
	const { RealFSProvider, VM } = await import("@earendil-works/gondolin");
	const workspace = await mkdtemp(join(cacheRoot, "build-tmp", `${recipe.taskId}-guest-smoke-`));
	const smokePath = join(workspace, "smoke.sh");
	let vm: VM | undefined;
	try {
		await mkdir(join(workspace, "research"), { recursive: true });
		await copyFile(join(recipe.imageDirectory, ...recipe.smokeScript.split("/")), smokePath);
		vm = await VM.create({
			sandbox: { imagePath: guest.assetDir },
			rootfs: { mode: "cow", size: guest.rootfsSize },
			vfs: { mounts: { "/workspace": new RealFSProvider(workspace) } },
		});
		const result = await vm.exec(["/bin/sh", "-lc", "/bin/sh /workspace/smoke.sh"], {
			cwd: "/workspace",
			env: { HOME: "/root", PATH: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" },
		});
		if (result.exitCode !== 0) throw new Error(`task guest smoke failed with exit code ${result.exitCode}: ${result.stderr}`);
		const proof = await readFile(join(workspace, "research", "proof.txt"), "utf8");
		if (!proof.trim()) throw new Error("task guest smoke did not write research/proof.txt");
		process.stdout.write(`${JSON.stringify({ status: "ok", imageRef: guest.imageRef, proof: proof.trim() })}\n`);
	} finally {
		await vm?.close();
		await rm(workspace, { recursive: true, force: true });
	}
}

await main();
