import path from "node:path";
import { RealFSProvider, VM, createHttpHooks } from "@earendil-works/gondolin";
import {
	createBashTool,
	createReadTool,
	createWriteTool,
	type BashOperations,
	type ExtensionFactory,
	type ReadOperations,
	type WriteOperations,
} from "@earendil-works/pi-coding-agent";

const GUEST_WORKSPACE = "/workspace";

export function guestPath(hostWorkspace: string, value: string): string {
	const input = value.trim();
	if (!input || input === ".") return GUEST_WORKSPACE;
	let candidate: string;
	if (input === GUEST_WORKSPACE || input.startsWith(`${GUEST_WORKSPACE}/`)) {
		candidate = path.posix.normalize(input);
	} else if (path.isAbsolute(input)) {
		const relative = path.relative(hostWorkspace, input);
		if (relative.startsWith("..") || path.isAbsolute(relative)) {
			throw new Error("path is outside the session workspace");
		}
		candidate = path.posix.join(GUEST_WORKSPACE, relative.split(path.sep).join(path.posix.sep));
	} else {
		candidate = path.posix.resolve(GUEST_WORKSPACE, input.split(path.sep).join(path.posix.sep));
	}
	if (candidate !== GUEST_WORKSPACE && !candidate.startsWith(`${GUEST_WORKSPACE}/`)) {
		throw new Error("path is outside the session workspace");
	}
	return candidate;
}

function readOperations(vm: VM, hostWorkspace: string): ReadOperations {
	return {
		readFile: async (filePath) => vm.fs.readFile(guestPath(hostWorkspace, filePath)),
		access: async (filePath) => {
			await vm.fs.access(guestPath(hostWorkspace, filePath));
		},
		detectImageMimeType: async () => null,
	};
}

function writeOperations(vm: VM, hostWorkspace: string): WriteOperations {
	return {
		writeFile: async (filePath, content) => {
			await vm.fs.writeFile(guestPath(hostWorkspace, filePath), content, { encoding: "utf8" });
		},
		mkdir: async (directory) => {
			await vm.fs.mkdir(guestPath(hostWorkspace, directory), { recursive: true });
		},
	};
}

function bashOperations(vm: VM, hostWorkspace: string): BashOperations {
	return {
		exec: async (command, cwd, { onData, signal, timeout }) => {
			if (signal?.aborted) throw new Error("aborted");
			const controller = new AbortController();
			const abort = () => controller.abort();
			signal?.addEventListener("abort", abort, { once: true });
			let timedOut = false;
			const timer = timeout && timeout > 0
				? setTimeout(() => {
					timedOut = true;
					controller.abort();
				}, timeout * 1000)
				: undefined;
			try {
				const process = vm.exec(["/bin/sh", "-lc", command], {
					cwd: guestPath(hostWorkspace, cwd),
					env: {
						HOME: "/root",
						PATH: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
					},
					signal: controller.signal,
					stdout: "pipe",
					stderr: "pipe",
				});
				for await (const chunk of process.output()) onData(chunk.data);
				const result = await process;
				return { exitCode: result.exitCode };
			} catch (error) {
				if (signal?.aborted) throw new Error("aborted");
				if (timedOut) throw new Error(`timeout:${timeout}`);
				throw error;
			} finally {
				if (timer) clearTimeout(timer);
				signal?.removeEventListener("abort", abort);
			}
		},
	};
}

export class GondolinController {
	private vm: VM | undefined;
	private starting: Promise<VM> | undefined;

	constructor(private readonly hostWorkspace: string) {}

	createExtension(): ExtensionFactory {
		return (pi) => {
			const hostWorkspace = this.hostWorkspace;
			const localRead = createReadTool(hostWorkspace);
			const localWrite = createWriteTool(hostWorkspace);
			const localBash = createBashTool(hostWorkspace);

			const ensureVm = async (): Promise<VM> => {
				if (this.vm) return this.vm;
				if (!this.starting) {
					this.starting = (async () => {
						const { httpHooks, env } = createHttpHooks({ allowedHosts: [] });
						const created = await VM.create({
							sessionLabel: `ldm-harness-${path.basename(hostWorkspace)}`,
							httpHooks,
							env,
							vfs: { mounts: { [GUEST_WORKSPACE]: new RealFSProvider(hostWorkspace) } },
						});
						this.vm = created;
						return created;
					})().finally(() => {
						this.starting = undefined;
					});
				}
				return this.starting;
			};

			pi.on("session_start", async () => {
				await ensureVm();
			});
			pi.on("session_shutdown", async () => {
				await this.close();
			});

			pi.registerTool({
				...localRead,
				async execute(id, params, signal, onUpdate) {
					const active = await ensureVm();
					return createReadTool(GUEST_WORKSPACE, {
						operations: readOperations(active, hostWorkspace),
					}).execute(id, params, signal, onUpdate);
				},
			});
			pi.registerTool({
				...localWrite,
				async execute(id, params, signal, onUpdate) {
					const active = await ensureVm();
					return createWriteTool(GUEST_WORKSPACE, {
						operations: writeOperations(active, hostWorkspace),
					}).execute(id, params, signal, onUpdate);
				},
			});
			pi.registerTool({
				...localBash,
				async execute(id, params, signal, onUpdate) {
					const active = await ensureVm();
					return createBashTool(GUEST_WORKSPACE, {
						operations: bashOperations(active, hostWorkspace),
					}).execute(id, params, signal, onUpdate);
				},
			});
		};
	}

	async close(): Promise<void> {
		const active = this.vm;
		this.vm = undefined;
		if (!active) return;
		await active.close();
	}
}
