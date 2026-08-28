import { createInterface } from "node:readline";
import { PiSessionPool } from "./session.js";
import { PROTOCOL_VERSION, ProtocolError, parseFrame } from "./protocol.js";
import { Redactor } from "./trace.js";

let apiKey: string | undefined;
let campaignId: string | undefined;
let pool: PiSessionPool | undefined;
let redactor = new Redactor([]);
let closing = false;

function respond(value: unknown): void {
	process.stdout.write(`${JSON.stringify(redactor.value(value))}\n`);
}

function respondTo(
	frame: { requestId: string; campaignId: string },
	type: string,
	fields: Record<string, unknown> = {},
): void {
	respond({
		type,
		requestId: frame.requestId,
		protocolVersion: PROTOCOL_VERSION,
		campaignId: frame.campaignId,
		...fields,
	});
}

function errorCode(error: unknown): string {
	return error instanceof ProtocolError ? error.code : "sidecar_error";
}

async function close(): Promise<void> {
	if (closing) return;
	closing = true;
	await pool?.close();
	pool = undefined;
	apiKey = undefined;
}

process.on("SIGTERM", () => {
	void close().finally(() => process.exit(143));
});
process.on("SIGINT", () => {
	void close().finally(() => process.exit(130));
});

async function handle(line: string): Promise<boolean> {
	const frame = parseFrame(line);
	if (frame.type === "bootstrap_secret") {
		if (apiKey || pool) throw new ProtocolError("invalid_state", "bootstrap_secret is accepted exactly once before initialize");
		apiKey = frame.apiKey;
		campaignId = frame.campaignId;
		redactor = new Redactor([apiKey]);
		respondTo(frame, "secret_bootstrapped");
		return true;
	}
	if (frame.campaignId !== campaignId) throw new ProtocolError("invalid_state", "campaignId changed after bootstrap");
	if (frame.type === "initialize") {
		if (!apiKey) throw new ProtocolError("invalid_state", "bootstrap_secret must precede initialize");
		if (pool) throw new ProtocolError("invalid_state", "sidecar is already initialized");
		pool = new PiSessionPool(frame, apiKey);
		try {
			await pool.initialize();
		} catch (error) {
			await pool.close();
			pool = undefined;
			throw error;
		}
		apiKey = undefined;
		respondTo(frame, "initialized", {
			profiles: frame.profiles.map((profile) => profile.profileId),
			manifest: "manifest.json",
		});
		return true;
	}
	if (frame.type === "run_turn") {
		if (!pool) throw new ProtocolError("invalid_state", "sidecar is not initialized");
		const turns = await pool.runTurns(frame.turns);
		respondTo(frame, "turn_committed", { turns });
		return true;
	}
	if (frame.type === "close") {
		await close();
		respondTo(frame, "closed");
		return false;
	}
	return true;
}

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity, terminal: false });
for await (const line of lines) {
	if (!line.trim()) continue;
	let requestId: string | undefined;
	let requestCampaignId: string | undefined;
	try {
		const parsed = JSON.parse(line) as { requestId?: unknown };
		if (typeof parsed.requestId === "string") requestId = parsed.requestId;
		if (typeof (parsed as { campaignId?: unknown }).campaignId === "string") {
			requestCampaignId = (parsed as { campaignId: string }).campaignId;
		}
	} catch {
		// parseFrame returns the authoritative protocol error below.
	}
	try {
		if (!(await handle(line))) break;
	} catch (error) {
		const response = {
			type: "error",
			requestId,
			protocolVersion: PROTOCOL_VERSION,
			campaignId: requestCampaignId ?? campaignId,
			error: { code: errorCode(error), message: redactor.text((error as Error).message) },
		};
		respond(response);
	}
}

await close();
