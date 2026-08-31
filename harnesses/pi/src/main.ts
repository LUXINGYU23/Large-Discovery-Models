import { createInterface } from "node:readline";
import { PiSessionPool } from "./session.js";
import {
	PROTOCOL_VERSION,
	ProtocolError,
	parseFrame,
	type InputFrame,
	type SubmissionValidationDecision,
	type SubmissionValidationRequest,
	type SubmissionValidationResultFrame,
} from "./protocol.js";
import { Redactor } from "./trace.js";

let apiKey: string | undefined;
let campaignId: string | undefined;
let pool: PiSessionPool | undefined;
let redactor = new Redactor([]);
let closing = false;

type CommandFrame = Exclude<InputFrame, SubmissionValidationResultFrame>;

class SubmissionValidationBroker {
	private readonly pending = new Map<string, {
		requestId: string;
		resolve: (decision: SubmissionValidationDecision) => void;
		reject: (error: Error) => void;
	}>();
	private nextId = 0;

	request(
		frame: { requestId: string; campaignId: string },
		request: SubmissionValidationRequest,
	): Promise<SubmissionValidationDecision> {
		this.nextId += 1;
		const validationId = `${frame.requestId}-validation-${this.nextId.toString().padStart(6, "0")}`;
		const result = new Promise<SubmissionValidationDecision>((resolve, reject) => {
			this.pending.set(validationId, { requestId: frame.requestId, resolve, reject });
		});
		respondTo(frame, "submission_validation_requested", { validationId, ...request });
		return result;
	}

	resolve(frame: SubmissionValidationResultFrame): void {
		const pending = this.pending.get(frame.validationId);
		if (!pending || pending.requestId !== frame.requestId) {
			throw new ProtocolError("invalid_state", `unknown submission validation: ${frame.validationId}`);
		}
		this.pending.delete(frame.validationId);
		pending.resolve({ accepted: frame.accepted, rejected: frame.rejected });
	}

	rejectAll(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}
}

const validations = new SubmissionValidationBroker();

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
	validations.rejectAll(new Error("harness sidecar is closing"));
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

async function handle(frame: CommandFrame): Promise<boolean> {
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
		const turns = await pool.runTurns(
			frame.turns,
			(request) => validations.request(frame, request),
		);
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
let activeCommand: Promise<void> | undefined;
for await (const line of lines) {
	if (!line.trim()) continue;
	let requestId: string | undefined;
	let requestCampaignId: string | undefined;
	let frame: InputFrame | undefined;
	try {
		const parsed = JSON.parse(line) as { requestId?: unknown; campaignId?: unknown };
		if (typeof parsed.requestId === "string") requestId = parsed.requestId;
		if (typeof parsed.campaignId === "string") {
			requestCampaignId = parsed.campaignId;
		}
	} catch {
		// parseFrame returns the authoritative protocol error below.
	}
	try {
		frame = parseFrame(line);
	} catch (error) {
		respond({
			type: "error",
			requestId,
			protocolVersion: PROTOCOL_VERSION,
			campaignId: requestCampaignId ?? campaignId,
			error: { code: errorCode(error), message: redactor.text((error as Error).message) },
		});
		continue;
	}
	if (frame.type === "submission_validation_result") {
		try {
			validations.resolve(frame);
		} catch (error) {
			respondTo(frame, "error", {
				error: { code: errorCode(error), message: redactor.text((error as Error).message) },
			});
		}
		continue;
	}
	if (activeCommand) {
		respondTo(frame, "error", {
			error: { code: "invalid_state", message: "sidecar is processing another command" },
		});
		continue;
	}
	const operation = (async () => {
		try {
			if (!(await handle(frame))) lines.close();
		} catch (error) {
			validations.rejectAll(error as Error);
			respondTo(frame, "error", {
				error: { code: errorCode(error), message: redactor.text((error as Error).message) },
			});
		}
	})();
	activeCommand = operation;
	void operation.finally(() => {
		if (activeCommand === operation) activeCommand = undefined;
	});
}

await activeCommand;
await close();
