import { request as httpRequest, type IncomingHttpHeaders, type IncomingMessage, type ServerResponse } from "node:http";
import { request as httpsRequest } from "node:https";
import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { ArtifactBudget, Redactor, TraceWriter, safeHeaders, sha256 } from "./trace.js";

interface ActiveTurn {
	profileId: string;
	sessionId: string;
	turnId: string;
	turnRoot: string;
	maxProviderCalls: number;
	requestCount: number;
	trace: TraceWriter;
	budget: ArtifactBudget;
}

export interface ProviderTurnSummary {
	providerCalls: number;
	artifactBytes: number;
}

function joinTargetUrl(baseUrl: URL, suffix: string, search: string): URL {
	const target = new URL(baseUrl);
	const basePath = target.pathname.replace(/\/+$/, "");
	const suffixPath = suffix.startsWith("/") ? suffix : `/${suffix}`;
	target.pathname = `${basePath}${suffixPath}` || "/";
	target.search = search;
	target.hash = "";
	return target;
}

function responseHeaders(headers: IncomingHttpHeaders): Record<string, string | string[]> {
	const result: Record<string, string | string[]> = {};
	for (const [name, value] of Object.entries(headers)) {
		if (value !== undefined) result[name] = value;
	}
	return result;
}

export class ProviderProxy {
	private readonly targetBaseUrl: URL;
	private readonly redactor: Redactor;
	private readonly activeTurns = new Map<string, ActiveTurn>();
	private server: Server | undefined;
	private port: number | undefined;

	constructor(baseUrl: string, private readonly apiKey: string, private readonly campaignId: string) {
		this.targetBaseUrl = new URL(baseUrl);
		if (this.targetBaseUrl.protocol !== "http:" && this.targetBaseUrl.protocol !== "https:") {
			throw new Error("provider base URL must use HTTP(S)");
		}
		if (this.targetBaseUrl.username || this.targetBaseUrl.password || this.targetBaseUrl.search || this.targetBaseUrl.hash) {
			throw new Error("provider base URL must not contain credentials, query parameters, or a fragment");
		}
		this.redactor = new Redactor([apiKey]);
	}

	async start(): Promise<void> {
		if (this.server) return;
		this.server = createServer((request, response) => {
			void this.forward(request, response);
		});
		await new Promise<void>((resolve, reject) => {
			this.server?.once("error", reject);
			this.server?.listen(0, "127.0.0.1", () => resolve());
		});
		const address = this.server.address();
		if (!address || typeof address === "string") throw new Error("provider proxy did not bind a TCP port");
		this.port = address.port;
	}

	baseUrl(profileId: string): string {
		if (!this.port) throw new Error("provider proxy is not started");
		return `http://127.0.0.1:${this.port}/${encodeURIComponent(profileId)}`;
	}

	async beginTurn(
		profileId: string,
		sessionId: string,
		turnId: string,
		turnRoot: string,
		maxProviderCalls: number,
		maxArtifactBytes: number,
	): Promise<void> {
		if (this.activeTurns.has(profileId)) throw new Error(`profile already has an active turn: ${profileId}`);
		const recovered = await existingTrace(turnRoot, turnId);
		const budget = new ArtifactBudget(maxArtifactBytes, recovered.artifactBytes);
		this.activeTurns.set(profileId, {
			profileId,
			sessionId,
			turnId,
			turnRoot,
			maxProviderCalls,
			requestCount: recovered.requestCount,
			budget,
			trace: new TraceWriter(turnRoot, this.redactor, budget),
		});
	}

	async recoveredTurnSummary(turnRoot: string, turnId: string): Promise<ProviderTurnSummary> {
		const recovered = await existingTrace(turnRoot, turnId);
		return { providerCalls: recovered.requestCount, artifactBytes: recovered.artifactBytes };
	}

	async endTurn(profileId: string): Promise<ProviderTurnSummary> {
		const active = this.activeTurns.get(profileId);
		if (!active) return { providerCalls: 0, artifactBytes: 0 };
		this.activeTurns.delete(profileId);
		await active.trace.flush();
		return { providerCalls: active.requestCount, artifactBytes: active.budget.bytes };
	}

	async close(): Promise<void> {
		if (!this.server) return;
		const server = this.server;
		this.server = undefined;
		await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
	}

	private async readRequest(request: IncomingMessage): Promise<Buffer> {
		const chunks: Buffer[] = [];
		for await (const chunk of request) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
		return Buffer.concat(chunks);
	}

	private async reject(response: ServerResponse, status: number, message: string): Promise<void> {
		response.writeHead(status, { "content-type": "application/json" });
		response.end(JSON.stringify({ error: { message, type: "ldm_harness_transport_error" } }));
	}

	private async forward(request: IncomingMessage, response: ServerResponse): Promise<void> {
		try {
			const incoming = new URL(request.url ?? "/", "http://127.0.0.1");
			const segments = incoming.pathname.split("/").filter(Boolean);
			const encodedProfile = segments.shift();
			if (!encodedProfile) return await this.reject(response, 404, "missing profile route");
			const profileId = decodeURIComponent(encodedProfile);
			const active = this.activeTurns.get(profileId);
			if (!active) return await this.reject(response, 409, "profile has no active turn");
			if (active.requestCount >= active.maxProviderCalls) {
				await active.trace.append("provider_index.jsonl", {
					type: "provider_request_blocked",
					campaignId: this.campaignId,
					profileId: active.profileId,
					sessionId: active.sessionId,
					turnId: active.turnId,
					reason: "provider_call_limit",
				});
				return await this.reject(response, 429, "provider call limit reached");
			}

			active.requestCount += 1;
			const requestId = `${active.turnId}-provider-${active.requestCount}`;
			const body = await this.readRequest(request);
			const capturedRequest = this.redactor.buffer(body);
			const target = joinTargetUrl(this.targetBaseUrl, segments.join("/"), incoming.search);
			await active.trace.writeRaw(join("provider", `${requestId}.request.bin`), capturedRequest);

			const headers: Record<string, string | string[]> = {};
			for (const [name, value] of Object.entries(request.headers)) {
				if (value !== undefined && name.toLowerCase() !== "host" && name.toLowerCase() !== "authorization") {
					headers[name] = value;
				}
			}
			headers.authorization = `Bearer ${this.apiKey}`;
			headers.host = target.host;
			headers["accept-encoding"] = "identity";

			const requester = target.protocol === "https:" ? httpsRequest : httpRequest;
			const upstream = requester(
				target,
				{ method: request.method ?? "POST", headers },
				(upstreamResponse) => {
					response.writeHead(upstreamResponse.statusCode ?? 502, upstreamResponse.headers);
					let chunkIndex = 0;
					let responseBytes = 0;
					const responseHash = createHash("sha256");
					const responseRedactor = this.redactor.stream();
					upstreamResponse.on("data", (value: Buffer | string) => {
						const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value);
						const captured = responseRedactor.update(chunk);
						chunkIndex += 1;
						responseBytes += captured.length;
						responseHash.update(captured);
						try {
							if (captured.length) {
								void active.trace.writeRaw(join("provider", `${requestId}.response.bin`), captured);
							}
							response.write(chunk);
						} catch (error) {
							upstreamResponse.destroy(error as Error);
						}
					});
					upstreamResponse.on("end", () => {
						const tail = responseRedactor.end();
						if (tail.length) {
							responseBytes += tail.length;
							responseHash.update(tail);
							void active.trace.writeRaw(join("provider", `${requestId}.response.bin`), tail);
						}
						void active.trace.append("provider_index.jsonl", {
							type: "provider_exchange",
							campaignId: this.campaignId,
							profileId: active.profileId,
							sessionId: active.sessionId,
							requestId,
							turnId: active.turnId,
							request: {
								method: request.method ?? "POST",
								url: target.toString(),
								headers: safeHeaders(request.headers),
								bytes: body.length,
								sha256: sha256(capturedRequest),
								artifact: `provider/${requestId}.request.bin`,
							},
							response: {
								status: upstreamResponse.statusCode,
								headers: safeHeaders(responseHeaders(upstreamResponse.headers)),
								chunks: chunkIndex,
								bytes: responseBytes,
								sha256: responseHash.digest("hex"),
								artifact: `provider/${requestId}.response.bin`,
							},
						});
						response.end();
					});
					upstreamResponse.on("error", (error) => response.destroy(error));
				},
			);
			upstream.on("error", async (error) => {
				await active.trace.append("provider_index.jsonl", {
					type: "provider_transport_error",
					campaignId: this.campaignId,
					profileId: active.profileId,
					sessionId: active.sessionId,
					requestId,
					turnId: active.turnId,
					message: this.redactor.text(error.message),
				});
				if (!response.headersSent) await this.reject(response, 502, "provider transport failed");
				else response.destroy(error);
			});
			upstream.end(body);
		} catch (error) {
			if (!response.headersSent) await this.reject(response, 500, this.redactor.text((error as Error).message));
			else response.destroy(error as Error);
		}
	}
}

async function existingTrace(turnRoot: string, turnId: string): Promise<{ requestCount: number; artifactBytes: number }> {
	let artifactBytes = 0;
	let requestCount = 0;
	const providerRoot = join(turnRoot, "provider");
	let names: string[] = [];
	try {
		names = await readdir(providerRoot);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
	for (const name of names) {
		artifactBytes += (await stat(join(providerRoot, name))).size;
		const prefix = turnId + "-provider-";
		if (!name.startsWith(prefix) || !name.endsWith(".request.bin")) continue;
		const value = Number(name.slice(prefix.length, -".request.bin".length));
		if (Number.isSafeInteger(value) && value > requestCount) requestCount = value;
	}
	try {
		artifactBytes += (await stat(join(turnRoot, "provider_index.jsonl"))).size;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
	return { requestCount, artifactBytes };
}
