import { createHash } from "node:crypto";
import { mkdir, open, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

export class ArtifactLimitError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ArtifactLimitError";
	}
}

export class Redactor {
	private readonly secrets: Buffer[];

	constructor(secrets: string[]) {
		this.secrets = secrets.filter(Boolean).map((secret) => Buffer.from(secret));
	}

	text(value: string): string {
		let redacted = value;
		for (const secret of this.secrets) redacted = redacted.split(secret.toString()).join("[REDACTED]");
		return redacted;
	}

	buffer(value: Buffer): Buffer {
		let redacted = Buffer.from(value);
		for (const secret of this.secrets) {
			let offset = redacted.indexOf(secret);
			while (offset >= 0) {
				replacement(secret.length).copy(redacted, offset);
				offset = redacted.indexOf(secret, offset + secret.length);
			}
		}
		return redacted;
	}

	stream(): StreamingRedactor {
		return new StreamingRedactor(this.secrets);
	}

	value(value: unknown): unknown {
		if (typeof value === "string") return this.text(value);
		if (Array.isArray(value)) return value.map((item) => this.value(item));
		if (value && typeof value === "object") {
			return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, this.value(item)]));
		}
		return value;
	}
}

export class StreamingRedactor {
	private pending = Buffer.alloc(0);
	private readonly longest: number;

	constructor(private readonly secrets: Buffer[]) {
		this.longest = Math.max(1, ...secrets.map((secret) => secret.length));
	}

	update(value: Buffer): Buffer {
		this.pending = Buffer.concat([this.pending, value]);
		return this.drain(false);
	}

	end(): Buffer {
		return this.drain(true);
	}

	private drain(flush: boolean): Buffer {
		const output: Buffer[] = [];
		while (this.pending.length > 0 && (flush || this.pending.length >= this.longest)) {
			const secret = this.secrets.find((item) => this.pending.subarray(0, item.length).equals(item));
			if (secret) {
				output.push(replacement(secret.length));
				this.pending = this.pending.subarray(secret.length);
			} else {
				output.push(this.pending.subarray(0, 1));
				this.pending = this.pending.subarray(1);
			}
		}
		return Buffer.concat(output);
	}
}

function replacement(length: number): Buffer {
	const value = Buffer.alloc(length, 42);
	Buffer.from("[REDACTED]").copy(value, 0, 0, Math.min(length, 10));
	return value;
}

export class ArtifactBudget {
	private used: number;

	constructor(readonly limit: number, used = 0) {
		this.used = used;
		if (used > limit) throw new ArtifactLimitError(`turn artifact limit exceeded: ${used} > ${limit}`);
	}

	consume(bytes: number): void {
		this.used += bytes;
		if (this.used > this.limit) {
			throw new ArtifactLimitError(`turn artifact limit exceeded: ${this.used} > ${this.limit}`);
		}
	}

	get bytes(): number {
		return this.used;
	}
}

export async function atomicJson(path: string, value: unknown): Promise<void> {
	await mkdir(dirname(path), { recursive: true });
	const body = `${JSON.stringify(value, null, 2)}\n`;
	const temporary = `${path}.tmp-${process.pid}-${Date.now()}`;
	await writeFile(temporary, body, { encoding: "utf8", mode: 0o600 });
	await rename(temporary, path);
}

export class TraceWriter {
	private queue: Promise<void> = Promise.resolve();

	constructor(
		readonly root: string,
		private readonly redactor: Redactor,
		private readonly budget?: ArtifactBudget,
	) {}

	append(relativePath: string, value: unknown): Promise<void> {
		const body = `${JSON.stringify(this.redactor.value(value))}\n`;
		this.budget?.consume(Buffer.byteLength(body));
		this.queue = this.queue.then(async () => {
			const path = join(this.root, relativePath);
			await mkdir(dirname(path), { recursive: true });
			const file = await open(path, "a", 0o600);
			try {
				await file.writeFile(body, "utf8");
			} finally {
				await file.close();
			}
		});
		return this.queue;
	}

	writeRaw(relativePath: string, value: Buffer): Promise<void> {
		const body = this.redactor.buffer(value);
		this.budget?.consume(body.length);
		this.queue = this.queue.then(async () => {
			const path = join(this.root, relativePath);
			await mkdir(dirname(path), { recursive: true });
			const file = await open(path, "a", 0o600);
			try {
				await file.write(body);
			} finally {
				await file.close();
			}
		});
		return this.queue;
	}

	flush(): Promise<void> {
		return this.queue;
	}
}

export function sha256(value: Buffer | string): string {
	return createHash("sha256").update(value).digest("hex");
}

export function canonicalSha256(value: unknown): string {
	return sha256(JSON.stringify(canonicalValue(value)));
}

function canonicalValue(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(canonicalValue);
	if (value && typeof value === "object") {
		return Object.fromEntries(
			Object.entries(value)
				.sort(([left], [right]) => left.localeCompare(right))
				.map(([key, item]) => [key, canonicalValue(item)]),
		);
	}
	return value;
}

export function safeHeaders(headers: Record<string, string | string[] | undefined>): Record<string, string | string[]> {
	const safe: Record<string, string | string[]> = {};
	const sensitive = new Set([
		"api-key",
		"authorization",
		"cookie",
		"proxy-authorization",
		"set-cookie",
		"x-api-key",
		"x-goog-api-key",
	]);
	for (const [key, value] of Object.entries(headers)) {
		if (value === undefined) continue;
		safe[key] = sensitive.has(key.toLowerCase()) ? "[REDACTED]" : value;
	}
	return safe;
}
