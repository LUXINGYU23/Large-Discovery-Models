import { readFileSync } from "node:fs";

const metadata = JSON.parse(
	readFileSync(new URL("../package.json", import.meta.url), "utf8"),
) as { version?: unknown };

if (typeof metadata.version !== "string" || !metadata.version) {
	throw new Error("Pi sidecar package.json must declare a release version");
}

export const SIDECAR_RELEASE_VERSION = metadata.version;
