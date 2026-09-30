import { readFile, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";

const revision = JSON.parse(await readFile(process.argv[2], "utf8")).sha;
if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error("Invalid SoL-Pi revision");
const response = await fetch(`https://raw.githubusercontent.com/NVlabs/SoL-Pi/${revision}/package.json`);
if (!response.ok) throw new Error(`SoL-Pi package lookup failed: ${response.status}`);
const upstream = await response.json();
const piVersion = upstream.devDependencies["@earendil-works/pi-coding-agent"];
if (!/^\d+\.\d+\.\d+$/.test(piVersion)) throw new Error("SoL-Pi must declare an exact tested Pi version");
const source = `https://codeload.github.com/NVlabs/SoL-Pi/tar.gz/${revision}`;
const result = spawnSync("npm", [
    "install", "--package-lock-only", "--ignore-scripts", "--no-audit", "--no-fund",
    "--save-exact", `--registry=${process.argv[3]}`,
    `sol-pi@${source}`,
    ...["pi-agent-core", "pi-ai", "pi-coding-agent", "pi-tui"].map(
        (name) => `@earendil-works/${name}@${piVersion}`,
    ),
], { stdio: "inherit" });
if (result.status !== 0) throw new Error("Unable to resolve the SoL-Pi/Pi dependency pair");
await writeFile("sol-pi-version.json", JSON.stringify({
    revision, version: upstream.version, piVersion, source,
}, null, 2) + "\n");
