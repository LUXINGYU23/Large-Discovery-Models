import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
	snapshotSubmissionArtifacts,
	verifySubmissionRecord,
} from "./submission.js";
import { canonicalJson, sha256 } from "./trace.js";

const RULE = {
	pathPointer: "/artifact_path",
	allowedSuffixes: [".py"],
	maxBytes: 32,
};

async function fixture(): Promise<{
	root: string;
	workspace: string;
	turnRoot: string;
}> {
	const root = await mkdtemp(join(tmpdir(), "ldm-submission-"));
	const workspace = join(root, "sessions", "policy", "workspace");
	const turnRoot = join(root, "turns", "turn-1");
	await mkdir(workspace, { recursive: true });
	return { root, workspace, turnRoot };
}

test("artifact snapshots are immutable and digest-bound", async () => {
	const { root, workspace, turnRoot } = await fixture();
	try {
		await writeFile(join(workspace, "optimization_policy.py"), "VALUE = 1\n");
		const submission = { artifact_path: "optimization_policy.py" };
		const artifacts = await snapshotSubmissionArtifacts(
			[RULE],
			workspace,
			root,
			turnRoot,
			1,
			submission,
		);
		const submissionJson = canonicalJson({ artifacts, submission });
		const submissionDigest = sha256(submissionJson);
		await verifySubmissionRecord({
			submissionStatus: "accepted",
			submissionJson,
			submissionDigest,
			submission,
			submittedArtifacts: artifacts,
			validationErrors: [],
		}, root);

		await writeFile(join(workspace, "optimization_policy.py"), "VALUE = 2\n");
		assert.equal(
			await readFile(join(root, artifacts[0]!.snapshotPath), "utf8"),
			"VALUE = 1\n",
		);
		await assert.rejects(
			verifySubmissionRecord({
				submissionStatus: "accepted",
				submissionJson,
				submissionDigest: "0".repeat(64),
				submission,
				submittedArtifacts: artifacts,
				validationErrors: [],
			}, root),
			/terminal submission digest mismatch/,
		);

		const snapshotPath = join(root, artifacts[0]!.snapshotPath);
		await chmod(snapshotPath, 0o600);
		await writeFile(snapshotPath, "tampered\n");
		await assert.rejects(
			verifySubmissionRecord({
				submissionStatus: "accepted",
				submissionJson,
				submissionDigest,
				submission,
				submittedArtifacts: artifacts,
				validationErrors: [],
			}, root),
			/artifact snapshot digest mismatch/,
		);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("submission JSON orders mixed-case condition keys deterministically", () => {
	assert.equal(
		canonicalJson({
			artifacts: [],
			submission: {
				candidates: [{
					dataset_id: "reductive_amination",
					conditions: {
						substrate: "S1",
						solvent: "NMP",
						reaction_concentration_mM: 100,
						TTIP_equiv: 2,
						AcOH_equiv: 1,
					},
				}],
			},
		}),
		'{"artifacts":[],"submission":{"candidates":[{"conditions":{"AcOH_equiv":1,"TTIP_equiv":2,"reaction_concentration_mM":100,"solvent":"NMP","substrate":"S1"},"dataset_id":"reductive_amination"}]}}',
	);
});

test("artifact rules allow submissions without an artifact reference", async () => {
	const { root, workspace, turnRoot } = await fixture();
	try {
		assert.deepEqual(
			await snapshotSubmissionArtifacts([RULE], workspace, root, turnRoot, 1, { action: "keep" }),
			[],
		);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("artifact snapshots reject traversal, invalid suffixes, and oversized files", async () => {
	const { root, workspace, turnRoot } = await fixture();
	try {
		await assert.rejects(
			snapshotSubmissionArtifacts([RULE], workspace, root, turnRoot, 1, { artifact_path: "../outside.py" }),
			/unsafe_artifact_path/,
		);
		await writeFile(join(workspace, "policy.txt"), "value\n");
		await assert.rejects(
			snapshotSubmissionArtifacts([RULE], workspace, root, turnRoot, 2, { artifact_path: "policy.txt" }),
			/invalid_artifact_suffix/,
		);
		await writeFile(join(workspace, "large.py"), "x".repeat(RULE.maxBytes + 1));
		await assert.rejects(
			snapshotSubmissionArtifacts([RULE], workspace, root, turnRoot, 3, { artifact_path: "large.py" }),
			/artifact_too_large/,
		);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("artifact snapshots reject symlink escapes", async (context) => {
	const { root, workspace, turnRoot } = await fixture();
	try {
		const external = join(root, "external");
		await mkdir(external);
		await writeFile(join(external, "policy.py"), "VALUE = 1\n");
		try {
			await symlink(external, join(workspace, "linked"), process.platform === "win32" ? "junction" : "dir");
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "EPERM") {
				context.skip("symlink creation is unavailable in this environment");
				return;
			}
			throw error;
		}
		await assert.rejects(
			snapshotSubmissionArtifacts([RULE], workspace, root, turnRoot, 1, { artifact_path: "linked/policy.py" }),
			/artifact_symlink_escape/,
		);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});
