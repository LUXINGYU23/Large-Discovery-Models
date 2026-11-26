// Guest-level integration check for WSL read-only /workspace/reference_solvers mount.
// Runs inside the real Pi sidecar container with PiAgent's mounts and Gondolin VM.
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, posix, resolve } from "node:path";
import { GondolinController } from "/app/dist/gondolin.js";
import { configureGuestCache, resolveGuestRuntime } from "/app/dist/guest-image.js";
import { parseFrame } from "/app/dist/protocol.js";

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

configureGuestCache("/runtime-home/.cache/gondolin");
const probeSpec = JSON.parse(await readFile("/probe/spec.json", "utf8"));
const runtime = await resolveGuestRuntime("copper_inversion", probeSpec.runtime);

// 1. Verify protocol.ts parseFrame accepts PiHarnessConfig's readOnlyWorkspaceMounts frame
//    and rejects invalid/escaping mounts.
const parsed = parseFrame(JSON.stringify(probeSpec.initializeFrame));
if (
  parsed.type !== "initialize" ||
  JSON.stringify(parsed.readOnlyWorkspaceMounts) !== JSON.stringify({ reference_solvers: "/artifacts/reference_solvers" })
) {
  throw new Error("parseFrame did not preserve readOnlyWorkspaceMounts");
}
for (const badMounts of [
  { "../escape": "/artifacts/reference_solvers" },
  { candidates: "/artifacts/reference_solvers" },
  { reference_solvers: "/artifacts/sessions/researcher_0/workspace" },
  { reference_solvers: "/etc" },
]) {
  let rejected = false;
  try {
    parseFrame(JSON.stringify({ ...probeSpec.initializeFrame, readOnlyWorkspaceMounts: badMounts }));
  } catch {
    rejected = true;
  }
  if (!rejected) {
    throw new Error("parseFrame accepted invalid readOnlyWorkspaceMounts: " + JSON.stringify(badMounts));
  }
}

// 2. Boot the real Gondolin VM with /workspace writable and /workspace/reference_solvers mounted read-only.
const workspace = "/artifacts/sessions/researcher_0/workspace";
await mkdir(join(workspace, ".ldm-resources"), { recursive: true });
const roMounts = Object.fromEntries(
  Object.entries(parsed.readOnlyWorkspaceMounts ?? {}).map(([guest, host]) => [
    posix.join("/workspace", guest),
    resolve(host),
  ]),
);
const controller = new GondolinController(
  workspace,
  join(workspace, ".ldm-resources"),
  { allowedHosts: [], deniedHosts: ["*"], forbiddenQueryPatterns: [] },
  runtime,
  roMounts,
);

try {
  const tools = await controller.toolOptions();

  // Read via Pi's read tool succeeds and matches pinned SHA-256.
  const refBytes = await tools.read.operations.readFile("reference_solvers/rbf_sdf.py");
  if (sha256(refBytes) !== probeSpec.expectedSha256.rbf_sdf) {
    throw new Error("read tool returned unexpected bytes for reference_solvers/rbf_sdf.py");
  }

  // Write via Pi's write tool to reference_solvers must fail.
  const writeToolDenied = {};
  for (const rel of ["reference_solvers/rbf_sdf.py", "reference_solvers/injected.py"]) {
    try {
      await tools.write.operations.writeFile(rel, "# tampered\n");
      writeToolDenied[rel] = false;
    } catch {
      writeToolDenied[rel] = true;
    }
  }
  if (!Object.values(writeToolDenied).every(Boolean)) {
    throw new Error("write tool modified read-only reference_solvers: " + JSON.stringify(writeToolDenied));
  }

  // Copying and editing a candidate under /workspace/candidates/0/solver.py via write tool succeeds.
  const editedText = refBytes.toString("utf8") + "\n# candidate adaptation in writable workspace\n";
  await tools.write.operations.mkdir("candidates/0");
  await tools.write.operations.writeFile("candidates/0/solver.py", editedText);
  const readBackCandidate = await tools.read.operations.readFile("candidates/0/solver.py");
  if (readBackCandidate.toString("utf8") !== editedText) {
    throw new Error("candidate copy/edit in writable /workspace/candidates/0/solver.py failed");
  }

  // Inside the guest Linux VM (bash + Python): verify reading works, editing candidate works,
  // and every overwrite/unlink/chmod/rmdir/rename/replace attempt on /workspace/reference_solvers fails.
  const guestScript = `import hashlib, json, os, pathlib, shutil, subprocess

expected = ${JSON.stringify(probeSpec.expectedSha256)}
ref_dir = pathlib.Path("/workspace/reference_solvers")
assert ref_dir.is_dir(), "reference_solvers missing in /workspace"
assert "reference_solvers" in os.listdir("/workspace"), "reference_solvers not listed in /workspace"

for name, exp_sha in expected.items():
    data = (ref_dir / f"{name}.py").read_bytes()
    assert hashlib.sha256(data).hexdigest() == exp_sha, f"digest mismatch for {name}"

# Copy into /workspace/candidates/1/solver.py and edit inside guest Python
cand1 = pathlib.Path("/workspace/candidates/1/solver.py")
cand1.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(ref_dir / "knn_idw.py", cand1)
cand1.write_text(cand1.read_text(encoding="utf-8") + "\\n# guest python edit\\n", encoding="utf-8")
assert "# guest python edit" in cand1.read_text(encoding="utf-8")

# Mutation attempts inside guest must all fail
attempts = {}
def must_fail(label, fn):
    try:
        fn()
    except OSError as exc:
        attempts[label] = type(exc).__name__
        return
    raise AssertionError(f"mutation unexpectedly succeeded: {label}")

must_fail("overwrite_file", lambda: (ref_dir / "rbf_sdf.py").write_text("bad", encoding="utf-8"))
must_fail("create_file", lambda: (ref_dir / "extra.py").write_text("bad", encoding="utf-8"))
must_fail("unlink_file", lambda: (ref_dir / "rbf_sdf.py").unlink())
must_fail("chmod_file", lambda: os.chmod(ref_dir / "rbf_sdf.py", 0o666))
must_fail("chmod_dir", lambda: os.chmod(ref_dir, 0o777))
must_fail("rename_inside", lambda: os.rename(ref_dir / "rbf_sdf.py", ref_dir / "renamed.py"))
must_fail("rename_dir", lambda: os.rename(ref_dir, "/workspace/reference_solvers_moved"))

for cmd_name, cmd in [
    ("rm_rf_dir", ["rm", "-rf", "/workspace/reference_solvers"]),
    ("mv_replace_dir", ["mv", "/workspace/candidates", "/workspace/reference_solvers"]),
]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0, f"{cmd_name} unexpectedly succeeded"
    attempts[cmd_name] = f"rc={proc.returncode}"

# Re-verify all reference files remain byte-identical after every mutation attempt
for name, exp_sha in expected.items():
    data = (ref_dir / f"{name}.py").read_bytes()
    assert hashlib.sha256(data).hexdigest() == exp_sha, f"post-attack digest mismatch for {name}"

print(json.dumps({"status": "passed", "attempts_blocked": attempts}, sort_keys=True))
`;
  await tools.write.operations.writeFile("wsl-guest-check.py", guestScript);
  let output = "";
  const execRes = await tools.bash.operations.exec("python /workspace/wsl-guest-check.py", "/workspace", {
    timeout: 45,
    onData: (chunk) => {
      output += chunk.toString();
    },
  });
  if (execRes.exitCode !== 0) {
    throw new Error(`guest check failed (exit ${execRes.exitCode}): ${output}`);
  }
  const summary = {
    ...JSON.parse(output.trim()),
    write_tool_denied: writeToolDenied,
    candidate_written_bytes: readBackCandidate.length,
  };
  await writeFile("/artifacts/wsl-guest-probe-result.json", JSON.stringify(summary, null, 2));
  process.stdout.write(JSON.stringify(summary) + "\n");
} finally {
  await controller.close();
}
