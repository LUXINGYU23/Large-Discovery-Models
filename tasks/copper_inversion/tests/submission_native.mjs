// Remote integration: actual Pi snapshot/commit code and Python task parser.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';

const { SubmissionController, verifySubmissionRecord } = await import(
  pathToFileURL(join(process.env.COPPER_HARNESS_DIST, 'submission.js')).href);
function python(code, value) {
  const result = spawnSync(process.env.COPPER_PYTHON, ['-c', code], {
    input: JSON.stringify(value), encoding: 'utf8', env: process.env,
  });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
const contract = python('import json; from tasks.copper_inversion.core.harness import submission_contract; print(json.dumps(submission_contract().to_dict()))');

test('array candidate snapshots cross the real Pi controller and Python parser', async () => {
  const root = await mkdtemp(join(tmpdir(), 'copper-array-'));
  try {
    const workspace = join(root, 'workspace');
    const turnRoot = join(root, 'turn');
    await mkdir(join(workspace, 'candidates/0'), { recursive: true });
    const source = join(workspace, 'candidates/0/solver.py');
    const payload = { candidates: [{ artifact_path: 'candidates/0/solver.py', parent_sha256: null,
      change_summary: 'fixture', rationale: 'native contract integration' }] };
    const config = { artifactRoot: root, submissionContract: contract, forceFirstToolCall: false };
    const validate = async request => python(`
import json,sys
from types import SimpleNamespace
from tasks.copper_inversion.core.harness import parse_snapshot
v=json.load(sys.stdin); r=v['request']
a=[SimpleNamespace(path_pointer=x['pathPointer'],relative_path=x['relativePath'],snapshot_path=x['snapshotPath'],sha256=x['sha256'],size_bytes=x['sizeBytes']) for x in r['artifacts']]
try:
 parse_snapshot(r['submission'],a,v['root'],set())
 result={'decision':'accept','errors':[]}
except ValueError as e:
 result={'decision':'retry','errors':[{'path':'/candidates/0/artifact_path','code':'invalid_solver','message':str(e)}]}
print(json.dumps(result))
`, { request, root });
    await writeFile(source, '# missing solve\n');
    const first = new SubmissionController(config, workspace);
    await first.begin('researcher', 'round_1', turnRoot, async () => {}, validate);
    await assert.rejects(first.tool().execute('bad', payload), /invalid_solver/);
    assert.equal(first.submission, undefined);
    // Reopen the same native turn and repair; the attempt counter must survive.
    const body = 'def solve(problem,budget,seed):\n    return problem["initial_level_set"].copy()\n';
    await writeFile(source, body);
    let persisted;
    const recovered = new SubmissionController(config, workspace);
    await recovered.begin('researcher', 'round_1', turnRoot, async value => { persisted = value; }, validate);
    await recovered.tool().execute('good', payload);
    assert.equal(persisted.submissionId, 'round_1-submission-2');
    assert.equal(persisted.submissionStatus, 'accepted');
    assert.equal(persisted.submittedArtifacts[0].pathPointer, '/candidates/0/artifact_path');
    await writeFile(source, '# changed after submission\n');
    assert.equal(await readFile(join(root, persisted.submittedArtifacts[0].snapshotPath), 'utf8'), body);
    await verifySubmissionRecord(persisted, root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
