// AT-09: run inside the rebuilt sidecar (tests/rsi_sidecar_probe.py) against the real Gondolin guest,
// with exactly the mounts PiAgent uses: this branch's root at /artifacts, /resources/rsi read-only.
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { GondolinController } from '/app/dist/gondolin.js';
import { configureGuestCache, resolveGuestRuntime } from '/app/dist/guest-image.js';

const exists = async (path) => access(path).then(() => true, () => false);
configureGuestCache('/runtime-home/.cache/gondolin');
const runtime = await resolveGuestRuntime('copper_inversion', JSON.parse(await readFile('/probe/runtime.json', 'utf8')));
const sibling = process.env.PROBE_SIBLING;  // host path of another branch's root and view
const runDir = process.env.PROBE_RUN_DIR;   // host path of the run directory
if (!sibling || !runDir) throw new Error('probe paths missing');

// Sidecar level: the narrow mount and per-branch roots (gap audit §3.9, plan §5).
const sidecar = {
  rsi_resources_present: await exists('/resources/rsi/AGENTS.md') && await exists('/resources/rsi/rsi.mjs'),
  old_resources_absent: !(await exists('/resources/v2')) && !(await exists('/resources/harness')),
  sibling_root_absent: !(await exists(sibling)) && !(await exists(runDir)),
};
if (!Object.values(sidecar).every(Boolean)) throw new Error('sidecar mounts: ' + JSON.stringify(sidecar));

const workspace = '/artifacts/sessions/researcher_0/workspace';
await mkdir(workspace, {recursive:true});
await mkdir('/artifacts/sessions/researcher_0/turns/round_001/provider', {recursive:true});
await writeFile('/artifacts/sessions/researcher_0/turns/round_001/provider/round_001-provider-1.request.bin',
  'own-request-canary');
const controller = new GondolinController(workspace, workspace + '/.ldm-resources',
  {allowedHosts:[], deniedHosts:['*']}, runtime);
try {
  const tools = await controller.toolOptions();
  const denied = {};
  for (const target of ['/artifacts/public/evidence/history.txt', '../turns/round_001/provider/round_001-provider-1.request.bin',
                        '/artifacts/sessions/researcher_0/turns/round_001/provider/round_001-provider-1.request.bin',
                        sibling + '/public/evidence/history.txt', '/resources/rsi/AGENTS.md']) {
    try { await tools.read.operations.readFile(target); denied[target] = false; }
    catch { denied[target] = true; }
  }
  if (!Object.values(denied).every(Boolean)) throw new Error('read tool escaped: ' + JSON.stringify(denied));
  const code = `import json, os, pathlib, urllib.request
targets = ['/artifacts', '/artifacts/public', '/artifacts/turns', '/artifacts/sessions',
           '/workspace/../turns', '/workspace/../../researcher_0', '/resources', '/probe',
           ${JSON.stringify(sibling)}, ${JSON.stringify(sibling + '/public/evidence/history.txt')},
           ${JSON.stringify(runDir)}, '/truth/truth.npy', '/var/run/docker.sock',
           '/Aidata/shared/xingyulu/rsi4geo/private']
for target in targets:
    p = pathlib.Path(target)
    if p.exists(): raise AssertionError('unexpected host, sibling or own-root path: ' + target)
    try: p.read_bytes()
    except OSError: pass
    else: raise AssertionError('path readable: ' + target)
listed = sorted(os.listdir('/workspace'))
assert 'LLM_API_KEY' not in os.environ and 'SSH_AUTH_SOCK' not in os.environ
try:
    urllib.request.urlopen('https://example.com', timeout=5).read(1)
except Exception: pass
else: raise AssertionError('guest network request succeeded')
print(json.dumps({'status':'passed','own_root_paths_absent':True,'sibling_paths_absent':True,
                  'private_paths_absent':True,'host_credentials_absent':True,
                  'http_egress_probe_failed':True,'workspace_entries':listed}))`;
  await tools.write.operations.writeFile('rsi-probe.py', code);
  let output = '';
  const result = await tools.bash.operations.exec('python /workspace/rsi-probe.py', '/workspace',
    {timeout:30, onData:chunk => { output += chunk.toString(); }});
  if (result.exitCode !== 0) throw new Error(`guest probe failed: ${output}`);
  const evidence = {...JSON.parse(output.trim()), sidecar, read_tool_escape_denied:denied,
    scope:'actual guest path, credential and network probes; not an exhaustive sandbox audit'};
  await writeFile('/artifacts/probe-result.json', JSON.stringify(evidence, null, 2));
  process.stdout.write(JSON.stringify(evidence) + '\n');
} finally { await controller.close(); }
