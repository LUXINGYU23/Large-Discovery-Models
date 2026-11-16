// Run inside the rebuilt sidecar against the real Gondolin guest.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { GondolinController } from '/app/dist/gondolin.js';
import { configureGuestCache, resolveGuestRuntime } from '/app/dist/guest-image.js';

configureGuestCache('/runtime-home/.cache/gondolin');
const runtime = await resolveGuestRuntime('copper_inversion', JSON.parse(await readFile('/probe/runtime.json', 'utf8')));
await mkdir('/artifacts/workspace', {recursive:true});
await mkdir('/artifacts/other-branch', {recursive:true});
await writeFile('/artifacts/other-branch/private.txt', 'sibling-private-canary');
const controller = new GondolinController('/artifacts/workspace', '/probe/resources',
  {allowedHosts:[], deniedHosts:['*']}, runtime);
try {
  const tools = await controller.toolOptions();
  let readDenied = false;
  try { await tools.read.operations.readFile('/artifacts/other-branch/private.txt'); }
  catch { readDenied = true; }
  if (!readDenied) throw new Error('read tool reached a sibling branch');
  const code = `import json, os, pathlib, urllib.request
for target in ('/artifacts', '/artifacts/other-branch/private.txt', '/truth/truth.npy',
               '/var/run/docker.sock', '/Aidata/shared/xingyulu/rsi4geo/private'):
    p = pathlib.Path(target)
    if p.exists(): raise AssertionError('unexpected host or sibling path: ' + target)
    try: p.read_bytes()
    except OSError: pass
    else: raise AssertionError('host or sibling path readable: ' + target)
assert 'LLM_API_KEY' not in os.environ and 'SSH_AUTH_SOCK' not in os.environ
try:
    urllib.request.urlopen('https://example.com', timeout=5).read(1)
except Exception: pass
else: raise AssertionError('guest network request succeeded')
print(json.dumps({'status':'passed','sibling_paths_absent':True,'private_paths_absent':True,
                  'host_credentials_absent':True,'http_egress_probe_failed':True}))`;
  await tools.write.operations.writeFile('rsi-probe.py', code);
  let output = '';
  const result = await tools.bash.operations.exec('python /workspace/rsi-probe.py', '/workspace',
    {timeout:30, onData:chunk => { output += chunk.toString(); }});
  if (result.exitCode !== 0) throw new Error(`guest probe failed: ${output}`);
  const evidence = {...JSON.parse(output.trim()), read_tool_escape_denied:readDenied,
    scope:'actual guest path, credential and network probes; not an exhaustive sandbox audit'};
  await writeFile('/artifacts/result.json', JSON.stringify(evidence, null, 2));
  process.stdout.write(JSON.stringify(evidence) + '\n');
} finally { await controller.close(); }
