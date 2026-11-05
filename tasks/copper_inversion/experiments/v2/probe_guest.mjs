// Run inside the pinned Pi sidecar; exercises its actual GondolinController.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { GondolinController } from '/app/dist/gondolin.js';
import { configureGuestCache, resolveGuestRuntime } from '/app/dist/guest-image.js';

configureGuestCache('/runtime-home/.cache/gondolin');
const config=JSON.parse(await readFile('/probe/runtime.json','utf8'));
const runtime=await resolveGuestRuntime('copper_inversion',config);
await mkdir('/artifacts/workspace',{recursive:true});
const controller=new GondolinController('/artifacts/workspace','/probe/resources',
    {allowedHosts:[],deniedHosts:['*']},runtime);
try {
    const tools=await controller.toolOptions();
    let denied=false;
    try { await tools.read.operations.readFile('/etc/hostname'); }
    catch { denied=true; }
    if (!denied) throw new Error('read tool escaped workspace');
    const source=`import json, os, pathlib, socket, urllib.request
r=pathlib.Path('/workspace/.ldm-resources')
assert (r/'canary.txt').read_text()=='readonly-canary'
denied=[]
for target in (r/'canary.txt', r/'new-file'):
    try: target.write_text('attempt')
    except OSError: denied.append(str(target))
    else: raise AssertionError('readonly resource write succeeded')
link=pathlib.Path('/workspace/resource-link')
link.symlink_to(r/'canary.txt')
try: link.write_text('attempt')
except OSError: denied.append('symlink')
else: raise AssertionError('symlink bypassed readonly mount')
assert 'LLM_API_KEY' not in os.environ
assert 'SSH_AUTH_SOCK' not in os.environ
for path in ('/truth/truth.npy','/var/run/docker.sock','/Aidata/shared/ldm/copper-v2-evaluator/datasets/access-canary.txt','/Aidata/shared/ldm/copper-v2-rpc-r2/evaluator.sock','/root/.ssh/id_rsa'):
    assert not pathlib.Path(path).exists(), 'unexpected private path'
    try:
        with open(path,'rb') as f: f.read(1)
    except OSError: pass
    else: raise AssertionError('private file read succeeded')
s=socket.socket(socket.AF_UNIX)
try: s.connect('/Aidata/shared/ldm/copper-v2-rpc-r2/evaluator.sock')
except OSError: pass
else: raise AssertionError('private service accessible')
finally: s.close()
try:
    with urllib.request.urlopen('https://example.com',timeout=5) as response: response.read(1)
except Exception: pass
else: raise AssertionError('guest network request succeeded')
print(json.dumps(dict(status='passed',readonly_write_denials=len(denied),private_paths_absent=True,host_credentials_absent=True,private_socket_inaccessible=True,http_egress_probe_failed=True)))
`;
    await tools.write.operations.writeFile('probe.py',source);
    let output='';
    const result=await tools.bash.operations.exec('python /workspace/probe.py','/workspace',
        {timeout:30,onData:chunk=>{output+=chunk.toString();}});
    if (result.exitCode!==0) throw new Error(`guest probe failed: ${output}`);
    const evidence={...JSON.parse(output.trim()),read_tool_escape_denied:denied,
        runtime:config,scope:'Actual GondolinController guest filesystem and environment probes; not an exhaustive sandbox escape or network audit'};
    await writeFile('/artifacts/result.json',JSON.stringify(evidence,null,2));
    process.stdout.write(JSON.stringify(evidence)+'\n');
} finally { await controller.close(); }
